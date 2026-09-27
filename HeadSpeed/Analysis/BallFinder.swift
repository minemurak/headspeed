import Foundation

struct BallFix: Equatable {
    var x: Double      // centre, pixels of the image it was found in
    var y: Double
    var d: Double      // diameter, pixels
}

/// Finds a golf ball at rest: a bright, round, compact blob inside the guide area.
enum BallFinder {
    /// Coarse search on a downscaled frame. `guide` is (x0, y0, x1, y1) in 0…1.
    static func search(_ img: GrayImage, guide: (Double, Double, Double, Double),
                       minD: Double, maxD: Double) -> BallFix? {
        let W = img.width, H = img.height
        let gx0 = max(0, Int(guide.0 * Double(W))), gx1 = min(W, Int(guide.2 * Double(W)))
        let gy0 = max(0, Int(guide.1 * Double(H))), gy1 = min(H, Int(guide.3 * Double(H)))
        guard gx1 - gx0 > 8, gy1 - gy0 > 8 else { return nil }

        var hist = [Int](repeating: 0, count: 256)
        for y in gy0..<gy1 { for x in gx0..<gx1 { hist[Int(img.pixels[y * W + x])] += 1 } }
        let total = (gx1 - gx0) * (gy1 - gy0)
        var acc = 0, p995 = 255
        for v in stride(from: 255, through: 0, by: -1) {
            acc += hist[v]
            if Double(acc) >= Double(total) * 0.005 { p995 = v; break }
        }
        let thr = max(140, p995)

        var mask = [Bool](repeating: false, count: W * H)
        for y in gy0..<gy1 { for x in gx0..<gx1 where Int(img.pixels[y * W + x]) >= thr { mask[y * W + x] = true } }

        var best: (fix: BallFix, score: Double)?
        for b in Blobs.find(mask: mask, width: W, height: H) {
            let d = 2 * (Double(b.area) / .pi).squareRoot()
            guard d >= minD, d <= maxD else { continue }
            let aspect = Double(max(b.boxW, b.boxH)) / Double(min(b.boxW, b.boxH))
            guard aspect <= 1.4, b.fill >= 0.6 else { continue }
            let inside = meanIn(img, cx: b.cx, cy: b.cy, r0: 0, r1: d * 0.3)
            let ring = meanIn(img, cx: b.cx, cy: b.cy, r0: d * 0.8, r1: d * 1.6)
            let contrast = inside - ring
            guard contrast >= 40 else { continue }
            let score = contrast * b.fill
            if best == nil || score > best!.score { best = (BallFix(x: b.cx, y: b.cy, d: d), score) }
        }
        return best?.fix
    }

    /// Sub-pixel refinement on a full-resolution crop around an approximate fix.
    static func refine(_ img: GrayImage, approx: BallFix) -> BallFix? {
        let W = img.width, H = img.height
        let inside = meanIn(img, cx: approx.x, cy: approx.y, r0: 0, r1: approx.d * 0.3)
        let ring = meanIn(img, cx: approx.x, cy: approx.y, r0: approx.d * 0.9, r1: approx.d * 1.4)
        guard inside - ring >= 30 else { return nil }
        let thr = (inside + ring) / 2
        let lim2 = approx.d * approx.d
        var mask = [Bool](repeating: false, count: W * H)
        for y in 0..<H {
            for x in 0..<W {
                let dx = Double(x) - approx.x, dy = Double(y) - approx.y
                if dx * dx + dy * dy < lim2 && Double(img.pixels[y * W + x]) > thr { mask[y * W + x] = true }
            }
        }
        var best: Blob?
        for b in Blobs.find(mask: mask, width: W, height: H) where best == nil || b.area > best!.area { best = b }
        guard let b = best, b.area > 20 else { return nil }
        let aspect = Double(max(b.boxW, b.boxH)) / Double(min(b.boxW, b.boxH))
        guard aspect <= 1.3, b.fill >= 0.6 else { return nil }
        return BallFix(x: b.cx, y: b.cy, d: 2 * (Double(b.area) / .pi).squareRoot())
    }

    private static func meanIn(_ img: GrayImage, cx: Double, cy: Double, r0: Double, r1: Double) -> Double {
        let W = img.width, H = img.height
        let x0 = max(0, Int(cx - r1)), x1 = min(W - 1, Int(cx + r1) + 1)
        let y0 = max(0, Int(cy - r1)), y1 = min(H - 1, Int(cy + r1) + 1)
        guard x0 <= x1, y0 <= y1 else { return 0 }
        var sum = 0.0, cnt = 0
        for y in y0...y1 {
            for x in x0...x1 {
                let dx = Double(x) - cx, dy = Double(y) - cy
                let r2 = dx * dx + dy * dy
                if r2 >= r0 * r0 && r2 <= r1 * r1 { sum += Double(img.pixels[y * W + x]); cnt += 1 }
            }
        }
        return cnt > 0 ? sum / Double(cnt) : 0
    }
}
