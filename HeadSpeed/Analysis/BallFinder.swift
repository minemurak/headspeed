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

    // MARK: Rear view
    // From behind, the ball is often backlit and darker than the mat, and it usually sits on a
    // tee. Mirrors search_rear / refine_rear in reference/ballfinder.py.

    /// Round blob of either polarity in the guide; several opening widths cut it loose from a
    /// tee. Picks the candidate nearest `anchor` (x, y in 0…1).
    static func searchRear(_ img: GrayImage, guide: (Double, Double, Double, Double),
                           minD: Double, maxD: Double, anchor: (Double, Double)) -> BallFix? {
        let W = img.width, H = img.height
        let gx0 = max(0, Int(guide.0 * Double(W))), gx1 = min(W, Int(guide.2 * Double(W)))
        let gy0 = max(0, Int(guide.1 * Double(H))), gy1 = min(H, Int(guide.3 * Double(H)))
        guard gx1 - gx0 > 8, gy1 - gy0 > 8 else { return nil }
        let loc = boxMean(img, radius: max(4, Int(maxD * 0.7)))
        var best: (fix: BallFix, dist: Double)?
        for pol in [1.0, -1.0] {
            var raw = [Bool](repeating: false, count: W * H)
            for y in gy0..<gy1 {
                for x in gx0..<gx1 where pol * (Double(img.pixels[y * W + x]) - loc[y * W + x]) >= 25 {
                    raw[y * W + x] = true
                }
            }
            for k in [2, 3, 5, 7] {
                let lo = max(minD, Double(k) / 0.6), hi = min(maxD, max(Double(k) / 0.3, 6))
                if lo > hi { continue }
                for b in Blobs.find(mask: openH(raw, width: W, height: H, k: k), width: W, height: H) {
                    let bw = Double(b.boxW), bh = Double(b.boxH)
                    guard bw >= lo, bw <= hi, bh >= 0.7 * bw, bh <= 1.3 * bw, b.fill >= 0.6 else { continue }
                    let d = (bw + bh) / 2
                    let contrast = pol * (meanIn(img, cx: b.cx, cy: b.cy, r0: 0, r1: d * 0.3)
                                          - meanIn(img, cx: b.cx, cy: b.cy, r0: d * 0.8, r1: d * 1.6))
                    guard contrast >= 35 else { continue }
                    let dist = hypot(b.cx / Double(W) - anchor.0, b.cy / Double(H) - anchor.1)
                    if best == nil || dist < best!.dist { best = (BallFix(x: b.cx, y: b.cy, d: d), dist) }
                }
            }
        }
        return best?.fix
    }

    /// Sub-pixel circle from the ball's upper edge (a tee or the ground can touch the lower
    /// half): rays from the approximate centre, threshold crossings, least-squares circle.
    static func refineRear(_ img: GrayImage, approx: BallFix) -> BallFix? {
        let W = img.width, H = img.height
        let d0 = approx.d
        let inside = meanIn(img, cx: approx.x, cy: approx.y, r0: 0, r1: d0 * 0.25)
        let ring = meanIn(img, cx: approx.x, cy: approx.y, r0: d0 * 0.9, r1: d0 * 1.4)
        guard abs(inside - ring) >= 30 else { return nil }
        let thr = (inside + ring) / 2, pol = inside > ring ? 1.0 : -1.0
        func at(_ x: Double, _ y: Double) -> Double? {
            let x0 = Int(x.rounded(.down)), y0 = Int(y.rounded(.down))
            guard x0 >= 0, y0 >= 0, x0 + 1 < W, y0 + 1 < H else { return nil }
            let fx = x - Double(x0), fy = y - Double(y0)
            func p(_ xx: Int, _ yy: Int) -> Double { Double(img.pixels[yy * W + xx]) }
            return (1 - fx) * (1 - fy) * p(x0, y0) + fx * (1 - fy) * p(x0 + 1, y0)
                + (1 - fx) * fy * p(x0, y0 + 1) + fx * fy * p(x0 + 1, y0 + 1)
        }
        var pts: [(Double, Double)] = []
        for deg in stride(from: -170.0, through: -10.0, by: 8) {      // upper half (y grows downward)
            let a = deg * .pi / 180, ux = cos(a), uy = sin(a)
            var prev: (r: Double, s: Double)?
            var r = 0.15 * d0
            while r < d0 {
                guard let v = at(approx.x + ux * r, approx.y + uy * r) else { break }
                let s = pol * (v - thr)
                if let pr = prev, pr.s > 0, s <= 0 {
                    let rr = pr.r + (r - pr.r) * pr.s / (pr.s - s)
                    pts.append((approx.x + ux * rr, approx.y + uy * rr))
                    break
                }
                prev = (r, s)
                r += 0.25
            }
        }
        guard pts.count >= 10 else { return nil }
        // Kåsa fit: x² + y² = 2ax + 2by + c.
        var A = [[Double]](repeating: [0, 0, 0], count: 3), rhs = [0.0, 0.0, 0.0]
        for (x, y) in pts {
            let row = [2 * x, 2 * y, 1.0], z = x * x + y * y
            for i in 0..<3 {
                rhs[i] += row[i] * z
                for j in 0..<3 { A[i][j] += row[i] * row[j] }
            }
        }
        guard let sol = RearAnalyzer.solve(A, rhs) else { return nil }
        let cx = sol[0], cy = sol[1]
        let r2 = sol[2] + cx * cx + cy * cy
        guard r2 > 0 else { return nil }
        let rad = r2.squareRoot()
        let res = pts.map { abs(hypot($0.0 - cx, $0.1 - cy) - rad) }.sorted()
        let medRes = res.count % 2 == 1 ? res[res.count / 2] : (res[res.count / 2 - 1] + res[res.count / 2]) / 2
        guard medRes <= 0.06 * rad + 0.3, 2 * rad > 0.7 * d0, 2 * rad < 1.3 * d0 else { return nil }
        return BallFix(x: cx, y: cy, d: 2 * rad)
    }

    /// Mean over a (2r+1)² box, clamped at the borders.
    private static func boxMean(_ img: GrayImage, radius r: Int) -> [Double] {
        let W = img.width, H = img.height
        var I = [Double](repeating: 0, count: (W + 1) * (H + 1))
        for y in 0..<H {
            var rowSum = 0.0
            for x in 0..<W {
                rowSum += Double(img.pixels[y * W + x])
                I[(y + 1) * (W + 1) + x + 1] = I[y * (W + 1) + x + 1] + rowSum
            }
        }
        var out = [Double](repeating: 0, count: W * H)
        for y in 0..<H {
            let y0 = max(0, y - r), y1 = min(H, y + r + 1)
            for x in 0..<W {
                let x0 = max(0, x - r), x1 = min(W, x + r + 1)
                let s = I[y1 * (W + 1) + x1] - I[y0 * (W + 1) + x1] - I[y1 * (W + 1) + x0] + I[y0 * (W + 1) + x0]
                out[y * W + x] = s / Double((y1 - y0) * (x1 - x0))
            }
        }
        return out
    }

    /// Opening with a horizontal segment of k pixels: removes anything narrower than k (a tee
    /// under the ball, thin lines) and keeps wider shapes.
    private static func openH(_ mask: [Bool], width W: Int, height H: Int, k: Int) -> [Bool] {
        guard k > 1, W >= k else { return mask }
        var out = [Bool](repeating: false, count: W * H)
        for y in 0..<H {
            var run = 0
            for x in 0..<W {
                run = mask[y * W + x] ? run + 1 : 0
                // A run of k ending at x: the segment x-k+1...x is fully set.
                if run >= k {
                    if run == k {
                        for j in (x - k + 1)...x { out[y * W + j] = true }
                    } else {
                        out[y * W + x] = true
                    }
                }
            }
        }
        return out
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
