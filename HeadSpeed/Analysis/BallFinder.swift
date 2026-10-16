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
        searchAll(img, guide: guide, minD: minD, maxD: maxD).first
    }

    /// Every bright, round, compact blob above a percentile threshold, best (contrast × fill) first.
    static func searchAll(_ img: GrayImage, guide: (Double, Double, Double, Double),
                          minD: Double, maxD: Double) -> [BallFix] {
        let W = img.width, H = img.height
        let gx0 = max(0, Int(guide.0 * Double(W))), gx1 = min(W, Int(guide.2 * Double(W)))
        let gy0 = max(0, Int(guide.1 * Double(H))), gy1 = min(H, Int(guide.3 * Double(H)))
        guard gx1 - gx0 > 8, gy1 - gy0 > 8 else { return [] }

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

        var found: [(fix: BallFix, score: Double)] = []
        for b in Blobs.find(mask: mask, width: W, height: H) {
            let d = 2 * (Double(b.area) / .pi).squareRoot()
            guard d >= minD, d <= maxD else { continue }
            let aspect = Double(max(b.boxW, b.boxH)) / Double(min(b.boxW, b.boxH))
            guard aspect <= 1.4, b.fill >= 0.6 else { continue }
            let inside = meanIn(img, cx: b.cx, cy: b.cy, r0: 0, r1: d * 0.3)
            let ring = meanIn(img, cx: b.cx, cy: b.cy, r0: d * 0.8, r1: d * 1.6)
            let contrast = inside - ring
            guard contrast >= 40 else { continue }
            found.append((BallFix(x: b.cx, y: b.cy, d: d), contrast * b.fill))
        }
        return stableSorted(found) { $0.score > $1.score }.map { $0.fix }
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
    // From behind, the ball sits on a tee, often in front of a split background (dark field
    // above, bright path below) or backlit, so it is neither reliably brighter nor darker than
    // its surroundings. Its outline is still a circle of edges: look for that instead of a blob.
    // Mirrors search_rear / refine_rear in reference/ballfinder.py.

    /// Directions on the tee's side (image y grows downward) carry no outline.
    private static let teeSide = (50.0 * .pi / 180, 130.0 * .pi / 180)

    /// Circle Hough on edge directions (either polarity) inside the guide, on a half-size
    /// frame. Candidates must have edges all round (a square sticker or a tee fails); among
    /// them, the one nearest `anchor` (x, y in 0…1) wins; specks near the minimum size count half.
    static func searchRear(_ img: GrayImage, guide: (Double, Double, Double, Double),
                           minD: Double, maxD: Double, anchor: (Double, Double),
                           edgeMin: Double = 10) -> BallFix? {
        searchRearAll(img, guide: guide, minD: minD, maxD: maxD, anchor: anchor, edgeMin: edgeMin).first
    }

    /// Every circle that passes the outline check, best (by the key above) first.
    static func searchRearAll(_ img: GrayImage, guide: (Double, Double, Double, Double),
                              minD: Double, maxD: Double, anchor: (Double, Double),
                              edgeMin: Double = 10) -> [BallFix] {
        let W = img.width, H = img.height
        let gx0 = max(0, Int(guide.0 * Double(W))), gx1 = min(W, Int(guide.2 * Double(W)))
        let gy0 = max(0, Int(guide.1 * Double(H))), gy1 = min(H, Int(guide.3 * Double(H)))
        let gw = gx1 - gx0, gh = gy1 - gy0
        guard gw > 8, gh > 8 else { return [] }
        let (gx, gy) = sobel(img)
        var mag = [Double](repeating: 0, count: W * H)
        for i in 0..<(W * H) { mag[i] = hypot(gx[i], gy[i]) }

        var ex: [Int] = [], ey: [Int] = [], nx: [Double] = [], ny: [Double] = []
        for y in gy0..<gy1 {
            for x in gx0..<gx1 where mag[y * W + x] >= edgeMin {
                let m = mag[y * W + x]
                ex.append(x); ey.append(y); nx.append(gx[y * W + x] / m); ny.append(gy[y * W + x] / m)
            }
        }
        var cands: [(score: Double, x: Int, y: Int, r: Double)] = []
        var acc = [Double](repeating: 0, count: gw * gh)
        var a3 = [Double](repeating: 0, count: gw * gh)
        var r = max(2.0, minD / 2)
        while r <= maxD / 2 + 0.01 {
            for i in 0..<acc.count { acc[i] = 0; a3[i] = 0 }
            for k in 0..<ex.count {
                for s in [1.0, -1.0] {
                    // half to even, as the reference (r steps by halves, so exact halves are common)
                    let cx = Int((Double(ex[k]) + s * r * nx[k]).rounded(.toNearestOrEven)) - gx0
                    let cy = Int((Double(ey[k]) + s * r * ny[k]).rounded(.toNearestOrEven)) - gy0
                    if cx >= 0, cx < gw, cy >= 0, cy < gh { acc[cy * gw + cx] += 1 }
                }
            }
            // 3×3 sum: votes scatter by a pixel or so.
            for y in 1..<(gh - 1) {
                for x in 1..<(gw - 1) {
                    var t = 0.0
                    for dy in -1...1 { for dx in -1...1 { t += acc[(y + dy) * gw + x + dx] } }
                    a3[y * gw + x] = t
                }
            }
            let norm = 2 * Double.pi * r
            for y in 0..<gh {
                for x in 0..<gw {
                    let sc = a3[y * gw + x] / norm
                    guard sc >= 0.6 else { continue }
                    var isMax = true
                    for yy in max(0, y - 1)...min(gh - 1, y + 1) {
                        for xx in max(0, x - 1)...min(gw - 1, x + 1) where a3[yy * gw + xx] / norm > sc { isMax = false }
                    }
                    if isMax { cands.append((sc, x + gx0, y + gy0, r)) }
                }
            }
            r += 1
        }
        // as the reference's tuple sort: score, then x, y, r, all descending
        cands.sort { ($0.score, $0.x, $0.y, $0.r) > ($1.score, $1.x, $1.y, $1.r) }
        var found: [(fix: BallFix, key: Double)] = []
        for c in cands.prefix(60) {
            let cov = coverage(gx, gy, mag, width: W, height: H, cx: Double(c.x), cy: Double(c.y), r: c.r, edgeMin: edgeMin)
            guard cov >= 0.75 else { continue }
            let dist = hypot(Double(c.x) / Double(W) - anchor.0, Double(c.y) / Double(H) - anchor.1)
            found.append((BallFix(x: Double(c.x), y: Double(c.y), d: 2 * c.r), cov * min(1, c.r / minD) / (1 + 3 * dist)))
        }
        return stableSorted(found) { $0.key > $1.key }.map { $0.fix }
    }

    /// Share of directions (tee side skipped) with a radial edge on the circle.
    private static func coverage(_ gx: [Double], _ gy: [Double], _ mag: [Double], width W: Int, height H: Int,
                                 cx: Double, cy: Double, r: Double, edgeMin: Double) -> Double {
        var hit = 0, n = 0
        for deg in stride(from: 0.0, to: 360.0, by: 10) {
            let a = deg * .pi / 180
            if a >= teeSide.0 && a <= teeSide.1 { continue }
            let ux = cos(a), uy = sin(a)
            n += 1
            for dr in [0.0, -1, 1, -1.5, 1.5] {
                let x = Int((cx + ux * (r + dr)).rounded(.toNearestOrEven)), y = Int((cy + uy * (r + dr)).rounded(.toNearestOrEven))
                guard x >= 0, x < W, y >= 0, y < H else { continue }
                let i = y * W + x, m = mag[i]
                if m >= edgeMin && abs(gx[i] * ux + gy[i] * uy) >= 0.93 * m { hit += 1; break }
            }
        }
        return n > 0 ? Double(hit) / Double(n) : 0
    }

    /// Sobel gradients (borders clamped), scaled to grey levels per pixel.
    private static func sobel(_ img: GrayImage) -> ([Double], [Double]) {
        let W = img.width, H = img.height
        var gx = [Double](repeating: 0, count: W * H), gy = [Double](repeating: 0, count: W * H)
        func p(_ x: Int, _ y: Int) -> Double { Double(img.pixels[min(H - 1, max(0, y)) * W + min(W - 1, max(0, x))]) }
        for y in 0..<H {
            for x in 0..<W {
                gx[y * W + x] = (p(x + 1, y - 1) + 2 * p(x + 1, y) + p(x + 1, y + 1)
                                 - p(x - 1, y - 1) - 2 * p(x - 1, y) - p(x - 1, y + 1)) / 8
                gy[y * W + x] = (p(x - 1, y + 1) + 2 * p(x, y + 1) + p(x + 1, y + 1)
                                 - p(x - 1, y - 1) - 2 * p(x, y - 1) - p(x + 1, y - 1)) / 8
            }
        }
        return (gx, gy)
    }

    /// Sub-pixel circle: on rays from the approximate centre (tee side skipped), the strongest
    /// radial edge with a parabolic peak, then a least-squares circle with outlier rejection.
    /// `strong`: keep only rays whose edge is at least this share of the strongest one.
    static func refineRear(_ img: GrayImage, approx: BallFix, strong: Double? = nil, minPts: Int = 12) -> BallFix? {
        let W = img.width, H = img.height
        let d0 = approx.d
        func at(_ x: Double, _ y: Double) -> Double? {
            let x0 = Int(x.rounded(.down)), y0 = Int(y.rounded(.down))
            guard x0 >= 0, y0 >= 0, x0 + 1 < W, y0 + 1 < H else { return nil }
            let fx = x - Double(x0), fy = y - Double(y0)
            func p(_ xx: Int, _ yy: Int) -> Double { Double(img.pixels[yy * W + xx]) }
            return (1 - fx) * (1 - fy) * p(x0, y0) + fx * (1 - fy) * p(x0 + 1, y0)
                + (1 - fx) * fy * p(x0, y0 + 1) + fx * fy * p(x0 + 1, y0 + 1)
        }
        var rs: [Double] = []
        var rr0 = 0.2 * d0
        while rr0 < 0.85 * d0 { rs.append(rr0); rr0 += 0.5 }
        guard rs.count >= 3 else { return nil }
        var pts: [(Double, Double)] = []
        var strength: [Double] = []
        rays: for deg in stride(from: 0.0, to: 360.0, by: 6) {
            let a = deg * .pi / 180
            if a >= teeSide.0 && a <= teeSide.1 { continue }
            let ux = cos(a), uy = sin(a)
            var v: [Double] = []
            for r in rs {
                guard let q = at(approx.x + ux * r, approx.y + uy * r) else { continue rays }
                v.append(q)
            }
            let n = v.count
            var g = [Double](repeating: 0, count: n)
            g[0] = abs(v[1] - v[0]) / 0.5
            g[n - 1] = abs(v[n - 1] - v[n - 2]) / 0.5
            for i in 1..<(n - 1) { g[i] = abs(v[i + 1] - v[i - 1]) }
            var i = 0
            for j in 1..<n where g[j] > g[i] { i = j }
            guard g[i] >= 6, i > 0, i < n - 1 else { continue }
            let den = g[i - 1] - 2 * g[i] + g[i + 1]
            let off = den < 0 ? 0.5 * (g[i - 1] - g[i + 1]) / den : 0
            let rr = rs[i] + 0.5 * off
            pts.append((approx.x + ux * rr, approx.y + uy * rr))
            strength.append(g[i])
        }
        if let strong, let gmax = strength.max() {
            pts = zip(pts, strength).filter { $0.1 >= strong * gmax }.map { $0.0 }
        }
        guard pts.count >= minPts else { return nil }
        var keep = [Bool](repeating: true, count: pts.count)
        var cx = 0.0, cy = 0.0, rad = 0.0
        var res = [Double](repeating: 0, count: pts.count)
        for _ in 0..<3 {
            // Kåsa fit: x² + y² = 2ax + 2by + c.
            var A = [[Double]](repeating: [0, 0, 0], count: 3), rhs = [0.0, 0.0, 0.0]
            for (k, (x, y)) in pts.enumerated() where keep[k] {
                let row = [2 * x, 2 * y, 1.0], z = x * x + y * y
                for i in 0..<3 {
                    rhs[i] += row[i] * z
                    for j in 0..<3 { A[i][j] += row[i] * row[j] }
                }
            }
            guard let sol = RearAnalyzer.solve(A, rhs) else { return nil }
            cx = sol[0]; cy = sol[1]
            rad = max(1e-9, sol[2] + cx * cx + cy * cy).squareRoot()
            res = pts.map { abs(hypot($0.0 - cx, $0.1 - cy) - rad) }
            let lim = max(0.6, 2.5 * median(zip(res, keep).filter { $0.1 }.map { $0.0 }))
            keep = res.map { $0 <= lim }
            guard keep.filter({ $0 }).count >= minPts else { return nil }
        }
        let medRes = median(zip(res, keep).filter { $0.1 }.map { $0.0 })
        guard medRes <= 0.04 * rad + 0.3, 2 * rad > 0.7 * d0, 2 * rad < 1.3 * d0 else { return nil }
        return BallFix(x: cx, y: cy, d: 2 * rad)
    }

    // MARK: Face-on, partly lit
    // With the sun behind the ball, only its near-left part is lit; the rest is in the ball's
    // own shade, as dark as sunlit grass, so the bright blob is a crescent that fails the
    // roundness checks above. Mirrors search_lit / refine_lit in reference/ballfinder.py.

    /// Centre-surround brightness (box means on an integral image), so the lit part still
    /// stands out on average, then a shape check on the bright part so a white shoe toe or a
    /// leg does not pass. With `anchor` (a tapped point) the nearest strong candidate wins;
    /// without, the lowest strong one (face-on, the ball lies in front of the feet).
    static func searchLit(_ img: GrayImage, guide: (Double, Double, Double, Double),
                          minD: Double, maxD: Double, anchor: (Double, Double)?) -> BallFix? {
        searchLitAll(img, guide: guide, minD: minD, maxD: maxD, anchor: anchor).first
    }

    /// Every candidate that passes the shape check, in the order searchLit prefers them: with an
    /// anchor, by strength over distance; without, the strong ones (≥ 0.8 × the strongest)
    /// lowest first, then the rest by strength.
    static func searchLitAll(_ img: GrayImage, guide: (Double, Double, Double, Double),
                             minD: Double, maxD: Double, anchor: (Double, Double)?) -> [BallFix] {
        let W = img.width, H = img.height
        let gx0 = max(0, Int(guide.0 * Double(W))), gx1 = min(W, Int(guide.2 * Double(W)))
        let gy0 = max(0, Int(guide.1 * Double(H))), gy1 = min(H, Int(guide.3 * Double(H)))
        guard gx1 - gx0 > 8, gy1 - gy0 > 8 else { return [] }
        var I = [Double](repeating: 0, count: (W + 1) * (H + 1))
        for y in 0..<H {
            var row = 0.0
            for x in 0..<W {
                row += Double(img.pixels[y * W + x])
                I[(y + 1) * (W + 1) + x + 1] = I[y * (W + 1) + x + 1] + row
            }
        }
        func box(_ x: Int, _ y: Int, _ h: Int) -> (Double, Int) {
            let x0 = max(0, x - h), x1 = min(W, x + h + 1), y0 = max(0, y - h), y1 = min(H, y + h + 1)
            let s = I[y1 * (W + 1) + x1] - I[y0 * (W + 1) + x1] - I[y1 * (W + 1) + x0] + I[y0 * (W + 1) + x0]
            return (s, (x1 - x0) * (y1 - y0))
        }
        let gw = gx1 - gx0, gh = gy1 - gy0
        var cands: [(s: Double, x: Int, y: Int, r: Int)] = []
        var sc = [Double](repeating: -1e9, count: gw * gh)
        for r in max(2, Int(minD / 2))...max(2, Int(maxD / 2)) {
            let hi = max(1, Int((0.6 * Double(r)).rounded())), hr = Int((1.25 * Double(r)).rounded()), ho = 2 * r
            for y in gy0..<gy1 {
                for x in gx0..<gx1 {
                    let (si, ni) = box(x, y, hi), (sm, nm) = box(x, y, hr), (so, no) = box(x, y, ho)
                    sc[(y - gy0) * gw + (x - gx0)] = no > nm ? si / Double(ni) - (so - sm) / Double(no - nm) : -1e9
                }
            }
            for y in 0..<gh {
                for x in 0..<gw where sc[y * gw + x] >= 60 {
                    let v = sc[y * gw + x]
                    var peak = true
                    for yy in max(0, y - 1)...min(gh - 1, y + 1) {
                        for xx in max(0, x - 1)...min(gw - 1, x + 1) where sc[yy * gw + xx] > v { peak = false }
                    }
                    if peak { cands.append((v, x + gx0, y + gy0, r)) }
                }
            }
        }
        cands.sort { ($0.s, $0.x, $0.y, $0.r) > ($1.s, $1.x, $1.y, $1.r) }
        var valid: [(s: Double, fix: BallFix)] = []
        for c in cands.prefix(40) {
            let w = Int((2.5 * Double(c.r)).rounded(.up))
            let x0 = max(0, c.x - w), x1 = min(W, c.x + w + 1), y0 = max(0, c.y - w), y1 = min(H, c.y + w + 1)
            let ww = x1 - x0, wh = y1 - y0
            let inner = meanIn(img, cx: Double(c.x), cy: Double(c.y), r0: 0, r1: 0.6 * Double(c.r))
            let ring = meanIn(img, cx: Double(c.x), cy: Double(c.y), r0: 1.25 * Double(c.r), r1: 2.0 * Double(c.r))
            let thr = (inner + ring) / 2
            var mask = [Bool](repeating: false, count: ww * wh)
            for y in y0..<y1 { for x in x0..<x1 where Double(img.pixels[y * W + x]) >= thr { mask[(y - y0) * ww + (x - x0)] = true } }
            let px = c.x - x0, py = c.y - y0
            var hit: Blob?
            for b in Blobs.find(mask: mask, width: ww, height: wh)
            where b.minX <= px && px <= b.maxX && b.minY <= py && py <= b.maxY && (hit == nil || b.area > hit!.area) { hit = b }
            guard let b = hit else { continue }
            // compact and on its own: not running on to the window's edge (a shoe, a leg)
            if b.minX == 0 || b.minY == 0 || b.maxX == ww - 1 || b.maxY == wh - 1 { continue }
            let aspect = Double(max(b.boxW, b.boxH)) / Double(min(b.boxW, b.boxH))
            guard aspect <= 1.6, b.fill >= 0.4 else { continue }
            let d = Double(max(b.boxW, b.boxH))
            guard d >= minD, d <= maxD else { continue }
            valid.append((c.s, BallFix(x: Double(x0) + b.cx, y: Double(y0) + b.cy, d: d)))
        }
        guard let top = valid.map({ $0.s }).max() else { return [] }
        if let a = anchor {
            func key(_ v: (s: Double, fix: BallFix)) -> Double {
                v.s / (1 + 3 * hypot(v.fix.x / Double(W) - a.0, v.fix.y / Double(H) - a.1))
            }
            return stableSorted(valid) { key($0) > key($1) }.map { $0.fix }
        }
        let strong = stableSorted(valid.filter { $0.s >= 0.8 * top }) { $0.fix.y > $1.fix.y }
        let rest = stableSorted(valid.filter { $0.s < 0.8 * top }) { $0.s > $1.s }
        return (strong + rest).map { $0.fix }
    }

    /// Only the lit arc has a real outline (the shaded side's edge is lost in the grass
    /// texture): fit the circle to the strong radial edges alone, starting from a few
    /// diameters since the lit part only hints at the size.
    static func refineLit(_ img: GrayImage, approx: BallFix) -> BallFix? {
        for k in [1.0, 1.25, 0.8] {
            if let r = refineRear(img, approx: BallFix(x: approx.x, y: approx.y, d: k * approx.d), strong: 0.5, minPts: 8) {
                return r
            }
        }
        return nil
    }

    /// Sort that keeps equal elements in their original order (as Python's sort does).
    static func stableSorted<T>(_ v: [T], by less: (T, T) -> Bool) -> [T] {
        v.enumerated().sorted { a, b in
            if less(a.element, b.element) { return true }
            if less(b.element, a.element) { return false }
            return a.offset < b.offset
        }.map { $0.element }
    }

    private static func median(_ v: [Double]) -> Double {
        guard !v.isEmpty else { return 0 }
        let s = v.sorted(), n = s.count
        return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
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
