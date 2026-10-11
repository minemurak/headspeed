import Foundation

/// Rear-view (down-the-line) analysis. Mirrors reference/analyzer_rear.py, which is
/// validated against synthetic 240fps rear-view swings (reference/test_rear.py).
///
/// From behind, the ball and the clubhead move mostly away from the camera, so everything
/// is placed in 3D: the ball's depth comes from its apparent size, and the head speed from
/// an arc fitted to the head's image track over the last frames before impact.
enum RearAnalyzer {
    static let ballM = 0.04267

    static func analyze(_ clip: SwingClip, club: Club) -> SwingMeasurement {
        var out = SwingMeasurement()
        let n = clip.frames.count
        guard n > 40, clip.times.count == n, (0..<n).contains(clip.triggerIndex), clip.focal > 0 else {
            out.notes.append("映像が短すぎます")
            return out
        }
        let W = clip.frames[0].width, H = clip.frames[0].height
        let bx = clip.ballX, by = clip.ballY, d = clip.ballD
        let trig = clip.triggerIndex
        let f = clip.focal, cx = clip.centerX, cy = clip.centerY
        let times = clip.times
        let p = min(max(clip.cameraPitch, -0.6), 1.2)
        let up = Vec3(0, -cos(p), -sin(p))           // world up in camera coordinates
        func ray(_ u: Double, _ v: Double) -> Vec3 { Vec3((u - cx) / f, (v - cy) / f, 1) }
        func px(_ k: Int, _ i: Int) -> Double { Double(clip.frames[k].pixels[i]) }

        // Background: per-pixel median of five frames well before impact.
        let bg: [Float] = {
            let src = [110, 95, 80, 65, 50].map { clip.frames[min(max(0, trig - $0), n - 1)].pixels }
            var out = [Float](repeating: 0, count: W * H)
            var v = [Int](repeating: 0, count: 5)
            for i in 0..<(W * H) {
                for j in 0..<5 { v[j] = Int(src[j][i]) }
                out[i] = Float(median5(&v))
            }
            return out
        }()
        func r2(_ x: Int, _ y: Int) -> Double {
            let dx = Double(x) - bx, dy = Double(y) - by
            return dx * dx + dy * dy
        }
        let restR2 = (0.75 * d) * (0.75 * d)
        func inRest(_ x: Int, _ y: Int) -> Bool { r2(x, y) < restR2 }

        // 1. Impact: first frame whose ball core changes (the head may cover the ball early).
        var coreIdx: [Int] = []
        for y in max(0, Int(by - d))..<min(H, Int(by + d) + 1) {
            for x in max(0, Int(bx - d))..<min(W, Int(bx + d) + 1) where r2(x, y) < (0.3 * d) * (0.3 * d) {
                coreIdx.append(y * W + x)
            }
        }
        guard !coreIdx.isEmpty else { out.notes.append("ボールの位置が不正です"); return out }
        var kImpOpt: Int?
        for k in max(1, trig - 24)..<min(n, trig + 12) {
            var sum = 0.0
            for i in coreIdx { sum += abs(px(k, i) - Double(bg[i])) }
            if sum / Double(coreIdx.count) > 18 { kImpOpt = k; break }
        }
        guard let kImp = kImpOpt else {
            out.notes.append("インパクトの瞬間を検出できませんでした")
            return out
        }
        let iBall = coreIdx.reduce(0.0) { $0 + Double(bg[$1]) } / Double(coreIdx.count)

        // 2. Ball diameter at rest from its coverage, against the spot after the ball has gone.
        var d0 = d
        do {
            let late = [24, 30, 36].map { clip.frames[min(n - 1, kImp + $0)].pixels }
            func g(_ i: Int) -> Double { median3(Double(late[0][i]), Double(late[1][i]), Double(late[2][i])) }
            let rWin = 0.5 * d + 2.5, rIn = 0.5 * d + 3.5, rOut = 0.5 * d + 7
            var win: [Int] = [], ring: [Double] = [], dens: [Double] = []
            for y in max(0, Int(by - rOut) - 1)..<min(H, Int(by + rOut) + 2) {
                for x in max(0, Int(bx - rOut) - 1)..<min(W, Int(bx + rOut) + 2) {
                    let q = r2(x, y), i = y * W + x
                    if q < rWin * rWin { win.append(i); dens.append(iBall - g(i)) }
                    else if q >= rIn * rIn && q < rOut * rOut { ring.append(Double(bg[i]) - g(i)) }
                }
            }
            // The ball may be brighter or (backlit) darker than what is behind it.
            if !win.isEmpty, abs(median(dens)) > 25 {
                // Not clipped at 0, so background noise around the ball cancels out.
                let off = ring.isEmpty ? 0 : median(ring)
                var s = 0.0
                for (j, i) in win.enumerated() {
                    s += min(max((Double(bg[i]) - g(i) - off) / (abs(dens[j]) < 1 ? 1 : dens[j]), -0.5), 1.2)
                }
                let est = 2 * (max(s, 0) / .pi).squareRoot()
                if est > 0.8 * d && est < 1.2 * d { d0 = est }
            }
        }
        let z0 = f * ballM / d0
        let b3 = ray(bx, by) * z0

        // 3. Ball in flight: compact blobs that look like the ball, sized by summed coverage.
        let a0 = Double.pi * d0 * d0 / 4
        let fr = n > 1 ? median((1..<n).map { times[$0] - times[$0 - 1] }) : 1.0 / 240
        func den(_ i: Int) -> Double { iBall - Double(bg[i]) }
        let rx0 = max(0, Int(bx - 15 * d0)), rx1 = min(W, Int(bx + 15 * d0))
        let ry0 = max(0, Int(by - 25 * d0)), ry1 = min(H, Int(by + 1.5 * d0))
        var bpts: [(t: Double, u: Double, v: Double, d: Double, ds: Double)] = []
        if rx0 < rx1 && ry0 < ry1 {
            let rw = rx1 - rx0, rh = ry1 - ry0
            for k in kImp..<min(n, kImp + 16) {
                var mask = [Bool](repeating: false, count: rw * rh)
                for y in ry0..<ry1 {
                    for x in rx0..<rx1 where !inRest(x, y) {
                        let i = y * W + x
                        let dn = den(i)
                        if abs(dn) > 20 && (px(k, i) - Double(bg[i])) / dn > 0.5 { mask[(y - ry0) * rw + (x - rx0)] = true }
                    }
                }
                var cands: [(u: Double, v: Double, d: Double, ds: Double)] = []
                for b in Blobs.find(mask: mask, width: rw, height: rh) {
                    let area = Double(b.area)
                    guard area >= 0.08 * a0, area <= 2.2 * a0 else { continue }
                    guard Double(max(b.boxW, b.boxH)) <= 2.6 * d0, b.fill >= 0.4 else { continue }
                    // Coverage over the blob plus a margin, after removing the frame's overall
                    // brightness offset (flicker), measured on a ring just outside the margin.
                    let x0 = max(0, b.minX + rx0 - 3), x1 = min(W, b.maxX + rx0 + 4)
                    let y0 = max(0, b.minY + ry0 - 3), y1 = min(H, b.maxY + ry0 + 4)
                    let X0 = max(0, x0 - 3), X1 = min(W, x1 + 3), Y0 = max(0, y0 - 3), Y1 = min(H, y1 + 3)
                    var ring: [Double] = []
                    for y in Y0..<Y1 {
                        for x in X0..<X1 where !(x >= x0 && x < x1 && y >= y0 && y < y1) {
                            let i = y * W + x
                            if abs(den(i)) > 20 && !inRest(x, y) { ring.append(px(k, i) - Double(bg[i])) }
                        }
                    }
                    let off = ring.count > 8 ? median(ring) : 0
                    var s = 0.0, sp = 0.0, su = 0.0, sv = 0.0, cmax = -1.0
                    var absDen: [Double] = []
                    for y in y0..<y1 {
                        for x in x0..<x1 {
                            let i = y * W + x
                            let dn = den(i)
                            absDen.append(abs(dn) > 20 ? abs(dn) : 1)
                            var c = 0.0
                            if abs(dn) > 20 && !inRest(x, y) {
                                c = min(max((px(k, i) - Double(bg[i]) - off) / dn, -0.5), 1.2)
                            }
                            s += c
                            cmax = max(cmax, c)
                            if c > 0 { sp += c; su += c * Double(x); sv += c * Double(y) }
                        }
                    }
                    guard s > 0, sp > 0 else { continue }
                    let dk = 2 * (s / .pi).squareRoot()
                    if dk > 1.1 * d0 { continue }
                    let u = su / sp, v = sv / sp
                    // Must not touch the resting spot (coverage there is unreliable).
                    let lim = 0.75 * d0 + dk / 2 + 1
                    if (u - bx) * (u - bx) + (v - by) * (v - by) < lim * lim { continue }
                    if cmax < 0.7 { continue }          // not as bright (or dark) as the ball
                    // Size noise grows as the ball's contrast with what is behind it drops.
                    cands.append((u, v, dk, max(1.0, 100 / median(absDen))))
                }
                // Follow the ball outward from its resting spot.
                let ref = bpts.last.map { ($0.u, $0.v) } ?? (bx, by)
                let gap = bpts.last.map { max(1.0, (times[k] - $0.t) / fr) } ?? 1
                let lim = bpts.isEmpty ? 6 * d0 : 5 * d0 * gap
                var best: (dist: Double, u: Double, v: Double, d: Double, ds: Double)?
                for c in cands {
                    let dist = hypot(c.u - ref.0, c.v - ref.1)
                    if dist > lim { continue }
                    if best == nil || dist < best!.dist { best = (dist, c.u, c.v, c.d, c.ds) }
                }
                if let b = best { bpts.append((times[k], b.u, b.v, b.d, b.ds)) }
            }
        }

        // Straight 3D line from the resting ball, fitted in image space: the rest position slides
        // along its ray (its apparent size at rest is only a soft hint for its depth), plus
        // velocity and launch time. Frames that do not fit are dropped one at a time (a ball
        // still partly hidden by the head looks too small, i.e. too far).
        let ray0 = ray(bx, by)
        func fitLine(_ pts: [(t: Double, u: Double, v: Double, d: Double, ds: Double)]) -> (v: Vec3, t0: Double, errs: [Double])? {
            let P = pts.map { ray($0.u, $0.v) * (f * ballM / $0.d) }
            // Depth error grows as the ball shrinks.
            let w = pts.map { ($0.d / d0) * ($0.d / d0) / ($0.ds * $0.ds) }
            var best: (e: Double, v: Vec3, t0: Double)?
            // Start: launch after the ball core first changed (minus slack), before the ball was seen.
            var tt0 = times[kImp] - 2 * fr
            while tt0 < pts[0].t {
                var num = Vec3(0, 0, 0), wsum = 0.0
                for j in 0..<pts.count {
                    let dt = pts[j].t - tt0
                    num = num + (P[j] - b3) * (w[j] * dt)
                    wsum += w[j] * dt * dt
                }
                if wsum > 0 {
                    let vc = num * (1 / wsum)
                    var e = 0.0
                    for j in 0..<pts.count {
                        let r = P[j] - b3 - vc * (pts[j].t - tt0)
                        e += w[j] * r.dot(r)
                    }
                    if best == nil || e < best!.e { best = (e, vc, tt0) }
                }
                tt0 += fr / 20
            }
            guard let start = best else { return nil }
            let resid: ([Double]) -> [Double]? = { q in
                let vq = Vec3(q[2], q[3], q[4])
                var r: [Double] = []
                for c in pts {
                    let P = ray0 * q[0] + vq * (c.t - q[1])
                    if P.z < 0.3 { return nil }
                    r += [(f * P.x / P.z + cx - c.u) / 0.3, (f * P.y / P.z + cy - c.v) / 0.3,
                          (f * ballM / P.z - c.d) / (0.12 * c.ds)]
                }
                r.append((q[0] - z0) / (0.015 * z0))
                return r
            }
            guard let fit = levenbergMarquardt(resid, [z0, start.t0, start.v.x, start.v.y, start.v.z],
                                               h: [1e-4, 1e-6, 1e-3, 1e-3, 1e-3]) else { return nil }
            let p = fit.p
            let vfit = Vec3(p[2], p[3], p[4])
            let errs = pts.map { q -> Double in
                let pq = ray0 * p[0] + vfit * (q.t - p[1])
                let dq = f * ballM / pq.z
                let uq = f * pq.x / pq.z + cx, vq = f * pq.y / pq.z + cy
                return abs(dq - q.d) / (0.3 * q.ds) + hypot(uq - q.u, vq - q.v) / 0.7
            }
            return (vfit, p[1], errs)
        }
        var cand = Array(bpts.filter { $0.d <= 1.05 * d0 }.prefix(10))
        var vel: Vec3?
        var t0 = times[kImp]
        while cand.count >= 3 {
            guard let fl = fitLine(cand) else { break }
            let i = fl.errs.indices.max { fl.errs[$0] < fl.errs[$1] }!
            if fl.errs[i] > max(2.5, 2.5 * median(fl.errs)) && cand.count > 3 { cand.remove(at: i); continue }
            if fl.errs[i] <= 6 { vel = fl.v; t0 = fl.t0 }
            break
        }
        if let v = vel {
            let speed = v.norm
            out.ballSpeed = speed
            out.launch = asin(max(-1, min(1, v.dot(up) / speed))) * 180 / .pi
        } else {
            out.notes.append("打ち出したボールを追跡できませんでした")
        }
        out.impactTime = t0

        // 4. Club path at contact: the horizontal part of the ball's direction.
        let th = vel.map { $0 - up * $0.dot(up) } ?? (Vec3(0, 0, 1) - up * up.z)
        let tHat = th * (1 / th.norm)
        let hc = b3 - tHat * 0.035

        // 5. Head: track backwards from contact against a per-frame moving background.
        let kFirst = (times.firstIndex { $0 >= t0 } ?? n) - 1
        var prev: [(t: Double, u: Double, v: Double)] = []
        var zp = z0
        var k = kFirst
        while k > max(12, kFirst - 9) {
            let rk = max(3.0, f * 0.06 / zp)          // about the clubhead's size
            var pu = bx, pv = by, sr = 3 * rk
            if prev.count == 1 {
                pu = prev[0].u; pv = prev[0].v; sr = 4 * rk
            } else if prev.count >= 2 {
                let a = prev[prev.count - 1], b = prev[prev.count - 2]
                // Going back in time the head moves faster across the image.
                pu = a.u + (a.u - b.u) * 1.3; pv = a.v + (a.v - b.v) * 1.3
                sr = max(2.5 * rk, 0.6 * hypot(a.u - b.u, a.v - b.v))
            }
            let x0 = Int(max(0, pu - sr)), x1 = Int(min(Double(W), pu + sr + 1))
            let y0 = Int(max(0, pv - sr)), y1 = Int(min(Double(H), pv + sr + 1))
            guard x0 < x1, y0 < y1 else { break }
            let f0 = clip.frames[k - 12].pixels, f1 = clip.frames[k - 9].pixels, f2 = clip.frames[k - 6].pixels
            var xs: [Double] = [], ys: [Double] = []
            for y in y0..<y1 {
                for x in x0..<x1 {
                    let i = y * W + x
                    // The resting ball is in the moving background too, so it drops out.
                    let m = median3(Double(f0[i]), Double(f1[i]), Double(f2[i]))
                    if abs(px(k, i) - m) > 28 { xs.append(Double(x)); ys.append(Double(y)) }
                }
            }
            guard xs.count >= 3 else { break }
            // Seed on the densest head-sized cluster, then mean-shift onto it.
            let step = max(1, xs.count / 400)
            var bestSeed = 0, bestCnt = -1
            for s in stride(from: 0, to: xs.count, by: step) {
                var c = 0
                for j in 0..<xs.count {
                    let dx = xs[j] - xs[s], dy = ys[j] - ys[s]
                    if dx * dx + dy * dy < rk * rk { c += 1 }
                }
                if c > bestCnt { bestCnt = c; bestSeed = s }
            }
            if Double(bestCnt) < 0.25 * .pi * rk * rk { break }     // nothing head-sized
            var u = xs[bestSeed], v = ys[bestSeed]
            for _ in 0..<6 {
                var su = 0.0, sv = 0.0, c = 0
                for j in 0..<xs.count {
                    let dx = xs[j] - u, dy = ys[j] - v
                    if dx * dx + dy * dy < rk * rk { su += xs[j]; sv += ys[j]; c += 1 }
                }
                if c == 0 { break }
                u = su / Double(c); v = sv / Double(c)
            }
            prev.append((times[k], u, v))
            // Rough depth for the next kernel size: the head comes toward the camera.
            zp = max(0.4, z0 - 0.15 * Double(prev.count))
            k -= 1
        }
        let hpts = Array(prev.reversed())
        if hpts.count >= 4, let fit = fitArc(hpts, t0: t0, hc: hc, tHat: tHat, up: up, f: f, cx: cx, cy: cy,
                                              r0: club.swingRadius) {
            out.headSpeed = fit.radius * fit.omega
            if fit.rms > 0.6 * d0 { out.notes.append("ヘッドの軌道が円弧に合いません（参考値）") }
        } else {
            out.notes.append("ヘッドを検出できませんでした")
        }
        if let h = out.headSpeed, let b = out.ballSpeed, h > 0 { out.smash = b / h }
        SwingAnalyzer.sanityCheck(&out)
        return out
    }

    /// Head centre positions for params (plane tilt, radius, angular speed at contact,
    /// its relative growth per second, path yaw relative to tHat).
    static func arcPoints(_ p: [Double], times: [Double], t0: Double, hc: Vec3, tHat: Vec3, up: Vec3) -> [Vec3] {
        let a = p[0], r = p[1], om = p[2], c = p[3], psi = p[4]
        let sHat = tHat.cross(up)
        let th = tHat * cos(psi) + sHat * sin(psi)
        let sh = th.cross(up)
        let nv = up * sin(a) + sh * cos(a)
        let center = hc + nv * r
        return times.map { t in
            let tau = t0 - t
            let ph = om * (tau - c * tau * tau / 2)
            return center - (nv * cos(ph) + th * sin(ph)) * r
        }
    }

    /// Least squares over arcPoints' params, in pixels. Soft priors keep the radius near the
    /// club's typical value (±15%), the path within ~5° of the ball's direction and the head
    /// still speeding up a little into the ball.
    private static func fitArc(_ pts: [(t: Double, u: Double, v: Double)], t0: Double, hc: Vec3, tHat: Vec3, up: Vec3,
                               f: Double, cx: Double, cy: Double, r0: Double) -> (radius: Double, omega: Double, rms: Double)? {
        let ts = pts.map { $0.t }
        let resid: ([Double]) -> [Double]? = { p in
            let P = arcPoints(p, times: ts, t0: t0, hc: hc, tHat: tHat, up: up)
            var r: [Double] = []
            for (j, q) in P.enumerated() {
                if q.z < 0.2 { return nil }
                r.append((f * q.x / q.z + cx - pts[j].u) / 0.7)
                r.append((f * q.y / q.z + cy - pts[j].v) / 0.7)
            }
            r += [(p[1] - r0) / (0.15 * r0), (p[3] - 1.5) / 3.0, p[4] / (5 * .pi / 180)]
            return r
        }
        var best: (cost: Double, p: [Double])?
        for deg in stride(from: -85, through: 85, by: 5) {
            for hs in [25.0, 35.0, 45.0, 55.0] {
                let p = [Double(deg) * .pi / 180, r0, hs / r0, 1.5, 0]
                guard let r = resid(p) else { continue }
                let c = r.reduce(0) { $0 + $1 * $1 }
                if best == nil || c < best!.cost { best = (c, p) }
            }
        }
        guard let start = best else { return nil }
        let clampR: ([Double]) -> [Double] = { q in
            var q = q
            q[1] = min(max(q[1], 0.6 * r0), 1.5 * r0)
            return q
        }
        guard let fit = levenbergMarquardt(resid, start.p, h: [1e-4, 1e-4, 1e-3, 1e-3, 1e-4], clamp: clampR) else { return nil }
        let pix = fit.r.dropLast(3)
        let rms = (pix.reduce(0) { $0 + ($1 * 0.7) * ($1 * 0.7) } / Double(max(1, pix.count))).squareRoot()
        return (fit.p[1], fit.p[2], rms)
    }

    /// Levenberg–Marquardt with a forward-difference Jacobian.
    static func levenbergMarquardt(_ resid: ([Double]) -> [Double]?, _ p0: [Double], h: [Double],
                                   iterations: Int = 60, clamp: (([Double]) -> [Double])? = nil) -> (p: [Double], r: [Double])? {
        var p = p0
        guard var r = resid(p) else { return nil }
        var cost = r.reduce(0) { $0 + $1 * $1 }
        var lam = 1e-2
        let m = p.count
        for _ in 0..<iterations {
            var J = [[Double]](repeating: [Double](repeating: 0, count: m), count: r.count)
            for j in 0..<m {
                var q = p
                q[j] += h[j]
                guard let rq = resid(q), rq.count == r.count else { return (p, r) }
                for i in 0..<r.count { J[i][j] = (rq[i] - r[i]) / h[j] }
            }
            var A = [[Double]](repeating: [Double](repeating: 0, count: m), count: m)
            var g = [Double](repeating: 0, count: m)
            for i in 0..<r.count {
                for a in 0..<m {
                    g[a] += J[i][a] * r[i]
                    for b in 0..<m { A[a][b] += J[i][a] * J[i][b] }
                }
            }
            var M = A
            for a in 0..<m { M[a][a] += lam * (A[a][a] + 1e-9) }
            guard let step = solve(M, g.map { -$0 }) else { break }
            var q = zip(p, step).map { $0 + $1 }
            if let clamp { q = clamp(q) }
            if let rq = resid(q), rq.count == r.count, rq.reduce(0, { $0 + $1 * $1 }) < cost {
                p = q; r = rq; cost = rq.reduce(0) { $0 + $1 * $1 }
                lam = max(lam / 3, 1e-7)
                if zip(step, h).allSatisfy({ abs($0) < $1 * 1e-2 }) { break }
            } else {
                lam *= 4
                if lam > 1e7 { break }
            }
        }
        return (p, r)
    }

    /// Gaussian elimination with partial pivoting.
    static func solve(_ A: [[Double]], _ b: [Double]) -> [Double]? {
        let n = b.count
        var M = A, y = b
        for c in 0..<n {
            var piv = c
            for r in (c + 1)..<max(c + 1, n) where abs(M[r][c]) > abs(M[piv][c]) { piv = r }
            guard abs(M[piv][c]) > 1e-300 else { return nil }
            M.swapAt(c, piv); y.swapAt(c, piv)
            for r in (c + 1)..<max(c + 1, n) {
                let fct = M[r][c] / M[c][c]
                if fct == 0 { continue }
                for k in c..<n { M[r][k] -= fct * M[c][k] }
                y[r] -= fct * y[c]
            }
        }
        var x = [Double](repeating: 0, count: n)
        for c in stride(from: n - 1, through: 0, by: -1) {
            var s = y[c]
            for k in (c + 1)..<max(c + 1, n) { s -= M[c][k] * x[k] }
            x[c] = s / M[c][c]
        }
        return x
    }

    private static func median(_ v: [Double]) -> Double {
        let s = v.sorted()
        guard !s.isEmpty else { return 0 }
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    @inline(__always)
    private static func median3(_ a: Double, _ b: Double, _ c: Double) -> Double {
        max(min(a, b), min(max(a, b), c))
    }

    @inline(__always)
    private static func median5(_ v: inout [Int]) -> Int {
        @inline(__always) func cs(_ a: Int, _ b: Int) { if v[a] > v[b] { v.swapAt(a, b) } }
        cs(0, 1); cs(3, 4); cs(2, 4); cs(2, 3); cs(1, 4); cs(0, 3); cs(0, 2); cs(1, 3); cs(1, 2)
        return v[2]
    }
}

struct Vec3 {
    var x, y, z: Double
    init(_ x: Double, _ y: Double, _ z: Double) { self.x = x; self.y = y; self.z = z }
    static func + (a: Vec3, b: Vec3) -> Vec3 { Vec3(a.x + b.x, a.y + b.y, a.z + b.z) }
    static func - (a: Vec3, b: Vec3) -> Vec3 { Vec3(a.x - b.x, a.y - b.y, a.z - b.z) }
    static func * (a: Vec3, s: Double) -> Vec3 { Vec3(a.x * s, a.y * s, a.z * s) }
    func dot(_ b: Vec3) -> Double { x * b.x + y * b.y + z * b.z }
    func cross(_ b: Vec3) -> Vec3 { Vec3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x) }
    var norm: Double { dot(self).squareRoot() }
}
