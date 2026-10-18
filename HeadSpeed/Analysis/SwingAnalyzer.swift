import Foundation

/// Frames around one swing: the horizontal strip that contains the ball (face-on), or the
/// frame above the ball at reduced resolution (rear view).
struct SwingClip {
    let frames: [GrayImage]
    let times: [Double]          // seconds, per frame (handles dropped frames)
    let ballX: Double            // ball-at-rest centre, strip pixels
    let ballY: Double
    let ballD: Double            // ball diameter, strip pixels
    let triggerIndex: Int        // first frame where the ball patch changed
    var cameraPitch = 0.0        // radians the camera looks down
    var rear = false             // filmed from behind the golfer (down the line)
    var overhead = false         // face-on from well above: speeds from the along-target motion only
    var focal = 0.0              // focal length, clip pixels (rear view only)
    var centerX = 0.0            // optical centre, clip pixels (rear view only)
    var centerY = 0.0
    /// How the ball was locked, for the exported swing (diagnosis only).
    var lock: LockInfo?

    struct LockInfo {
        var frameWidth: Int, frameHeight: Int     // full camera frame
        var stripY0: Int, stripStep: Int          // strip row 0 = frame row stripY0; strip px = stripStep frame px
        var fieldOfView: Double                   // horizontal, degrees
        var kind: String                          // BallKind of the locked candidate
        var seconds: Double                       // from first seeing the candidate to the lock
        var frame: GrayImage?                     // the full frame at the lock
    }
}

struct SwingMeasurement {
    var headSpeed: Double?
    var ballSpeed: Double?
    var smash: Double?
    var launch: Double?
    var attack: Double?
    var impactTime: Double?
    var notes: [String] = []
}

/// Fully automatic face-on analysis. Mirrors reference/analyzer.py, which is
/// validated against synthetic 240fps swings (reference/test.py, test_overhead.py). sanityCheck
/// is an extra app-only guard that drops physically impossible values.
///
/// Overhead (上から): face-on looking down steeply. The image's vertical motion then mixes
/// rising with moving toward or away from the camera, so only the horizontal (along the target
/// line) motion is used, and no launch or attack angle is given.
enum SwingAnalyzer {
    static func analyze(_ clip: SwingClip, club: Club) -> SwingMeasurement {
        if clip.rear { return RearAnalyzer.analyze(clip, club: club) }
        var out = SwingMeasurement()
        let n = clip.frames.count
        guard n > 20, clip.times.count == n, (0..<n).contains(clip.triggerIndex) else {
            out.notes.append("映像が短すぎます")
            return out
        }
        let W = clip.frames[0].width, H = clip.frames[0].height
        var bx = clip.ballX, by = clip.ballY, d = clip.ballD
        let trig = clip.triggerIndex
        // Looking down at the swing shrinks vertical motion by cos(pitch); divide it back out.
        let yCorr = 1 / cos(min(max(clip.cameraPitch, 0), 60 * .pi / 180))

        // Background: per-pixel median of five frames well before impact.
        let bgIdx = [110, 95, 80, 65, 50].map { max(0, trig - $0) }
        var bg = [Int16](repeating: 0, count: W * H)
        do {
            let src = bgIdx.map { clip.frames[min($0, n - 1)].pixels }
            var v = [Int](repeating: 0, count: 5)
            for i in 0..<(W * H) {
                for j in 0..<5 { v[j] = Int(src[j][i]) }
                bg[i] = Int16(median5(&v))
            }
        }
        if let sil = silhouette(clip, bg: bg, ball: (bx, by, d)), abs(sil.d / d - 1) > 0.1 {
            // The finder saw only part of the ball (the lit side); every speed scales with d.
            (bx, by, d) = (sil.x, sil.y, sil.d)
        }
        let mmPerPx = ballDiameterMM / d

        let restR2 = (0.75 * d) * (0.75 * d)
        let coreR2 = (0.35 * d) * (0.35 * d)
        func inRest(_ x: Int, _ y: Int) -> Bool {
            let dx = Double(x) - bx, dy = Double(y) - by
            return dx * dx + dy * dy < restR2
        }

        // 1. Impact frame: first frame whose ball core differs from the background.
        var coreIdx: [Int] = []
        for y in max(0, Int(by - d))..<min(H, Int(by + d) + 1) {
            for x in max(0, Int(bx - d))..<min(W, Int(bx + d) + 1) {
                let dx = Double(x) - bx, dy = Double(y) - by
                if dx * dx + dy * dy < coreR2 { coreIdx.append(y * W + x) }
            }
        }
        guard !coreIdx.isEmpty else { out.notes.append("ボールの位置が不正です"); return out }
        var kImpOpt: Int?
        for k in max(1, trig - 24)..<min(n, trig + 12) {
            let px = clip.frames[k].pixels
            var sum = 0
            for i in coreIdx { sum += abs(Int(px[i]) - Int(bg[i])) }
            if Double(sum) / Double(coreIdx.count) > 18 { kImpOpt = k; break }
        }
        guard let kImp = kImpOpt else {
            out.notes.append("インパクトの瞬間を検出できませんでした")
            return out
        }
        out.impactTime = clip.times[kImp]

        // 2. Head: leading edge of the moving region in a band at ball height.
        let y0 = max(0, Int(by - 2.5 * d)), y1 = min(H, Int(by + 0.8 * d))
        let minCnt = max(2, Int((0.2 * d).rounded(.toNearestOrEven)))
        let shaft = max(3, Int(0.17 * d) | 1)   // odd, wider than the shaft
        var direction: Int?
        var pts: [(t: Double, e: Double, y: Double)] = []
        for k in max(1, kImp - 6)..<kImp {
            let px = clip.frames[k].pixels, prev = clip.frames[k - 1].pixels
            var cnt = [Int](repeating: 0, count: W)
            var band = [Bool](repeating: false, count: (y1 - y0) * W)
            for y in y0..<y1 {
                for x in 0..<W {
                    let i = y * W + x
                    // Only what is moving fast right now: different from the background and
                    // from the previous frame (a foot or shadow shifting slowly is not).
                    if abs(Int(px[i]) - Int(bg[i])) > 28 && abs(Int(px[i]) - Int(prev[i])) > 28 && !inRest(x, y) {
                        band[(y - y0) * W + x] = true
                    }
                }
            }
            // The shaft leans ahead of the head and crosses the band first: open the mask
            // with a square wider than the shaft so only the head's bulk is left.
            band = openMask(band, width: W, height: y1 - y0, k: shaft)
            for y in 0..<(y1 - y0) { for x in 0..<W where band[y * W + x] { cnt[x] += 1 } }
            var anyCol = false, wsum = 0.0, csum = 0.0
            for x in 0..<W where cnt[x] >= minCnt {
                anyCol = true
                wsum += Double(x * cnt[x])
                csum += Double(cnt[x])
            }
            if !anyCol { continue }
            if direction == nil { direction = bx > wsum / csum ? 1 : -1 }
            let dir = direction!
            var edge: Int?
            if dir > 0 {
                var x = Int((bx - 0.2 * d).rounded(.down))
                x = min(x, W - 1)
                while x >= 0 { if cnt[x] >= minCnt { edge = x; break }; x -= 1 }
            } else {
                var x = Int((bx + 0.2 * d).rounded(.up))
                x = max(x, 0)
                while x < W { if cnt[x] >= minCnt { edge = x; break }; x += 1 }
            }
            guard let e = edge else { continue }
            let lo = dir > 0 ? Double(e) - 0.5 * d : Double(e)
            let hi = dir > 0 ? Double(e) : Double(e) + 0.5 * d
            var ySum = 0.0, yCnt = 0
            for y in y0..<y1 {
                for x in max(0, Int(lo.rounded(.up)))...min(W - 1, Int(hi.rounded(.down))) where band[(y - y0) * W + x] {
                    ySum += Double(y); yCnt += 1
                }
            }
            pts.append((clip.times[k], Double(e), yCnt > 0 ? ySum / Double(yCnt) : .nan))
        }

        guard let dir = direction else {
            out.notes.append("ヘッドを検出できませんでした")
            return out
        }
        // Keep the last frames that advance monotonically toward the ball at a steady pace
        // (an earlier frame can catch the shaft crossing the band ahead of the head).
        var clean: [(t: Double, e: Double, y: Double)] = []
        for p in pts.reversed() {
            if let l = clean.last, Double(dir) * (l.e - p.e) <= 0.1 * d { continue }
            if clean.count >= 2 {
                let a = clean[clean.count - 2], b = clean[clean.count - 1]
                let last = Double(dir) * (a.e - b.e) / (a.t - b.t)
                let step = Double(dir) * (b.e - p.e) / (b.t - p.t)
                if !(step >= 0.6 * last && step <= 1.6 * last) { break }
            }
            clean.append(p)
            if clean.count == 4 { break }
        }
        clean.reverse()
        var headPxS: Double?
        var tContact: Double?
        if clean.count >= 2 {
            let t = clean.map { $0.t }
            var vx = linearFit(t, clean.map { $0.e }).slope
            var vy = 0.0
            if !clean.contains(where: { $0.y.isNaN }) { vy = linearFit(t, clean.map { $0.y }).slope * yCorr }
            vy = max(-abs(vx) * 0.27, min(abs(vx) * 0.27, vy))
            var tMean = t.reduce(0, +) / Double(t.count)
            let eMean = clean.map { $0.e }.reduce(0, +) / Double(clean.count)
            let intercept = eMean - vx * tMean
            let eContact = bx - Double(dir) * 0.5 * d
            let tC = vx != 0 ? (eContact - intercept) / vx : tMean
            if vx != 0 { tContact = tC }
            // The head's whole-body displacement over the last frames, when it agrees roughly.
            let hm = headMatch(clip, bg: bg, ball: (bx, by, d), band: (y0, y1), kImp: kImp, dir: dir, shaft: shaft)
            if !hm.isEmpty, vx != 0 {
                let mvx = hm.map { $0.vx }.reduce(0, +) / Double(hm.count)
                let mvy = hm.map { $0.vy }.reduce(0, +) / Double(hm.count) * yCorr
                if mvx / vx >= 0.6 && mvx / vx <= 2.0 {
                    // the whole velocity along the arc (a few frames out it still dips noticeably)
                    vx = mvx
                    vy = max(-abs(mvx) * 0.6, min(abs(mvx) * 0.6, mvy))
                    tMean = hm.map { $0.t }.reduce(0, +) / Double(hm.count)
                }
            }
            if clip.overhead {
                // The head a few frames before contact still moves along the arc, tilted from
                // the target line by the club's usual attack angle plus its turn until contact;
                // its horizontal part is the speed times the cosine of that tilt.
                let turn = abs(vx) * mmPerPx / 1000 / club.swingRadius * max(0, tC - tMean)
                let th = min(25 * Double.pi / 180, turn - club.typicalAttack * .pi / 180)
                let pxs = abs(vx) / cos(th)
                headPxS = pxs
                out.headSpeed = pxs * mmPerPx / 1000
            } else {
                let pxs = hypot(vx, vy)
                headPxS = pxs
                let hs = pxs * mmPerPx / 1000
                out.headSpeed = hs
                // The fitted line is a chord of the swing arc: rotate it to the tangent at contact.
                let chord = atan2(-vy, abs(vx)) * 180 / .pi
                out.attack = chord + (hs / club.swingRadius) * (tC - tMean) * 180 / .pi
            }
        } else {
            out.notes.append("ヘッドを検出できませんでした")
        }

        // 3. Ball: most forward bright compact blob on the target side.
        let A0 = Double.pi * d * d / 4
        func findBall(thr: Int, maxArea: Double, maxSide: Double, minFill: Double) -> [(t: Double, x: Double, y: Double)] {
            var found: [(t: Double, x: Double, y: Double)] = []
            let xMin = dir > 0 ? Int(bx + 0.5 * d) : 0
            let xMax = dir > 0 ? W : Int(bx - 0.5 * d)       // exclusive
            // from above, a ball starting away from the golfer (toward the camera) goes down
            let rowMin = max(0, Int(by - 14 * d)), rowMax = min(H, Int(by + (clip.overhead ? 4.0 : 1.0) * d))
            for k in kImp..<min(n, kImp + 10) {
                let px = clip.frames[k].pixels, prev = clip.frames[max(0, k - 1)].pixels
                var mask = [Bool](repeating: false, count: W * H)
                if rowMin < rowMax && max(0, xMin) < min(W, xMax) {
                    for y in rowMin..<rowMax {
                        for x in max(0, xMin)..<min(W, xMax) {
                            let i = y * W + x
                            // Brighter than the background and changed since the previous frame:
                            // the ball in flight moves several widths per frame, while a glove or
                            // trouser leg that shifted at address sits still.
                            if Int(px[i]) - Int(bg[i]) > thr && abs(Int(px[i]) - Int(prev[i])) > thr
                                && !inRest(x, y) { mask[i] = true }
                        }
                    }
                }
                var best: (x: Double, y: Double)?
                for b in Blobs.find(mask: mask, width: W, height: H) {
                    let area = Double(b.area)
                    guard area >= 0.35 * A0, area <= maxArea * A0 else { continue }
                    guard Double(max(b.boxW, b.boxH)) <= maxSide * d, b.fill >= minFill else { continue }
                    guard b.minX > 0, b.minY > 0, b.maxX < W - 1, b.maxY < H - 1 else { continue }
                    if best == nil || Double(dir) * b.cx > Double(dir) * best!.x { best = (b.cx, b.cy) }
                }
                if let b = best { found.append((clip.times[k], b.x, b.y)) }
            }
            return found
        }
        func forward(_ pts: [(t: Double, x: Double, y: Double)]) -> [(t: Double, x: Double, y: Double)] {
            var kept: [(t: Double, x: Double, y: Double)] = []
            for p in pts where kept.isEmpty || Double(dir) * (p.x - kept[kept.count - 1].x) > 0.3 * d { kept.append(p) }
            return Array(kept.prefix(4))
        }
        let strict = findBall(thr: 35, maxArea: 4.0, maxSide: 3.5, minFill: 0.45)
        var cleanB = forward(strict)
        if strict.count < 2 {
            // Slow shutter: the ball is a long, faint streak.
            let loose = forward(findBall(thr: 12, maxArea: 12.0, maxSide: 10.0, minFill: 0.25))
            if loose.count >= 2 {
                cleanB = loose
                out.notes.append("シャッターが遅くボールがブレています（精度低下）")
            }
        }
        let steps = zip(clip.times.dropFirst(), clip.times).map { $0 - $1 }.sorted()
        let dtFrame = steps[steps.count / 2]
        var ballV: (vx: Double, vy: Double)?
        if cleanB.count >= 2 {
            let t = cleanB.map { $0.t }
            ballV = (linearFit(t, cleanB.map { $0.x }).slope, linearFit(t, cleanB.map { $0.y }).slope * yCorr)
        } else if let p = cleanB.first, let tC = tContact, p.t - tC >= 0.3 * dtFrame, p.t - tC <= 2.5 * dtFrame,
                  Double(dir) * (p.x - bx) > d {
            // One flight frame only (the ball left the picture straight away): it left the tee
            // when the head reached it, so take the displacement from there.
            let vx = (p.x - bx) / (p.t - tC)
            ballV = (vx, (p.y - by) / (p.t - tC) * yCorr)
            let next = p.x + vx * dtFrame
            if next >= 0 && next < Double(W) {
                out.notes.append("ボールが1コマしか写らず推定値です（精度低下）")
            } else {
                out.notes.append("ボールがすぐ画面外に出たため1コマから推定しました（精度低下）。ボールを画面の中央寄りに置いてください")
            }
        } else {
            out.notes.append("打ち出したボールを追跡できませんでした")
        }
        var ballPxS: Double?
        if let v = ballV {
            let pxs = clip.overhead ? abs(v.vx) / cos(club.typicalLaunch * .pi / 180) : hypot(v.vx, v.vy)
            ballPxS = pxs
            out.ballSpeed = pxs * mmPerPx / 1000
            if !clip.overhead { out.launch = atan2(-v.vy, abs(v.vx)) * 180 / .pi }
        }

        if let h = headPxS, let b = ballPxS, h > 0 { out.smash = b / h }
        sanityCheck(&out)
        return out
    }

    /// Drop values that cannot come from a real golf swing.
    static func sanityCheck(_ m: inout SwingMeasurement) {
        if let hs = m.headSpeed, !(8...75).contains(hs) {
            m.headSpeed = nil; m.attack = nil; m.smash = nil
            m.notes.append("ヘッドスピードが範囲外のため除外しました")
        }
        if let bs = m.ballSpeed, !(8...100).contains(bs) {
            m.ballSpeed = nil; m.launch = nil; m.smash = nil
            m.notes.append("ボール初速が範囲外のため除外しました")
        }
        if let s = m.smash, !(0.7...1.62).contains(s) {
            m.smash = nil
            m.notes.append("ミート率が範囲外です（ヘッドかボールの誤検出の可能性）")
        }
    }

    /// The ball's size and place from what is gone once it has left: the frames at the end of
    /// the clip against the background, so a ball lit only on one side (its shaded part as
    /// dark as the grass, invisible to a brightness search) still shows whole. Width across,
    /// since the tee hangs below. Nil when the end frames are missing or the change there is
    /// not a lone ball. Mirrors silhouette() in reference/analyzer.py.
    static func silhouette(_ clip: SwingClip, bg: [Int16], ball: (Double, Double, Double)) -> (x: Double, y: Double, d: Double)? {
        let (bx, by, d) = ball
        let n = clip.frames.count, W = clip.frames[0].width, H = clip.frames[0].height
        let first = clip.triggerIndex + 30
        guard n - first >= 5 else { return nil }
        let post = Array(max(first, n - 8)..<n)
        let x0 = max(0, Int(bx - 2 * d)), x1 = min(W, Int(bx + 2 * d) + 1)
        let y0 = max(0, Int(by - 2 * d)), y1 = min(H, Int(by + 2 * d) + 1)
        let ww = x1 - x0, wh = y1 - y0
        guard ww > 2, wh > 2 else { return nil }
        var mask = [Bool](repeating: false, count: ww * wh)
        var v = [Double](repeating: 0, count: post.count)
        for y in y0..<y1 {
            for x in x0..<x1 {
                let i = y * W + x
                for (j, k) in post.enumerated() { v[j] = Double(clip.frames[k].pixels[i]) }
                v.sort()
                let m = v.count % 2 == 1 ? v[v.count / 2] : (v[v.count / 2 - 1] + v[v.count / 2]) / 2
                mask[(y - y0) * ww + (x - x0)] = abs(m - Double(bg[i])) > 25
            }
        }
        let cx = bx - Double(x0), cy = by - Double(y0)
        var hit: Blob?
        for b in Blobs.find(mask: mask, width: ww, height: wh) where b.area >= 10 {
            if Double(b.minX) <= cx && cx <= Double(b.maxX) && Double(b.minY) <= cy && cy <= Double(b.maxY),
               hit == nil || b.area > hit!.area { hit = b }
        }
        guard let b = hit, b.minX > 0, b.minY > 0, b.maxX < ww - 1, b.maxY < wh - 1 else { return nil }
        let w = Double(b.boxW)
        guard w >= 0.5 * d, w <= 2.5 * d else { return nil }
        return (Double(x0) + Double(b.minX + b.maxX) / 2, Double(y0) + Double(b.minY) + w / 2, w)
    }

    /// Head displacement between consecutive frames shortly before contact, by matching the
    /// head's own pixels (the moving region within a head's length of its front) in the next
    /// frame. Uses the whole head, so the hosel and shaft that lead the face no longer set the
    /// speed, as they can for a single leading-edge column. Mirrors head_match() in
    /// reference/analyzer.py. Velocities in px/s.
    static func headMatch(_ clip: SwingClip, bg: [Int16], ball: (Double, Double, Double), band: (Int, Int),
                          kImp: Int, dir: Int, shaft: Int) -> [(t: Double, vx: Double, vy: Double)] {
        let (bx, by, d) = ball
        let (y0, y1) = band
        let W = clip.frames[0].width, H = clip.frames[0].height
        guard y1 > y0 else { return [] }
        let restR2 = (0.75 * d) * (0.75 * d)
        func inRest(_ x: Int, _ y: Int) -> Bool {
            let dx = Double(x) - bx, dy = Double(y) - by
            return dx * dx + dy * dy < restR2
        }
        let bh = y1 - y0
        let grow = max(3, Int(0.6 * d) | 1), p = grow / 2
        // rows the dilated head can reach
        let r0 = max(0, y0 - p), r1 = min(H, y1 + p), rh = r1 - r0
        var out: [(t: Double, vx: Double, vy: Double)] = []
        let lo = max(1, kImp - 4), hi = kImp - 2
        guard lo < hi, hi < clip.frames.count else { return [] }
        for k in lo..<hi {
            let px = clip.frames[k].pixels, prev = clip.frames[k - 1].pixels, nx = clip.frames[k + 1].pixels
            var m = [Bool](repeating: false, count: bh * W)
            for y in y0..<y1 {
                for x in 0..<W {
                    let i = y * W + x
                    if abs(Int(px[i]) - Int(bg[i])) > 28 && abs(Int(px[i]) - Int(prev[i])) > 28 && !inRest(x, y) {
                        m[(y - y0) * W + x] = true
                    }
                }
            }
            m = openMask(m, width: W, height: bh, k: shaft)
            let cs = Blobs.find(mask: m, width: W, height: bh).filter { Double($0.area) >= 0.3 * d * d }
            guard !cs.isEmpty else { continue }
            // the head: everything moving within a head's length of the front-most part
            let xa: Int, xb: Int
            if dir > 0 {
                let front = cs.map { $0.maxX }.max()!
                xa = max(0, Int(Double(front) - 3 * d)); xb = min(W, front + 1)
            } else {
                let front = cs.map { $0.minX }.min()!
                xa = max(0, front); xb = min(W, Int(Double(front) + 3 * d) + 1)
            }
            guard xa < xb else { continue }
            // with a margin of background round it, so a plain (untextured) head is pinned by
            // its outline; compared as differences from the background, ~0 off the head
            var hm = [Bool](repeating: false, count: rh * W)
            var count = 0
            for y in y0..<y1 {
                for x in xa..<xb where m[(y - y0) * W + x] { hm[(y - r0) * W + x] = true; count += 1 }
            }
            guard Double(count) >= 0.3 * d * d else { continue }
            hm = dilateMask(hm, width: W, height: rh, k: grow)
            var idx: [(x: Int, y: Int, v: Int)] = []
            for y in r0..<r1 {
                for x in 0..<W where hm[(y - r0) * W + x] && !inRest(x, y) {
                    idx.append((x, y, Int(px[y * W + x]) - Int(bg[y * W + x])))
                }
            }
            guard !idx.isEmpty else { continue }
            func cost(_ dx: Int, _ dy: Int) -> Double? {
                var sum = 0, n = 0
                for q in idx {
                    let X = q.x + dx, Y = q.y + dy
                    guard X >= 0, X < W, Y >= 0, Y < H, !inRest(X, Y) else { continue }
                    let i = Y * W + X
                    sum += abs(Int(nx[i]) - Int(bg[i]) - q.v)
                    n += 1
                }
                guard Double(n) >= 0.6 * Double(idx.count) else { return nil }
                return Double(sum) / Double(n)
            }
            var best: (c: Double, dx: Int, dy: Int)?
            let r = Int(1.5 * d)
            for dy in stride(from: -r, through: r, by: 2) {          // coarse, then refine
                for a in stride(from: Int(0.5 * d), to: Int(8 * d), by: 2) {
                    if let c = cost(dir * a, dy), best == nil || c < best!.c { best = (c, dir * a, dy) }
                }
            }
            guard var b = best else { continue }
            let (cdx, cdy) = (b.dx, b.dy)
            for dy in (cdy - 2)...(cdy + 2) {
                for dx in (cdx - 2)...(cdx + 2) {
                    if let c = cost(dx, dy), c < b.c { b = (c, dx, dy) }
                }
            }
            let dt = clip.times[k + 1] - clip.times[k]
            guard dt > 0 else { continue }
            out.append(((clip.times[k] + clip.times[k + 1]) / 2, Double(b.dx) / dt, Double(b.dy) / dt))
        }
        return out
    }

    /// Binary dilation with a k × k square (k odd).
    static func dilateMask(_ m: [Bool], width W: Int, height H: Int, k: Int) -> [Bool] {
        guard k > 1 else { return m }
        let p = k / 2
        var a = [Bool](repeating: false, count: m.count)
        for y in 0..<H {
            for x in 0..<W where m[y * W + x] {
                for yy in max(0, y - p)...min(H - 1, y + p) { a[yy * W + x] = true }
            }
        }
        var b = [Bool](repeating: false, count: m.count)
        for y in 0..<H {
            for x in 0..<W where a[y * W + x] {
                for xx in max(0, x - p)...min(W - 1, x + p) { b[y * W + xx] = true }
            }
        }
        return b
    }

    /// Binary opening with a k × k square (k odd): removes anything thinner than k.
    static func openMask(_ m: [Bool], width W: Int, height H: Int, k: Int) -> [Bool] {
        guard k > 1 else { return m }
        let p = k / 2
        func pass(_ a: [Bool], horizontal: Bool, all: Bool) -> [Bool] {
            var out = [Bool](repeating: false, count: a.count)
            for y in 0..<H {
                for x in 0..<W {
                    var r = all
                    for o in -p...p {
                        let xx = horizontal ? x + o : x, yy = horizontal ? y : y + o
                        let v = xx >= 0 && xx < W && yy >= 0 && yy < H ? a[yy * W + xx] : false
                        if all ? !v : v { r = !all; break }
                    }
                    out[y * W + x] = r
                }
            }
            return out
        }
        let eroded = pass(pass(m, horizontal: false, all: true), horizontal: true, all: true)
        return pass(pass(eroded, horizontal: false, all: false), horizontal: true, all: false)
    }

    /// Median of five values using a 9-comparator sorting network.
    @inline(__always)
    private static func median5(_ v: inout [Int]) -> Int {
        @inline(__always) func cs(_ a: Int, _ b: Int) { if v[a] > v[b] { v.swapAt(a, b) } }
        cs(0, 1); cs(3, 4); cs(2, 4); cs(2, 3); cs(1, 4); cs(0, 3); cs(0, 2); cs(1, 3); cs(1, 2)
        return v[2]
    }
}
