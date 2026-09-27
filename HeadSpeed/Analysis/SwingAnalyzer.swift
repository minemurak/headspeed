import Foundation

/// Frames around one swing, cropped to the horizontal strip that contains the ball.
struct SwingClip {
    let frames: [GrayImage]
    let times: [Double]          // seconds, per frame (handles dropped frames)
    let ballX: Double            // ball-at-rest centre, strip pixels
    let ballY: Double
    let ballD: Double            // ball diameter, strip pixels
    let triggerIndex: Int        // first frame where the ball patch changed
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
/// validated against synthetic 240fps swings (reference/test.py). sanityCheck is
/// an extra app-only guard that drops physically impossible values.
enum SwingAnalyzer {
    static func analyze(_ clip: SwingClip, club: Club) -> SwingMeasurement {
        var out = SwingMeasurement()
        let n = clip.frames.count
        guard n > 20, clip.times.count == n, (0..<n).contains(clip.triggerIndex) else {
            out.notes.append("映像が短すぎます")
            return out
        }
        let W = clip.frames[0].width, H = clip.frames[0].height
        let bx = clip.ballX, by = clip.ballY, d = clip.ballD
        let trig = clip.triggerIndex
        let mmPerPx = ballDiameterMM / d

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
        var direction: Int?
        var pts: [(t: Double, e: Double, y: Double)] = []
        for k in max(0, kImp - 6)..<kImp {
            let px = clip.frames[k].pixels
            var cnt = [Int](repeating: 0, count: W)
            var band = [Bool](repeating: false, count: (y1 - y0) * W)
            for y in y0..<y1 {
                for x in 0..<W {
                    let i = y * W + x
                    if abs(Int(px[i]) - Int(bg[i])) > 28 && !inRest(x, y) {
                        band[(y - y0) * W + x] = true
                        cnt[x] += 1
                    }
                }
            }
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
        // Keep the last frames that advance monotonically toward the ball.
        var clean: [(t: Double, e: Double, y: Double)] = []
        for p in pts.reversed() {
            if clean.isEmpty || Double(dir) * (clean[clean.count - 1].e - p.e) > 0.1 * d { clean.append(p) }
            if clean.count == 4 { break }
        }
        clean.reverse()
        var headPxS: Double?
        if clean.count >= 2 {
            let t = clean.map { $0.t }
            let vx = linearFit(t, clean.map { $0.e }).slope
            var vy = 0.0
            if !clean.contains(where: { $0.y.isNaN }) { vy = linearFit(t, clean.map { $0.y }).slope }
            vy = max(-abs(vx) * 0.27, min(abs(vx) * 0.27, vy))
            let pxs = hypot(vx, vy)
            headPxS = pxs
            let hs = pxs * mmPerPx / 1000
            out.headSpeed = hs
            // The fitted line is a chord of the swing arc: rotate it to the tangent at contact.
            let chord = atan2(-vy, abs(vx)) * 180 / .pi
            let tMean = t.reduce(0, +) / Double(t.count)
            let eMean = clean.map { $0.e }.reduce(0, +) / Double(clean.count)
            let intercept = eMean - vx * tMean
            let eContact = bx - Double(dir) * 0.5 * d
            let tContact = vx != 0 ? (eContact - intercept) / vx : tMean
            out.attack = chord + (hs / club.swingRadius) * (tContact - tMean) * 180 / .pi
        } else {
            out.notes.append("ヘッドを検出できませんでした")
        }

        // 3. Ball: most forward bright compact blob on the target side.
        let A0 = Double.pi * d * d / 4
        func findBall(thr: Int, maxArea: Double, maxSide: Double, minFill: Double) -> [(t: Double, x: Double, y: Double)] {
            var found: [(t: Double, x: Double, y: Double)] = []
            let xMin = dir > 0 ? Int(bx + 0.5 * d) : 0
            let xMax = dir > 0 ? W : Int(bx - 0.5 * d)       // exclusive
            let rowMin = max(0, Int(by - 14 * d)), rowMax = min(H, Int(by + 1.0 * d))
            for k in kImp..<min(n, kImp + 10) {
                let px = clip.frames[k].pixels
                var mask = [Bool](repeating: false, count: W * H)
                if rowMin < rowMax && max(0, xMin) < min(W, xMax) {
                    for y in rowMin..<rowMax {
                        for x in max(0, xMin)..<min(W, xMax) {
                            let i = y * W + x
                            if Int(px[i]) - Int(bg[i]) > thr && !inRest(x, y) { mask[i] = true }
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
        var bpts = findBall(thr: 35, maxArea: 4.0, maxSide: 3.5, minFill: 0.45)
        if bpts.count < 2 {
            // Slow shutter: the ball is a long, faint streak.
            bpts = findBall(thr: 12, maxArea: 12.0, maxSide: 10.0, minFill: 0.25)
            if bpts.count >= 2 { out.notes.append("シャッターが遅くボールがブレています（精度低下）") }
        }
        var cleanB: [(t: Double, x: Double, y: Double)] = []
        for p in bpts where cleanB.isEmpty || Double(dir) * (p.x - cleanB[cleanB.count - 1].x) > 0.3 * d {
            cleanB.append(p)
        }
        if cleanB.count > 4 { cleanB = Array(cleanB.prefix(4)) }
        var ballPxS: Double?
        if cleanB.count >= 2 {
            let t = cleanB.map { $0.t }
            let vx = linearFit(t, cleanB.map { $0.x }).slope
            let vy = linearFit(t, cleanB.map { $0.y }).slope
            let pxs = hypot(vx, vy)
            ballPxS = pxs
            out.ballSpeed = pxs * mmPerPx / 1000
            out.launch = atan2(-vy, abs(vx)) * 180 / .pi
        } else {
            out.notes.append("打ち出したボールを追跡できませんでした")
        }

        if let h = headPxS, let b = ballPxS, h > 0 { out.smash = b / h }
        sanityCheck(&out)
        return out
    }

    /// Drop values that cannot come from a real golf swing.
    private static func sanityCheck(_ m: inout SwingMeasurement) {
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

    /// Median of five values using a 9-comparator sorting network.
    @inline(__always)
    private static func median5(_ v: inout [Int]) -> Int {
        @inline(__always) func cs(_ a: Int, _ b: Int) { if v[a] > v[b] { v.swapAt(a, b) } }
        cs(0, 1); cs(3, 4); cs(2, 4); cs(2, 3); cs(1, 4); cs(0, 3); cs(0, 2); cs(1, 3); cs(1, 2)
        return v[2]
    }
}
