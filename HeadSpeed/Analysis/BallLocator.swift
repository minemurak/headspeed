import Foundation

/// How a ball candidate was found, which decides how it is refined again later.
enum BallKind: String {
    case round      // face-on bright blob with an all-round outline
    case plain      // face-on bright blob without one (a ball partly hidden, or a shoe toe)
    case lit        // face-on, lit only on one side (sun behind it)
    case rear       // rear view, found by its outline
}

struct BallCandidate {
    var fix: BallFix
    var kind: BallKind
}

/// From a full-resolution frame to ball candidates, best first. Mirrors locate / track in
/// reference/balllock.py.
///
/// Every detector returns all it finds and the choice is made on all of them: an all-round
/// outline beats a bright blob without one; face-on, the ball lies in front of the feet, so
/// lower wins; the spot of the last measured ball wins (the next ball goes on the same tee);
/// nothing far above the horizon (the ball lies on the ground) is kept.
enum BallLocator {
    static let frontGuide = (0.15, 0.30, 0.85, 0.97)
    // From behind, the ball sits low and in the middle; this keeps sky and the range out.
    static let rearGuide = (0.25, 0.45, 0.75, 0.97)

    /// Search area around a tapped point: about three ball widths each way at the usual distance.
    static func pickArea(_ p: (Double, Double)) -> (Double, Double, Double, Double) {
        (max(0, p.0 - 0.05), max(0, p.1 - 0.07), min(1, p.0 + 0.05), min(1, p.1 + 0.07))
    }

    /// Image row of the horizon for a camera looking down by `pitch` radians.
    static func horizonRow(width: Int, height: Int, fieldOfView: Double, pitch: Double) -> Double {
        let f = Double(width) / 2 / tan(fieldOfView * .pi / 360)
        return Double(height) / 2 - f * tan(pitch)
    }

    /// Refine on a full-resolution crop around `approx`.
    static func cropRefine(_ full: GrayImage, _ approx: BallFix, _ fn: (GrayImage, BallFix) -> BallFix?) -> BallFix? {
        let W = full.width, H = full.height
        let half = Int(approx.d * 2.1) + 2
        let x0 = max(0, Int(approx.x) - half), x1 = min(W, Int(approx.x) + half)
        let y0 = max(0, Int(approx.y) - half), y1 = min(H, Int(approx.y) + half)
        guard x1 - x0 > 4, y1 - y0 > 4 else { return nil }
        let cw = x1 - x0
        var crop = [UInt8](repeating: 0, count: cw * (y1 - y0))
        full.pixels.withUnsafeBufferPointer { src in
            crop.withUnsafeMutableBufferPointer { dst in
                for y in y0..<y1 {
                    for x in x0..<x1 { dst[(y - y0) * cw + (x - x0)] = src[y * W + x] }
                }
            }
        }
        let local = BallFix(x: approx.x - Double(x0), y: approx.y - Double(y0), d: approx.d)
        guard let r = fn(GrayImage(width: cw, height: y1 - y0, pixels: crop), local) else { return nil }
        return BallFix(x: r.x + Double(x0), y: r.y + Double(y0), d: r.d)
    }

    static func refine(_ img: GrayImage, _ approx: BallFix, kind: BallKind) -> BallFix? {
        switch kind {
        case .round, .plain: return BallFinder.refine(img, approx: approx)
        case .lit: return BallFinder.refineLit(img, approx: approx)
        case .rear: return BallFinder.refineRear(img, approx: approx)
        }
    }

    /// The ball near `fix` on a new frame, by its detector's refine, or nil.
    static func track(_ full: GrayImage, _ fix: BallFix, kind: BallKind) -> BallFix? {
        guard let r = cropRefine(full, fix, { refine($0, $1, kind: kind) }),
              hypot(r.x - fix.x, r.y - fix.y) <= 0.5 * fix.d else { return nil }
        return r
    }

    /// Every ball candidate in `full`, best first. anchor: tapped point (normalized);
    /// home: the last measured ball, looked at directly first (a full search may not rank it
    /// among its top candidates); horizon: image row of the horizon, nil when the tilt is unknown.
    static func locate(_ full: GrayImage, rear: Bool, anchor: (Double, Double)? = nil,
                       home: BallCandidate? = nil, horizon: Double? = nil) -> [BallCandidate] {
        let W = full.width, H = full.height
        let guide = anchor.map(pickArea) ?? (rear ? rearGuide : frontGuide)
        let f = rear ? 2 : 4
        let sw = W / f, sh = H / f
        var px = [UInt8](repeating: 0, count: sw * sh)
        full.pixels.withUnsafeBufferPointer { src in
            for y in 0..<sh { for x in 0..<sw { px[y * sw + x] = src[y * f * W + x * f] } }
        }
        let small = GrayImage(width: sw, height: sh, pixels: px)
        func up(_ c: BallFix) -> BallFix {
            BallFix(x: c.x * Double(f) + Double(f - 1) / 2, y: c.y * Double(f) + Double(f - 1) / 2, d: c.d * Double(f))
        }
        var found: [BallCandidate] = []
        func add(_ r: BallFix?, _ kind: BallKind) {
            guard let r else { return }
            if found.contains(where: { hypot(r.x - $0.fix.x, r.y - $0.fix.y) < 0.3 * max(r.d, $0.fix.d) }) { return }
            found.append(BallCandidate(fix: r, kind: kind))
        }
        // a tap says which ball, whatever was measured before
        let home = anchor == nil ? home : nil
        if let home { add(track(full, home.fix, kind: home.kind), home.kind) }
        let fd = Double(f)
        if rear {
            for c in BallFinder.searchRearAll(small, guide: guide, minD: 14 / fd, maxD: 70 / fd, anchor: anchor ?? (0.5, 0.75)) {
                add(cropRefine(full, up(c)) { BallFinder.refineRear($0, approx: $1) }, .rear)
            }
        } else {
            for c in BallFinder.searchAll(small, guide: guide, minD: 12 / fd, maxD: 110 / fd) {
                guard let r = cropRefine(full, up(c), { BallFinder.refine($0, approx: $1) }) else { continue }
                // an all-round outline (tee side aside): a ball has one, a shoe toe or a trouser leg not
                let outlined = cropRefine(full, r) { BallFinder.refineRear($0, approx: $1) } != nil
                add(r, outlined ? .round : .plain)
            }
            if !found.contains(where: { $0.kind == .round }) {
                // a ball lit only on one side is a crescent to the bright search
                for c in BallFinder.searchLitAll(small, guide: guide, minD: 12 / fd, maxD: 110 / fd, anchor: anchor) {
                    add(cropRefine(full, up(c)) { BallFinder.refineLit($0, approx: $1) }, .lit)
                }
            }
        }
        if let horizon {
            // on the ground: not far above the horizon (margin for tilt error and a camera set low)
            found = found.filter { $0.fix.y >= horizon - (0.15 * Double(H) + 3 * $0.fix.d) }
        }
        func atHome(_ q: BallCandidate) -> Bool {
            guard let home else { return false }
            return hypot(q.fix.x - home.fix.x, q.fix.y - home.fix.y) < 2 * home.fix.d && abs(q.fix.d / home.fix.d - 1) < 0.25
        }
        func rank(_ k: BallKind) -> Int {
            switch k {
            case .round, .rear: return 0
            case .lit: return 1
            case .plain: return 2
            }
        }
        let keyed: [(h: Int, r: Int, key: Double, c: BallCandidate)] = found.enumerated().map { i, q in
            let key: Double
            if let a = anchor { key = hypot(q.fix.x / Double(W) - a.0, q.fix.y / Double(H) - a.1) }
            else if rear || q.kind == .lit { key = Double(i) }       // the detector's own ranking
            else { key = -q.fix.y }                                  // face-on bright blobs: lowest first
            return (atHome(q) ? 0 : 1, rank(q.kind), key, q)
        }
        return BallFinder.stableSorted(keyed) { ($0.h, $0.r, $0.key) < ($1.h, $1.r, $1.key) }.map { $0.c }
    }
}

/// The ball and a little of what is round it (a square 1.2 d wide, 11 × 11 samples) as first
/// seen, found again on a later frame within a quarter ball either way. Works for any look of
/// the ball (bright, dark, half lit, split background), since it only compares with itself.
struct BallTemplate {
    static let n = 11
    let x: Double, y: Double, d: Double
    private let offs: [(Int, Int)]
    private let ref: [Double]

    init(_ full: GrayImage, x: Double, y: Double, d: Double) {
        self.x = x; self.y = y; self.d = d
        let h = Double(Self.n - 1) / 2
        var o: [(Int, Int)] = []
        for j in 0..<Self.n {
            for i in 0..<Self.n {
                o.append((Int(((Double(i) - h) / h * 0.6 * d).rounded()), Int(((Double(j) - h) / h * 0.6 * d).rounded())))
            }
        }
        offs = o
        ref = Self.sample(full, offs: o, cx: Int(x.rounded()), cy: Int(y.rounded()))
    }

    private static func sample(_ full: GrayImage, offs: [(Int, Int)], cx: Int, cy: Int) -> [Double] {
        let W = full.width, H = full.height
        return offs.map { u, v in Double(full.pixels[min(H - 1, max(0, cy + v)) * W + min(W - 1, max(0, cx + u))]) }
    }

    /// The best place and the mean |difference| there, in grey levels.
    func match(_ full: GrayImage) -> (x: Double, y: Double, change: Double) {
        let R = max(1, Int(0.25 * d))
        let cx = Int(x.rounded()), cy = Int(y.rounded())
        let W = full.width, H = full.height
        var best = (x: x, y: y, change: Double.infinity)
        full.pixels.withUnsafeBufferPointer { p in
            for dy in -R...R {
                for dx in -R...R {
                    var s = 0.0
                    for (k, (u, v)) in offs.enumerated() {
                        let yy = min(H - 1, max(0, cy + dy + v)), xx = min(W - 1, max(0, cx + dx + u))
                        s += abs(Double(p[yy * W + xx]) - ref[k])
                    }
                    let c = s / Double(offs.count)
                    if c < best.change { best = (x + Double(dx), y + Double(dy), c) }
                }
            }
        }
        return best
    }
}

/// When to lock, from one look at a time. Mirrors BallLock in reference/balllock.py.
struct BallLock {
    static let hits = 4          // fixes needed ...
    static let span = 0.25       // ... spread over at least this many seconds
    static let maxMisses = 2     // looks in a row without the ball that start over
    static let tolPos = 0.2      // × ball diameter: how far a fix may sit from the run's first
    static let still = 12.0      // grey levels: change of the ball's look allowed within a run
    static let gone = 24.0       // grey levels: beyond this the ball is not there (covered, taken)

    private(set) var run: [(t: Double, fix: BallFix)] = []
    private(set) var misses = 0

    mutating func reset() { run = []; misses = 0 }

    /// fix nil = not seen. Returns the locked ball (medians of the run) once it has been seen
    /// `hits` times over `span` seconds without moving or changing.
    mutating func update(t: Double, fix: BallFix?, change: Double) -> BallFix? {
        guard let fix else {
            misses += 1
            if misses >= Self.maxMisses { reset() }
            return nil
        }
        misses = 0
        if let first = run.first?.fix,
           hypot(fix.x - first.x, fix.y - first.y) > Self.tolPos * first.d || change > Self.still {
            reset()
        }
        run.append((t, fix))
        guard run.count >= Self.hits, t - run[0].t >= Self.span - 1e-9 else { return nil }
        func med(_ v: [Double]) -> Double {
            let s = v.sorted(), n = s.count
            return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
        }
        return BallFix(x: med(run.map { $0.fix.x }), y: med(run.map { $0.fix.y }), d: med(run.map { $0.fix.d }))
    }
}

/// From a chosen candidate to a lock, one frame at a time: the template says where the ball is
/// and whether it stayed put, the detector's refine measures its size where it can, and the
/// lock takes the median size of the run. Mirrors Follower in reference/balllock.py.
struct BallFollower {
    enum Step { case locked(BallFix), following(BallFix?), lost }

    let kind: BallKind
    let d: Double
    private var template: BallTemplate
    private(set) var lock = BallLock()

    init(_ full: GrayImage, _ cand: BallCandidate) {
        kind = cand.kind
        d = cand.fix.d
        template = BallTemplate(full, x: cand.fix.x, y: cand.fix.y, d: cand.fix.d)
    }

    mutating func step(t: Double, _ full: GrayImage) -> Step {
        let m = template.match(full)
        if m.change > BallLock.gone {
            _ = lock.update(t: t, fix: nil, change: 0)
            return lock.run.isEmpty && lock.misses == 0 ? .lost : .following(nil)
        }
        var size = d
        if let r = BallLocator.track(full, BallFix(x: m.x, y: m.y, d: d), kind: kind),
           hypot(r.x - m.x, r.y - m.y) < 0.25 * d { size = r.d }
        let n0 = lock.run.count
        let got = lock.update(t: t, fix: BallFix(x: m.x, y: m.y, d: size), change: m.change)
        if n0 > 0 && lock.run.count <= n0 {
            // started over (moved or changed): look at it afresh from here
            template = BallTemplate(full, x: m.x, y: m.y, d: d)
        }
        if let got { return .locked(got) }
        return .following(BallFix(x: m.x, y: m.y, d: size))
    }
}
