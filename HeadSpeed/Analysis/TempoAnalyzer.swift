import Foundation

/// Swing tempo from the per-frame motion energy of the whole picture.
/// Takeaway = energy rises from rest, top = the pause before the downswing.
/// Mirrors reference/tempo.py.
enum TempoAnalyzer {
    static func tempo(times: [Double], energy: [Double], impact: Double) -> (backswing: Double, downswing: Double)? {
        var ts: [Double] = [], raw: [Double] = []
        for (t, e) in zip(times, energy) where t >= impact - 4.0 && t <= impact {
            ts.append(t); raw.append(e)
        }
        let n = raw.count
        guard n >= 60 else { return nil }
        let w = 5
        var sm = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let lo = max(0, i - w), hi = min(n, i + w + 1)
            var s = 0.0
            for j in lo..<hi { s += raw[j] }
            sm[i] = s / Double(hi - lo)
        }
        func argExt(_ lo: Double, _ hi: Double, _ better: (Double, Double) -> Bool) -> Int? {
            var best: Int?
            for i in 0..<n where ts[i] >= lo && ts[i] <= hi {
                if best == nil || better(sm[i], sm[best!]) { best = i }
            }
            return best
        }
        guard let iTop = argExt(impact - 0.7, impact - 0.12, { $0 < $1 }) else { return nil }
        let tTop = ts[iTop]
        guard let iPeak = argExt(tTop - 1.5, tTop - 0.1, { $0 > $1 }) else { return nil }
        guard let iBase = argExt(tTop - 3.0, ts[iPeak], { $0 < $1 }) else { return nil }
        let peak = sm[iPeak], base = sm[iBase]
        guard peak > base * 1.5 else { return nil }
        let thr = base + 0.1 * (peak - base)
        var i0: Int?
        var i = iPeak
        while i >= 0 { if sm[i] < thr { i0 = i; break }; i -= 1 }
        guard let start = i0 else { return nil }
        let bsw = tTop - ts[start], dsw = impact - tTop
        guard (0.4...2.0).contains(bsw), (0.15...0.6).contains(dsw) else { return nil }
        return (bsw, dsw)
    }
}
