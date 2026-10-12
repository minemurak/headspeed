import Foundation

/// Was the camera itself moved (bumped, or the phone picked up) when the trigger fired?
/// A swing changes part of the picture over a still background; a moved camera shifts all of
/// it, so a small whole-frame shift lines the frames up much better than no shift at all.
/// Mirrors reference/shake.py.
enum CameraShake {
    private static let step = 4, maxShift = 4, margin = 8

    /// (dx, dy, diff at the best shift, diff at no shift): mean |cur - ref| on a sparse grid.
    static func bestShift(_ ref: GrayImage, _ cur: GrayImage) -> (dx: Int, dy: Int, best: Double, still: Double) {
        let W = ref.width, H = ref.height
        guard cur.width == W, cur.height == H, W > 2 * margin, H > 2 * margin else { return (0, 0, 0, 0) }
        var best = (dx: 0, dy: 0, best: Double.infinity, still: 0.0)
        var still = 0.0
        ref.pixels.withUnsafeBufferPointer { r in
            cur.pixels.withUnsafeBufferPointer { c in
                for dy in -maxShift...maxShift {
                    for dx in -maxShift...maxShift {
                        var sum = 0, n = 0
                        for y in stride(from: margin, to: H - margin, by: step) {
                            let cRow = y * W, rRow = (y + dy) * W + dx
                            for x in stride(from: margin, to: W - margin, by: step) {
                                sum += abs(Int(c[cRow + x]) - Int(r[rRow + x]))
                                n += 1
                            }
                        }
                        let d = Double(sum) / Double(max(n, 1))
                        if dx == 0 && dy == 0 { still = d }
                        if d < best.best { best = (dx, dy, d, 0) }
                    }
                }
            }
        }
        best.still = still
        return best
    }

    /// True when, in at least 2 of the frames from the trigger to 8 after (every other one), a
    /// whole-frame shift explains the change: shifted diff < 0.7 × unshifted, with more than 1.5
    /// grey levels of change to explain. Reference: 12 frames before the trigger.
    static func moved(_ frames: [GrayImage], trigger: Int) -> Bool {
        guard trigger >= 12, trigger < frames.count else { return false }
        let ref = frames[trigger - 12]
        var hits = 0
        for k in stride(from: trigger, to: min(frames.count, trigger + 9), by: 2) {
            let s = bestShift(ref, frames[k])
            if (s.dx != 0 || s.dy != 0) && s.still > 1.5 && s.best < 0.7 * s.still { hits += 1 }
        }
        return hits >= 2
    }
}
