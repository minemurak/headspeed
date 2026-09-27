import Foundation

/// Fixed-size ring of cropped luminance frames (the strip around the ball).
/// Written only from the capture queue.
final class FrameRing {
    let capacity: Int
    let width: Int
    let height: Int
    private var buffers: [[UInt8]]
    private var times: [Double]
    /// Number of frames ever written; the absolute index of the next frame.
    private(set) var total = 0

    init(capacity: Int, width: Int, height: Int) {
        self.capacity = capacity
        self.width = width
        self.height = height
        buffers = (0..<capacity).map { _ in [UInt8](repeating: 0, count: width * height) }
        times = [Double](repeating: 0, count: capacity)
    }

    func write(time: Double, _ fill: (UnsafeMutablePointer<UInt8>) -> Void) {
        let slot = total % capacity
        buffers[slot].withUnsafeMutableBufferPointer { fill($0.baseAddress!) }
        times[slot] = time
        total += 1
    }

    /// Frames in time order and the absolute index of the first one.
    func snapshot() -> (frames: [GrayImage], times: [Double], firstIndex: Int) {
        let count = min(total, capacity)
        let first = total - count
        var frames: [GrayImage] = []
        var ts: [Double] = []
        frames.reserveCapacity(count)
        for a in first..<total {
            let slot = a % capacity
            frames.append(GrayImage(width: width, height: height, pixels: buffers[slot]))
            ts.append(times[slot])
        }
        return (frames, ts, first)
    }
}
