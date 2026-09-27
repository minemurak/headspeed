import Foundation

/// 8-bit luminance image, row-major.
struct GrayImage {
    let width: Int
    let height: Int
    var pixels: [UInt8]

    init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height)
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    init(width: Int, height: Int) {
        self.init(width: width, height: height, pixels: [UInt8](repeating: 0, count: width * height))
    }
}

/// A 4-connected region of a binary mask.
struct Blob {
    var area = 0
    var sumX = 0.0
    var sumY = 0.0
    var minX = Int.max
    var maxX = Int.min
    var minY = Int.max
    var maxY = Int.min

    var cx: Double { sumX / Double(area) }
    var cy: Double { sumY / Double(area) }
    var boxW: Int { maxX - minX + 1 }
    var boxH: Int { maxY - minY + 1 }
    var fill: Double { Double(area) / Double(boxW * boxH) }
}

enum Blobs {
    /// 4-neighbour connected components of `mask` (row-major, width × height).
    static func find(mask: [Bool], width: Int, height: Int) -> [Blob] {
        var visited = [Bool](repeating: false, count: mask.count)
        var blobs: [Blob] = []
        var stack: [Int] = []
        stack.reserveCapacity(1024)
        for start in 0..<mask.count where mask[start] && !visited[start] {
            var b = Blob()
            visited[start] = true
            stack.append(start)
            while let i = stack.popLast() {
                let x = i % width, y = i / width
                b.area += 1
                b.sumX += Double(x)
                b.sumY += Double(y)
                if x < b.minX { b.minX = x }
                if x > b.maxX { b.maxX = x }
                if y < b.minY { b.minY = y }
                if y > b.maxY { b.maxY = y }
                if x > 0 { let j = i - 1; if mask[j] && !visited[j] { visited[j] = true; stack.append(j) } }
                if x < width - 1 { let j = i + 1; if mask[j] && !visited[j] { visited[j] = true; stack.append(j) } }
                if y > 0 { let j = i - width; if mask[j] && !visited[j] { visited[j] = true; stack.append(j) } }
                if y < height - 1 { let j = i + width; if mask[j] && !visited[j] { visited[j] = true; stack.append(j) } }
            }
            blobs.append(b)
        }
        return blobs
    }
}

/// Least-squares line v = slope · t + intercept.
func linearFit(_ t: [Double], _ v: [Double]) -> (slope: Double, intercept: Double) {
    let n = Double(t.count)
    let tm = t.reduce(0, +) / n
    let vm = v.reduce(0, +) / n
    var num = 0.0, den = 0.0
    for i in 0..<t.count {
        num += (t[i] - tm) * (v[i] - vm)
        den += (t[i] - tm) * (t[i] - tm)
    }
    let slope = den > 0 ? num / den : 0
    return (slope, vm - slope * tm)
}
