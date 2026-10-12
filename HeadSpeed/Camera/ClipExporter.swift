import AVFoundation
import Foundation

/// Saves the last analysed swing (the grey frames the analyzer saw, plus its inputs and
/// result) so it can be shared and replayed with reference/replay.py. One swing is kept;
/// each new one replaces it.
enum ClipExporter {
    static var directory: URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("LastSwing", isDirectory: true)
    }

    /// Writes swing.mov (frames in order, 30 fps playback, near-lossless H.264) and swing.json
    /// (real frame times, ball, camera geometry, result). Returns both URLs, or nil on failure.
    static func export(_ clip: SwingClip, club: Club, result: SwingResult) -> [URL]? {
        guard let first = clip.frames.first, first.width >= 16, first.height >= 16 else { return nil }
        let fm = FileManager.default
        let dir = directory
        try? fm.removeItem(at: dir)
        do { try fm.createDirectory(at: dir, withIntermediateDirectories: true) } catch { return nil }
        let movURL = dir.appendingPathComponent("swing.mov")
        let jsonURL = dir.appendingPathComponent("swing.json")
        // H.264 wants even dimensions; the odd column / row (if any) is dropped.
        let W = first.width & ~1, H = first.height & ~1
        guard writeMovie(clip.frames, width: W, height: H, to: movURL) else { return nil }

        var info: [String: Any] = [
            "format": 1,
            "width": W, "height": H,
            "times": clip.times,
            "ballX": clip.ballX, "ballY": clip.ballY, "ballD": clip.ballD,
            "triggerIndex": clip.triggerIndex,
            "cameraPitch": clip.cameraPitch,
            "rear": clip.rear,
            "focal": clip.focal, "centerX": clip.centerX, "centerY": clip.centerY,
            "club": club.rawValue,
            "notes": result.notes,
            "build": Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "",
        ]
        if let v = result.headSpeed { info["headSpeed"] = v }
        if let v = result.ballSpeed { info["ballSpeed"] = v }
        if let v = result.launch { info["launch"] = v }
        if let v = result.attack { info["attack"] = v }
        if let v = result.shutter { info["shutter"] = v }
        guard JSONSerialization.isValidJSONObject(info),
              let data = try? JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys]),
              (try? data.write(to: jsonURL)) != nil else { return nil }
        return [movURL, jsonURL]
    }

    private static func writeMovie(_ frames: [GrayImage], width W: Int, height H: Int, to url: URL) -> Bool {
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return false }
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: W,
            AVVideoHeightKey: H,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 40_000_000,
                AVVideoMaxKeyFrameIntervalKey: 30,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ],
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                kCVPixelBufferWidthKey as String: W,
                kCVPixelBufferHeightKey as String: H,
            ])
        guard writer.canAdd(input) else { return false }
        writer.add(input)
        guard writer.startWriting() else { return false }
        writer.startSession(atSourceTime: .zero)

        for (k, f) in frames.enumerated() {
            while !input.isReadyForMoreMediaData {
                if writer.status != .writing { return false }
                Thread.sleep(forTimeInterval: 0.002)
            }
            guard let pool = adaptor.pixelBufferPool else { return false }
            var pbOut: CVPixelBuffer?
            guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pbOut) == kCVReturnSuccess, let pb = pbOut else { return false }
            CVPixelBufferLockBaseAddress(pb, [])
            if let yBase = CVPixelBufferGetBaseAddressOfPlane(pb, 0)?.assumingMemoryBound(to: UInt8.self) {
                let bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
                f.pixels.withUnsafeBufferPointer { src in
                    for y in 0..<min(H, f.height) {
                        memcpy(yBase + y * bpr, src.baseAddress! + y * f.width, min(W, f.width))
                    }
                }
            }
            if let uv = CVPixelBufferGetBaseAddressOfPlane(pb, 1) {
                let bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)
                memset(uv, 128, bpr * CVPixelBufferGetHeightOfPlane(pb, 1))   // no colour
            }
            CVPixelBufferUnlockBaseAddress(pb, [])
            if !adaptor.append(pb, withPresentationTime: CMTime(value: CMTimeValue(k), timescale: 30)) { return false }
        }
        input.markAsFinished()
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()
        return writer.status == .completed
    }
}
