import AVFoundation
import Foundation
import ImageIO

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
            "overhead": clip.overhead,
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
        var files = [movURL, jsonURL]
        if let l = clip.lock {
            info["lock"] = ["frameWidth": l.frameWidth, "frameHeight": l.frameHeight,
                            "stripY0": l.stripY0, "stripStep": l.stripStep,
                            "fieldOfView": l.fieldOfView, "kind": l.kind, "seconds": l.seconds] as [String: Any]
            // the whole frame at the lock, to tune the ball search on what the camera really saw
            let jpgURL = dir.appendingPathComponent("lock.jpg")
            if let f = l.frame, writeJPEG(f, to: jpgURL) { files.append(jpgURL) }
        }
        guard JSONSerialization.isValidJSONObject(info),
              let data = try? JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys]),
              (try? data.write(to: jsonURL)) != nil else { return nil }
        return files
    }

    private static func writeJPEG(_ img: GrayImage, to url: URL) -> Bool {
        guard img.width > 0, img.height > 0,
              let provider = CGDataProvider(data: Data(img.pixels) as CFData),
              let cg = CGImage(width: img.width, height: img.height, bitsPerComponent: 8, bitsPerPixel: 8,
                               bytesPerRow: img.width, space: CGColorSpaceCreateDeviceGray(),
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.jpeg" as CFString, 1, nil) else { return false }
        CGImageDestinationAddImage(dest, cg, [kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary)
        return CGImageDestinationFinalize(dest)
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
