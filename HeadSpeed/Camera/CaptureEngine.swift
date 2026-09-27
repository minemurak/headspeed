import AVFoundation
import Combine
import CoreMedia
import Foundation

enum MeasureState: Equatable {
    case starting
    case unavailable(String)
    case searching      // looking for a ball at rest in the guide area
    case armed          // ball locked, waiting for the swing
    case analyzing
}

/// 240fps capture → ball detection → automatic trigger → analysis.
/// Everything except @Published properties runs on `queue`.
final class CaptureEngine: NSObject, ObservableObject {
    // MARK: Published (main thread)
    @Published private(set) var state: MeasureState = .starting
    /// Ball in normalized frame coordinates (x, y in 0…1; d relative to width).
    @Published private(set) var ballOverlay: BallFix?
    @Published private(set) var frameAspect: Double = 16.0 / 9.0
    @Published private(set) var formatLabel = ""
    @Published private(set) var shutterLabel = ""
    @Published private(set) var shutterTooSlow = false
    @Published private(set) var ballTooSmall = false
    @Published var club: Club = .driver { didSet { lock.lock(); clubValue = club; lock.unlock() } }

    /// Called on the main thread for every analyzed swing.
    var onResult: ((SwingResult) -> Void)?

    /// Guide area where the ball must be placed (x0, y0, x1, y1), normalized.
    static let guide = (0.15, 0.30, 0.85, 0.97)

    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "headspeed.capture", qos: .userInteractive)
    private let analysisQueue = DispatchQueue(label: "headspeed.analysis", qos: .userInitiated)
    private let lock = NSLock()
    private var clubValue: Club = .driver
    private var device: AVCaptureDevice?
    private var configured = false

    // MARK: Capture-queue state
    private var qState: MeasureState = .starting
    private var frameW = 0, frameH = 0
    private var frameIndex = 0
    private var energyPrev: [UInt8] = []
    private var energyTimes: [Double] = []
    private var energyVals: [Double] = []
    private var candidate: BallFix?
    private var stableCount = 0
    private var armedBall: BallFix?
    private var strip: (y0: Int, step: Int, width: Int, height: Int)?
    private var ring: FrameRing?
    private var coreIdx: [Int] = []          // full-res offsets (y * bytesPerRow + x)
    private var coreRef: [Double] = []
    private var overCount = 0
    private var firstOverIndex = 0
    private var triggerIndex = 0
    private var postRemaining = 0
    private var resumeAt = 0.0
    private var armedShutter: Double?
    private var lastShutterPublish = 0.0

    // MARK: Lifecycle

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self else { return }
            guard granted else {
                self.publishState(.unavailable("カメラへのアクセスが許可されていません。設定アプリの「ヘッドスピード」でカメラを許可してください。"))
                return
            }
            self.queue.async {
                if !self.configured { self.configure() }
                if self.configured && !self.session.isRunning { self.session.startRunning() }
            }
        }
    }

    func stop() {
        queue.async { if self.session.isRunning { self.session.stopRunning() } }
    }

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: dev), session.canAddInput(input) else {
            session.commitConfiguration()
            publishState(.unavailable("背面カメラを使えません。"))
            return
        }
        session.addInput(input)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            publishState(.unavailable("カメラの出力を設定できませんでした。"))
            return
        }
        session.addOutput(output)

        guard let picked = Self.pickFormat(dev) else {
            session.commitConfiguration()
            publishState(.unavailable("この端末は120fps以上の撮影に対応していません。"))
            return
        }
        let (format, fps) = picked
        do {
            try dev.lockForConfiguration()
            dev.activeFormat = format
            let frame = CMTime(value: 1, timescale: CMTimeScale(fps))
            dev.activeVideoMinFrameDuration = frame
            dev.activeVideoMaxFrameDuration = frame
            // Cap the shutter at 1/1000 s so the clubhead and ball are not smeared.
            dev.activeMaxExposureDuration = CMTimeMaximum(format.minExposureDuration, CMTime(value: 1, timescale: 1000))
            if dev.isExposureModeSupported(.continuousAutoExposure) { dev.exposureMode = .continuousAutoExposure }
            if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
            dev.unlockForConfiguration()
        } catch {
            session.commitConfiguration()
            publishState(.unavailable("カメラを設定できませんでした。"))
            return
        }
        session.commitConfiguration()
        device = dev
        configured = true

        let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let label = "\(dims.width)×\(dims.height) \(fps)fps"
        let aspect = Double(dims.width) / Double(dims.height)
        DispatchQueue.main.async {
            self.formatLabel = label
            self.frameAspect = aspect
        }
        qState = .searching
        publishState(.searching)
    }

    /// Highest frame rate up to 240fps, preferring 1080p, then 720p.
    private static func pickFormat(_ dev: AVCaptureDevice) -> (AVCaptureDevice.Format, Int)? {
        var best: (format: AVCaptureDevice.Format, fps: Int, score: Int)?
        for f in dev.formats {
            let sub = CMFormatDescriptionGetMediaSubType(f.formatDescription)
            guard sub == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || sub == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange else { continue }
            let dims = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            guard dims.width <= 1920, dims.width >= 1280 else { continue }
            let maxFps = f.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 0
            guard maxFps >= 120 else { continue }
            let fps = maxFps >= 240 ? 240 : 120
            let score = fps * 10_000 + Int(dims.width) + (sub == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ? 1 : 0)
            if best == nil || score > best!.score { best = (f, fps, score) }
        }
        return best.map { ($0.format, $0.fps) }
    }

    // MARK: Helpers

    private func publishState(_ s: MeasureState) {
        DispatchQueue.main.async { self.state = s }
    }

    private func publishBall(_ fix: BallFix?) {
        let w = Double(frameW), h = Double(frameH)
        let norm = fix.map { BallFix(x: $0.x / w, y: $0.y / h, d: $0.d / w) }
        let tooSmall = (fix?.d ?? 99) < 14
        DispatchQueue.main.async {
            self.ballOverlay = norm
            self.ballTooSmall = tooSmall
        }
    }

    private func setExposureLocked(_ locked: Bool) {
        guard let dev = device, (try? dev.lockForConfiguration()) != nil else { return }
        if locked {
            if dev.isExposureModeSupported(.locked) { dev.exposureMode = .locked }
            if dev.isWhiteBalanceModeSupported(.locked) { dev.whiteBalanceMode = .locked }
            if dev.isFocusModeSupported(.locked) { dev.focusMode = .locked }
        } else {
            if dev.isExposureModeSupported(.continuousAutoExposure) { dev.exposureMode = .continuousAutoExposure }
            if dev.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { dev.whiteBalanceMode = .continuousAutoWhiteBalance }
            if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
        }
        dev.unlockForConfiguration()
    }
}

// MARK: - Per-frame processing

extension CaptureEngine: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let t = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0) else { return }
        let W = CVPixelBufferGetWidthOfPlane(pb, 0)
        let H = CVPixelBufferGetHeightOfPlane(pb, 0)
        let bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        let y = base.assumingMemoryBound(to: UInt8.self)
        if W != frameW || H != frameH {
            frameW = W; frameH = H
            energyPrev = []
        }
        frameIndex += 1
        recordEnergy(y, W: W, H: H, bpr: bpr, t: t)
        publishShutterIfNeeded(t)

        switch qState {
        case .searching:
            if t >= resumeAt && frameIndex % 24 == 0 { searchBall(y, W: W, H: H, bpr: bpr, t: t) }
        case .armed:
            writeStrip(y, bpr: bpr, t: t)
            checkTrigger(y, t: t)
        case .analyzing:
            if postRemaining > 0 {
                writeStrip(y, bpr: bpr, t: t)
                postRemaining -= 1
                if postRemaining == 0 { runAnalysis() }
            }
        default:
            break
        }
    }

    /// Mean absolute change of a coarse grid; used for swing tempo.
    private func recordEnergy(_ y: UnsafePointer<UInt8>, W: Int, H: Int, bpr: Int, t: Double) {
        let step = 16
        let gw = W / step, gh = H / step
        if energyPrev.count != gw * gh { energyPrev = [UInt8](repeating: 0, count: gw * gh) }
        var sum = 0
        for gy in 0..<gh {
            let row = y + gy * step * bpr
            for gx in 0..<gw {
                let v = row[gx * step]
                let i = gy * gw + gx
                sum += abs(Int(v) - Int(energyPrev[i]))
                energyPrev[i] = v
            }
        }
        energyTimes.append(t)
        energyVals.append(Double(sum) / Double(max(1, gw * gh)))
        if energyTimes.count > 2400 {
            energyTimes.removeFirst(600)
            energyVals.removeFirst(600)
        }
    }

    private func publishShutterIfNeeded(_ t: Double) {
        guard t - lastShutterPublish > 0.5, let dev = device else { return }
        lastShutterPublish = t
        let s = dev.exposureDuration.seconds
        guard s > 0 else { return }
        let label = "1/\(Int((1 / s).rounded()))秒"
        let slow = s > 1.0 / 700
        DispatchQueue.main.async {
            self.shutterLabel = label
            self.shutterTooSlow = slow
        }
    }

    // MARK: Ball search

    private func searchBall(_ y: UnsafePointer<UInt8>, W: Int, H: Int, bpr: Int, t: Double) {
        let f = 4
        let sw = W / f, sh = H / f
        var small = [UInt8](repeating: 0, count: sw * sh)
        for sy in 0..<sh {
            let row = y + sy * f * bpr
            for sx in 0..<sw { small[sy * sw + sx] = row[sx * f] }
        }
        let img = GrayImage(width: sw, height: sh, pixels: small)
        guard let coarse = BallFinder.search(img, guide: Self.guide, minD: 12.0 / Double(f), maxD: 110.0 / Double(f)) else {
            candidate = nil; stableCount = 0
            publishBall(nil)
            return
        }
        let approx = BallFix(x: coarse.x * Double(f) + 1.5, y: coarse.y * Double(f) + 1.5, d: coarse.d * Double(f))
        // Full-resolution crop around the coarse fix for a sub-pixel diameter.
        let half = Int(approx.d * 1.6) + 2
        let x0 = max(0, Int(approx.x) - half), x1 = min(W, Int(approx.x) + half)
        let y0 = max(0, Int(approx.y) - half), y1 = min(H, Int(approx.y) + half)
        guard x1 - x0 > 4, y1 - y0 > 4 else { return }
        var crop = [UInt8](repeating: 0, count: (x1 - x0) * (y1 - y0))
        for cy in y0..<y1 {
            let row = y + cy * bpr
            for cx in x0..<x1 { crop[(cy - y0) * (x1 - x0) + (cx - x0)] = row[cx] }
        }
        let cropImg = GrayImage(width: x1 - x0, height: y1 - y0, pixels: crop)
        let local = BallFix(x: approx.x - Double(x0), y: approx.y - Double(y0), d: approx.d)
        guard let r = BallFinder.refine(cropImg, approx: local) else {
            candidate = nil; stableCount = 0
            publishBall(nil)
            return
        }
        let fix = BallFix(x: r.x + Double(x0), y: r.y + Double(y0), d: r.d)
        if let c = candidate, abs(c.x - fix.x) < 0.15 * fix.d, abs(c.y - fix.y) < 0.15 * fix.d, abs(c.d - fix.d) < 0.15 * fix.d {
            stableCount += 1
        } else {
            stableCount = 1
        }
        candidate = fix
        publishBall(fix)
        if stableCount >= 5 && fix.d >= 14 { arm(fix, y: y, bpr: bpr, t: t) }
    }

    // MARK: Armed

    private func arm(_ fix: BallFix, y: UnsafePointer<UInt8>, bpr: Int, t: Double) {
        let y0 = max(0, Int(fix.y - 14 * fix.d))
        let y1 = min(frameH, Int(fix.y + 3 * fix.d))
        let rows = y1 - y0
        var step = 1
        while (frameW / step) * (rows / step) > 460_000 { step += 1 }
        strip = (y0, step, frameW / step, rows / step)
        // ~1 s of history at 240fps; the analyzer needs 110 frames before impact.
        ring = FrameRing(capacity: 240, width: frameW / step, height: rows / step)

        coreIdx = []
        coreRef = []
        let r = 0.35 * fix.d
        for yy in max(0, Int(fix.y - r))...min(frameH - 1, Int(fix.y + r) + 1) {
            for xx in max(0, Int(fix.x - r))...min(frameW - 1, Int(fix.x + r) + 1) {
                let dx = Double(xx) - fix.x, dy = Double(yy) - fix.y
                if dx * dx + dy * dy < r * r {
                    let i = yy * bpr + xx
                    coreIdx.append(i)
                    coreRef.append(Double(y[i]))
                }
            }
        }
        armedBall = fix
        overCount = 0
        setExposureLocked(true)
        armedShutter = device?.exposureDuration.seconds
        qState = .armed
        publishState(.armed)
    }

    private func writeStrip(_ y: UnsafePointer<UInt8>, bpr: Int, t: Double) {
        guard let g = strip, let ring else { return }
        ring.write(time: t) { dst in
            for oy in 0..<g.height {
                let src = y + (g.y0 + oy * g.step) * bpr
                let drow = dst + oy * g.width
                if g.step == 1 {
                    memcpy(drow, src, g.width)
                } else {
                    for ox in 0..<g.width { drow[ox] = src[ox * g.step] }
                }
            }
        }
    }

    private func checkTrigger(_ y: UnsafePointer<UInt8>, t: Double) {
        guard let ring, !coreIdx.isEmpty else { return }
        var sum = 0.0
        for (j, i) in coreIdx.enumerated() { sum += abs(Double(y[i]) - coreRef[j]) }
        let diff = sum / Double(coreIdx.count)
        if diff > 18 {
            if overCount == 0 { firstOverIndex = ring.total - 1 }
            overCount += 1
            if overCount >= 2 {
                if ring.total < 150 {
                    // The ball was disturbed right after locking on: start over.
                    disarm(resumeAfter: t + 0.5)
                    return
                }
                triggerIndex = firstOverIndex
                postRemaining = 40          // keep ~0.17 s after impact for the ball flight
                qState = .analyzing
                publishState(.analyzing)
            }
        } else {
            overCount = 0
            // Follow slow lighting changes (clouds) while waiting.
            for (j, i) in coreIdx.enumerated() { coreRef[j] = coreRef[j] * 0.98 + Double(y[i]) * 0.02 }
        }
    }

    private func disarm(resumeAfter: Double) {
        ring = nil
        strip = nil
        armedBall = nil
        candidate = nil
        stableCount = 0
        overCount = 0
        postRemaining = 0
        resumeAt = resumeAfter
        setExposureLocked(false)
        qState = .searching
        publishState(.searching)
        publishBall(nil)
    }

    // MARK: Analysis

    private func runAnalysis() {
        guard let ring, let g = strip, let ball = armedBall else { disarm(resumeAfter: 0); return }
        let snap = ring.snapshot()
        let clip = SwingClip(frames: snap.frames, times: snap.times,
                             ballX: ball.x / Double(g.step),
                             ballY: (ball.y - Double(g.y0)) / Double(g.step),
                             ballD: ball.d / Double(g.step),
                             triggerIndex: triggerIndex - snap.firstIndex)
        let eTimes = energyTimes, eVals = energyVals
        let shutter = armedShutter
        lock.lock(); let club = clubValue; lock.unlock()
        // Release the ring before analysis so its buffers are not duplicated.
        self.ring = nil

        analysisQueue.async {
            let m = SwingAnalyzer.analyze(clip, club: club)
            let impact = m.impactTime ?? clip.times[min(max(clip.triggerIndex, 0), clip.times.count - 1)]
            let tempo = TempoAnalyzer.tempo(times: eTimes, energy: eVals, impact: impact)
            var result = SwingResult(club: club)
            result.headSpeed = m.headSpeed
            result.ballSpeed = m.ballSpeed
            result.smash = m.smash
            result.launch = m.launch
            result.attack = m.attack
            result.backswing = tempo?.backswing
            result.downswing = tempo?.downswing
            result.shutter = shutter
            result.notes = m.notes
            if !result.hasMeasurement {
                result.notes = ["スイングを検出できませんでした。ボールが動いただけの可能性があります。"] + m.notes
            }
            DispatchQueue.main.async { self.onResult?(result) }
            self.queue.async {
                // Give the ball time to leave and the golfer time to settle.
                let now = self.energyTimes.last ?? 0
                self.disarm(resumeAfter: now + 1.5)
            }
        }
    }
}
