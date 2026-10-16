import AVFoundation
import Combine
import CoreMedia
import CoreMotion
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
    /// swing.mov + swing.json of the last analysed swing, for sharing (see ClipExporter).
    @Published private(set) var lastClip: [URL] = []
    @Published var club: Club = .driver { didSet { lock.lock(); clubValue = club; lock.unlock() } }
    /// Where the user tapped to say which ball to measure (normalized frame x, y), or nil for
    /// the automatic choice. Kept across swings, since the next ball goes in the same spot.
    @Published private(set) var pickedPoint: CGPoint?
    /// Camera position. Changing it drops a locked ball so the next lock uses the new layout.
    @Published var view: CameraView = CaptureEngine.savedView {
        didSet {
            UserDefaults.standard.set(view.rawValue, forKey: Self.viewKey)
            lock.lock(); viewValue = view; pickValue = nil; lock.unlock()
            pickedPoint = nil
            queue.async { if self.qState == .armed { self.disarm(resumeAfter: 0) } }
        }
    }
    private static let viewKey = "headspeed.view"
    private static var savedView: CameraView {
        UserDefaults.standard.string(forKey: viewKey).flatMap(CameraView.init(rawValue:)) ?? .front
    }

    /// Called on the main thread for every analyzed swing.
    var onResult: ((SwingResult) -> Void)?

    /// Guide area where the ball must be placed (x0, y0, x1, y1), normalized.
    /// Search area around a tapped point: about three ball widths each way at the usual distance.
    static func pickArea(_ p: CGPoint) -> (Double, Double, Double, Double) {
        (max(0, Double(p.x) - 0.05), max(0, Double(p.y) - 0.07), min(1, Double(p.x) + 0.05), min(1, Double(p.y) + 0.07))
    }

    /// Main thread. Measures the ball nearest `p` (normalized frame coordinates) from now on;
    /// nil goes back to the automatic choice. A locked ball is dropped so the search reruns.
    func pickBall(at p: CGPoint?) {
        pickedPoint = p
        lock.lock(); pickValue = p.map { (Double($0.x), Double($0.y)) }; lock.unlock()
        queue.async {
            switch self.qState {
            case .armed: self.disarm(resumeAfter: 0)
            case .searching: self.candidate = nil; self.stableCount = 0
            default: break
            }
        }
    }

    static func guide(for view: CameraView) -> (Double, Double, Double, Double) {
        switch view {
        case .front: return (0.15, 0.30, 0.85, 0.97)
        // From behind, the ball sits low and in the middle; this keeps sky and the range out.
        case .rear: return (0.25, 0.45, 0.75, 0.97)
        }
    }

    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "headspeed.capture", qos: .userInteractive)
    private let analysisQueue = DispatchQueue(label: "headspeed.analysis", qos: .userInitiated)
    private let lock = NSLock()
    private var clubValue: Club = .driver
    private var viewValue: CameraView = CaptureEngine.savedView
    private var pickValue: (Double, Double)?
    private var device: AVCaptureDevice?
    private let motion = CMMotionManager()
    private var configured = false

    // MARK: Capture-queue state
    private var qState: MeasureState = .starting
    private var frameW = 0, frameH = 0
    private var fieldOfView = 0.0           // horizontal, degrees, of the active format
    private var armedView: CameraView = .front
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
    // Rear: the club set down behind the ball at address hides it from the camera.
    private var recentDiffs: [Double] = []   // ball-patch change over the last 3 frames
    private var covered = false
    private var coveredSince = 0.0
    private var clearCount = 0
    private var coveredFrames = 0
    private var firstOverIndex = 0
    private var triggerIndex = 0
    private var postRemaining = 0
    private var resumeAt = 0.0
    private var armedShutter: Double?
    private var armedPitch = 0.0
    private var lastShutterPublish = 0.0
    private var lastFrameTime = 0.0

    // MARK: Lifecycle

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self else { return }
            guard granted else {
                self.publishState(.unavailable("カメラへのアクセスが許可されていません。設定アプリの「ヘッドスピード」でカメラを許可してください。"))
                return
            }
            DispatchQueue.main.async {
                if self.motion.isDeviceMotionAvailable && !self.motion.isDeviceMotionActive {
                    self.motion.deviceMotionUpdateInterval = 0.2
                    self.motion.startDeviceMotionUpdates()
                }
            }
            self.queue.async {
                if !self.configured { self.configure() }
                if self.configured && !self.session.isRunning { self.session.startRunning() }
            }
        }
    }

    func stop() {
        DispatchQueue.main.async { self.motion.stopDeviceMotionUpdates() }
        queue.async {
            if self.session.isRunning { self.session.stopRunning() }
            // An in-flight analysis works on its own snapshot and disarms when it finishes.
            if self.qState == .armed { self.disarm(resumeAfter: 0) }
        }
    }

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: dev), session.canAddInput(input) else {
            abortConfiguration("背面カメラを使えません。", input: nil, output: nil)
            return
        }
        session.addInput(input)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else {
            abortConfiguration("カメラの出力を設定できませんでした。", input: input, output: nil)
            return
        }
        session.addOutput(output)

        guard let picked = Self.pickFormat(dev) else {
            abortConfiguration("このiPhoneは240fps撮影に対応していないため計測できません。", input: input, output: output)
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
            abortConfiguration("カメラを設定できませんでした。", input: input, output: output)
            return
        }
        session.commitConfiguration()
        device = dev
        fieldOfView = Double(format.videoFieldOfView)
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

    /// Removes what was added so a later configure() can start from scratch.
    private func abortConfiguration(_ message: String, input: AVCaptureInput?, output: AVCaptureOutput?) {
        if let output { session.removeOutput(output) }
        if let input { session.removeInput(input) }
        session.commitConfiguration()
        publishState(.unavailable(message))
    }

    /// 240fps format (the analyzer's frame counts assume it), preferring 1080p, then 720p.
    private static func pickFormat(_ dev: AVCaptureDevice) -> (AVCaptureDevice.Format, Int)? {
        var best: (format: AVCaptureDevice.Format, fps: Int, score: Int)?
        for f in dev.formats {
            let sub = CMFormatDescriptionGetMediaSubType(f.formatDescription)
            guard sub == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || sub == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange else { continue }
            let dims = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            guard dims.width <= 1920, dims.width >= 1280 else { continue }
            let maxFps = f.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 0
            guard maxFps >= 240 else { continue }
            let fps = 240
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
        let gap = t - lastFrameTime
        lastFrameTime = t
        if qState == .armed && gap > 0.1 {
            // Frames stopped (background, interruption): the ring no longer holds a continuous swing.
            disarm(resumeAfter: t + 0.5)
            return
        }
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
        lock.lock(); let rear = viewValue == .rear; let pick = pickValue; lock.unlock()
        // Rear: the ball's outline is searched on a half-size frame (a quarter is too coarse
        // for a ~30 px ball's edges); face-on keeps the quarter-size bright-blob search.
        let f = rear ? 2 : 4
        let sw = W / f, sh = H / f
        var small = [UInt8](repeating: 0, count: sw * sh)
        for sy in 0..<sh {
            let row = y + sy * f * bpr
            for sx in 0..<sw { small[sy * sw + sx] = row[sx * f] }
        }
        let img = GrayImage(width: sw, height: sh, pixels: small)
        // A tapped point narrows the search to the area around it (another ball nearby, such
        // as one left on the auto-tee next to the mat, is otherwise a valid pick too).
        let area = pick.map { Self.pickArea(CGPoint(x: $0.0, y: $0.1)) }
        let found = rear
            ? BallFinder.searchRear(img, guide: area ?? Self.guide(for: .rear), minD: 14.0 / Double(f), maxD: 70.0 / Double(f), anchor: pick ?? (0.5, 0.75))
            : BallFinder.search(img, guide: area ?? Self.guide(for: .front), minD: 12.0 / Double(f), maxD: 110.0 / Double(f))
        guard let coarse = found else {
            candidate = nil; stableCount = 0
            publishBall(nil)
            return
        }
        let approx = BallFix(x: coarse.x * Double(f) + Double(f - 1) / 2, y: coarse.y * Double(f) + Double(f - 1) / 2, d: coarse.d * Double(f))
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
        guard let r = rear ? BallFinder.refineRear(cropImg, approx: local) : BallFinder.refine(cropImg, approx: local) else {
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
        lock.lock(); armedView = viewValue; lock.unlock()
        // Face-on: a strip around the ball. Rear: everything above the ball too, since the
        // head comes down from high in the frame (at half resolution for a 1080p format).
        let rear = armedView == .rear
        let y0 = rear ? 0 : max(0, Int(fix.y - 14 * fix.d))
        let y1 = min(frameH, Int(fix.y + 3 * fix.d))
        let rows = y1 - y0
        let budget = rear ? 520_000 : 460_000
        var step = 1
        while (frameW / step) * (rows / step) > budget { step += 1 }
        let sw = frameW / step, sh = rows / step
        strip = (y0, step, sw, sh)
        // ~1 s of history at 240fps; the analyzer needs 110 frames before impact.
        if let r = ring, r.capacity == 240, r.width == sw, r.height == sh {
            r.reset()
        } else {
            ring = nil
            ring = FrameRing(capacity: 240, width: sw, height: sh)
        }

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
        recentDiffs = []
        covered = false
        clearCount = 0
        setExposureLocked(true)
        if let s = device?.exposureDuration.seconds, s.isFinite, s > 0 {
            armedShutter = s
        } else {
            armedShutter = nil
        }
        armedPitch = cameraPitch()
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
        let threshold = 18.0
        let rear = armedView == .rear
        let prev = recentDiffs.last
        defer {
            recentDiffs.append(diff)
            if recentDiffs.count > 3 { recentDiffs.removeFirst() }
        }

        if covered {
            coveredFrames += 1
            // Just after a cover began, a big one-frame jump is the clubhead arriving at
            // impact after all (it can graze the ball patch a frame early).
            if coveredFrames <= 3, let p = prev, diff - p >= threshold, ring.total >= 150 {
                covered = false
                startAnalysis()
                return
            }
            // Wait for the ball to show again (the takeaway) before watching for the hit.
            clearCount = diff < threshold * 0.5 ? clearCount + 1 : 0
            if clearCount >= 12 {
                covered = false
                clearCount = 0
            } else if t - coveredSince > 20 {
                // Hidden for too long: the ball was probably moved. Look for it again.
                disarm(resumeAfter: t)
            }
            return
        }

        if diff > threshold {
            if overCount == 0 {
                firstOverIndex = ring.total - 1
                // From behind, a club set down behind the ball at address slides over it in
                // several frames; at impact the patch changes by a lot in a single frame.
                if rear, let p = prev, diff - p < threshold {
                    markCovered(t)
                    return
                }
                // Face-on: at address the clubhead's shadow falls on the ball, and the takeaway
                // lifts it again; the patch drifts over many frames. The hit changes it at once.
                // A slow crossing is taken as the new look of the ball at rest.
                if !rear, recentDiffs.count == 3, diff - recentDiffs[0] < threshold * 0.5 {
                    for (j, i) in coreIdx.enumerated() { coreRef[j] = Double(y[i]) }
                    return
                }
            }
            overCount += 1
            if overCount >= 2 {
                if ring.total < 150 {
                    // The ball was disturbed right after locking on: start over.
                    disarm(resumeAfter: t + 0.5)
                    return
                }
                startAnalysis()
            }
        } else {
            overCount = 0
            // Follow slow lighting changes (clouds) while waiting; from behind, not while a
            // club is creeping over the ball.
            if !rear || diff < threshold * 0.5 {
                for (j, i) in coreIdx.enumerated() { coreRef[j] = coreRef[j] * 0.98 + Double(y[i]) * 0.02 }
            }
        }
    }

    /// Whether the whole picture moved fast just before `t` (a downswing), compared with
    /// the second or two before: a clubhead at 30–50 m/s changes far more of the frame
    /// than setting the club down does.
    private static func motionBeforeTrigger(_ times: [Double], _ energy: [Double], at t: Double) -> Bool {
        var recent: [Double] = [], before: [Double] = []
        for (ti, e) in zip(times, energy) {
            if ti > t - 0.3 && ti <= t + 0.01 { recent.append(e) }
            else if ti > t - 2.3 && ti <= t - 0.3 { before.append(e) }
        }
        guard let peak = recent.max(), before.count >= 60 else { return true }   // unknown: treat as a swing
        let base = before.sorted()[before.count / 2]
        return peak > 3 * base + 2
    }

    private func markCovered(_ t: Double) {
        covered = true
        coveredSince = t
        coveredFrames = 0
        clearCount = 0
        overCount = 0
    }

    private func startAnalysis() {
        triggerIndex = firstOverIndex
        postRemaining = 40          // keep ~0.17 s after impact for the ball flight
        qState = .analyzing
        publishState(.analyzing)
    }

    /// Rear view: the "hit" was the club set down over the ball, not a swing. Go back to
    /// waiting with the same ball, as if the cover had been recognised at once.
    private func resumeCovered(_ t: Double) {
        guard armedBall != nil, strip != nil else { disarm(resumeAfter: t); return }
        postRemaining = 0
        recentDiffs = []
        markCovered(t)
        qState = .armed
        publishState(.armed)
    }

    private func disarm(resumeAfter: Double) {
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

    /// How far the camera looks down, in radians (0 = level, face-on from the side).
    /// The rear camera looks along the device's -z axis, so the downward tilt is asin(-gravity.z).
    private func cameraPitch() -> Double {
        guard let g = motion.deviceMotion?.gravity else { return 0 }
        return asin(max(-1, min(1, -g.z)))
    }

    // MARK: Analysis

    private func runAnalysis() {
        guard let ring, let g = strip, let ball = armedBall else { disarm(resumeAfter: 0); return }
        let snap = ring.snapshot()
        var clip = SwingClip(frames: snap.frames, times: snap.times,
                             ballX: ball.x / Double(g.step),
                             ballY: (ball.y - Double(g.y0)) / Double(g.step),
                             ballD: ball.d / Double(g.step),
                             triggerIndex: triggerIndex - snap.firstIndex,
                             cameraPitch: armedPitch)
        let view = armedView
        if view == .rear {
            let fov = fieldOfView > 1 && fieldOfView < 179 ? fieldOfView : 70
            clip.rear = true
            clip.focal = Double(frameW) / 2 / tan(fov * .pi / 360) / Double(g.step)
            clip.centerX = Double(frameW) / 2 / Double(g.step)
            clip.centerY = (Double(frameH) / 2 - Double(g.y0)) / Double(g.step)
        }
        let eTimes = energyTimes, eVals = energyVals
        let shutter = armedShutter
        lock.lock(); let club = clubValue; lock.unlock()

        analysisQueue.async {
            let m = SwingAnalyzer.analyze(clip, club: club)
            let impact = m.impactTime ?? clip.times[min(max(clip.triggerIndex, 0), clip.times.count - 1)]
            let tempo = TempoAnalyzer.tempo(times: eTimes, energy: eVals, impact: impact)
            var result = SwingResult(club: club)
            result.view = view
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
            if !result.hasMeasurement && CameraShake.moved(clip.frames, trigger: clip.triggerIndex) {
                // Nothing measured and the whole picture shifted: the phone was bumped or picked
                // up (often while walking over to look at the last result). Not a swing; look
                // for the ball again, since it may now sit elsewhere in the frame.
                self.queue.async { self.disarm(resumeAfter: (self.energyTimes.last ?? 0) + 1.0) }
                return
            }
            if view == .rear && !result.hasMeasurement
                && !Self.motionBeforeTrigger(eTimes, eVals, at: clip.times[min(max(clip.triggerIndex, 0), clip.times.count - 1)]) {
                // Nothing measured and no fast motion before the "hit": the club was set down
                // over the ball at address. Keep waiting instead of reporting a failed swing.
                self.queue.async { self.resumeCovered(self.energyTimes.last ?? 0) }
                return
            }
            DispatchQueue.main.async {
                self.lastClip = []
                self.onResult?(result)
            }
            // After the result is shown: encoding takes a second or two.
            let files = ClipExporter.export(clip, club: club, result: result) ?? []
            DispatchQueue.main.async { self.lastClip = files }
            self.queue.async {
                // Give the ball time to leave and the golfer time to settle.
                let now = self.energyTimes.last ?? 0
                self.disarm(resumeAfter: now + 1.5)
            }
        }
    }
}
