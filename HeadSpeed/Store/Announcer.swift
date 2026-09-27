import AVFoundation
import Combine

/// Reads the result aloud so the golfer does not have to walk to the phone.
final class Announcer: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    private let synth = AVSpeechSynthesizer()

    override init() {
        super.init()
        synth.delegate = self
        try? AVAudioSession.sharedInstance().setCategory(.playback, options: [.mixWithOthers, .duckOthers])
    }

    func announce(_ r: SwingResult) {
        var parts: [String] = []
        if let hs = r.headSpeed { parts.append("ヘッドスピード \(String(format: "%.1f", hs))") }
        if let s = r.smash { parts.append("ミート率 \(String(format: "%.2f", s))") }
        if parts.isEmpty { parts.append("計測できませんでした") }
        let u = AVSpeechUtterance(string: parts.joined(separator: "、"))
        u.voice = AVSpeechSynthesisVoice(language: "ja-JP")
        u.rate = 0.52
        synth.stopSpeaking(at: .immediate)
        synth.speak(u)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        deactivateIfIdle(synthesizer)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        deactivateIfIdle(synthesizer)
    }

    /// Lets other audio (music) return to full volume after ducking.
    private func deactivateIfIdle(_ synthesizer: AVSpeechSynthesizer) {
        guard !synthesizer.isSpeaking else { return }
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}
