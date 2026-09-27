import AVFoundation
import Combine

/// Reads the result aloud so the golfer does not have to walk to the phone.
final class Announcer: ObservableObject {
    private let synth = AVSpeechSynthesizer()

    init() {
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
}
