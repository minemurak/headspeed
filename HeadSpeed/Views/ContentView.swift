import SwiftUI

struct ContentView: View {
    @StateObject private var engine = CaptureEngine()
    @StateObject private var history = HistoryStore()
    @StateObject private var announcer = Announcer()
    @State private var result: SwingResult?
    @State private var showHistory = false
    @AppStorage("headspeed.speak") private var speak = true

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(session: engine.session).ignoresSafeArea()
            GuideOverlay(engine: engine).ignoresSafeArea()

            VStack(spacing: 10) {
                HStack(alignment: .top, spacing: 10) {
                    StatusPill(state: engine.state, ballTooSmall: engine.ballTooSmall, shutterTooSlow: engine.shutterTooSlow)
                    Spacer()
                    Picker("クラブ", selection: $engine.club) {
                        ForEach(Club.allCases) { Text($0.label).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 320)
                    Button { speak.toggle() } label: {
                        Image(systemName: speak ? "speaker.wave.2.fill" : "speaker.slash.fill")
                            .frame(width: 36, height: 32)
                    }
                    .accessibilityLabel(speak ? "読み上げをオフ" : "読み上げをオン")
                    Button { showHistory = true } label: {
                        Image(systemName: "list.bullet").frame(width: 36, height: 32)
                    }
                    .accessibilityLabel("履歴")
                }
                .buttonStyle(.bordered)
                .tint(.white)

                Spacer()

                HStack(alignment: .bottom) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(engine.formatLabel)
                        Text("シャッター \(engine.shutterLabel)")
                            .foregroundStyle(engine.shutterTooSlow ? .orange : .white.opacity(0.7))
                    }
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.7))
                    Spacer()
                    if let result {
                        ResultCard(result: result)
                            .transition(.move(edge: .trailing).combined(with: .opacity))
                    }
                }
            }
            .padding()
        }
        .animation(.easeOut(duration: 0.25), value: result)
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            engine.onResult = { r in
                result = r
                history.add(r)
                if speak { announcer.announce(r) }
            }
            engine.start()
        }
        .onDisappear {
            engine.stop()
            UIApplication.shared.isIdleTimerDisabled = false
        }
        .sheet(isPresented: $showHistory) { HistoryView(store: history) }
    }
}

/// Guide box for ball placement and the detected ball.
struct GuideOverlay: View {
    @ObservedObject var engine: CaptureEngine

    var body: some View {
        GeometryReader { geo in
            let rect = fitted(geo.size, aspect: engine.frameAspect)
            let g = CaptureEngine.guide
            let guideRect = CGRect(x: rect.minX + rect.width * CGFloat(g.0), y: rect.minY + rect.height * CGFloat(g.1),
                                   width: rect.width * CGFloat(g.2 - g.0), height: rect.height * CGFloat(g.3 - g.1))
            ZStack {
                Path { $0.addRoundedRect(in: guideRect, cornerSize: CGSize(width: 12, height: 12)) }
                    .stroke(.white.opacity(engine.state == .armed ? 0.15 : 0.45), style: StrokeStyle(lineWidth: 1.5, dash: [8, 6]))
                if let b = engine.ballOverlay {
                    let size = max(CGFloat(18), CGFloat(b.d) * rect.width * 1.5)
                    Circle()
                        .stroke(engine.state == .armed ? Color.green : Color.yellow, lineWidth: 3)
                        .frame(width: size, height: size)
                        .position(x: rect.minX + CGFloat(b.x) * rect.width, y: rect.minY + CGFloat(b.y) * rect.height)
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func fitted(_ size: CGSize, aspect: Double) -> CGRect {
        guard size.width > 0, size.height > 0 else { return .zero }
        if size.width / size.height > CGFloat(aspect) {
            let w = size.height * CGFloat(aspect)
            return CGRect(x: (size.width - w) / 2, y: 0, width: w, height: size.height)
        } else {
            let h = size.width / CGFloat(aspect)
            return CGRect(x: 0, y: (size.height - h) / 2, width: size.width, height: h)
        }
    }
}

struct StatusPill: View {
    let state: MeasureState
    let ballTooSmall: Bool
    let shutterTooSlow: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Circle().fill(color).frame(width: 10, height: 10)
                Text(title).font(.headline)
            }
            if let hint {
                Text(hint).font(.caption).foregroundStyle(.white.opacity(0.8))
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .frame(maxWidth: 380, alignment: .leading)
    }

    private var title: String {
        switch state {
        case .starting: return "カメラを準備しています"
        case .unavailable: return "撮影できません"
        case .searching: return "枠の中にボールを置いてください"
        case .armed: return "準備OK　いつでも打ってください"
        case .analyzing: return "解析中…"
        }
    }

    private var hint: String? {
        if case .unavailable(let msg) = state { return msg }
        if ballTooSmall { return "ボールが小さく写っています。カメラを近づけてください。" }
        if shutterTooSlow { return "暗いためシャッターが遅く、精度が落ちます。明るい場所で撮影してください。" }
        if state == .searching { return "正面から、ボールと同じ高さで2〜3m離して固定してください。" }
        return nil
    }

    private var color: Color {
        switch state {
        case .armed: return .green
        case .analyzing: return .blue
        case .unavailable: return .red
        default: return .yellow
        }
    }
}
