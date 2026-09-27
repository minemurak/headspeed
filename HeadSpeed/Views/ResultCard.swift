import SwiftUI

struct ResultCard: View {
    let result: SwingResult

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                VStack(alignment: .leading, spacing: 0) {
                    Text("ヘッドスピード").font(.caption).foregroundStyle(.secondary)
                    HStack(alignment: .firstTextBaseline, spacing: 4) {
                        Text(Fmt.num(result.headSpeed, 1)).font(.system(size: 44, weight: .bold, design: .rounded).monospacedDigit())
                        Text("m/s").font(.subheadline).foregroundStyle(.secondary)
                    }
                    if let hs = result.headSpeed {
                        Text("\(Fmt.num(hs * 2.237, 1)) mph").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
                Spacer(minLength: 16)
                VStack(alignment: .trailing, spacing: 0) {
                    Text("ミート率").font(.caption).foregroundStyle(.secondary)
                    Text(Fmt.num(result.smash, 2)).font(.system(size: 34, weight: .bold, design: .rounded).monospacedDigit())
                        .foregroundStyle(smashColor)
                    Text("目安 \(Fmt.num(result.club.smashRange.lowerBound, 2))〜\(Fmt.num(result.club.smashRange.upperBound, 2))")
                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                }
            }

            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 4) {
                GridRow {
                    metric("ボール初速", result.ballSpeed.map { "\(Fmt.num($0, 1)) m/s" })
                    metric("打ち出し角", result.launch.map { "\(Fmt.num($0, 1))°" })
                }
                GridRow {
                    metric("入射角", result.attack.map { "\(Fmt.signed($0, 1))°" })
                    metric("推定キャリー", result.carryYards.map { "\(Int($0.rounded())) yd" })
                }
                GridRow {
                    metric("テンポ", result.tempo.map { "\(Fmt.num($0, 1)) : 1" })
                    metric("シャッター", result.shutter.map { "1/\(Int((1 / $0).rounded()))" })
                }
            }

            ForEach(result.notes, id: \.self) { note in
                Label(note, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
            }
        }
        .padding(16)
        .frame(width: 380)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }

    private func metric(_ label: String, _ value: String?) -> some View {
        HStack(spacing: 6) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(value ?? "—").font(.callout.monospacedDigit().weight(.semibold))
        }
    }

    private var smashColor: Color {
        guard let s = result.smash else { return .secondary }
        let r = result.club.smashRange
        if s > r.upperBound + 0.04 { return .orange }
        if s >= r.lowerBound { return .green }
        if s >= r.lowerBound - 0.06 { return .yellow }
        return .red
    }
}
