import SwiftUI

struct HistoryView: View {
    @ObservedObject var store: HistoryStore
    @Environment(\.dismiss) private var dismiss
    @State private var confirmClear = false

    var body: some View {
        NavigationStack {
            List {
                Section("クラブ別の平均") {
                    ForEach(Club.allCases.filter { store.count($0) > 0 }) { club in
                        HStack {
                            Text(club.label).frame(width: 90, alignment: .leading)
                            Text("\(store.count(club))球").foregroundStyle(.secondary).frame(width: 50, alignment: .leading)
                            Spacer()
                            stat("HS", Fmt.num(store.average(club) { $0.headSpeed }, 1))
                            stat("初速", Fmt.num(store.average(club) { $0.ballSpeed }, 1))
                            stat("ミート率", Fmt.num(store.average(club) { $0.smash }, 2))
                        }
                    }
                    if store.results.isEmpty {
                        Text("まだ記録がありません。ボールを置いて打つと自動で保存されます。").foregroundStyle(.secondary)
                    }
                }
                Section("すべての計測") {
                    ForEach(store.results) { r in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(r.date, format: .dateTime.month().day().hour().minute())
                                Text(r.club.label).font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            stat("HS", Fmt.num(r.headSpeed, 1))
                            stat("初速", Fmt.num(r.ballSpeed, 1))
                            stat("ミート率", Fmt.num(r.smash, 2))
                            stat("打出", r.launch.map { "\(Fmt.num($0, 1))°" } ?? "—")
                        }
                    }
                    .onDelete(perform: store.remove)
                }
            }
            .navigationTitle("履歴")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("閉じる") { dismiss() } }
                ToolbarItemGroup(placement: .primaryAction) {
                    ShareLink(item: csvFile, preview: SharePreview("ヘッドスピード履歴.csv")) {
                        Image(systemName: "square.and.arrow.up")
                    }
                    Button(role: .destructive) { confirmClear = true } label: { Image(systemName: "trash") }
                        .disabled(store.results.isEmpty)
                }
            }
            .confirmationDialog("履歴をすべて削除しますか？", isPresented: $confirmClear, titleVisibility: .visible) {
                Button("すべて削除", role: .destructive) { store.removeAll() }
            }
        }
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .trailing, spacing: 0) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.callout.monospacedDigit())
        }
        .frame(width: 64, alignment: .trailing)
    }

    /// UTF-8 with BOM and CRLF so Excel opens it correctly.
    private var csvFile: URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ヘッドスピード履歴.csv")
        try? Data(("\u{FEFF}" + csv).utf8).write(to: url, options: .atomic)
        return url
    }

    private var csv: String {
        var lines = ["日時,クラブ,ヘッドスピード(m/s),ボール初速(m/s),ミート率,打ち出し角,入射角,テンポ"]
        let df = DateFormatter()
        df.locale = Locale(identifier: "en_US_POSIX")
        df.timeZone = .current
        df.dateFormat = "yyyy/MM/dd HH:mm"
        for r in store.results {
            lines.append([df.string(from: r.date), r.club.label,
                          Fmt.num(r.headSpeed, 1), Fmt.num(r.ballSpeed, 1), Fmt.num(r.smash, 2),
                          Fmt.num(r.launch, 1), Fmt.num(r.attack, 1), Fmt.num(r.tempo, 2)]
                .map { $0 == "—" ? "" : $0 }.joined(separator: ","))
        }
        return lines.joined(separator: "\r\n") + "\r\n"
    }
}
