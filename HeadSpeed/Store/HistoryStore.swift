import Foundation
import SwiftUI

/// Saved swings, kept on the device (UserDefaults, JSON).
final class HistoryStore: ObservableObject {
    @Published private(set) var results: [SwingResult] = []
    private let key = "headspeed.history.v1"

    init() {
        if let data = UserDefaults.standard.data(forKey: key),
           let saved = try? JSONDecoder().decode([SwingResult].self, from: data) {
            results = saved
        }
    }

    func add(_ r: SwingResult) {
        guard r.hasMeasurement else { return }
        results.insert(r, at: 0)
        save()
    }

    func remove(at offsets: IndexSet) {
        results.remove(atOffsets: offsets)
        save()
    }

    func removeAll() {
        results = []
        save()
    }

    func average(_ club: Club, _ value: (SwingResult) -> Double?) -> Double? {
        let v = results.filter { $0.club == club }.compactMap(value)
        return v.isEmpty ? nil : v.reduce(0, +) / Double(v.count)
    }

    func count(_ club: Club) -> Int { results.filter { $0.club == club }.count }

    private func save() {
        if let data = try? JSONEncoder().encode(results) { UserDefaults.standard.set(data, forKey: key) }
    }
}
