import Foundation
import SwiftUI

/// Saved swings, kept on the device (UserDefaults, JSON).
final class HistoryStore: ObservableObject {
    @Published private(set) var results: [SwingResult] = []
    private let key = "headspeed.history.v1"

    init() {
        guard let data = UserDefaults.standard.data(forKey: key) else { return }
        if let saved = try? JSONDecoder().decode([SwingResult].self, from: data) {
            results = saved
        } else {
            // Keep unreadable data so the next save does not silently destroy it.
            let stamp = Int(Date().timeIntervalSince1970)
            UserDefaults.standard.set(data, forKey: key + ".backup-\(stamp)")
        }
    }

    func add(_ result: SwingResult) {
        let r = Self.sanitized(result)
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

    /// JSONEncoder rejects NaN and infinity, which would make every later save fail.
    private static func sanitized(_ r: SwingResult) -> SwingResult {
        func finite(_ v: Double?) -> Double? {
            guard let v, v.isFinite else { return nil }
            return v
        }
        var s = r
        s.headSpeed = finite(r.headSpeed)
        s.ballSpeed = finite(r.ballSpeed)
        s.smash = finite(r.smash)
        s.launch = finite(r.launch)
        s.attack = finite(r.attack)
        s.backswing = finite(r.backswing)
        s.downswing = finite(r.downswing)
        s.shutter = finite(r.shutter)
        return s
    }

    private func save() {
        if let data = try? JSONEncoder().encode(results) { UserDefaults.standard.set(data, forKey: key) }
    }
}
