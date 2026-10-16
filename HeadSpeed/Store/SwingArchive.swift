import Foundation

/// Keeps each saved swing's exported files (see ClipExporter) next to its history record, so a
/// swing can be exported again later from the history. swing.json is kept for every record;
/// the movie only for the most recent `keepMovies` swings, since each one is tens of MB.
enum SwingArchive {
    static let keepMovies = 10

    static var root: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("Swings", isDirectory: true)
    }

    private static func folder(_ id: UUID) -> URL {
        root.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    /// Copies the exported files (swing.mov / swing.json) for record `id`, named after its date
    /// so several swings exported together don't collide. Call off the main thread.
    static func store(_ files: [URL], id: UUID, date: Date) {
        guard !files.isEmpty else { return }
        let fm = FileManager.default
        let dir = folder(id)
        do { try fm.createDirectory(at: dir, withIntermediateDirectories: true) } catch { return }
        let name = "swing-" + stamp(date)
        for src in files {
            let dst = dir.appendingPathComponent(name).appendingPathExtension(src.pathExtension)
            try? fm.removeItem(at: dst)
            try? fm.copyItem(at: src, to: dst)
        }
        pruneMovies()
    }

    /// The files kept for record `id` (movie first), or empty.
    static func files(_ id: UUID) -> [URL] {
        let items = (try? FileManager.default.contentsOfDirectory(at: folder(id), includingPropertiesForKeys: nil)) ?? []
        return items.filter { ["mov", "json"].contains($0.pathExtension.lowercased()) }
            .sorted { $0.pathExtension.lowercased() == "mov" && $1.pathExtension.lowercased() != "mov" }
    }

    static func remove(_ id: UUID) {
        try? FileManager.default.removeItem(at: folder(id))
    }

    static func removeAll() {
        try? FileManager.default.removeItem(at: root)
    }

    /// Deletes movies beyond the newest `keepMovies`; their swing.json stays.
    private static func pruneMovies() {
        let fm = FileManager.default
        let dirs = (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.creationDateKey])) ?? []
        let movies = dirs.compactMap { dir -> (URL, Date)? in
            let items = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
            guard let mov = items.first(where: { $0.pathExtension.lowercased() == "mov" }) else { return nil }
            let created = (try? dir.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            return (mov, created)
        }.sorted { $0.1 > $1.1 }
        for (mov, _) in movies.dropFirst(keepMovies) { try? fm.removeItem(at: mov) }
    }

    private static func stamp(_ date: Date) -> String {
        let df = DateFormatter()
        df.locale = Locale(identifier: "en_US_POSIX")
        df.dateFormat = "yyyyMMdd-HHmmss"
        return df.string(from: date)
    }
}
