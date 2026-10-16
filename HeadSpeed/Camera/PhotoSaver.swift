import Foundation
import Photos

/// Saves an exported swing movie to the Camera Roll. Asks only for add-only access, so the
/// app never sees the user's library.
enum PhotoSaver {
    enum Failure: Error { case denied, failed }

    static func saveVideo(at url: URL) async throws {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else { throw Failure.denied }
        do {
            try await PHPhotoLibrary.shared().performChanges {
                let req = PHAssetCreationRequest.forAsset()
                let opts = PHAssetResourceCreationOptions()
                opts.shouldMoveFile = false
                req.addResource(with: .video, fileURL: url, options: opts)
            }
        } catch {
            throw Failure.failed
        }
    }
}
