import AVFoundation
import SwiftUI

/// Live camera preview. The app runs in landscape (home side right), which matches
/// the back camera's native buffer orientation, so preview and analysis frames line up.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspect
        v.backgroundColor = .black
        v.observe(session)
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

        /// The session is configured asynchronously, so the connection may not exist at first layout.
        func observe(_ session: AVCaptureSession) {
            NotificationCenter.default.addObserver(self, selector: #selector(sessionDidStartRunning),
                                                   name: AVCaptureSession.didStartRunningNotification, object: session)
        }

        @objc private func sessionDidStartRunning(_ note: Notification) {
            DispatchQueue.main.async { [weak self] in
                self?.setNeedsLayout()
                self?.applyRotation()
            }
        }

        deinit {
            NotificationCenter.default.removeObserver(self)
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            applyRotation()
        }

        private func applyRotation() {
            guard let c = previewLayer.connection else { return }
            if #available(iOS 17.0, *) {
                if c.isVideoRotationAngleSupported(0) { c.videoRotationAngle = 0 }
            } else if c.isVideoOrientationSupported {
                c.videoOrientation = .landscapeRight
            }
        }
    }
}
