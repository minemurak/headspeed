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
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

        override func layoutSubviews() {
            super.layoutSubviews()
            guard let c = previewLayer.connection else { return }
            if #available(iOS 17.0, *) {
                if c.isVideoRotationAngleSupported(0) { c.videoRotationAngle = 0 }
            } else if c.isVideoOrientationSupported {
                c.videoOrientation = .landscapeRight
            }
        }
    }
}
