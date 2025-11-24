import UIKit
import AVFoundation

class CameraViewController: UIViewController {
    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    let network = NetworkClient(server: "http://127.0.0.1:8000/segment") // change host/ip as needed

    override func viewDidLoad() {
        super.viewDidLoad()

        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .high

        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        guard let videoInput = try? AVCaptureDeviceInput(device: videoDevice) else { return }
        if captureSession.canAddInput(videoInput) { captureSession.addInput(videoInput) }

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoOutput)

        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        view.addGestureRecognizer(tap)

        captureSession.startRunning()
    }

    var latestImage: UIImage?

    @objc func handleTap(_ gesture: UITapGestureRecognizer) {
        let p = gesture.location(in: view)
        guard let image = latestImage else { return }
        // Convert tap point from view coords to image coords if needed
        // For simplicity pass view coords; server should be aware of image size
        network.sendFrame(image, click: p) { result in
            switch result {
            case .success(let json):
                print("Got response", json)
                if let masks = json["masks"] as? [String], masks.count > 0 {
                    let b64 = masks[0]
                    DispatchQueue.main.async {
                        if let maskData = Data(base64Encoded: b64), let maskImg = UIImage(data: maskData) {
                            self.showMaskOverlay(maskImg)
                        }
                    }
                }
            case .failure(let err):
                print("Network error", err)
            }
        }
    }

    func showMaskOverlay(_ mask: UIImage) {
        let iv = UIImageView(image: mask)
        iv.frame = view.bounds
        iv.contentMode = .scaleAspectFill
        iv.alpha = 0.6
        iv.backgroundColor = .clear
        view.addSubview(iv)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            iv.removeFromSuperview()
        }
    }
}

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            let uiImage = UIImage(cgImage: cgImage)
            self.latestImage = uiImage
        }
    }
}
