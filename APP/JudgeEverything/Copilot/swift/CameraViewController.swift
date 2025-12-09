import UIKit
import AVFoundation

class CameraViewController: UIViewController {
    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    let segmentationEngine = SegmentationEngine()
    
    var latestImage: UIImage?
    var maskOverlayView: UIImageView?

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupGestureRecognizers()
    }

    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .high

        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("No camera available")
            return
        }
        
        guard let videoInput = try? AVCaptureDeviceInput(device: videoDevice) else {
            print("Failed to create video input")
            return
        }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }

        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        captureSession.startRunning()
    }

    func setupGestureRecognizers() {
        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        view.addGestureRecognizer(tap)
    }

    @objc func handleTap(_ gesture: UITapGestureRecognizer) {
        let tapPoint = gesture.location(in: view)
        guard let image = latestImage else {
            print("No image available")
            return
        }
        
        // Step 1: Run YOLOv9 detection
        if let detections = segmentationEngine.detectObjects(in: image) {
            // Find detection box closest to tap point or containing tap point
            var selectedBox: CGRect? = nil
            
            for detection in detections {
                if detection.bbox.contains(tapPoint) {
                    selectedBox = detection.bbox
                    break
                }
            }
            
            if selectedBox == nil && !detections.isEmpty {
                // If no box contains tap, use closest by center distance
                let tapCenter = CGPoint(x: tapPoint.midX, y: tapPoint.midY)
                var minDistance = CGFloat.infinity
                
                for detection in detections {
                    let center = CGPoint(x: detection.bbox.midX, y: detection.bbox.midY)
                    let distance = hypot(center.x - tapCenter.x, center.y - tapCenter.y)
                    if distance < minDistance {
                        minDistance = distance
                        selectedBox = detection.bbox
                    }
                }
            }
            
            // Step 2: Run MobileSAM mask generation on selected box
            if let box = selectedBox {
                if let maskImage = segmentationEngine.generateMask(for: image, in: box) {
                    showMaskOverlay(maskImage)
                }
            }
        } else {
            print("No detections available")
        }
    }

    func showMaskOverlay(_ mask: UIImage) {
        // Remove previous overlay if exists
        maskOverlayView?.removeFromSuperview()
        
        let iv = UIImageView(image: mask)
        iv.frame = view.bounds
        iv.contentMode = .scaleAspectFill
        iv.alpha = 0.6
        iv.backgroundColor = .clear
        view.addSubview(iv)
        self.maskOverlayView = iv
        
        // Auto-remove after 1 second
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            iv.removeFromSuperview()
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            let uiImage = UIImage(cgImage: cgImage)
            DispatchQueue.main.async {
                self.latestImage = uiImage
            }
        }
    }
}

