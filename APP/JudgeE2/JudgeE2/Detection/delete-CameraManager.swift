import AVFoundation
import Accelerate
import Combine
import CoreImage
import CoreML
import Foundation
import SwiftUI
import MachO
import UIKit

final class CameraManager: NSObject, ObservableObject {
    let session = AVCaptureSession()
    @Published var boxes: [CGRect] = []
    @Published var maskImage: CGImage?
    @Published var maskRotationAngle: CGFloat = 0
    @Published var maskMirrored: Bool = false

    var currentMode: AppMode = .detectionOnly

    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private let videoQueue = DispatchQueue(label: "camera.video.queue")
    private let ciContext = CIContext()
    private var model: yolov9_c?
    private var isProcessing = false
    private var currentPosition: AVCaptureDevice.Position = .back
    private var backend: InferenceBackend = .all
    private var videoConnection: AVCaptureConnection?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var lastRotationAngle: CGFloat = 0

    private var inferenceTimesMs: [Double] = []
    private let inferenceStatsWindow = 100

    // SAM encoder sliding-window stats (n=100, cold-start excluded)
    // Only accessed on encoderQueue — no extra lock needed.
    private var samEncoderTimesMs: [Double] = []
    private let samEncoderStatsWindow = 100
    private var samEncoderCallCount = 0    // first call is ANE cold-start; skipped from stats

    // SAM decoder sliding-window stats (n=100, cold-start excluded)
    // Only accessed on decoderQueue — no extra lock needed.
    private var samDecoderTimesMs: [Double] = []
    private let samDecoderStatsWindow = 100
    private var samDecoderCallCount = 0    // first call is ANE cold-start; skipped from stats

    private var lastPreprocessMs: Double = 0

    struct Detection {
        let x1: Float
        let y1: Float
        let x2: Float
        let y2: Float
        let score: Float
        let classId: Int
    }

    private struct LetterboxInfo {
        let origW: Float
        let origH: Float
        let scale: Float
        let padX: Float
        let padY: Float
        let inputSize: Float
    }
    private var lastLetterbox: LetterboxInfo?
    private var latestInputBuffer: CVPixelBuffer?
    private var latestCameraBuffer: CVPixelBuffer?

    // MARK: - Segmentation queues & locks
    // Encoder and decoder run on *separate* serial queues so a slow encode
    // never delays mask updates from the decoder.
    private let encoderQueue = DispatchQueue(label: "sam.encoder.queue", qos: .userInitiated)
    private let decoderQueue = DispatchQueue(label: "sam.decoder.queue", qos: .userInitiated)

    // stateLock guards: isEncoding, isDecoding, embeddingCache (written on encoderQueue, read on videoQueue)
    private let stateLock = NSLock()
    private var isEncoding = false
    private var isDecoding = false

    // Pre-allocated transpose buffer: [8400 × 84] Floats = 2.8 MB, reused every frame.
    // Avoids the per-frame heap alloc that caused Post ~250-400 ms regression.
    private var transposeBuffer = [Float](repeating: 0, count: 8400 * 84)

    // Pre-allocated 640×640 BGRA letterbox output buffer (reused every frame on videoQueue).
    // Eliminates per-frame CVPixelBufferCreate heap alloc that caused Pre spikes of 10-16 ms.
    private var letterboxOutputBuffer: CVPixelBuffer?

    private var samEncoder: SAMEncoder?
    private var samDecoder: SAMDecoder?
    private let maskRenderer = MaskRenderer()

    private var encoderEveryNFrames: Int = 12    // ~4 s at 3 fps; refresh embedding regularly
    private var decoderEveryNFrames: Int = 2     // decode every 2nd frame → ~1.5 Hz mask refresh
    private var frameIndex: Int = 0

    // Access to embeddingCache must be done while holding stateLock.
    private var embeddingCache: (embedding: MLMultiArray, timestampMs: Double)?

    private var encoderHitCount: Int = 0
    private var encoderMissCount: Int = 0
    private var decoderCount: Int = 0
    private let debugSegmentation: Bool = true
    private var debugFrameCount: Int = 0

    // TemporalManager owns: primary-object selection, drift classification,
    // mask cache + TTL (default 1200 ms), geometry-signature tracking,
    // and embedding TTL validation (default 8000 ms).
    private let temporal = TemporalManager()

    override init() {
        super.init()
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(handleOrientationChange),
                                               name: UIDevice.orientationDidChangeNotification,
                                               object: nil)
        sessionQueue.async { [weak self] in
            self?.configureSession()
        }
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        UIDevice.current.endGeneratingDeviceOrientationNotifications()
    }

    func start() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if !self.session.isRunning {
                self.session.startRunning()
            }
        }
    }

    func setBackend(_ backend: InferenceBackend) {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.backend = backend
            self.reloadModel()
        }
    }

    func setMode(_ mode: AppMode) {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.currentMode = mode
            if mode == .segmentation {
                self.warmupSegmentationIfPossible()
            } else {
                // Clear all caches when leaving segmentation mode to free memory.
                self.stateLock.lock()
                self.embeddingCache = nil
                self.stateLock.unlock()
                self.temporal.invalidateMask()
                DispatchQueue.main.async {
                    self.maskImage = nil
                }
            }
        }
    }

    private func warmupSegmentationIfPossible() {
        // §2.1 fix: dispatch to videoQueue first to take safe snapshots of
        // latestCameraBuffer and lastLetterbox (both written on videoQueue).
        // Previously this ran on sessionQueue, causing cross-queue reads without locks.
        videoQueue.async { [weak self] in
            guard let self = self else { return }
            guard let cameraBuffer = self.latestCameraBuffer else { return }  // safe: on videoQueue
            let capturedLetterbox = self.lastLetterbox                         // snapshot on videoQueue

            if self.samEncoder == nil { self.samEncoder = SAMEncoder(computeUnits: self.backend.computeUnits) }
            if self.samDecoder == nil { self.samDecoder = SAMDecoder(computeUnits: self.backend.computeUnits) }

            // Claim the encoder slot so the main pipeline knows encoding is in progress.
            self.stateLock.lock()
            guard !self.isEncoding else { self.stateLock.unlock(); return }
            self.isEncoding = true
            self.stateLock.unlock()

            self.encoderQueue.async { [weak self] in
                guard let self = self, let encoder = self.samEncoder else {
                    self?.stateLock.lock(); self?.isEncoding = false; self?.stateLock.unlock()
                    return
                }
                let t0 = PerfLogger.nowMs()
                if let embedding = encoder.encode(pixelBuffer: cameraBuffer) {
                    let t1 = PerfLogger.nowMs()
                    self.stateLock.lock()
                    self.embeddingCache = (embedding: embedding, timestampMs: t1)
                    self.isEncoding = false
                    self.stateLock.unlock()
                    print(String(format: "SAM encoder warmup latency: %.2f ms", t1 - t0))

                    // Warm up decoder using the videoQueue snapshot (§2.1 fix: no more
                    // self.lastLetterbox read on encoderQueue).
                    if let decoder = self.samDecoder,
                       let lb = capturedLetterbox,
                       let prompt = PromptBuilder.buildBoxPrompt(
                            x1: 0.25 * lb.origW, y1: 0.25 * lb.origH,
                            x2: 0.75 * lb.origW, y2: 0.75 * lb.origH,
                            origW: lb.origW, origH: lb.origH, inputSize: 1024) {
                        let t2 = PerfLogger.nowMs()
                        _ = decoder.decode(embedding: embedding, prompt: prompt)
                        print(String(format: "SAM decoder warmup latency: %.2f ms", PerfLogger.nowMs() - t2))
                    }
                } else {
                    print("SAM encoder warmup failed")
                    self.stateLock.lock(); self.isEncoding = false; self.stateLock.unlock()
                }
            }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    // MARK: - Phase 3 Tap-to-Segment (Day 2: geometry exposure + tap intake)

    /// Snapshot of the current canonical frame geometry for TouchHandler.
    /// Canonical = original camera frame in display orientation (origW/origH already
    /// carry device rotation, matching the mask pipeline). Returns nil until the
    /// first letterbox pass has populated `lastLetterbox`.
    /// NOTE: `lastLetterbox` is written on videoQueue; this is a lightweight value
    /// read for a UI gesture and is treated as an eventually-consistent snapshot.
    func currentFrameGeometry() -> FrameGeometry? {
        guard let info = lastLetterbox else { return nil }
        return FrameGeometry(origW: CGFloat(info.origW),
                             origH: CGFloat(info.origH),
                             mirrored: (currentPosition == .front),
                             rotation: lastRotationAngle)
    }

    /// Day 2: single-tap intake. No SAM call yet — the coordinate transform is
    /// already logged by TouchHandler; this confirms the value reached the pipeline.
    /// Encoder/decoder wiring lands in Day 3.
    func handleTap(canonicalPoint: CGPoint) {
        guard currentMode == .tapToSegment else { return }
        print(String(format: "[TAP] pipeline received canonical=(%.1f,%.1f) — Day 2 (no SAM call)",
                     canonicalPoint.x, canonicalPoint.y))
    }

    /// Day 2: double-tap → clear all tap masks. TapInstanceManager arrives Day 5;
    /// for now this only clears the current mask overlay so the gesture is verifiable.
    func handleClearAllTapMasks() {
        guard currentMode == .tapToSegment else { return }
        print("[TAP] pipeline received clearAll — Day 2 (clears overlay)")
        DispatchQueue.main.async { [weak self] in self?.maskImage = nil }
    }

    @objc private func handleOrientationChange() {
        sessionQueue.async { [weak self] in self?.updateRotation() }
    }

    private func desiredRotationAngle() -> CGFloat {
        switch UIDevice.current.orientation {
        case .landscapeRight:       return 0
        case .landscapeLeft:        return 180
        case .portraitUpsideDown:   return 270
        default:                    return 90
        }
    }

    private func updateRotation() {
        guard let output = videoOutput,
              let connection = output.connection(with: .video) else { return }
        let angle = desiredRotationAngle()
        if #available(iOS 17.0, *) {
            if connection.isVideoRotationAngleSupported(angle) {
                connection.videoRotationAngle = angle
                lastRotationAngle = angle
            }
        } else {
            if connection.isVideoOrientationSupported {
                let orientation: AVCaptureVideoOrientation
                switch UIDevice.current.orientation {
                case .landscapeLeft:        orientation = .landscapeLeft
                case .landscapeRight:       orientation = .landscapeRight
                case .portraitUpsideDown:   orientation = .portraitUpsideDown
                default:                    orientation = .portrait
                }
                connection.videoOrientation = orientation
                lastRotationAngle = orientation == .portrait ? 90
                    : orientation == .portraitUpsideDown ? 270
                    : orientation == .landscapeLeft ? 180 : 0
            }
        }
        if debugSegmentation {
            print(String(format: "[DBG] device rotation updated: %.0f", lastRotationAngle))
        }
        DispatchQueue.main.async { [weak self] in
            self?.maskRotationAngle = self?.lastRotationAngle ?? 0
        }
    }

    func toggleCamera() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.currentPosition = (self.currentPosition == .back) ? .front : .back
            self.session.beginConfiguration()
            for input in self.session.inputs { self.session.removeInput(input) }
            if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: self.currentPosition),
               let input = try? AVCaptureDeviceInput(device: device),
               self.session.canAddInput(input) {
                self.session.addInput(input)
            } else {
                print("Camera input unavailable when toggling")
            }
            self.session.commitConfiguration()
            self.updateRotation()
            if let output = self.videoOutput,
               let conn = output.connection(with: .video),
               conn.isVideoMirroringSupported {
                conn.automaticallyAdjustsVideoMirroring = false
                conn.isVideoMirrored = (self.currentPosition == .front)
            }
            DispatchQueue.main.async { [weak self] in
                self?.maskMirrored = (self?.currentPosition == .front)
            }
        }
    }

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .high
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: currentPosition),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            print("Camera input unavailable")
            return
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            print("Camera output unavailable")
            return
        }
        session.addOutput(output)
        videoOutput = output
        updateRotation()
        if let conn = output.connection(with: .video), conn.isVideoMirroringSupported {
            conn.isVideoMirrored = (currentPosition == .front)
        }
        DispatchQueue.main.async { [weak self] in
            self?.maskMirrored = (self?.currentPosition == .front)
        }
        session.commitConfiguration()
        reloadModel()
    }

    private func reloadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = backend.computeUnits
            model = try yolov9_c(configuration: config)
            if let loaded = model {
                PerfLogger.logComputeUnits(config)
                PerfLogger.logInputInfo(loaded.model)
            }
        } catch {
            print("Model load failed in CameraManager: \(error)")
        }
    }
}

// MARK: - Camera delegate
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        PerfLogger.logFpsAndMemoryEverySecond()
        guard model != nil else { return }
        if isProcessing { return }
        isProcessing = true
        defer { isProcessing = false }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        latestCameraBuffer = pixelBuffer

        let tPreStart = PerfLogger.nowMs()
        guard let inputBuffer = letterboxToSquare(pixelBuffer: pixelBuffer, size: 640) else { return }
        lastPreprocessMs = PerfLogger.nowMs() - tPreStart
        latestInputBuffer = inputBuffer

        if debugSegmentation {
            debugFrameCount += 1
            if debugFrameCount % 30 == 0 {
                let camW = CVPixelBufferGetWidth(pixelBuffer)
                let camH = CVPixelBufferGetHeight(pixelBuffer)
                let inW  = CVPixelBufferGetWidth(inputBuffer)
                let inH  = CVPixelBufferGetHeight(inputBuffer)
                if let info = lastLetterbox {
                    print(String(format: "[DBG] camera=%dx%d | modelInput=%dx%d | letterbox scale=%.4f padX=%.2f padY=%.2f | rot=%.0f mirrored=%@",
                                 camW, camH, inW, inH, info.scale, info.padX, info.padY,
                                 lastRotationAngle, (currentPosition == .front) ? "true" : "false"))
                } else {
                    print(String(format: "[DBG] camera=%dx%d | modelInput=%dx%d | rot=%.0f mirrored=%@",
                                 camW, camH, inW, inH, lastRotationAngle, (currentPosition == .front) ? "true" : "false"))
                }
            }
        }
        runDetectionPipeline()
    }

    func runDetectionPipeline() {
        guard let model = model, let inputBuffer = latestInputBuffer else { return }

        let tInferStart = PerfLogger.nowMs()
        do {
            let output = try model.prediction(image: inputBuffer)
            let tInferEnd = PerfLogger.nowMs()
            let inferMs = tInferEnd - tInferStart

            PerfLogger.logOutputShapesOnce(var_3019: output.var_3019, var_3022: output.var_3022)

            inferenceTimesMs.append(inferMs)
            if inferenceTimesMs.count >= inferenceStatsWindow {
                let mean = inferenceTimesMs.reduce(0.0, +) / Double(inferenceTimesMs.count)
                let sorted = inferenceTimesMs.sorted()
                let p95 = sorted[max(0, Int(ceil(0.95 * Double(sorted.count))) - 1)]
                print(String(format: "Inference time stats (n=%d): mean=%.2f ms | p95=%.2f ms",
                             sorted.count, mean, p95))
                inferenceTimesMs.removeAll(keepingCapacity: true)
            }

            let tPostStart = PerfLogger.nowMs()
            let detections = decodeDetections(from: output.var_3019, confidenceThreshold: 0.25)
            let topK = 10
            let topDetections = detections.sorted { $0.score > $1.score }.prefix(topK)
            let nmsDetections = classAwareNMS(Array(topDetections), iouThreshold: 0.45)

            let top5 = Array(nmsDetections.prefix(5))
            print(String(format: "Frame inference time: %.2f ms | raw_in_boxes: %d | topK: %d | final_detections: %d",
                         inferMs, detections.count, min(topK, detections.count), nmsDetections.count))
            for (idx, det) in top5.enumerated() {
                print(String(format: "det[%d]: class=%d score=%.3f box=[%.1f, %.1f, %.1f, %.1f]",
                             idx, det.classId, det.score, det.x1, det.y1, det.x2, det.y2))
            }

            let rects = top5.compactMap { mapToMetadataRect($0) }
            DispatchQueue.main.async { [weak self] in self?.boxes = rects }

            if currentMode != .segmentation {
                DispatchQueue.main.async { [weak self] in self?.maskImage = nil }
            }
            if currentMode == .segmentation {
                runSegmentationPipeline(using: nmsDetections)
            }

            let postMs = PerfLogger.nowMs() - tPostStart
            PerfLogger.logTimings(preMs: lastPreprocessMs, inferMs: inferMs, postMs: postMs)
        } catch {
            print("Frame inference failed: \(error)")
        }
    }
}

// MARK: - Segmentation pipeline
extension CameraManager {

    func runSegmentationPipeline(using detections: [Detection]) {
        frameIndex += 1
        let nowMs = PerfLogger.nowMs()

        guard let info = lastLetterbox else { return }

        // 1. Geometry change → full invalidation
        let geoSig = TemporalManager.GeometrySignature(
            origW: info.origW, origH: info.origH,
            scale: info.scale, padX: info.padX, padY: info.padY,
            rotation: lastRotationAngle,
            mirrored: (currentPosition == .front),
            inputSize: info.inputSize)
        if temporal.geometryChanged(geoSig) {
            invalidateEmbeddingAndMask(reason: "geometry change")
            temporal.resetPrimary()
        }

        // 2. Primary object selection (via TemporalManager)
        let prevPrimaryRect  = temporal.primaryRect
        let prevPrimaryScore = temporal.primaryScore
        let tmInputs = detections.map {
            TemporalManager.DetectionInput(x1: $0.x1, y1: $0.y1, x2: $0.x2, y2: $0.y2,
                                           score: $0.score, classId: $0.classId)
        }
        guard let selection = temporal.selectPrimary(
            from: tmInputs,
            toOrigRect: { [weak self] det in
                guard let self = self else { return nil }
                return self.detectionToOriginalRect(Detection(
                    x1: det.x1, y1: det.y1, x2: det.x2, y2: det.y2,
                    score: det.score, classId: det.classId))
            },
            origW: info.origW,
            origH: info.origH
        ) else { return }

        if selection.primaryChanged {
            invalidateEmbeddingAndMask(reason: "primary changed")
        }

        if debugSegmentation {
            print(String(format: "[DET] primary box orig=%.1f,%.1f,%.1f,%.1f score=%.3f class=%d",
                         selection.rect.minX, selection.rect.minY,
                         selection.rect.maxX, selection.rect.maxY,
                         selection.detection.score, selection.detection.classId))
            print(String(format: "[DET] letterbox scale=%.4f padX=%.2f padY=%.2f",
                         info.scale, info.padX, info.padY))
        }

        // 3. Drift classification (via TemporalManager)
        // Skip drift check when primary just changed — the embedding is already
        // invalidated and we must re-encode regardless; drift on a new object is meaningless.
        var needsEncoder = selection.primaryChanged
        var needsImmediateDecode = false
        if !selection.primaryChanged, let prev = prevPrimaryRect, prev != selection.rect {
            let drift = temporal.classifyDrift(
                prev: prev, current: selection.rect,
                prevScore: prevPrimaryScore, currentScore: selection.detection.score,
                origW: info.origW, origH: info.origH,
                debugEnabled: debugSegmentation)
            switch drift {
            case .heavyDrift:
                // Large jump — embedding unreliable, must re-encode and clear mask.
                temporal.invalidateMask()
                if debugSegmentation { print("[SEG] invalidate mask: heavy drift") }
                needsEncoder = true
            case .lightDrift:
                // Moderate shift — embedding still valid, prompt box changed.
                // Keep existing mask as visual fallback; force a decoder run this frame.
                needsImmediateDecode = true
                if debugSegmentation { print("[SEG] schedule re-decode: light drift (mask retained)") }
            case .noDrift:
                break
            }
        }

        // 4. Embedding validity check
        stateLock.lock()
        let embeddingEntry = embeddingCache.map {
            EmbeddingEntry(embedding: $0.embedding, timestampMs: $0.timestampMs)
        }
        let hasValidEmbedding = temporal.isEmbeddingValid(entry: embeddingEntry, nowMs: nowMs)
        let currentlyEncoding = isEncoding
        stateLock.unlock()

        let needsRencode = !hasValidEmbedding || needsEncoder || (frameIndex % encoderEveryNFrames == 0)
        if hasValidEmbedding { encoderHitCount += 1 } else { encoderMissCount += 1 }
        logEncoderCacheStatsIfNeeded()
        if needsRencode { scheduleEncoder(cameraBuffer: latestCameraBuffer) }

        // 5. Fallback if no embedding yet
        if currentlyEncoding && !hasValidEmbedding {
            if debugSegmentation { print("[SEG] fallback: bbox-only (encoding in progress, no valid embedding)") }
            return
        }

        stateLock.lock()
        let cachedEmbedding = embeddingCache?.embedding
        stateLock.unlock()
        guard let embedding = cachedEmbedding else { return }

        // 6. Expire stale mask
        if !temporal.isMaskValid(nowMs: nowMs) {
            temporal.invalidateMask()
        }

        // 7. Decoder cadence
        // Run decoder when:
        //   a) cadence tick
        //   b) encoder just fired (embedding is fresh)
        //   c) light drift — prompt box changed, run decoder immediately regardless of cadence
        //   d) no mask at all
        let maskValid = (temporal.maskCache != nil)
        let shouldDecodeByCadence = (frameIndex % decoderEveryNFrames == 0)
        if maskValid && !shouldDecodeByCadence && !needsEncoder && !needsImmediateDecode { return }

        if samDecoder == nil {
            samDecoder = SAMDecoder(computeUnits: backend.computeUnits)
        }
        guard let decoder = samDecoder else { return }

        stateLock.lock()
        guard !isDecoding else { stateLock.unlock(); return }
        isDecoding = true
        stateLock.unlock()

        let selectedRect  = selection.rect
        let capturedInfo  = info
        let rotAngle      = lastRotationAngle
        let isFront       = (currentPosition == .front)

        decoderQueue.async { [weak self] in
            guard let self = self else { return }

            guard let prompt = PromptBuilder.buildBoxPrompt(
                x1: Float(selectedRect.minX), y1: Float(selectedRect.minY),
                x2: Float(selectedRect.maxX), y2: Float(selectedRect.maxY),
                origW: capturedInfo.origW, origH: capturedInfo.origH,
                inputSize: 1024
            ) else {
                self.stateLock.lock(); self.isDecoding = false; self.stateLock.unlock()
                print("SAM decoder: prompt build failed — isDecoding reset")
                return
            }

            let t0 = PerfLogger.nowMs()
            guard let mask = decoder.decode(embedding: embedding, prompt: prompt) else {
                self.stateLock.lock(); self.isDecoding = false; self.stateLock.unlock()
                print("SAM decoder failed to produce mask — isDecoding reset")
                return
            }
            let t1 = PerfLogger.nowMs()
            let latency = t1 - t0

            // Record mask in TemporalManager — also updates lastMaskRefreshMs.
            let refreshInterval = self.temporal.recordMask(mask, timestampMs: t1)
            self.decoderCount += 1

            if let rendered = self.maskRenderer.renderMask(
                lowResMask: mask,
                origW: Int(capturedInfo.origW),
                origH: Int(capturedInfo.origH),
                box: selectedRect) {
                DispatchQueue.main.async {
                    self.maskImage         = rendered
                    self.maskRotationAngle = rotAngle
                    self.maskMirrored      = isFront
                }
            }

            // SAM decoder sliding-window stats (Task 1)
            let isFirstDecodeCall = (self.samDecoderCallCount == 0)
            self.samDecoderCallCount += 1
            if isFirstDecodeCall {
                // First run contains ANE compilation; exclude from stats.
                if refreshInterval > 0 {
                    print(String(format: "SAM decoder latency: %.2f ms (cold start — excluded from stats) | mask refresh: %.2f ms (%.2f Hz)",
                                 latency, refreshInterval, 1000.0 / refreshInterval))
                } else {
                    print(String(format: "SAM decoder latency: %.2f ms (cold start — excluded from stats)", latency))
                }
            } else {
                if refreshInterval > 0 {
                    print(String(format: "SAM decoder latency: %.2f ms | mask refresh: %.2f ms (%.2f Hz)",
                                 latency, refreshInterval, 1000.0 / refreshInterval))
                } else {
                    print(String(format: "SAM decoder latency: %.2f ms", latency))
                }
                self.samDecoderTimesMs.append(latency)
                if self.samDecoderTimesMs.count >= self.samDecoderStatsWindow {
                    let mean = self.samDecoderTimesMs.reduce(0.0, +) / Double(self.samDecoderTimesMs.count)
                    let sorted = self.samDecoderTimesMs.sorted()
                    let p95 = sorted[max(0, Int(ceil(0.95 * Double(sorted.count))) - 1)]
                    print(String(format: "SAM decoder stats (n=%d): mean=%.2f ms | p95=%.2f ms",
                                 self.samDecoderTimesMs.count, mean, p95))
                    self.samDecoderTimesMs.removeAll(keepingCapacity: true)
                }
            }

            self.stateLock.lock(); self.isDecoding = false; self.stateLock.unlock()
        }
    }

    /// Dispatches an encode task to encoderQueue.
    /// If an encode is already running the request is silently dropped —
    /// the in-flight encode will produce a fresh embedding for subsequent frames.
    private func scheduleEncoder(cameraBuffer: CVPixelBuffer?) {
        guard let buffer = cameraBuffer else { return }

        stateLock.lock()
        guard !isEncoding else { stateLock.unlock(); return }
        isEncoding = true
        stateLock.unlock()

        // NOTE: encoderMissCount already incremented by runSegmentationPipeline; don't double-count here.

        // Capture backend before leaving videoQueue.
        let capturedBackend = backend

        encoderQueue.async { [weak self] in
            guard let self = self else {
                return
            }
            // Lazy-load encoder INSIDE encoderQueue — keeps videoQueue unblocked.
            if self.samEncoder == nil {
                self.samEncoder = SAMEncoder(computeUnits: capturedBackend.computeUnits)
            }
            guard let encoder = self.samEncoder else {
                self.stateLock.lock(); self.isEncoding = false; self.stateLock.unlock()
                return
            }
            let t0 = PerfLogger.nowMs()
            if let embedding = encoder.encode(pixelBuffer: buffer) {
                let t1 = PerfLogger.nowMs()
                self.stateLock.lock()
                self.embeddingCache = (embedding: embedding, timestampMs: t1)
                self.isEncoding = false
                self.stateLock.unlock()
                // SAM encoder sliding-window stats (Task 1)
            let latencyMs = t1 - t0
            let isFirstEncoderCall = (self.samEncoderCallCount == 0)
            self.samEncoderCallCount += 1
            if isFirstEncoderCall {
                // First run contains ANE compilation; exclude from stats.
                print(String(format: "SAM encoder latency: %.2f ms (cold start — excluded from stats)", latencyMs))
            } else {
                self.samEncoderTimesMs.append(latencyMs)
                if self.samEncoderTimesMs.count >= self.samEncoderStatsWindow {
                    let mean = self.samEncoderTimesMs.reduce(0.0, +) / Double(self.samEncoderTimesMs.count)
                    let sorted = self.samEncoderTimesMs.sorted()
                    let p95 = sorted[max(0, Int(ceil(0.95 * Double(sorted.count))) - 1)]
                    print(String(format: "SAM encoder stats (n=%d): mean=%.2f ms | p95=%.2f ms",
                                 self.samEncoderTimesMs.count, mean, p95))
                    self.samEncoderTimesMs.removeAll(keepingCapacity: true)
                } else {
                    print(String(format: "SAM encoder latency: %.2f ms", latencyMs))
                }
            }
            } else {
                print("SAM encoder failed to produce embedding")
                self.stateLock.lock(); self.isEncoding = false; self.stateLock.unlock()
            }
        }
    }

    private func logEncoderCacheStatsIfNeeded() {
        let total = encoderHitCount + encoderMissCount
        guard total > 0, total % 30 == 0 else { return }
        let hitRate = Double(encoderHitCount) / Double(total) * 100.0
        print(String(format: "SAM encoder cache hit rate: %.1f%% (%d/%d)",
                     hitRate, encoderHitCount, total))
    }

    private func invalidateEmbeddingAndMask(reason: String) {
        stateLock.lock()
        embeddingCache = nil
        stateLock.unlock()
        temporal.invalidateMask()
        if debugSegmentation { print("[SEG] invalidate embedding+mask: \(reason)") }
    }

    // MARK: Geometry helpers

    private func detectionToOriginalRect(_ det: Detection) -> CGRect? {
        guard let info = lastLetterbox else { return nil }
        let x1 = (det.x1 - info.padX) / info.scale
        let y1 = (det.y1 - info.padY) / info.scale
        let x2 = (det.x2 - info.padX) / info.scale
        let y2 = (det.y2 - info.padY) / info.scale
        let bx = max(0, min(info.origW, x1))
        let by = max(0, min(info.origH, y1))
        let bw = max(0, min(info.origW, x2)) - bx
        let bh = max(0, min(info.origH, y2)) - by
        if bw <= 1 || bh <= 1 { return nil }
        return CGRect(x: CGFloat(bx), y: CGFloat(by), width: CGFloat(bw), height: CGFloat(bh))
    }

    private func mapToMetadataRect(_ det: Detection) -> CGRect? {
        guard let info = lastLetterbox else { return nil }
        let x1 = (det.x1 - info.padX) / info.scale
        let y1 = (det.y1 - info.padY) / info.scale
        let x2 = (det.x2 - info.padX) / info.scale
        let y2 = (det.y2 - info.padY) / info.scale
        let bx = max(0, min(info.origW, x1))
        let by = max(0, min(info.origH, y1))
        let bw = max(0, min(info.origW, x2)) - bx
        let bh = max(0, min(info.origH, y2)) - by
        if bw <= 1 || bh <= 1 { return nil }

        var rect: CGRect
        switch Int(lastRotationAngle) {
        case 90:
            let normX = by / info.origH
            let normY = (info.origW - bx - bw) / info.origW
            let normW = bh / info.origH
            let normH = bw / info.origW
            rect = CGRect(x: CGFloat(normX), y: CGFloat(normY), width: CGFloat(normW), height: CGFloat(normH))
        case 180:
            rect = CGRect(x: CGFloat((info.origW - bx - bw) / info.origW),
                          y: CGFloat((info.origH - by - bh) / info.origH),
                          width: CGFloat(bw / info.origW),
                          height: CGFloat(bh / info.origH))
        case 270:
            let normX = (info.origH - by - bh) / info.origH
            let normY = bx / info.origW
            let normW = bh / info.origH
            let normH = bw / info.origW
            rect = CGRect(x: CGFloat(normX), y: CGFloat(normY), width: CGFloat(normW), height: CGFloat(normH))
        default:
            rect = CGRect(x: CGFloat(bx / info.origW), y: CGFloat(by / info.origH),
                          width: CGFloat(bw / info.origW), height: CGFloat(bh / info.origH))
        }
        if currentPosition == .front {
            rect = CGRect(x: 1.0 - rect.origin.x - rect.size.width,
                          y: rect.origin.y,
                          width: rect.size.width, height: rect.size.height)
        }
        return rect
    }
}

// MARK: - Detection decode (optimised)
extension CameraManager {

    /// Decodes YOLO output [1, 84, 8400] into Detection structs.
    ///
    /// **Optimisations:**
    /// 1. Transpose src[84×8400] → transposeBuffer[8400×84] using vDSP_mtrans (SIMD,
    ///    ~3 ms) into a pre-allocated instance buffer — zero per-frame heap allocation.
    /// 2. Single sigmoid call per location (argmax on raw logits, sigmoid only on winner).
    /// Expected Post: ~8 ms.
    private func decodeDetections(from multiArray: MLMultiArray,
                                   confidenceThreshold: Float) -> [Detection] {
        let shape = multiArray.shape.map { $0.intValue }
        guard shape.count == 3, shape[1] == 84 else { return [] }
        let L = shape[2]   // 8400 locations
        let C = shape[1]   // 84 channels
        let strides = multiArray.strides.map { $0.intValue }
        let sC = strides[1]
        let sI = strides[2]
        let src = multiArray.dataPointer.bindMemory(to: Float.self, capacity: multiArray.count)

        // ── Transpose into pre-allocated buffer using vDSP_mtrans ───────────────
        // vDSP_mtrans requires contiguous row-major input; if strides are unit we
        // can call it directly, otherwise fall back to scalar copy.
        transposeBuffer.withUnsafeMutableBufferPointer { dstBuf in
            let dst = dstBuf.baseAddress!
            if sC == L && sI == 1 {
                // Contiguous [C × L] → [L × C]: one vDSP call (~0.5 ms for 84×8400)
                vDSP_mtrans(src, 1, dst, 1, vDSP_Length(L), vDSP_Length(C))
            } else {
                // Fallback scalar path (rare)
                for c in 0..<C {
                    let srcRow = src + c * sC
                    for i in 0..<L { dst[i * C + c] = srcRow[i * sI] }
                }
            }
        }

        let logitThreshold: Float = log(confidenceThreshold / (1.0 - confidenceThreshold))

        var detections = [Detection]()
        detections.reserveCapacity(256)

        for i in 0..<L {
            let base = i * C

            // Argmax on class logits (ch 4…83) — sequential read in transposed buffer.
            var bestRaw: Float = -Float.greatestFiniteMagnitude
            var bestClass = -1
            for c in 4..<C {
                let raw = transposeBuffer[base + c]
                if raw > bestRaw { bestRaw = raw; bestClass = c - 4 }
            }
            if bestRaw < logitThreshold { continue }

            let bestScore = sigmoid(bestRaw)
            if bestScore < confidenceThreshold { continue }

            var cx = transposeBuffer[base + 0]
            var cy = transposeBuffer[base + 1]
            var w  = transposeBuffer[base + 2]
            var h  = transposeBuffer[base + 3]

            if cx >= -0.1 && cx <= 1.5 { cx *= 640; cy *= 640; w *= 640; h *= 640 }

            let x1 = max(0, min(640, cx - w / 2))
            let y1 = max(0, min(640, cy - h / 2))
            let x2 = max(0, min(640, cx + w / 2))
            let y2 = max(0, min(640, cy + h / 2))
            if x2 - x1 <= 1 || y2 - y1 <= 1 { continue }

            detections.append(Detection(x1: x1, y1: y1, x2: x2, y2: y2,
                                        score: bestScore, classId: bestClass))
        }
        return detections
    }

    private func classAwareNMS(_ detections: [Detection], iouThreshold: Float) -> [Detection] {
        var grouped = [Int: [Detection]]()
        for det in detections { grouped[det.classId, default: []].append(det) }
        var results = [Detection]()
        for (_, dets) in grouped {
            let sorted = dets.sorted { $0.score > $1.score }
            var kept = [Detection]()
            for det in sorted {
                if !kept.contains(where: { iou(det, $0) > iouThreshold }) { kept.append(det) }
            }
            results.append(contentsOf: kept)
        }
        return results
    }

    private func iou(_ a: Detection, _ b: Detection) -> Float {
        let xA = max(a.x1, b.x1); let yA = max(a.y1, b.y1)
        let xB = min(a.x2, b.x2); let yB = min(a.y2, b.y2)
        let inter = max(0, xB - xA) * max(0, yB - yA)
        let union = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter
        return union <= 0 ? 0 : inter / union
    }

    @inline(__always) private func sigmoid(_ x: Float) -> Float {
        1.0 / (1.0 + exp(-x))
    }

    // MARK: Letterbox

    private func letterboxToSquare(pixelBuffer: CVPixelBuffer, size: Int) -> CVPixelBuffer? {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let width = image.extent.width, height = image.extent.height
        let scale = min(CGFloat(size) / width, CGFloat(size) / height)
        let scaled = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let x = (CGFloat(size) - scaled.extent.width) / 2.0
        let y = (CGFloat(size) - scaled.extent.height) / 2.0
        let translated = scaled.transformed(by: CGAffineTransform(translationX: x, y: y))

        lastLetterbox = LetterboxInfo(
            origW: Float(width), origH: Float(height),
            scale: Float(scale), padX: Float(x), padY: Float(y),
            inputSize: Float(size))

        // Reuse pre-allocated output buffer; allocate once on first call.
        if letterboxOutputBuffer == nil {
            let outputAttrs: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey: size,
                kCVPixelBufferHeightKey: size,
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true
            ]
            var newBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(kCFAllocatorDefault, size, size,
                                            kCVPixelFormatType_32BGRA,
                                            outputAttrs as CFDictionary, &newBuffer)
            guard status == kCVReturnSuccess else { return nil }
            letterboxOutputBuffer = newBuffer
        }
        guard let buffer = letterboxOutputBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        let background = CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: size, height: size))
        let composed = translated.composited(over: background)
        ciContext.render(composed, to: buffer)
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }
}