import Foundation
import CoreML
import QuartzCore
import MachO

enum PerfLogger {
    // FPS window
    private static var frameCount = 0
    private static var lastFpsTime = CFAbsoluteTimeGetCurrent()

    // Output shapes logged once
    private static var didLogOutputShapes = false

    @inline(__always) static func nowMs() -> Double {
        return CACurrentMediaTime() * 1000.0
    }

    static func logComputeUnits(_ config: MLModelConfiguration) {
        perfLog("CoreML computeUnits = \(config.computeUnits)")
    }

    static func logInputInfo(_ model: MLModel) {
        let desc = model.modelDescription
        for (name, input) in desc.inputDescriptionsByName {
            if let image = input.imageConstraint {
                perfLog("Model input: \(name) image \(image.pixelsWide)x\(image.pixelsHigh)")
            } else if let ma = input.multiArrayConstraint {
                let shape = ma.shape.map { $0.intValue }
                perfLog("Model input: \(name) shape=\(shape) dtype=\(ma.dataType)")
            }
        }
    }

    static func logOutputShapesOnce(var_3019: MLMultiArray, var_3022: MLMultiArray) {
        guard !didLogOutputShapes else { return }
        didLogOutputShapes = true
        let s1 = var_3019.shape.map { $0.intValue }
        let s2 = var_3022.shape.map { $0.intValue }
        perfLog("Output var_3019 shape=\(s1)")
        perfLog("Output var_3022 shape=\(s2)")
    }

    static func logFpsAndMemoryEverySecond() {
        frameCount += 1
        let now = CFAbsoluteTimeGetCurrent()
        if now - lastFpsTime >= 1.0 {
            let fps = Double(frameCount) / (now - lastFpsTime)
            let mem = currentMemoryMB()
            perfLog(String(format: "FPS: %.2f | Memory: %.1f MB", fps, mem))
            frameCount = 0
            lastFpsTime = now
        }
    }

    static func logTimings(preMs: Double, inferMs: Double, postMs: Double) {
        let totalMs = preMs + inferMs + postMs
        let fps = 1000.0 / max(1.0, totalMs)
        let mem = currentMemoryMB()
        perfLog(String(format: "Pre=%.2fms | Infer=%.2fms | Post=%.2fms | Total=%.2fms | FPS=%.2f | Mem=%.1f MB",
                     preMs, inferMs, postMs, totalMs, fps, mem))
    }

    static func currentMemoryMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if kerr == KERN_SUCCESS {
            return Double(info.resident_size) / 1024.0 / 1024.0
        }
        return -1
    }
}
