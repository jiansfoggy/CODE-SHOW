import Foundation
import CoreML

enum AppMode: String, CaseIterable, Identifiable {
    case detectionOnly
    case segmentation
    case tapToSegment       // Phase 3 — user-triggered tap segmentation

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .detectionOnly: return "Detection"
        case .segmentation:  return "Segmentation"
        case .tapToSegment:  return "Tap to Segment"
        }
    }
}

enum InferenceBackend: String, CaseIterable, Identifiable {
    case cpuOnly
    case cpuAndGPU
    case cpuAndNeuralEngine
    case all

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .cpuOnly: return "CPU"
        case .cpuAndGPU: return "CPU+GPU"
        case .cpuAndNeuralEngine: return "CPU+NeuralEngine"
        case .all: return "All"
        }
    }

    var computeUnits: MLComputeUnits {
        switch self {
        case .cpuOnly: return .cpuOnly
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .all: return .all
        }
    }
}
