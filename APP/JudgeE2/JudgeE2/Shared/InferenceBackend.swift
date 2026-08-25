import Foundation
import CoreML

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
