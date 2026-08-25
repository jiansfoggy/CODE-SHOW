//
//  SAMConfiguration.swift
//  JudgeE2
//
//  Phase 3 Day 4 — Encoder resolution AB-test configuration.
//
//  Authorised by Architect §8.6 (architect_output.md): 768×768 conditionally
//  approved for Builder AB testing.  Default stays 1024 (quality baseline,
//  constraint C-1).  Switching to 768 only via `encoderInputSize = 768`.
//
//  IMPORTANT CONTRACTS (Architect §8.3 constraints):
//    C-1: Default resolution MUST remain 1024.  768 is opt-in for AB test only.
//    C-2: Point prompt coordinates are ALWAYS computed in 1024 space, regardless
//         of encoderInputSize (Decoder is fixed at 1024 normalisation).
//         → PointPromptBuilder uses inputSize=1024 unconditionally.
//    C-3: 768 encoder output [1,256,48,48] MUST be bilinearly upsampled to
//         [1,256,64,64] in Swift before the Decoder (Decoder fixed at 64×64).
//    C-4: 768 model export MUST apply milfix (done: fp16_milfix_768).
//
//  This is a lightweight global config; the AB test toggles it at launch and
//  never mid-session (changing it invalidates the embedding cache geometry).

import Foundation

/// Global SAM encoder configuration for the resolution AB test (Phase 3 Day 4).
enum SAMConfiguration {

    /// Supported encoder input resolutions.
    /// - `.res1024`: Phase 3 default (quality baseline, Architect C-1).
    /// - `.res768`:  AB-test variant (conditionally approved, Architect §8.3).
    ///   512 is deliberately NOT offered here — Phase 3 rejected it (Architect §8.4).
    enum EncoderResolution: Int {
        case res1024 = 1024
        case res768  = 768

        /// Feature-grid side length the encoder produces (= inputSize / 16).
        /// 1024 → 64, 768 → 48.
        var featureSize: Int { rawValue / 16 }

        /// Bundle resource name of the milfix encoder for this resolution.
        var encoderResourceName: String {
            switch self {
            case .res1024: return "MobileSAM_ImageEncoder_fp16_milfix"
            case .res768:  return "MobileSAM_ImageEncoder_fp16_milfix_768"
            }
        }
    }

    /// Active encoder input resolution.
    ///
    /// Default = 1024 (Architect C-1).  Flip to `.res768` to run the AB test.
    /// Changing this after launch requires clearing the embedding cache (geometry
    /// signature includes inputSize) — CameraManager handles that on mode reset.
    static var encoderResolution: EncoderResolution = .res1024

    /// Convenience: the raw encoder input side length (1024 or 768).
    static var encoderInputSize: Int { encoderResolution.rawValue }

    /// Decoder always consumes a 64×64 embedding (fixed, Architect C-3).
    static let decoderEmbeddingSize: Int = 64

    /// Point-prompt coordinate space is ALWAYS 1024 (Architect C-2 / model_plan §C.4).
    static let pointPromptSpace: Int = 1024
}
