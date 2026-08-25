//
//  PinRecordV1.swift
//  JudgeE2
//
//  Phase 4B — Day 4 (Builder) · B-19 / B-20 / B-25
//
//  The persistence DTOs, the manifest envelope, the explicitly configured
//  JSON coders, and the (currently empty, but structurally present) migration
//  chain.  Contract: architect_output.md §19.2 / §19.5.
//
//  WHY THE DTOs ARE SEPARATE TYPES (rule P-A, §19.2.1).
//  `FrameGeometry` is a runtime member of the geometry chain (§3).  If it were
//  `: Codable`, the on-disk format would change every time someone added or
//  removed a field there — silently, with no review and no version bump.  The
//  two mapping functions in this file are the only bridge, and they are pure.
//
//  WHY EVERY CODING DETAIL IS SPELLED OUT (§19.5.3).
//  Coordinates are explicit `pointX`/`pointY` Doubles, never `CGPoint`'s
//  synthesised Codable (which is an *unkeyed two-element array* on Apple
//  platforms — an implementation detail, not a disk format).  Times are epoch
//  seconds as `Double`, never `Date` (whose encoding follows whatever
//  `JSONEncoder.dateEncodingStrategy` happens to be set to globally).  Floats
//  are `Double`, never `CGFloat` (platform-dependent width).  The coders below
//  set every strategy explicitly and rely on no SDK default.
//

import CoreGraphics
import Foundation

// MARK: - PinGeometryV1

/// Geometry snapshot persisted with a Pin (§19.2.3).
///
/// Source-checked against `PointPromptBuilder.buildPointPrompt`: re-deriving a
/// point prompt needs **only** `origSize` and `inputSize`; `scale`, `padX` and
/// `padY` are computed from those two inside the builder.  That is exactly why
/// `frameGeometry: FrameGeometry` (the tasks.md draft) is insufficient — it
/// carries no `promptSpace` — and why the letterbox fields below are
/// diagnostics that must never be read back into a transform.
/// B-34 (§22.1.7): `nonisolated` — a pure Codable value type with no actor
/// affinity of its own; used from both `pin.store.io` (encode/decode) and the
/// UI (construction from `FrameGeometry` at Pin-creation time).
nonisolated struct PinGeometryV1: Codable, Equatable {

    // Required for the re-decode mathematics.
    let origW: Double
    let origH: Double
    /// `SAMConfiguration.pointPromptSpace` at creation time (currently 1024).
    let promptSpace: Int

    // Required for compatibility adjudication (gate G1, §19.4.3).  They take no
    // part in the arithmetic — `canonicalPoint` is already post-inversion — but
    // Canonical space's *meaning* is defined by rotation + mirroring, so the
    // same coordinate under a different one of these points somewhere else
    // entirely.  There is no remapping between them; what differs is the
    // picture, not the axes.
    let rotationDeg: Double
    let mirrored: Bool

    // Diagnostics only.
    /// 1024/768 AB switch at creation time.  Because embeddings are never
    /// persisted (PIN-4), a revisit always uses the *current* embedding, so
    /// this constrains nothing — it exists for post-hoc attribution.
    let encoderInputSize: Int
    /// ⛔ **Must never take part in the re-decode mathematics** (§19.2.3).
    /// These three are redundant functions of `origW`/`origH`/`promptSpace`;
    /// reading them into a transform would constitute a second transform path
    /// and break §2.3's "exactly one transform path".
    let letterboxScale: Double?
    let letterboxPadX: Double?
    let letterboxPadY: Double?

    // MARK: B-20 — forward mapping (pure)

    /// `FrameGeometry` + prompt space + encoder size ⇒ persisted snapshot.
    ///
    /// `@MainActor`: an explicit override of this (`nonisolated`) type's
    /// default for this one initializer only.  `FrameGeometry` and
    /// `SAMConfiguration` are both main-actor-isolated by the module default
    /// (neither is part of B-34's seven-type list — both stay untouched, out
    /// of scope), and this initializer's only real call site
    /// (`PinFactory.makeRecord`, Day 5) is itself only ever invoked from the
    /// UI on the main thread — so this matches actual usage exactly, and is
    /// cheaper than widening `nonisolated` to two files this ruling never
    /// asked to touch.
    ///
    /// ⚠️ No default values on `promptSpace`/`encoderInputSize`, unlike Day
    /// 4's draft: a default-argument expression is isolation-checked in its
    /// own right, independent of the initializer's own `@MainActor` (a Swift
    /// quirk, not a design choice), so `SAMConfiguration.pointPromptSpace`
    /// read as a default value reintroduced the exact warning marking this
    /// initializer `@MainActor` was meant to remove.  Callers pass both
    /// explicitly instead — one extra argument each, at the single call site.
    @MainActor init(from geometry: FrameGeometry,
         promptSpace: Int,
         encoderInputSize: Int)
    {
        self.origW            = Double(geometry.origW)
        self.origH            = Double(geometry.origH)
        self.promptSpace      = promptSpace
        self.rotationDeg      = geometry.videoRotationAngle
        self.mirrored         = geometry.mirrored
        self.encoderInputSize = encoderInputSize
        self.letterboxScale   = Double(geometry.scale)
        self.letterboxPadX    = Double(geometry.letterboxOffset.x)
        self.letterboxPadY    = Double(geometry.letterboxOffset.y)
    }

    /// Memberwise construction, for tests and for decoding.
    init(origW: Double, origH: Double, promptSpace: Int,
         rotationDeg: Double, mirrored: Bool, encoderInputSize: Int,
         letterboxScale: Double? = nil,
         letterboxPadX: Double? = nil,
         letterboxPadY: Double? = nil)
    {
        self.origW            = origW
        self.origH            = origH
        self.promptSpace      = promptSpace
        self.rotationDeg      = rotationDeg
        self.mirrored         = mirrored
        self.encoderInputSize = encoderInputSize
        self.letterboxScale   = letterboxScale
        self.letterboxPadX    = letterboxPadX
        self.letterboxPadY    = letterboxPadY
    }

    // MARK: B-20 — reverse mapping (pure)

    /// The complete set of quantities a re-decode may read out of a stored
    /// geometry.  Deliberately a two-field tuple: it is not possible to reach
    /// `letterboxScale` / `letterboxPadX` / `letterboxPadY` through this API,
    /// which is how "no second transform path" is enforced structurally rather
    /// than by comment.
    func decodeInputs() -> (origSize: CGSize, promptSpace: Int) {
        (CGSize(width: origW, height: origH), promptSpace)
    }

    /// Gate G1 (§19.4.3): rotation, mirroring and prompt space must match
    /// bit-for-bit before a stored `canonicalPoint` means anything in the
    /// current scene.  ⚠️ Day 6 owns the actual revisit flow; this predicate is
    /// here because it is a property of the record, not of the flow.
    func isCompatible(withRotationDeg rotation: Double,
                      mirrored otherMirrored: Bool,
                      promptSpace otherPromptSpace: Int) -> Bool
    {
        rotationDeg == rotation && mirrored == otherMirrored && promptSpace == otherPromptSpace
    }
}

// MARK: - PinRecordV1

/// One persisted Pin (§19.2.2).  Every field is either frozen at creation or
/// one of the two user-editable strings; there is no third category.
///
/// B-34 (§22.1.7): `nonisolated` — its real execution domain is `pin.store.io`.
nonisolated struct PinRecordV1: Codable, Equatable {

    // Frozen.
    let id: String                  // UUID, lowercased — also the blob filename stem
    let pointX: Double              // Canonical pixel space (origW × origH)
    let pointY: Double
    let geometry: PinGeometryV1
    let maskFile: String?           // "masks/<id>.png", or nil = no visual fallback
    let maskWidth: Int              // v1: 256
    let maskHeight: Int             // v1: 256
    let maskNonZero: Int
    let iouPredAtCreation: Double?  // ⛔ diagnostic only, never a gate or sort key
    let createdAt: Double           // epoch seconds

    // Mutable.
    var tag: String?
    var note: String?
    var updatedAt: Double           // epoch seconds

    /// Reserved slot for a cropped RGB preview (R29).
    /// ⛔ **Day 4/5 must keep this nil.**  RGB frames are not among the three
    /// data sources §18.3.3 hard isolation 1 permits; whether it may ever be
    /// filled is adjudicated on Day 5 together with R27, privacy and size.
    let previewFile: String?

    var uuid: UUID? { UUID(uuidString: id) }

    var canonicalPoint: CGPoint { CGPoint(x: pointX, y: pointY) }

    /// The blob is usable only at the v1 dimensions.  A record whose blob is
    /// some other size is `degraded`: metadata stays fully usable, the blob is
    /// excluded from every composite (§19.5.4 — the direct lesson of §35, where
    /// a dimension premise lived only in comments).
    var isDegraded: Bool { maskWidth != MaskPNGCodec.side || maskHeight != MaskPNGCodec.side }

    /// The relative blob path this record's id implies.
    static func maskFileName(for id: String) -> String { "masks/\(id).png" }
}

// MARK: - Pin ⇄ PinRecordV1

extension Pin {
    /// DTO ⇒ runtime.  Pure.
    ///
    /// B-34: `nonisolated`.  `Pin` itself keeps the module's default isolation
    /// (it is not one of the seven types §22.1.7 names), but this specific
    /// initializer is called from `FilePinStore.publishSnapshot`, which runs
    /// on `pin.store.io` — the same pure-conversion reasoning as B-20's
    /// `PinGeometryV1` mapping functions applies here too.
    nonisolated init(record: PinRecordV1) {
        self.id                = record.uuid ?? UUID()
        self.canonicalPoint    = record.canonicalPoint
        self.geometry          = record.geometry
        self.maskFile          = record.maskFile
        self.maskWidth         = record.maskWidth
        self.maskHeight        = record.maskHeight
        self.maskNonZero       = record.maskNonZero
        self.iouPredAtCreation = record.iouPredAtCreation
        self.createdAt         = Date(timeIntervalSince1970: record.createdAt)
        self.tag               = record.tag
        self.note              = record.note
        self.updatedAt         = Date(timeIntervalSince1970: record.updatedAt)
        self.isDegraded        = record.isDegraded
    }
}

extension PinRecordV1 {
    /// Runtime ⇒ DTO.  Pure.  Used by tests and by any caller that already
    /// holds a `Pin`; the creation path builds its record through B-27 instead.
    nonisolated init(pin: Pin) {
        self.id                = pin.id.uuidString.lowercased()
        self.pointX            = Double(pin.canonicalPoint.x)
        self.pointY            = Double(pin.canonicalPoint.y)
        self.geometry          = pin.geometry
        self.maskFile          = pin.maskFile
        self.maskWidth         = pin.maskWidth
        self.maskHeight        = pin.maskHeight
        self.maskNonZero       = pin.maskNonZero
        self.iouPredAtCreation = pin.iouPredAtCreation
        self.createdAt         = pin.createdAt.timeIntervalSince1970
        self.tag               = pin.tag
        self.note              = pin.note
        self.updatedAt         = pin.updatedAt.timeIntervalSince1970
        self.previewFile       = nil            // R29: reserved, always nil
    }
}

// MARK: - Manifest envelope (§19.5.1)

/// The whole manifest.  The version number lives on the **envelope**, not on
/// each record: the file is rewritten atomically as a unit, so a mixed-version
/// state cannot exist.
/// B-34 (§22.1.7): `nonisolated` — decoded/encoded exclusively on `pin.store.io`.
nonisolated struct PinManifestV1: Codable {
    let schemaVersion: Int
    let writtenAt: Double        // epoch seconds
    var pins: [PinRecordV1]
}

/// Just enough of the envelope to read the version out of a file whose body may
/// be in a format this build cannot decode.
nonisolated private struct PinManifestHeader: Decodable {
    let schemaVersion: Int
}

// MARK: - Schema + migration scaffolding (B-25, §19.5)

/// Not one of the seven §22.1.7 types by name, but the same reasoning applies:
/// pure, stateless, used only from `pin.store.io`.
nonisolated enum PinSchema {

    /// The schema version this build writes.
    static let currentVersion = 1

    /// One numbered step of the migration chain.  Steps are pure functions over
    /// the raw file bytes and are applied **in order, never skipping a
    /// version** (§19.5.2 rule 2).
    struct Migration {
        let from: Int
        let to: Int
        let apply: (Data) throws -> Data
    }

    /// The ordered chain.  **Empty at v1 — the structure exists now so that the
    /// first schema change is a data change, not an architecture change.**
    /// Additive evolution (new `Optional` field with a default) needs no entry
    /// here at all, only a version bump (§19.5.2 rule 1).
    static let migrations: [Migration] = []

    /// Read the envelope version without committing to the body's shape.
    static func peekVersion(in data: Data) -> Int? {
        (try? PinCoders.decoder.decode(PinManifestHeader.self, from: data))?.schemaVersion
    }

    /// Apply the chain from `version` up to `currentVersion`.
    /// - Returns: the migrated bytes, or the input unchanged when already current.
    /// - Throws: when a required step is missing — a gap must fail loudly, not
    ///   be papered over by decoding whatever happens to parse.
    static func migrate(_ data: Data, from version: Int) throws -> Data {
        guard version < currentVersion else { return data }
        var bytes = data
        var v = version
        while v < currentVersion {
            guard let step = migrations.first(where: { $0.from == v }) else {
                throw MigrationError.missingStep(from: v, to: currentVersion)
            }
            bytes = try step.apply(bytes)
            v = step.to
        }
        return bytes
    }

    enum MigrationError: Error, CustomStringConvertible {
        case missingStep(from: Int, to: Int)
        var description: String {
            switch self {
            case .missingStep(let f, let t): return "missingMigrationStep(from:\(f) to:\(t))"
            }
        }
    }
}

// MARK: - Coders (§19.5.3 — nothing is left to a default)

/// B-34 (§22.1.7): `nonisolated` — the coders exist only to serve `pin.store.io`.
nonisolated enum PinCoders {

    /// ⚠️ Every strategy is set explicitly.  Defaults belong to the SDK, not to
    /// us; a future SDK that changes one would silently change our disk format.
    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting        = [.sortedKeys, .withoutEscapingSlashes]
        e.dateEncodingStrategy    = .secondsSince1970     // no Date fields exist; pinned anyway
        e.dataEncodingStrategy    = .base64               // no Data fields exist; pinned anyway
        e.keyEncodingStrategy     = .useDefaultKeys
        e.nonConformingFloatEncodingStrategy = .throw     // NaN must fail, not become a string
        return e
    }()

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy    = .secondsSince1970
        d.dataDecodingStrategy    = .base64
        d.keyDecodingStrategy     = .useDefaultKeys
        d.nonConformingFloatDecodingStrategy = .throw
        return d
    }()
}
