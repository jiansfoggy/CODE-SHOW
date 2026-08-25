//
//  Pin.swift
//  JudgeE2
//
//  Phase 4B — Day 4 (Builder) · B-18 / B-24
//
//  Runtime value type + store interface + error surface for the Pin subsystem.
//  Contract: shared/architect_output.md §19 (PinStore 存储方案裁决).
//
//  LAYERING (§19.2.1, rule P-A) — three types, deliberately not two:
//
//      TapInstance (Phase 3, frozen, in memory)
//            │  exactly three quantities are read: maskAlpha / canonicalPoint /
//            │  createdAt  ← §18.3.3 hard isolation 1
//            ▼
//      Pin  (runtime value type; what UI and PinStore exchange)   ← THIS FILE
//            │  one-to-one, no logic
//            ▼
//      PinRecordV1 + PinGeometryV1 (persistence DTOs, Codable)    ← PinRecordV1.swift
//            ▼
//      masks/<uuid>.png  (256×256 sidecar blob, immutable)        ← MaskPNGCodec.swift
//
//  `Pin` is deliberately NOT `Codable`.  Making a runtime structure directly
//  Codable pins the on-disk format to a structure that exists for other reasons
//  (§19.2.1 rule P-A) — the moment somebody adds or removes a field, the disk
//  format changes *silently*.  Conversion is two explicit pure functions (B-20 /
//  `Pin.init(record:)` / `PinRecordV1.init(pin:)`).
//
//  SCOPE GUARD.  Nothing in this subsystem may reference `TapInstance`'s
//  re-anchor fields (`anchorSignature` / `lastReAnchorEmbeddingGen` /
//  `lastReAnchorAtMs` / `originAlpha`) — §18.3.3 hard isolation 1, still in
//  force after §20.3.4 lifted only the third condition.
//

import CoreGraphics
import Foundation

// MARK: - Pin (runtime)

/// One saved anchor: "the user pointed *here*, in *this* geometry, and this is
/// what was segmented at the time".
///
/// A Pin is **not a screenshot**.  The embedding is never persisted (invariant
/// PIN-4, §19.4.2), so revisiting a Pin re-segments *the current scene* at a
/// remembered pixel location.  The stored mask blob is a degraded static
/// fallback and a list thumbnail — and nothing else (invariant PIN-2).
struct Pin: Identifiable, Equatable {

    // MARK: Frozen (written once at creation; PIN-1)

    let id: UUID

    /// Tap location in **Canonical pixel space** (`origW × origH`, e.g.
    /// 1080×1920), display orientation — the same space `FrameGeometry`
    /// produces and `PointPromptBuilder` consumes.
    ///
    /// ⚠️ This is *not* 1024-space.  The →1024 normalisation happens inside
    /// `PointPromptBuilder` (§2.3), so storing a 1024-space number here would
    /// leave the coordinate space of the persisted value undefined and make it
    /// impossible to know, at revisit time, whether to scale once more.
    ///
    /// **Invariant PIN-1 (§19.2.4): frozen.** No code path may rewrite it.
    /// Editing `tag`/`note` must not touch it; revisiting must not "correct"
    /// it; a future tracker that wants to move an anchor creates a *new* Pin.
    let canonicalPoint: CGPoint

    /// Geometry snapshot at creation time.  Carries everything a re-decode
    /// needs (`origW`/`origH`/`promptSpace`) plus the two compatibility
    /// discriminators (`rotationDeg`/`mirrored`), plus diagnostics.
    let geometry: PinGeometryV1

    /// Relative path of the mask blob (`masks/<id>.png`), or nil when this Pin
    /// has no visual fallback.  A Pin without a blob is still fully legal — the
    /// re-decode path does not need it.
    let maskFile: String?

    /// Mask blob dimensions, persisted **as data** rather than left to a
    /// comment (long-standing agreement A-15, §19.5.4).  v1 = 256 × 256.
    let maskWidth: Int
    let maskHeight: Int

    /// Non-zero pixel count inside the 256×256 mask space.  Makes the blob's
    /// round-trip self-certifying and gives revisit IoU an independent bound.
    let maskNonZero: Int

    /// `iou_pred` of the tap that produced this mask.
    /// ⛔ **Diagnostic only.** It must never take part in any gate or ordering.
    let iouPredAtCreation: Double?

    let createdAt: Date

    // MARK: Mutable

    var tag: String?
    var note: String?
    var updatedAt: Date

    // MARK: Derived state (not persisted)

    /// True when the record's blob dimensions are not the v1 invariant, i.e.
    /// the blob must not be used for any compositing.  Metadata stays fully
    /// usable — re-decoding does not need the blob (§19.5.4).
    let isDegraded: Bool
}

// MARK: - Field limits (§19.2.5 — reject, never truncate)

nonisolated enum PinFieldLimits {
    static let maxTagCharacters  = 64
    static let maxNoteCharacters = 2000

    /// B-35 (§22.1.8) — canonical character-count function for the 64/2000
    /// limits above: **UTF-16 code units**, not extended grapheme clusters.
    /// `String.count` (grapheme clusters) has no predictable relationship to
    /// on-disk byte size — a combined emoji sequence can be one grapheme
    /// cluster and a dozen UTF-16 units — which is exactly the property
    /// §19.1's byte-budget arithmetic depends on.  `FilePinStore` and the
    /// Day 5 `PinCreationSheet` input limiter both call this one function so
    /// "64 characters" cannot mean two different things in two places.
    static func length(of s: String) -> Int { s.utf16.count }
}

// MARK: - Errors (B-24, §19.7)

/// The complete error surface of `PinStore`.  Six cases, closed set.
///
/// B-34 (§22.1.7): `nonisolated`.  This is a pure value enum with no stored
/// state and no actor affinity; its real usage sites are all on `pin.store.io`
/// (`FilePinStore`'s mutating methods) with delivery to main only as an
/// already-nonisolated `Result` payload.  Left to the module's default
/// (`SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`), the compiler infers this
/// type as main-actor-isolated purely because of where the source file
/// happens to sit — which is exactly backwards from where it is actually used
/// — and flags every `pin.store.io`-side use as a nonisolated-context
/// violation (§37.3.2, ISSUE-D4-BUILD-1).
nonisolated enum PinStoreError: Error {
    /// The store could not be brought up at all (directory creation failed,
    /// or the on-disk schema is newer than this build so the store is
    /// read-only).  ⛔ Never degrade silently to "in-memory pins" — that lets
    /// the user believe a save succeeded.
    case unavailable

    /// `tag` > 64 or `note` > 2000 characters.  §19.2.5: reject, do not
    /// truncate — silently shrinking user data is the same failure class as
    /// silently clearing a loading indicator.
    case fieldTooLong(field: String)

    case notFound

    /// An attempt to change a field frozen at creation time (PIN-1).
    case frozenFieldMutation(field: String)

    case io(underlying: Error)

    /// The manifest was written by a newer build.  The store opens read-only
    /// and ⛔ never rewrites the file — one downgrade must not truncate all of
    /// the user's data (§19.5.2 rule 3).
    case schemaTooNew(found: Int, supported: Int)
}

// B-34: `nonisolated` on the enum's primary declaration does not cascade to a
// separate `extension` — Swift infers each extension's isolation on its own.
// Without this, the `description` witness for `CustomStringConvertible`
// (itself a `nonisolated` protocol requirement) is still inferred `@MainActor`
// by the module default, reproducing the exact warning marking the enum was
// meant to fix.
nonisolated extension PinStoreError: CustomStringConvertible {
    var description: String {
        switch self {
        case .unavailable:                    return "unavailable"
        case .fieldTooLong(let field):        return "fieldTooLong(\(field))"
        case .notFound:                       return "notFound"
        case .frozenFieldMutation(let field): return "frozenFieldMutation(\(field))"
        case .io(let underlying):             return "io(\(underlying))"
        case .schemaTooNew(let f, let s):     return "schemaTooNew(found:\(f) supported:\(s))"
        }
    }
}

/// Integrity violations detected while validating a record against its blob.
/// Surfaced through `PinStoreError.io` so the error surface stays at the six
/// cases §19.7 enumerates.
///
/// B-34: `nonisolated` — same reasoning as `PinStoreError`; it lives and dies
/// entirely inside `FilePinStore`'s `pin.store.io`-confined validation.
nonisolated enum PinIntegrityError: Error, CustomStringConvertible {
    case maskSizeMismatch(expected: Int, got: Int)
    case maskDimensionsNotV1(width: Int, height: Int)
    case nonZeroCountMismatch(recorded: Int, computed: Int)
    case maskFilePresenceMismatch

    var description: String {
        switch self {
        case .maskSizeMismatch(let e, let g):     return "maskSizeMismatch(expected:\(e) got:\(g))"
        case .maskDimensionsNotV1(let w, let h):  return "maskDimensionsNotV1(\(w)x\(h))"
        case .nonZeroCountMismatch(let r, let c): return "nonZeroCountMismatch(recorded:\(r) computed:\(c))"
        case .maskFilePresenceMismatch:           return "maskFilePresenceMismatch"
        }
    }
}

// MARK: - PinStore

/// The storage boundary for Pins.
///
/// This protocol is the single hedge taken in §19.1: `FilePinStore` is the only
/// implementation today, and if cross-device sync ever becomes a requirement it
/// is this one file that gets replaced (N ≤ 1,000 migrates in one pass).  The
/// hedge costs nothing, which is why it is here at all.
///
/// **Threading contract.**
///   • `fetchAll()` / `fetch(id:)` / `pins` / `isLoaded` / `lastWriteError` are
///     **main thread only**.  They serve the in-memory snapshot and never touch
///     the disk (§19.3.2), so "read < 50 ms" holds by construction.
///   • every mutating API completes with `Result<Void, PinStoreError>` **on the
///     main thread** (§19.3.4).  A `.success` means the change is on disk —
///     completions are held until the manifest flush that contains them lands.
///   • ⛔ no PinStore symbol may appear inside any closure executing on
///     `videoQueue` / `decoderQueue` / `encoderQueue` (constraint PIN-3,
///     statically checkable; §21.3 gives it a measured 0.1 ms of headroom).
protocol PinStore: AnyObject {

    /// Main-thread snapshot of every Pin, ascending by `createdAt`.
    @MainActor var pins: [Pin] { get }

    /// False until the asynchronous startup load has published its first
    /// snapshot.  ⛔ While false the UI must show a loading state, **never**
    /// "no pins yet" — painting "not read yet" as "no data" is the same silent
    /// failure class this project has been bitten by before (§19.3.2).
    @MainActor var isLoaded: Bool { get }

    /// Last mutating failure, for a persistent error affordance (§19.3.4).
    @MainActor var lastWriteError: PinStoreError? { get }

    /// Kick off the asynchronous startup load.  Safe to call more than once;
    /// only the first call does work.
    nonisolated func load()

    @MainActor func fetchAll() -> [Pin]
    @MainActor func fetch(id: UUID) -> Pin?

    /// Create a new Pin from a fully-formed record plus (optionally) the mask
    /// bytes the record describes.
    ///
    /// ⛔ There is no overwrite-style `save(_:)`: a whole-record write could
    /// rewrite frozen fields, which PIN-1 forbids.  If `record.id` already
    /// exists, every frozen field must match bit-for-bit or the call fails with
    /// `.frozenFieldMutation`.
    nonisolated func create(_ record: PinRecordV1,
                maskAlpha: [UInt8]?,
                completion: @escaping (Result<Void, PinStoreError>) -> Void)

    /// Commit the only two mutable fields.  `nil` means "leave unchanged";
    /// clearing is expressed as `""`.
    nonisolated func update(id: UUID,
                tag: String?,
                note: String?,
                completion: @escaping (Result<Void, PinStoreError>) -> Void)

    nonisolated func delete(id: UUID, completion: @escaping (Result<Void, PinStoreError>) -> Void)

    /// Lazily read one mask blob (256×256, values {0,255}).  Blobs are never
    /// read during the full load (§19.3.2).
    nonisolated func loadMaskImage(id: UUID,
                       completion: @escaping (Result<[UInt8], PinStoreError>) -> Void)

    /// Forced flush point.  Call with `waitUntilDone: true` from
    /// `scenePhase → .background` and `willTerminate` (§19.3.3).
    nonisolated func flush(waitUntilDone: Bool)
}
