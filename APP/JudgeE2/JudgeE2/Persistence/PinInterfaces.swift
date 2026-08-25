//
//  PinInterfaces.swift
//  JudgeE2
//
//  Phase 4B — Day 4 (Builder) interface, Day 5 (Builder) implementation ·
//  B-27 implemented this round; B-28 remains interface-only (Day 6).
//
//  Contract: architect_output.md §19.7 (B-27) + §22.2.2/§22.2.3 (PIN-6,
//  the single-snapshot discipline that governs how a caller may use this).
//
//  ⛔ This file still touches no render path and no UI framework.  The
//  PinCreationSheet (Day 5 UI, `UI/PinCreationSheet.swift`) is the only caller
//  of `makeRecord`/`maskAlpha`, and it must call each of them **once**, off a
//  single `TapInstance` value already captured under `TapInstanceManager`'s
//  lock (PIN-6, §22.2.3) — never re-reading `TapInstanceManager` between the
//  sheet's preview and its "Save" action.
//

import CoreGraphics
import Foundation

// MARK: - B-27 — TapInstance ⇒ PinRecordV1 (Day 5)

enum PinFactory {

    /// Build a persistable record from a live tap instance.
    ///
    /// **It reads exactly three quantities from `instance`:**
    ///   • `maskAlpha`       → the 256×256 blob, `maskNonZero`, `maskWidth/Height`
    ///   • `canonicalPoint`  → `pointX` / `pointY` (frozen from here on, PIN-1)
    ///   • `createdAt`       → `createdAt`
    ///
    /// ⛔ **Not one re-anchor field** — `anchorSignature`,
    /// `lastReAnchorEmbeddingGen`, `lastReAnchorAtMs` and `originAlpha` are out
    /// of bounds (§18.3.3 hard isolation 1, still in force; only the third
    /// condition was lifted by §20.3.4).
    ///
    /// ⛔ **B-48 (§24.4): same exclusion covers Capability C's tracking
    /// fields** — `trackedPoint`, `lastReAnchorTrackedPoint`, `trackState`.
    /// `pointX`/`pointY` are sourced from `canonicalPoint` ONLY, below, and
    /// must stay that way even when tracking is engaged and `trackedPoint` has
    /// wandered away from it — PIN-1 fixes a Pin's saved coordinates to the
    /// user's original tap, never to wherever the tracker currently believes
    /// the object is (§24.1.4: this is *why* Capability C was designed with a
    /// separate `trackedPoint` field instead of letting `canonicalPoint` move
    /// — so this function's contract is untouched by that design, not merely
    /// "still correct by luck"). If `TapInstance` ever grows further fields,
    /// they must be classified at the type level as either provenance
    /// (readable here) or tracking (never readable here) — don't rely on a
    /// reader remembering which is which from a comment alone.
    ///
    /// Two fields are deliberately **not** sourced from `instance` even though
    /// data exists there, because the "exactly three quantities" restriction is
    /// literal, not just "avoid re-anchor fields":
    ///   • `id` — a fresh `UUID()`, not `instance.id`.  A Pin's identity is not
    ///     tied to the `TapInstance` that produced it (the instance can be
    ///     FIFO-evicted or cleared while the Pin it produced lives on, §22.2.4).
    ///   • `iouPredAtCreation` — left `nil`.  It is documented on `PinRecordV1`
    ///     as "⛔ diagnostic only, never a gate or ordering", and reading it
    ///     would be a fourth quantity beyond the three this function is
    ///     chartered to read.
    ///
    /// R27(i/ii/iii) (§22.2): resolved.  R27(i)'s "saved at the instant a
    /// re-anchor is replacing the mask" is not specially arbitrated here — the
    /// caller's single locked snapshot (PIN-6) already makes "whichever version
    /// was current at snapshot time" the only meaningful answer.  R27(ii)/(iii)
    /// live entirely in the caller (long-press hit-test + no promote, no
    /// decode) — this function has no opinion about them.
    ///
    /// `previewFile` is always nil (R29).
    ///
    /// - Returns: nil when `instance` has no placed mask yet, or the mask is
    ///   not the v1 256×256 shape (defensive; the caller's hit-test already
    ///   requires a placed mask before this can be reached in practice).
    static func makeRecord(from instance: TapInstance,
                           geometry: FrameGeometry,
                           tag: String? = nil,
                           note: String? = nil) -> PinRecordV1?
    {
        guard let alpha = instance.maskAlpha, alpha.count == MaskPNGCodec.pixelCount else {
            return nil
        }
        let id = UUID().uuidString.lowercased()
        let createdAtEpoch = instance.createdAt.timeIntervalSince1970
        let geo = PinGeometryV1(from: geometry,
                                promptSpace: SAMConfiguration.pointPromptSpace,
                                encoderInputSize: SAMConfiguration.encoderInputSize)
        let record = PinRecordV1(
            id: id,
            pointX: Double(instance.canonicalPoint.x),
            pointY: Double(instance.canonicalPoint.y),
            geometry: geo,
            maskFile: PinRecordV1.maskFileName(for: id),
            maskWidth: MaskPNGCodec.side,
            maskHeight: MaskPNGCodec.side,
            maskNonZero: MaskPNGCodec.nonZeroCount(alpha),
            iouPredAtCreation: nil,
            createdAt: createdAtEpoch,
            tag: tag,
            note: note,
            updatedAt: createdAtEpoch,
            previewFile: nil)
        // B-48 (§24.4): defensive PIN-1 check, re-reading `canonicalPoint`
        // independently rather than trusting the values just assigned above —
        // if a future edit sources `pointX`/`pointY` from `trackedPoint`
        // instead (accidentally or otherwise) without updating this line too,
        // this catches the drift in DEBUG instead of it silently persisting.
        assert(record.pointX == Double(instance.canonicalPoint.x) &&
               record.pointY == Double(instance.canonicalPoint.y),
               "PIN-1 violation: pointX/pointY must be sourced from instance.canonicalPoint only, never trackedPoint")
        return record
    }

    /// The mask bytes that accompany a record into `PinStore.create`.
    /// Same three-quantity restriction as `makeRecord` — reads only `maskAlpha`.
    ///
    /// PIN-6 (§22.2.3): the caller must pass the **same** `instance` value it
    /// already used for `makeRecord` and for the sheet's preview — this
    /// function performs no locking of its own and must not be used to take a
    /// second, independent look at `TapInstanceManager`.
    static func maskAlpha(from instance: TapInstance) -> [UInt8]? {
        instance.maskAlpha
    }
}

// MARK: - B-28 — revisit entry (Day 6 implements)

/// Why a revisit was refused (§19.4.3 gates G1–G4, §19.4.4 outcome R-B).
///
/// ⚠️ Wording constraint (§19.4.4, tightened by §23.1.4 / PIN-7): user-facing
/// revisit copy may state *provenance* only — the permitted family is
/// `From Pin "<tag>"` / `From an untagged Pin` / `Re-segmented at this Pin's
/// saved point` (+ its in-flight `…ing` form); on the promote outcome R-D,
/// `That spot is already covered by a selection on screen.`  ⛔ "restore" /
/// "find your object back" / "tracking" / a bare tag / any similarity number
/// are forbidden — the system holds no stored quantity that could establish
/// object identity (R28), because the embedding is never persisted (PIN-4).
/// A revisit re-segments **the current scene** at a remembered pixel.
enum PinRevisitRefusal {
    /// G1 — rotation / mirroring / prompt space do not match the record.
    /// Canonical space's meaning is defined by those, so the same coordinate
    /// points somewhere else entirely; there is no remapping between them.
    /// The associated string carries the specific reason (orientation vs.
    /// camera vs. prompt-space vs. aspect ratio) — §19.4.4 requires the R-B
    /// text to state *why*, not just that it was refused.
    case geometryIncompatible(detail: String)
    /// G2 — not in `.tapToSegment`.  ⛔ Never switch modes automatically.
    case wrongMode
    /// G3 — no current frame geometry (camera not ready).
    case cameraNotReady

    var userMessage: String {
        switch self {
        case .geometryIncompatible(let detail): return detail
        case .wrongMode:      return "Switch to Tap to Segment mode to revisit this Pin."
        case .cameraNotReady: return "Camera not ready — try again in a moment."
        }
    }
}

// MARK: - Day 6 — revisit completion multiplexer

/// `CameraManager.onTapMaskPlaced` / `onTapFailed` / `onTapPromoted` are each
/// a single closure slot that fires for **every** tap, screen-originated or
/// revisit-originated alike (§19.4.6 + B-40's third hook, both in
/// `CameraManager.swift`). This multiplexes them by tap-generation number so
/// more than one outstanding revisit (or a revisit racing an ordinary screen
/// tap) does not stomp on the single slot. Main-thread only — all three hooks
/// fire from `CameraManager`'s existing main-thread blocks — so a plain
/// dictionary needs no lock, per §19.7's "no new lock".
///
/// **PIN-8 (§23.5):** this is the SOLE installer of the three hooks in the
/// whole app; nothing else may assign them.  The hooks are pure observation
/// exits — nothing here feeds back into tap routing or decode — and every
/// registered `gen` must be released by exactly one of placed / failed /
/// promoted (each handler below removes the gen from all three maps, which is
/// also what closes §23.1.9's ~64 KB/revisit leak: before B-40 the promote
/// outcome fired no hook, so its two entries — one closure holding the
/// origin blob read back from disk — stayed forever).
private final class PinRevisitTracker {
    static let shared = PinRevisitTracker()
    private init() {}

    private var placed: [Int: (CameraManager.TapPath, UUID, [UInt8]) -> Void] = [:]
    private var failed: [Int: (String) -> Void] = [:]
    private var promoted: [Int: (UUID) -> Void] = [:]
    private weak var hookedManager: CameraManager?

    /// B-41 — session-scoped revisit sequence number.  Allocated at
    /// `handleTap(fromPin:)` ENTRY, before any gate, so a revisit that dies
    /// mid-flight still consumes its number: that is what makes P-7's
    /// "seq covers 1…R with no gaps" check able to detect a lost line.
    private var seqCounter = 0

    func allocateSeq() -> Int {
        assert(Thread.isMainThread, "PinRevisitTracker.allocateSeq must run on main thread")
        seqCounter += 1
        return seqCounter
    }

    func register(gen: Int,
                  cameraManager: CameraManager,
                  onPlaced: @escaping (CameraManager.TapPath, UUID, [UInt8]) -> Void,
                  onFailed: @escaping (String) -> Void,
                  onPromoted: @escaping (UUID) -> Void) {
        assert(Thread.isMainThread, "PinRevisitTracker.register must run on main thread")
        placed[gen] = onPlaced
        failed[gen] = onFailed
        promoted[gen] = onPromoted
        installHooks(on: cameraManager)
    }

    private func installHooks(on cm: CameraManager) {
        guard hookedManager !== cm else { return }
        hookedManager = cm
        cm.onTapMaskPlaced = { [weak self] gen, path, instanceID, alpha in
            guard let self = self, let cb = self.placed.removeValue(forKey: gen) else { return }
            self.failed.removeValue(forKey: gen)
            self.promoted.removeValue(forKey: gen)
            cb(path, instanceID, alpha)
        }
        cm.onTapFailed = { [weak self] gen, message in
            guard let self = self, let cb = self.failed.removeValue(forKey: gen) else { return }
            self.placed.removeValue(forKey: gen)
            self.promoted.removeValue(forKey: gen)
            cb(message)
        }
        cm.onTapPromoted = { [weak self] gen, promotedID in
            guard let self = self, let cb = self.promoted.removeValue(forKey: gen) else { return }
            self.placed.removeValue(forKey: gen)
            self.failed.removeValue(forKey: gen)
            cb(promotedID)
        }
    }
}

extension CameraManager {

    /// Revisit a Pin: G1–G4, the one permitted coordinate derivation (§19.4.5 —
    /// same orientation, same aspect ratio, resolution only), then **a call
    /// through to the existing `handleTap(canonicalPoint:viewPoint:)`**.
    ///
    /// ⛔ **Constraint PIN-5: this must never become a second decode entry
    /// point.**  Routing through the existing tap inherits, for free, every
    /// Phase 3 invariant: `tapGeneration` / `requestGen` ordering, `inFlightTaps`
    /// accounting, the fast/slow path decision, park/drain, the timeouts,
    /// Requirement C failure visibility, geometry-change pool clearing, FIFO,
    /// and "a tap inside an existing mask promotes rather than re-decodes".  A
    /// parallel path would have to re-derive all of them, and this project has
    /// three recorded instances of re-derived invariants being the defect (§18.9).
    ///
    /// The derived coordinate is ⛔ never written back to the Pin (PIN-1); it is
    /// a one-shot value produced at revisit time.
    ///
    /// - Parameter store: needed only to load the Pin's stored blob for the
    ///   §19.4.6 IoU log field — "记录并展示，不判定" (computed and shown,
    ///   never gates anything). The Day 4 stub's signature did not carry this
    ///   because the log requirement had not been specified yet; see the
    ///   Day 6 builder report for this interpretation.
    /// - Returns: the tap sequence number the underlying `handleTap` assigned,
    ///   or 0 when G1/G2/G3 rejected the revisit before any tap was issued.
    @discardableResult
    func handleTap(fromPin pin: Pin, store: PinStore) -> Int {
        assert(Thread.isMainThread, "handleTap(fromPin:) must run on main thread")
        let shortID = String(pin.id.uuidString.prefix(8))

        // B-41 (§23.2.4) — `seq` is allocated at ENTRY, before ANY gate, so a
        // revisit that dies mid-flight still consumes its number.  That is
        // what makes P-7's completeness gate ("initiated count == line count,
        // seq covers 1…R with no gaps") able to detect a lost line.
        let seq = PinRevisitTracker.shared.allocateSeq()
        let pinField = String(format: "(%.1f,%.1f)",
                              pin.canonicalPoint.x, pin.canonicalPoint.y)

        // B-41 — the single line grammar every outcome goes through (§23.2.4
        // REPLACES §19.4.6's format; the old heterogeneous
        // `rejected reason=…` shape on the refusal paths is retired, so one
        // grep over `[PIN] revisit` now counts every revisit).  `pt` is the
        // derived canonical point actually sent into the prompt (n/a when the
        // revisit never dispatched one); `pin` is the record's stored point —
        // together they make the §19.4.5 coordinate derivation checkable
        // straight from the log (`geo=ok` ⇒ pt == pin).
        func revisitLine(outcome: String, reason: String, geo: String,
                         path: String, pt: CGPoint?, iou: String,
                         newPx: Int?) -> String {
            let ptField = pt.map { String(format: "(%.1f,%.1f)", $0.x, $0.y) } ?? "n/a"
            let newField = newPx.map { "\($0)px" } ?? "n/a"
            return "[PIN] revisit id=\(shortID) seq=\(seq) outcome=\(outcome)"
                 + " reason=\(reason) geo=\(geo) path=\(path) pt=\(ptField)"
                 + " pin=\(pinField) iou=\(iou) origin=\(pin.maskNonZero)px new=\(newField)"
        }

        // G3 — camera ready.  Checked first: G1 has nothing to compare against
        // without a current geometry snapshot.
        guard let geo = currentFrameGeometry() else {
            pinFault(revisitLine(outcome: "refused", reason: "cameraNotReady",
                                 geo: "n/a", path: "n/a", pt: nil,
                                 iou: "n/a", newPx: nil))
            pinRevisitEvent = PinRevisitEvent(
                pinID: pin.id,
                kind: .refused(reason: PinRevisitRefusal.cameraNotReady.userMessage))
            return 0
        }

        // G2 — mode.  ⛔ Never auto-switch (§19.4.3).
        guard currentMode == .tapToSegment else {
            pinFault(revisitLine(outcome: "refused", reason: "mode",
                                 geo: "n/a", path: "n/a", pt: nil,
                                 iou: "n/a", newPx: nil))
            pinRevisitEvent = PinRevisitEvent(
                pinID: pin.id,
                kind: .refused(reason: PinRevisitRefusal.wrongMode.userMessage))
            return 0
        }

        // G1 — geometry compatibility (§19.4.3), plus the ONE allowed
        // remapping (§19.4.5): same rotation/mirrored/promptSpace, only the
        // resolution differs, and the aspect ratio is preserved.
        let curRotation    = geo.videoRotationAngle
        let curMirrored    = geo.mirrored
        let curPromptSpace = SAMConfiguration.pointPromptSpace
        let sameOrientation = pin.geometry.isCompatible(withRotationDeg: curRotation,
                                                        mirrored: curMirrored,
                                                        promptSpace: curPromptSpace)

        let geoTag: String          // "ok" | "remapped" | "refused" — §19.4.6 log field
        var derivedPoint = pin.canonicalPoint   // never written back to the Pin (PIN-1)

        if !sameOrientation {
            geoTag = "refused"
        } else if pin.geometry.origW == Double(geo.origW) && pin.geometry.origH == Double(geo.origH) {
            geoTag = "ok"
        } else {
            let arPin = pin.geometry.origW / pin.geometry.origH
            let arNow = Double(geo.origW) / Double(geo.origH)
            if abs(arNow - arPin) < 1e-3 {
                let sx = Double(geo.origW) / pin.geometry.origW
                let sy = Double(geo.origH) / pin.geometry.origH
                var x = pin.canonicalPoint.x * CGFloat(sx)
                var y = pin.canonicalPoint.y * CGFloat(sy)
                // §2.3 Step 5 boundary clamp, re-applied to the derived point —
                // same clamp `FrameGeometry.invertViewPoint` performs, not a
                // second transform path (both just bound to [0, orig-1]).
                x = min(max(0, x), geo.origW - 1)
                y = min(max(0, y), geo.origH - 1)
                derivedPoint = CGPoint(x: x, y: y)
                geoTag = "remapped"
            } else {
                geoTag = "refused"   // aspect ratio changed — falls through to R-B
            }
        }

        guard geoTag != "refused" else {
            let rotMismatch    = pin.geometry.rotationDeg != curRotation
            let camMismatch    = pin.geometry.mirrored != curMirrored
            let promptMismatch = pin.geometry.promptSpace != curPromptSpace
            // B-41 reason token, single value per §23.2.4's grammar.  When a
            // geometry mismatch coexists with a promptSpace/aspect one the
            // orientation/camera token wins (it is the dimension the tester
            // controls, P-6 C2/C3); promptSpace next; aspect is the residual.
            let reasonToken: String
            if rotMismatch && camMismatch { reasonToken = "orientationCamera" }
            else if rotMismatch           { reasonToken = "orientation" }
            else if camMismatch           { reasonToken = "camera" }
            else if promptMismatch        { reasonToken = "promptSpace" }
            else                          { reasonToken = "aspect" }

            let pinPortrait = Int(pin.geometry.rotationDeg).isMultiple(of: 180) == false
            let curPortrait = Int(curRotation).isMultiple(of: 180) == false
            let pinCam = pin.geometry.mirrored ? "front" : "back"
            let curCam = curMirrored ? "front" : "back"
            var reasons: [String] = []
            if rotMismatch || camMismatch {
                reasons.append("This Pin was recorded in \(pinPortrait ? "portrait" : "landscape") orientation, \(pinCam) camera — that doesn't match the current \(curPortrait ? "portrait" : "landscape") orientation, \(curCam) camera.")
            }
            if promptMismatch {
                reasons.append("The segmentation configuration has changed since this Pin was recorded.")
            }
            if reasons.isEmpty {
                reasons.append("The camera resolution changed and no longer matches this Pin's aspect ratio.")
            }
            pinFault(revisitLine(outcome: "refused", reason: reasonToken,
                                 geo: "refused", path: "n/a", pt: nil,
                                 iou: "n/a", newPx: nil))
            let refusal = PinRevisitRefusal.geometryIncompatible(detail: reasons.joined(separator: " "))
            pinRevisitEvent = PinRevisitEvent(pinID: pin.id, kind: .refused(reason: refusal.userMessage))
            return 0
        }

        // G4 (pool capacity) needs no special case here — the call below goes
        // through the exact same FIFO(max=3) as every other tap (§3.2).

        // B-37 route A1 (§23.1.4): project the derived canonical point back
        // into view space so the ordinary tap path gets the `viewPoint` it
        // keys every anchor affordance off — in-flight pulse, ripple, and the
        // anchor marker itself.  Passing nil here was the whole of §23.1.1's
        // "spatially anonymous revisit" defect (M-23.1); the projection is
        // `FrameGeometry.projectCanonicalPoint`, the algebraic inverse of
        // `invertViewPoint`, reading the same geometry snapshot.  A nil
        // return (no preview layer yet) degrades to the old markerless
        // behaviour rather than blocking the revisit.
        let projectedViewPoint = viewPoint(forCanonicalPoint: derivedPoint)

        let gen = self.handleTap(canonicalPoint: derivedPoint, viewPoint: projectedViewPoint)
        guard gen > 0 else {
            // Defensive only: G2 ran on this same thread, so the mode guard
            // inside handleTap cannot have flipped in between.  `pt=n/a` —
            // no prompt was dispatched.
            pinFault(revisitLine(outcome: "failed", reason: "decodeFailed",
                                 geo: geoTag, path: "n/a", pt: nil,
                                 iou: "n/a", newPx: nil))
            pinRevisitEvent = PinRevisitEvent(pinID: pin.id, kind: .failed)
            return gen
        }

        // Join: the outcome=mask line needs BOTH the origin alpha (loaded
        // from the Pin's blob, if it has one) and the placed mask.  The
        // failed / promote outcomes log immediately — their lines carry no
        // iou.  All callbacks run on main; no lock needed.  `logged` is the
        // exactly-one-line discipline (§23.2.4: every entry into this
        // function produces exactly one line, at some terminal).
        final class Join {
            var origin: [UInt8]??               // .some(nil) = record has no blob
            var placement: (String, [UInt8])?   // (path label, new alpha)
            var didPromote = false
            var logged = false
        }
        let join = Join()

        // §19.4.4 R-A copy switch. Deferred onto the main queue (not a direct
        // write) so it runs AFTER the reset-to-false the call above already
        // enqueued for this same tap — GCD preserves FIFO order for
        // `DispatchQueue.main.async` calls issued from the same thread in
        // sequence, so this lands after both the normal-path reset AND the
        // promote-path reset/`onTapPromoted` block (both were enqueued inside
        // the handleTap call above).
        //
        // R-D guard (B-40, §23.1.9): when the tap resolved as a promote, the
        // promote block has therefore ALREADY run and set `join.didPromote`
        // by the time this executes — so the flag stays false and no
        // "Re-segmenting…" copy can appear for a revisit that decoded
        // nothing (PIN-7 table, last row: that copy would be a decidable
        // falsehood).
        DispatchQueue.main.async { [weak self] in
            guard let self = self, !join.didPromote else { return }
            self.lastTapWasRevisit = true
        }

        func finishMaskLine() {
            guard !join.logged,
                  let (pathLabel, alpha) = join.placement,
                  join.origin != nil else { return }
            join.logged = true
            let iouStr: String
            if let origin = join.origin ?? nil, origin.count == MaskPNGCodec.pixelCount {
                // §23.2.4 keeps §19.4.6's computation verbatim: 256×256
                // stored space, stride 1 full traversal, DriftDetector's
                // existing implementation — logged, never a gate (§19.4.3).
                let iou = DriftDetector.alphaIoU(origin, alpha,
                                                 width: MaskPNGCodec.side,
                                                 height: MaskPNGCodec.side,
                                                 stride: 1)
                iouStr = String(format: "%.2f", iou)
            } else {
                iouStr = "n/a"
            }
            pinLog(revisitLine(outcome: "mask", reason: "n/a", geo: geoTag,
                               path: pathLabel, pt: derivedPoint, iou: iouStr,
                               newPx: MaskPNGCodec.nonZeroCount(alpha)))
        }

        PinRevisitTracker.shared.register(
            gen: gen,
            cameraManager: self,
            onPlaced: { [weak self] path, instanceID, alpha in
                // B-37: decorate the revisit product — ↻ glyph + provenance
                // label, lifecycle identical to 📌 (§22.2.4 / §23.1.4 四).
                self?.markInstanceRevisitOrigin(id: instanceID, pinTag: pin.tag)
                // Grammar note: `path` in §23.2.4 is (fast|slow|n/a); a
                // parked tap rode someone else's encode but the user waited
                // it out, so it reports as slow — same mapping the pre-B-41
                // line used.
                join.placement = (path == .fast ? "fast" : "slow", alpha)
                finishMaskLine()
            },
            onFailed: { [weak self] message in
                guard !join.logged else { return }
                join.logged = true
                let reason = message.contains("timed out") ? "timeout" : "decodeFailed"
                pinFault(revisitLine(outcome: "failed", reason: reason,
                                     geo: geoTag, path: "n/a", pt: derivedPoint,
                                     iou: "n/a", newPx: nil))
                self?.pinRevisitEvent = PinRevisitEvent(pinID: pin.id, kind: .failed)
            },
            onPromoted: { [weak self] _ in
                guard !join.logged else { return }
                join.logged = true
                join.didPromote = true
                // Outcome R-D (§23.1.9): the derived point landed inside an
                // existing live mask — promoted, no decode.  ⛔ No
                // "Re-segment(ing)" wording anywhere on this path.
                pinLog(revisitLine(outcome: "promote", reason: "n/a",
                                   geo: geoTag, path: "n/a", pt: derivedPoint,
                                   iou: "n/a", newPx: nil))
                self?.pinRevisitEvent = PinRevisitEvent(
                    pinID: pin.id, kind: .promoted(fromPinTag: pin.tag))
            })

        if pin.maskFile != nil {
            store.loadMaskImage(id: pin.id) { result in
                join.origin = .some(try? result.get())
                finishMaskLine()
            }
        } else {
            join.origin = .some(nil)
        }

        return gen
    }
}
