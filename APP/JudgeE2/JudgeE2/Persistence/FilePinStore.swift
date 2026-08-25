//
//  FilePinStore.swift
//  JudgeE2
//
//  Phase 4B — Day 4 (Builder) · B-18 / B-21 / B-23 / B-24 / B-25
//
//  The one and only `PinStore` implementation: a manifest of metadata plus
//  immutable sidecar mask blobs.  Contract: architect_output.md §19.
//
//      <Application Support>/Pins/
//      ├── manifest.json        ← all metadata, single file, atomic write
//      ├── manifest.v<N>.bak    ← pre-migration backup (only when migrating)
//      └── masks/
//          └── <uuid>.png       ← 256×256 8-bit grey, one per Pin, immutable
//
//  WHY NOT CoreData / SwiftData (§19.1.4).  `NSManagedObject` / `@Model` are
//  context-bound reference types, and every cross-queue discipline in this
//  project is built on value-type snapshots (`capturedEmbedding`,
//  `tapGeometryMirror`, `tapInstances.snapshot()`).  Bringing a family of
//  context-bound objects in would re-invite the exact defect class this project
//  has already been bitten by three times.  Also: `.xcdatamodeld` is a binary
//  artefact, and contracts here survive by being reviewable.
//
//  WHY BLOBS ARE SEPARATE FILES.  A mask is immutable; `tag`/`note` are not.
//  Inlining the blob would base64 it into the manifest (+33 %) and make one tag
//  edit rewrite every byte of every mask.  Split apart, a tag edit rewrites
//  ≤ 60 KB of JSON and touches no blob at all.
//
//  THREADING (§19.3).
//    • `pin.store.io` — one serial queue at `.utility`, created here and
//      nowhere else.  **Serial is the mutual exclusion: this subsystem adds no
//      lock.**  Every mutation of the on-disk state and of the in-memory index
//      happens on it.
//    • ⛔ It is not `decoderQueue`.  §21.3 measured tap `qwait` reaching 4.9 ms
//      with only two kinds of work on that queue — 0.1 ms from the reopen
//      trigger.  A flush is orders of magnitude larger than 0.1 ms, so putting
//      Pin IO there would consume that headroom predictably, not statistically.
//    • ⛔ It is not `videoQueue` / `sessionQueue` (real-time) nor `encoderQueue`
//      (the single encoder slot).
//    • `.utility` sits strictly below `decoderQueue`'s `.userInitiated`, so
//      under CPU/IO contention the scheduler yields to inference by itself.
//    • Constraint PIN-3: no symbol from this file may appear inside a closure
//      running on `videoQueue` / `decoderQueue` / `encoderQueue`.
//
//  DURABILITY (§19.3.3 + acceptance P-3(ii)).  A `.success` callback means the
//  change is *on disk*.  Writes still coalesce — mutations mark the manifest
//  dirty and their completions are parked; the ≤ 250 ms window then performs one
//  atomic write and releases every parked completion together.  So a burst of
//  N saves costs one manifest rewrite, and "the callback said success" still
//  survives an immediate SIGKILL.
//

import Combine
import Foundation

final class FilePinStore: ObservableObject, PinStore {

    // MARK: - Published (main thread only) — B-34: explicit @MainActor

    /// Value-type snapshot of the index, ascending by `createdAt`.
    @MainActor @Published private(set) var pins: [Pin] = []

    /// ⛔ While this is false the UI must show a loading state, never "no pins".
    @MainActor @Published private(set) var isLoaded: Bool = false

    @MainActor @Published private(set) var lastWriteError: PinStoreError?

    /// Main-thread lookup mirror.  Kept in lockstep with `pins`, so `fetch(id:)`
    /// is a dictionary hit with no queue hop and no lock.
    ///
    /// B-34: `nonisolated`, not `@MainActor` — its only writer is the
    /// `DispatchQueue.main.async` block inside `publishSnapshot` (plain GCD,
    /// not Swift-actor-checked), and its only reader is `fetch(id:)`, itself
    /// always called from main.  The actual thread discipline here has always
    /// been GCD, not the actor system; this annotation just stops the compiler
    /// from asserting a discipline (@MainActor-synchronous access) nothing in
    /// this file actually follows.
    nonisolated(unsafe) private var mainIndex: [UUID: Pin] = [:]

    // MARK: - Queue (B-21 — exactly one new queue, no new locks)

    nonisolated private let queue = DispatchQueue(label: "pin.store.io", qos: .utility)

    // MARK: - State owned by `queue` — B-34: nonisolated (real execution domain
    // is `pin.store.io`, never main; see §22.1.7).

    nonisolated(unsafe) private var records: [String: PinRecordV1] = [:]
    /// Ids ordered by `createdAt` ascending, then by id for a total order.
    nonisolated(unsafe) private var order: [String] = []

    nonisolated(unsafe) private var manifestDirty = false
    nonisolated(unsafe) private var flushScheduled = false
    /// Completions parked until the flush that contains their change lands.
    nonisolated(unsafe) private var pendingCompletions: [(Result<Void, PinStoreError>) -> Void] = []

    /// Directory creation failed ⇒ every mutating call fails with `.unavailable`.
    /// ⛔ Never degrade to "in-memory pins" — that would let the UI report a
    /// save that did not happen (§19.5.5).
    nonisolated(unsafe) private var storeUnavailable = false
    /// Manifest written by a newer build ⇒ read-only, and ⛔ never rewritten.
    nonisolated(unsafe) private var forwardVersion: Int?

    nonisolated(unsafe) private var loadStarted = false

    /// Lazily-read mask blobs, ≤ 32 entries (≈ 2 MiB decoded, 65,536 B × 32).
    /// Queue-confined.  (B-33: the number was 32 all along; only the byte
    /// estimate in this comment — and in §19.3.2 — was wrong by 8×, having
    /// been computed for the *encoded PNG* rather than the decoded `[UInt8]`
    /// this cache actually holds.)
    nonisolated(unsafe) private var blobCache: [String: [UInt8]] = [:]
    nonisolated(unsafe) private var blobLRU: [String] = []
    nonisolated private static let blobCacheCapacity = 32

    nonisolated private static let coalesceWindow: DispatchTimeInterval = .milliseconds(250)

    // MARK: - Paths

    nonisolated private let rootURL: URL
    nonisolated private let masksURL: URL
    nonisolated private let manifestURL: URL

    /// - Parameter directory: overridable for tests; production uses
    ///   `Application Support/Pins` — **not** `Documents` (which
    ///   `UIFileSharingEnabled` exposes to the Files app) and **not** `Caches`
    ///   (which the system may reclaim; Pins are user data) — §19.5.5.
    nonisolated init(directory: URL? = nil) {
        let base = directory ?? FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Pins", isDirectory: true)
        self.rootURL     = base
        self.masksURL    = base.appendingPathComponent("masks", isDirectory: true)
        self.manifestURL = base.appendingPathComponent("manifest.json")
    }

    // MARK: - Load (B-23)

    nonisolated func load() {
        var alreadyStarted = false
        queue.sync {
            alreadyStarted = loadStarted
            loadStarted = true
        }
        guard !alreadyStarted else { return }

        queue.async { [weak self] in
            guard let self = self else { return }
            let t0 = PerfLogger.nowMs()

            // 1. Directory.
            do {
                try FileManager.default.createDirectory(at: self.masksURL,
                                                        withIntermediateDirectories: true)
            } catch {
                self.storeUnavailable = true
                pinFault("[PIN] load FAILED err=\(PinStoreError.io(underlying: error))")
                self.publishSnapshot(loaded: true)
                return
            }

            // 2. Manifest.
            let manifestStamp = self.readManifestStamp()
            guard let data = try? Data(contentsOf: self.manifestURL) else {
                // No manifest yet is the normal first-launch state, not a fault.
                // mtime/size are naturally absent here — "-"/0 rather than forced.
                pinLog(String(format: "[PIN] load ok pins=0 orphans=0 new=1 ms=%.1f mtime=%@ size=%dB",
                              PerfLogger.nowMs() - t0, manifestStamp.mtime, manifestStamp.size))
                self.publishSnapshot(loaded: true)
                return
            }

            let onDiskVersion = PinSchema.peekVersion(in: data) ?? PinSchema.currentVersion

            // 3. Forward version ⇒ read-only, never rewrite (§19.5.2 rule 3).
            if onDiskVersion > PinSchema.currentVersion {
                self.forwardVersion = onDiskVersion
                let decoded = try? PinCoders.decoder.decode(PinManifestV1.self, from: data)
                self.installRecords(decoded?.pins ?? [])
                pinFault("[PIN] load FAILED err=\(PinStoreError.schemaTooNew(found: onDiskVersion, supported: PinSchema.currentVersion)) readOnly=1 pins=\(self.records.count)")
                self.publishSnapshot(loaded: true)
                return
            }

            // 4. Migration (B-25).  The chain is empty at v1; the structure is
            //    here so the first schema change is a data change only.
            var bytes = data
            var migrated = false
            if onDiskVersion < PinSchema.currentVersion {
                do {
                    try self.backupManifest(version: onDiskVersion, bytes: data)
                    bytes = try PinSchema.migrate(data, from: onDiskVersion)
                    migrated = true
                } catch {
                    // B-30 (§22.1.2): a failed migration must not leave the
                    // store writable — the very next `create` would otherwise
                    // atomically overwrite `manifest.json` with a one-record
                    // file, and the following launch's orphan GC would delete
                    // every blob that isn't in it.  `backupManifest` above
                    // already wrote the forensic `.bak` for this path.
                    self.storeUnavailable = true
                    pinFault("[PIN] load FAILED err=\(PinStoreError.io(underlying: error)) stage=migrate from=\(onDiskVersion)")
                    self.publishSnapshot(loaded: true)
                    self.publishUnavailableNow()
                    return
                }
            }

            // 5. Decode.  ⛔ No blob is read here (§19.3.2) — a full load that
            //    opens N files would show up as the P-2 linearity failure.
            do {
                let manifest = try PinCoders.decoder.decode(PinManifestV1.self, from: bytes)
                self.installRecords(manifest.pins)
            } catch {
                // B-30 (§22.1.2): same two-step irreversible loss as the
                // migrate branch — a decode failure must block further writes,
                // not just log and fall through to "0 records, fully writable".
                // This path previously had no forensic backup at all (only the
                // migrate branch did); write one now, best-effort.
                self.storeUnavailable = true
                self.writeCorruptBackup(bytes: bytes, schemaVersion: onDiskVersion)
                pinFault("[PIN] load FAILED err=\(PinStoreError.io(underlying: error)) stage=decode")
                self.publishSnapshot(loaded: true)
                self.publishUnavailableNow()
                return
            }

            // 6. GC orphaned blobs (file present, no record refers to it).
            let orphans = self.collectOrphanBlobs()

            if migrated {
                self.manifestDirty = true
                self.flushNow()
            }

            pinLog(String(format: "[PIN] load ok pins=%d orphans=%d migratedFrom=%d ms=%.1f mtime=%@ size=%dB",
                          self.records.count, orphans,
                          migrated ? onDiskVersion : PinSchema.currentVersion,
                          PerfLogger.nowMs() - t0, manifestStamp.mtime, manifestStamp.size))
            self.publishSnapshot(loaded: true)
        }
    }

    /// Queue-confined.  Rebuilds `records` + `order` from a decoded list.
    nonisolated private func installRecords(_ list: [PinRecordV1]) {
        records.removeAll(keepingCapacity: true)
        for r in list { records[r.id] = r }
        rebuildOrder()
    }

    nonisolated private func rebuildOrder() {
        order = records.values
            .sorted { ($0.createdAt, $0.id) < ($1.createdAt, $1.id) }
            .map(\.id)
    }

    /// Queue-confined.  Deletes blob files no record refers to.
    /// - Returns: how many were removed.
    nonisolated private func collectOrphanBlobs() -> Int {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: masksURL,
                                                      includingPropertiesForKeys: nil)
        else { return 0 }
        var removed = 0
        for url in files where url.pathExtension.lowercased() == "png" {
            let stem = url.deletingPathExtension().lastPathComponent
            if records[stem] == nil {
                if (try? fm.removeItem(at: url)) != nil { removed += 1 }
            }
        }
        return removed
    }

    /// R34 (debug_report.md §39.7): the file identity a pasted `[PIN] load ok`
    /// line needs to answer "is this the manifest that was just written, or a
    /// stale one" without a device-container pull.  Best-effort on top of the
    /// stat `Data(contentsOf:)` already performs a few lines below — if
    /// `attributesOfItem` can't be read (e.g. no manifest yet), this returns
    /// placeholders rather than becoming a new fault source.
    /// `mtime` is an absolute wall-clock string (device-local timezone), not
    /// the process-relative `[t=…]` monotonic prefix `PerfLogger` adds — the
    /// two clocks aren't comparable across process boundaries, which is
    /// exactly what this field exists to let a reader do.
    nonisolated private func readManifestStamp() -> (mtime: String, size: Int) {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: manifestURL.path) else {
            return ("-", 0)
        }
        let size = (attrs[.size] as? NSNumber)?.intValue ?? 0
        guard let modDate = attrs[.modificationDate] as? Date else {
            return ("-", size)
        }
        return (Self.manifestStampFormatter.string(from: modDate), size)
    }

    nonisolated private static let manifestStampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS ZZZZZ"
        f.timeZone = TimeZone.current
        return f
    }()

    nonisolated private func backupManifest(version: Int, bytes: Data) throws {
        let url = rootURL.appendingPathComponent("manifest.v\(version).bak")
        try? FileManager.default.removeItem(at: url)     // keep the latest 1
        try bytes.write(to: url, options: [.atomic])
    }

    /// B-30 (§22.1.2 point 2) — forensic, best-effort backup of bytes that
    /// failed to *decode* (as opposed to `backupManifest`, which backs up
    /// bytes that decoded fine but were a stale schema).  `try?`: this is not
    /// on any success path, and a failure to write the forensic copy must not
    /// mask or replace the original decode failure being reported.
    nonisolated private func writeCorruptBackup(bytes: Data, schemaVersion: Int?) {
        let tag = schemaVersion.map(String.init) ?? "unknown"
        let url = rootURL.appendingPathComponent("manifest.corrupt-\(tag).bak")
        try? bytes.write(to: url, options: [.atomic])
    }

    /// B-30 (§22.1.2 point 3) — `storeUnavailable` must be observable
    /// immediately, not only the next time a write is attempted.  Reuses the
    /// existing `@Published lastWriteError`; no new published property.
    nonisolated private func publishUnavailableNow() {
        Self.onMain { [weak self] in
            self?.lastWriteError = .unavailable
        }
    }

    /// B-34 — bridges a callback already running on the main thread (via
    /// `DispatchQueue.main.async`, this file's only mechanism for reaching
    /// main, unchanged since Day 4) into `@MainActor` isolation, so the
    /// `@MainActor` properties above can be written from it without a false
    /// "cannot be used in nonisolated context" diagnostic.  Purely a
    /// compile-time bridge — `MainActor.assumeIsolated` performs a runtime
    /// check that the calling thread is in fact the main actor's executor and
    /// then runs `work` synchronously; it does not change *how* main is
    /// reached, which is the runtime-behaviour-preservation §37.3.2 requires.
    nonisolated private static func onMain(_ work: @escaping @MainActor () -> Void) {
        DispatchQueue.main.async {
            MainActor.assumeIsolated(work)
        }
    }

    // MARK: - Reads (main thread, memory index, never disk)

    @MainActor func fetchAll() -> [Pin] { pins }

    @MainActor func fetch(id: UUID) -> Pin? { mainIndex[id] }

    // MARK: - Create (B-18)

    nonisolated func create(_ inputRecord: PinRecordV1,
                maskAlpha: [UInt8]?,
                completion: @escaping (Result<Void, PinStoreError>) -> Void)
    {
        // B-31 (§22.1.3): idempotent — if `load()` already ran, this is a
        // single `queue.sync` check and nothing else.  Guarantees `load()`'s
        // `installRecords` closure is enqueued strictly before this call's own
        // `queue.async` below, on the same serial queue — so a `create` that
        // races ahead of the first `load()` can no longer have its write
        // silently erased by `installRecords`' `removeAll` (ISSUE-D4-2 / P0-2).
        load()
        // B-35 (§22.1.8): canonical empty-string normalisation, aligned with
        // `update` — `create` must not be the one path that persists `""`.
        var record = inputRecord
        record.tag  = Self.normalizeEmptyField(record.tag)
        record.note = Self.normalizeEmptyField(record.note)

        queue.async { [weak self] in
            guard let self = self else { return }

            if let failure = self.availabilityFailure() {
                self.finish(.failure(failure), op: "create", id: record.id, completion: completion)
                return
            }
            if let failure = Self.fieldLengthFailure(tag: record.tag, note: record.note) {
                self.finish(.failure(failure), op: "create", id: record.id, completion: completion)
                return
            }
            if let existing = self.records[record.id],
               let field = Self.firstFrozenMismatch(existing: existing, incoming: record) {
                // PIN-1: an id that already exists may not have its frozen
                // fields rewritten — this is why there is no `save(_:)`.
                self.finish(.failure(.frozenFieldMutation(field: field)),
                            op: "create", id: record.id, completion: completion)
                return
            }
            if let integrity = Self.integrityFailure(record: record, alpha: maskAlpha) {
                self.finish(.failure(.io(underlying: integrity)),
                            op: "create", id: record.id, completion: completion)
                return
            }

            // Blob first, immediately, and once — an immutable file has no
            // concurrent writer, no overwrite race and nothing to merge.
            var blobBytes = 0
            if let alpha = maskAlpha, record.maskFile != nil {
                do {
                    let png = try MaskPNGCodec.encode(alpha)
                    blobBytes = png.count
                    try png.write(to: self.blobURL(for: record.id), options: [.atomic])
                    self.cacheBlob(alpha, for: record.id)
                } catch {
                    self.finish(.failure(.io(underlying: error)),
                                op: "create", id: record.id, completion: completion)
                    return
                }
            }

            self.records[record.id] = record
            self.rebuildOrder()
            self.markDirtyAndPark(completion,
                                  logSuccess: { count in
                pinLog("[PIN] create id=\(Self.short(record.id)) ok blob=\(blobBytes)B nz=\(record.maskNonZero) pins=\(count)")
            },
                                  logFailure: { err in
                pinFault("[PIN] create id=\(Self.short(record.id)) FAILED err=\(err)")
            })
        }
    }

    // MARK: - Update (mutable fields only)

    nonisolated func update(id: UUID,
                tag: String?,
                note: String?,
                completion: @escaping (Result<Void, PinStoreError>) -> Void)
    {
        load()   // B-31 — idempotent; see `create` for the full rationale.
        let key = id.uuidString.lowercased()
        queue.async { [weak self] in
            guard let self = self else { return }

            if let failure = self.availabilityFailure() {
                self.finish(.failure(failure), op: "update", id: key, completion: completion)
                return
            }
            guard var record = self.records[key] else {
                self.finish(.failure(.notFound), op: "update", id: key, completion: completion)
                return
            }
            if let failure = Self.fieldLengthFailure(tag: tag, note: note) {
                self.finish(.failure(failure), op: "update", id: key, completion: completion)
                return
            }

            // nil = leave unchanged; "" = clear.  Frozen fields are not even
            // addressable through this API (PIN-1 enforced by the signature).
            // B-35: same normalisation function `create` now uses too.
            if let tag  = tag  { record.tag  = Self.normalizeEmptyField(tag) }
            if let note = note { record.note = Self.normalizeEmptyField(note) }
            record.updatedAt = Date().timeIntervalSince1970

            self.records[key] = record
            self.markDirtyAndPark(completion,
                                  logSuccess: { count in
                pinLog("[PIN] update id=\(Self.short(key)) ok tagLen=\(record.tag?.count ?? 0) noteLen=\(record.note?.count ?? 0) pins=\(count)")
            },
                                  logFailure: { err in
                pinFault("[PIN] update id=\(Self.short(key)) FAILED err=\(err)")
            })
        }
    }

    // MARK: - Delete

    nonisolated func delete(id: UUID, completion: @escaping (Result<Void, PinStoreError>) -> Void) {
        load()   // B-31 — idempotent; see `create` for the full rationale.
        let key = id.uuidString.lowercased()
        queue.async { [weak self] in
            guard let self = self else { return }

            if let failure = self.availabilityFailure() {
                self.finish(.failure(failure), op: "delete", id: key, completion: completion)
                return
            }
            guard self.records[key] != nil else {
                self.finish(.failure(.notFound), op: "delete", id: key, completion: completion)
                return
            }

            self.records.removeValue(forKey: key)
            self.rebuildOrder()
            self.evictBlob(key)
            try? FileManager.default.removeItem(at: self.blobURL(for: key))

            self.markDirtyAndPark(completion,
                                  logSuccess: { count in
                pinLog("[PIN] delete id=\(Self.short(key)) ok pins=\(count)")
            },
                                  logFailure: { err in
                pinFault("[PIN] delete id=\(Self.short(key)) FAILED err=\(err)")
            })
        }
    }

    // MARK: - Blob lazy load + LRU (B-23)

    nonisolated func loadMaskImage(id: UUID,
                       completion: @escaping (Result<[UInt8], PinStoreError>) -> Void)
    {
        load()   // B-31 — idempotent; see `create` for the full rationale.
        let key = id.uuidString.lowercased()
        queue.async { [weak self] in
            guard let self = self else { return }

            func deliver(_ result: Result<[UInt8], PinStoreError>) {
                DispatchQueue.main.async { completion(result) }
            }

            guard let record = self.records[key] else {
                pinFault("[PIN] blob id=\(Self.short(key)) FAILED err=\(PinStoreError.notFound)")
                deliver(.failure(.notFound))
                return
            }
            guard record.maskFile != nil, !record.isDegraded else {
                // A degraded blob (not 256×256) stays out of every composite;
                // the metadata is untouched and still fully usable (§19.5.4).
                pinFault("[PIN] blob id=\(Self.short(key)) FAILED err=\(PinStoreError.notFound) degraded=\(record.isDegraded)")
                deliver(.failure(.notFound))
                return
            }
            if let cached = self.cachedBlob(key) {
                deliver(.success(cached))
                return
            }
            do {
                let data  = try Data(contentsOf: self.blobURL(for: key))
                let alpha = try MaskPNGCodec.decode(data)
                self.cacheBlob(alpha, for: key)
                deliver(.success(alpha))
            } catch {
                pinFault("[PIN] blob id=\(Self.short(key)) FAILED err=\(PinStoreError.io(underlying: error))")
                deliver(.failure(.io(underlying: error)))
            }
        }
    }

    nonisolated private func blobURL(for id: String) -> URL {
        masksURL.appendingPathComponent("\(id).png")
    }

    nonisolated private func cachedBlob(_ key: String) -> [UInt8]? {
        guard let alpha = blobCache[key] else { return nil }
        if let i = blobLRU.firstIndex(of: key) { blobLRU.remove(at: i) }
        blobLRU.append(key)
        return alpha
    }

    nonisolated private func cacheBlob(_ alpha: [UInt8], for key: String) {
        blobCache[key] = alpha
        if let i = blobLRU.firstIndex(of: key) { blobLRU.remove(at: i) }
        blobLRU.append(key)
        while blobLRU.count > Self.blobCacheCapacity {
            let evicted = blobLRU.removeFirst()
            blobCache.removeValue(forKey: evicted)
        }
    }

    nonisolated private func evictBlob(_ key: String) {
        blobCache.removeValue(forKey: key)
        if let i = blobLRU.firstIndex(of: key) { blobLRU.remove(at: i) }
    }

    // MARK: - Coalesced write (B-21)

    /// Queue-confined.  Marks the manifest dirty and parks the completion.
    /// Arms the ≤ 250 ms window if it is not already armed.
    ///
    /// B-32 (§22.1.1): does **not** publish here any more.  §19.3.4 ("a
    /// success state must be driven by the callback, never optimistically
    /// pre-rendered") outranks §19.3.2's silence on *when* `pins` publishes —
    /// publishing here meant a write that later failed still left the failed
    /// record sitting in `@Published pins`, which is exactly the UI-visible
    /// inconsistency §19.3.4 forbids.  `publishSnapshot` now runs only from
    /// `flushNow`'s success branch, so the `pins` channel and the completion
    /// channel are on the same clock: both wait for a real flush to land.
    nonisolated private func markDirtyAndPark(_ completion: @escaping (Result<Void, PinStoreError>) -> Void,
                                  logSuccess: @escaping (Int) -> Void,
                                  logFailure: @escaping (PinStoreError) -> Void)
    {
        manifestDirty = true
        let count = records.count
        pendingCompletions.append { result in
            switch result {
            case .success: logSuccess(count)
            case .failure(let err): logFailure(err)
            }
            completion(result)
        }
        scheduleFlush()
    }

    nonisolated private func scheduleFlush() {
        guard !flushScheduled else { return }
        flushScheduled = true
        queue.asyncAfter(deadline: .now() + Self.coalesceWindow) { [weak self] in
            guard let self = self else { return }
            self.flushScheduled = false
            self.flushNow()
        }
    }

    /// Queue-confined.  One atomic full rewrite, then every parked completion
    /// is released — which is what makes a `.success` callback mean "on disk".
    /// B-32: the success branch is now also the *only* place a mutation's
    /// result reaches `@Published pins` — see `markDirtyAndPark`.
    nonisolated private func flushNow() {
        guard manifestDirty else { return }
        guard forwardVersion == nil else {
            // ⛔ A manifest from a newer build is never rewritten: one downgrade
            // must not truncate the user's data.
            releasePending(.failure(.schemaTooNew(found: forwardVersion!,
                                                  supported: PinSchema.currentVersion)))
            manifestDirty = false
            return
        }

        let manifest = PinManifestV1(schemaVersion: PinSchema.currentVersion,
                                     writtenAt: Date().timeIntervalSince1970,
                                     pins: order.compactMap { records[$0] })
        do {
            let data = try PinCoders.encoder.encode(manifest)
            try data.write(to: manifestURL, options: [.atomic])   // temp file + rename
            manifestDirty = false
            publishSnapshot(loaded: true)   // B-32 — moved from markDirtyAndPark
            releasePending(.success(()))
        } catch {
            // ⛔ Never `try?` a write away.  The manifest stays dirty so the next
            // forced flush retries, and the failure reaches the UI.  `pins` is
            // NOT published here — it stays at the last snapshot that really
            // made it to disk (B-32).
            releasePending(.failure(.io(underlying: error)))
        }
    }

    nonisolated private func releasePending(_ result: Result<Void, PinStoreError>) {
        let waiting = pendingCompletions
        pendingCompletions.removeAll(keepingCapacity: true)
        guard !waiting.isEmpty else { return }
        Self.onMain { [weak self] in
            if case .failure(let err) = result { self?.lastWriteError = err }
            for c in waiting { c(result) }
        }
    }

    /// Forced flush.  `waitUntilDone: true` at `scenePhase → .background` and
    /// `willTerminate`; at those moments no real-time inference is running, so
    /// blocking on the serial queue costs nothing that matters (§19.3.3).
    nonisolated func flush(waitUntilDone: Bool) {
        if waitUntilDone {
            queue.sync { self.flushNow() }
        } else {
            queue.async { self.flushNow() }
        }
    }

    // MARK: - Validation

    nonisolated private func availabilityFailure() -> PinStoreError? {
        if storeUnavailable { return .unavailable }
        if let v = forwardVersion {
            return .schemaTooNew(found: v, supported: PinSchema.currentVersion)
        }
        return nil
    }

    /// §19.2.5 — reject, never truncate.  B-35 (§22.1.8): character count is
    /// UTF-16 code units via `PinFieldLimits.length(of:)` — the same function
    /// the Day 5 UI's input limiter calls, so the two layers can never disagree
    /// about what "64 characters" means.
    nonisolated private static func fieldLengthFailure(tag: String?, note: String?) -> PinStoreError? {
        if let tag = tag, PinFieldLimits.length(of: tag) > PinFieldLimits.maxTagCharacters {
            return .fieldTooLong(field: "tag")
        }
        if let note = note, PinFieldLimits.length(of: note) > PinFieldLimits.maxNoteCharacters {
            return .fieldTooLong(field: "note")
        }
        return nil
    }

    /// B-35 (§22.1.8) — canonical empty-string normalisation: `""` persists as
    /// `nil`.  `create` and `update` must use the same function so the same
    /// user state (no tag) never has two different on-disk representations.
    nonisolated private static func normalizeEmptyField(_ s: String?) -> String? {
        guard let s = s, !s.isEmpty else { return nil }
        return s
    }

    /// PIN-1 — the frozen half of the record, field by field.  Returns the name
    /// of the first field that differs, so a failure can be reported by name
    /// rather than as a generic mismatch.
    nonisolated private static func firstFrozenMismatch(existing: PinRecordV1,
                                            incoming: PinRecordV1) -> String?
    {
        if existing.pointX            != incoming.pointX            { return "pointX" }
        if existing.pointY            != incoming.pointY            { return "pointY" }
        if existing.geometry          != incoming.geometry          { return "geometry" }
        if existing.maskFile          != incoming.maskFile          { return "maskFile" }
        if existing.maskWidth         != incoming.maskWidth         { return "maskWidth" }
        if existing.maskHeight        != incoming.maskHeight        { return "maskHeight" }
        if existing.maskNonZero       != incoming.maskNonZero       { return "maskNonZero" }
        if existing.iouPredAtCreation != incoming.iouPredAtCreation { return "iouPredAtCreation" }
        if existing.createdAt         != incoming.createdAt         { return "createdAt" }
        return nil
    }

    /// A-15: the blob's shape and its self-check travel *with* the data, so a
    /// changed mask-space invariant is detected instead of misread.
    nonisolated private static func integrityFailure(record: PinRecordV1, alpha: [UInt8]?) -> PinIntegrityError? {
        guard let alpha = alpha else {
            return record.maskFile == nil ? nil : .maskFilePresenceMismatch
        }
        guard record.maskFile != nil else { return .maskFilePresenceMismatch }
        guard record.maskWidth == MaskPNGCodec.side,
              record.maskHeight == MaskPNGCodec.side else {
            return .maskDimensionsNotV1(width: record.maskWidth, height: record.maskHeight)
        }
        guard alpha.count == MaskPNGCodec.pixelCount else {
            return .maskSizeMismatch(expected: MaskPNGCodec.pixelCount, got: alpha.count)
        }
        let computed = MaskPNGCodec.nonZeroCount(alpha)
        guard computed == record.maskNonZero else {
            return .nonZeroCountMismatch(recorded: record.maskNonZero, computed: computed)
        }
        return nil
    }

    // MARK: - Publishing (value-type snapshots, main thread)

    /// Queue-confined.  Builds an immutable snapshot and hands it to main.
    nonisolated private func publishSnapshot(loaded: Bool) {
        let snapshot = order.compactMap { records[$0] }.map(Pin.init(record:))
        Self.onMain { [weak self] in
            guard let self = self else { return }
            self.pins = snapshot
            var index: [UUID: Pin] = [:]
            index.reserveCapacity(snapshot.count)
            for p in snapshot { index[p.id] = p }
            self.mainIndex = index
            if loaded { self.isLoaded = true }
        }
    }

    // MARK: - Completion + logging helpers

    /// Immediate (non-write) outcome: log it and hand it to main.
    ///
    /// The log call moved inside `onMain` (it used to run directly in this
    /// nonisolated body): `pinFault`/`pinLog` ultimately call `faultLog`/
    /// `perfLog` (`Shared/PerfLogging.swift`, outside B-34's scope, module-
    /// default-isolated like everything not explicitly annotated), and a
    /// *direct, synchronous* call from nonisolated code into a main-actor
    /// function is always diagnosed — unlike a closure literal handed to a
    /// GCD API, which Swift 5 mode does not check this strictly.  Logging
    /// from inside `onMain`'s already-main-actor closure sidesteps that
    /// without touching any file this ruling did not authorise changing.
    nonisolated private func finish(_ result: Result<Void, PinStoreError>,
                        op: String,
                        id: String,
                        completion: @escaping (Result<Void, PinStoreError>) -> Void)
    {
        Self.onMain { [weak self] in
            if case .failure(let err) = result {
                pinFault("[PIN] \(op) id=\(Self.short(id)) FAILED err=\(err)")
                self?.lastWriteError = err
            }
            completion(result)
        }
    }

    nonisolated private static func short(_ id: String) -> String { String(id.prefix(8)) }
}

// MARK: - [PIN] logging (B-24)

//  Both helpers route through `PerfLogging`, so every line carries the B-11
//  monotonic `[t=…]` line-start prefix.  Success and failure are distinguishable
//  *by shape*, not only by wording: a success line carries `ok` plus its
//  operation's own fields, a failure line carries `FAILED err=<case>` and no
//  result fields.  §35.9's lesson is exactly this — a branch that logs in the
//  same shape as the normal path turned a gate into an undetectable no-op.
//
//      [t=1234.5] [PIN pid=1234] create id=ab12cd34 ok blob=1873B nz=4210 pins=7
//      [t=1234.5] [PIN pid=1234] create id=ab12cd34 FAILED err=fieldTooLong(tag)
//
//  R34 (debug_report.md §39.7): every `[PIN]` line is stamped with the emitting
//  process's pid so a pasted log can distinguish "same process, two events"
//  from "two processes, interleaved output" without a device-container pull.
//  Computed once (global `let`s are lazily, thread-safely initialised) and
//  spliced in here — the single choke point both helpers already share — so
//  none of the ~16 call sites across this file (and the two other files that
//  call through these same functions) needed to change.

/// Fixed for the lifetime of the process; the system already hands this to us,
/// so there is nothing new to generate or persist.
private let pinPIDTag = "[PIN pid=\(ProcessInfo.processInfo.processIdentifier)]"

/// Rewrites a message's leading `"[PIN]"` literal into `"[PIN pid=…]"`.
/// Matches both the plain `"[PIN] …"` shape and the `"[PIN][FIXTURE] …"` /
/// `"[PIN] revisit …"` shapes used elsewhere, since all of them share the same
/// 5-character `"[PIN]"` prefix.  A message that doesn't start with it (there
/// are none today) passes through unchanged rather than being misformatted.
@inline(__always)
private func pinTagged(_ message: String) -> String {
    guard message.hasPrefix("[PIN]") else { return message }
    return pinPIDTag + message.dropFirst(5)
}

@inline(__always)
func pinLog(_ message: @autoclosure () -> String) { perfLog(pinTagged(message())) }

@inline(__always)
func pinFault(_ message: @autoclosure () -> String) { faultLog(pinTagged(message())) }
