//
//  AnnotationView.swift
//  JudgeE2
//
//  Phase 4B — Day 6 (Builder)
//
//  Full-screen editor for one Pin (tasks.md Day 6, cross-checked against
//  architect_output.md §19.2 / §19.2.6). Content: the 256×256 mask thumbnail
//  (via `PinStore.loadMaskImage`, NOT 128×128 — same correction as Day 5's
//  PinCreationSheet), a tag editor, a note editor, the modification time
//  (`updatedAt` — §19.2.2 added this field specifically for this screen), and
//  a delete button. Save writes through `PinStore.update(id:tag:note:)`;
//  delete through `PinStore.delete(id:)`. Field-length feedback reuses
//  `PinFieldLimits.length(of:)` (B-35, UTF-16 counting) rather than
//  duplicating the check.
//
//  ── 2026-08-23 — device-test bug fix (presentation layer only) ─────────────
//  Symptom: editing tag/note and saving left the "Modified" row showing the
//  pre-edit timestamp.  The write path was never at fault — `FilePinStore`
//  sets `record.updatedAt` on every `update`, flushes, and republishes
//  `@Published pins` *before* releasing the parked completion.  The defect was
//  here: this view held `let pin: Pin`, a value-type snapshot frozen at
//  presentation time, and rendered every Info row off it.  A store republish
//  cannot reach a `let` on a struct that nobody re-initialises.
//
//  Fix: identity in, values from the store.  The view now keeps only the Pin's
//  `id` as its subject and reads displayed fields through `livePin`
//  (`store.fetch(id:)`, the store's own O(1) main-thread mirror).  The
//  presentation-time snapshot is retained under an explicit name for exactly
//  two jobs — seeding the `@State` editing buffers once, and covering the few
//  frames between "delete succeeded" and "sheet dismissed" — and nothing else.
//  See `livePin` for why that fallback cannot re-introduce staleness.
//

import SwiftUI

struct AnnotationView: View {
    /// The subject of this editor.  Identity only — every *value* comes from
    /// the store (see `livePin`), so there is nothing here that can go stale.
    let pinID: UUID

    /// The Pin as it looked when the sheet was presented.
    ///
    /// ⛔ Not for display.  Its only two legitimate uses are the `@State` seed
    /// in `init` and the delete-window fallback in `livePin`; reading a
    /// displayed field off it is precisely the 2026-08-23 bug.
    private let presentedSnapshot: Pin

    @ObservedObject var store: FilePinStore
    let onDismiss: () -> Void

    @State private var tag: String
    @State private var note: String
    @State private var isSaving = false
    @State private var isDeleting = false
    @State private var errorMessage: String?
    @State private var thumbnail: UIImage?
    @State private var thumbnailLoadFailed = false
    @State private var showDeleteConfirm = false

    init(pin: Pin, store: FilePinStore, onDismiss: @escaping () -> Void) {
        self.pinID = pin.id
        self.presentedSnapshot = pin
        self.store = store
        self.onDismiss = onDismiss
        // Seeded once, then owned by the user.  ⛔ Deliberately NOT re-synced
        // from the store afterwards: a republish landing mid-typing (this
        // view's own save, or the 250 ms coalescing window closing) would
        // otherwise overwrite characters the user is still entering.
        _tag = State(initialValue: pin.tag ?? "")
        _note = State(initialValue: pin.note ?? "")
    }

    /// The record as it exists in the store *right now*.
    ///
    /// `FilePinStore.fetch(id:)` is the main-thread `mainIndex` mirror — a
    /// dictionary hit, no queue hop, no lock (§19.3.2) — and it is written in
    /// the same main-thread block as `@Published pins`, whose `objectWillChange`
    /// is what re-runs this body.  `@ObservedObject` invalidates on the whole
    /// object rather than per property, so reading through `fetch` is exactly
    /// as reactive as scanning `store.pins` would be, and O(1) instead of O(N).
    ///
    /// **Fallback semantics — deliberate and narrow.**  `fetch` returns nil
    /// only when the id is *absent from the store*.  This sheet is presented
    /// solely from a fully-loaded `PinListView`, and nothing but `delete`
    /// removes a record, so nil here means "this Pin was just deleted".  We
    /// keep painting `presentedSnapshot` for the handful of frames between the
    /// delete completion and the sheet actually going away, so the Form neither
    /// blanks out nor renders a half-empty record on its way off screen.
    ///
    /// ⛔ The fallback cannot mask the staleness bug this property exists to
    /// fix: staleness requires the record to be *present* with a newer value,
    /// and whenever it is present `fetch` returns that newer value.  The two
    /// conditions are mutually exclusive by construction.
    private var livePin: Pin { store.fetch(id: pinID) ?? presentedSnapshot }

    /// Second-granular on purpose (2026-08-23).  `timeStyle = .short` renders
    /// only to the minute, so two writes inside the same minute produce an
    /// identical string — which makes a real `updatedAt` change invisible and
    /// is, on screen, indistinguishable from the staleness bug fixed above.  A
    /// "Modified" row that cannot show that it was modified is not doing its
    /// job.  Both rows share the formatter: `Created` vs `Modified` is read as
    /// a comparison ("has this Pin ever been edited?"), and at minute
    /// granularity that comparison returns false "never edited" answers.
    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .medium   // ⛔ not .short — see above
        return f
    }()

    var body: some View {
        NavigationView {
            Form {
                Section {
                    HStack {
                        Spacer()
                        thumbnailView
                            .frame(width: 180, height: 180)
                            .background(Color.black.opacity(0.85))
                            .cornerRadius(10)
                        Spacer()
                    }
                    .listRowBackground(Color.clear)
                }

                Section("Tag") {
                    TextField("Tag (optional)", text: $tag)
                        .onChange(of: tag) { _, newValue in
                            if PinFieldLimits.length(of: newValue) > PinFieldLimits.maxTagCharacters {
                                tag = String(newValue.prefix(PinFieldLimits.maxTagCharacters))
                            }
                        }
                    Text("\(PinFieldLimits.length(of: tag)) / \(PinFieldLimits.maxTagCharacters)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Section("Note") {
                    TextEditor(text: $note)
                        .frame(minHeight: 120)
                        .onChange(of: note) { _, newValue in
                            if PinFieldLimits.length(of: newValue) > PinFieldLimits.maxNoteCharacters {
                                note = String(newValue.prefix(PinFieldLimits.maxNoteCharacters))
                            }
                        }
                    Text("\(PinFieldLimits.length(of: note)) / \(PinFieldLimits.maxNoteCharacters)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                // ⛔ Every row here reads `livePin`, never `presentedSnapshot`
                // — that is the 2026-08-23 fix.  `createdAt` / `maskNonZero`
                // are frozen at creation (PIN-1) so they cannot differ, but
                // routing them the same way keeps the rule "displayed values
                // come from the store" true of the whole section rather than
                // of one hand-picked row.
                Section("Info") {
                    let live = livePin
                    LabeledContent("Created", value: Self.dateFormatter.string(from: live.createdAt))
                    LabeledContent("Modified", value: Self.dateFormatter.string(from: live.updatedAt))
                    LabeledContent("Mask area", value: "\(live.maskNonZero) px")
                }

                if let errorMessage = errorMessage {
                    Section {
                        Text(errorMessage)
                            .foregroundColor(.red)
                            .font(.footnote)
                    }
                }

                Section {
                    Button(role: .destructive) {
                        showDeleteConfirm = true
                    } label: {
                        HStack {
                            Spacer()
                            if isDeleting { ProgressView() } else { Text("Delete Pin") }
                            Spacer()
                        }
                    }
                    .disabled(isSaving || isDeleting)
                }
            }
            .navigationTitle("Edit Pin")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { onDismiss() }
                        .disabled(isSaving || isDeleting)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button {
                        save()
                    } label: {
                        if isSaving { ProgressView() } else { Text("Save") }
                    }
                    .disabled(isSaving || isDeleting)
                }
            }
            .alert("Delete this Pin?", isPresented: $showDeleteConfirm) {
                Button("Delete", role: .destructive) { delete() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This removes the saved mask and cannot be undone.")
            }
            .onAppear(perform: loadThumbnail)
        }
    }

    @ViewBuilder
    private var thumbnailView: some View {
        if let thumbnail = thumbnail {
            Image(uiImage: thumbnail)
                .resizable()
                .interpolation(.none)   // §19.2.6 — no resampling in the preview either
                .scaledToFit()
                .padding(8)
        } else if thumbnailLoadFailed {
            Text("No preview")
                .foregroundColor(.white.opacity(0.6))
        } else {
            ProgressView()
                .tint(.white)
        }
    }

    private func loadThumbnail() {
        guard livePin.maskFile != nil else {
            thumbnailLoadFailed = true
            return
        }
        store.loadMaskImage(id: pinID) { result in
            switch result {
            case .success(let alpha):
                if let png = try? MaskPNGCodec.encode(alpha), let img = UIImage(data: png) {
                    thumbnail = img
                } else {
                    thumbnailLoadFailed = true
                }
            case .failure:
                thumbnailLoadFailed = true
            }
        }
    }

    private func save() {
        isSaving = true
        errorMessage = nil
        store.update(id: pinID, tag: tag, note: note) { result in
            isSaving = false
            switch result {
            case .success:
                onDismiss()
            case .failure(let err):
                errorMessage = "Save failed: \(err)"
            }
        }
    }

    private func delete() {
        isDeleting = true
        errorMessage = nil
        store.delete(id: pinID) { result in
            isDeleting = false
            switch result {
            case .success:
                onDismiss()
            case .failure(let err):
                errorMessage = "Delete failed: \(err)"
            }
        }
    }
}
