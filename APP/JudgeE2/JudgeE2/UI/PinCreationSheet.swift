//
//  PinCreationSheet.swift
//  JudgeE2
//
//  Phase 4B — Day 5 (Builder)
//
//  Bottom half-sheet for turning a live tap instance into a persisted Pin.
//  Contract: architect_output.md §22.2 (the long-press decision tree that
//  opens this sheet), §22.2.3 (PIN-6 — single locked snapshot), §22.2.5
//  (B-36 — the static R32-mitigation hint), and tasks.md Day 5 ("mask 缩略图 +
//  标签输入框 + 保存/取消按钮").
//
//  ⛔ Thumbnail is 256×256 lossless via `MaskPNGCodec` (§19/§22 — NOT the
//  128×128 tasks.md draft; see `MaskPNGCodec.swift`'s header for why 128
//  corresponds to no invariant in this system).  SwiftUI displays it scaled
//  down; no separate downscaled asset is ever produced or persisted.
//

import SwiftUI

/// PIN-6 (§22.2.3): `draft.instance` is the single snapshot `CameraManager`
/// took, under `TapInstanceManager`'s existing lock, at long-press time. This
/// view must read it exactly once (captured in `draft` at construction) and
/// must NOT ask `TapInstanceManager` again before Save — the thumbnail shown
/// here and the bytes hitting `PinStore.create` are required to be the same
/// bytes.
struct PinCreationSheet: View {
    let draft: CameraManager.PinCreationDraft
    @ObservedObject var store: FilePinStore

    /// Invoked once, on success, so the caller can flip the on-screen anchor
    /// marker to its 📌 style and attach its label
    /// (`CameraManager.markInstancePinned(id:tag:)`). Deliberately a plain
    /// closure rather than this view reaching into CameraManager directly
    /// with a Persistence-flavoured API — keeps the dependency direction "UI
    /// calls PinStore", not "CameraManager knows about PinStore" (PIN-3's
    /// whole-file invariant on CameraManager.swift). Day 6: also carries the
    /// saved tag, so the on-screen marker can show it beside the pin icon.
    let onSaved: (UUID, String?) -> Void
    let onDismiss: () -> Void

    @State private var tag: String = ""
    @State private var isSaving = false
    @State private var saveErrorMessage: String?

    private var thumbnailImage: UIImage? {
        guard let alpha = draft.instance.maskAlpha,
              let png = try? MaskPNGCodec.encode(alpha) else { return nil }
        return UIImage(data: png)
    }

    /// B-30 (§22.1.2 point 3): the persistent "storage unavailable" state must
    /// disable every write-triggering control here, not just fail silently
    /// when Save is pressed.
    private var storeUnavailable: Bool {
        if case .unavailable = store.lastWriteError { return true }
        return false
    }

    var body: some View {
        VStack(spacing: 16) {
            Capsule()
                .fill(Color.secondary.opacity(0.4))
                .frame(width: 36, height: 5)
                .padding(.top, 8)

            if storeUnavailable {
                Label("Storage unavailable — Pin cannot be saved right now", systemImage: "exclamationmark.triangle.fill")
                    .font(.footnote)
                    .foregroundColor(.red)
                    .padding(8)
                    .frame(maxWidth: .infinity)
                    .background(Color.red.opacity(0.12))
                    .cornerRadius(8)
            }

            Group {
                if let thumbnailImage = thumbnailImage {
                    Image(uiImage: thumbnailImage)
                        .resizable()
                        .interpolation(.none)      // §19.2.6 — no resampling in the preview either
                        .scaledToFit()
                } else {
                    Color.black.opacity(0.6)
                        .overlay(Text("No preview").foregroundColor(.white.opacity(0.6)))
                }
            }
            .frame(width: 140, height: 140)
            .background(Color.black.opacity(0.85))
            .cornerRadius(10)

            TextField("Tag (optional)", text: $tag)
                .textFieldStyle(.roundedBorder)
                .onChange(of: tag) { _, newValue in
                    // B-35 (§22.1.8): same counting function the store enforces
                    // — `PinFieldLimits.length(of:)`, UTF-16 code units — so the
                    // UI can never allow what the store would reject.
                    if PinFieldLimits.length(of: newValue) > PinFieldLimits.maxTagCharacters {
                        tag = String(newValue.prefix(PinFieldLimits.maxTagCharacters))
                    }
                }

            // B-36 (§22.2.5) — unconditional, static hint mitigating R32
            // (the frozen-mask recovery path being real but undiscoverable).
            // Deliberately NOT conditioned on any frozen-state signal: doing
            // so would require reading `originAlpha` / re-anchor-derived
            // state, which §18.3.3 hard isolation 1 forbids this subsystem
            // from touching. Always shown, worded as generic guidance rather
            // than a diagnostic claim.
            Text("If the mask no longer matches the object on screen, tap elsewhere on it to re-select")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 8)

            if let saveErrorMessage = saveErrorMessage {
                Text(saveErrorMessage)
                    .font(.caption)
                    .foregroundColor(.red)
            }

            HStack(spacing: 16) {
                Button("Cancel") { onDismiss() }
                    .buttonStyle(.bordered)
                    .disabled(isSaving)

                Button {
                    save()
                } label: {
                    HStack(spacing: 6) {
                        if isSaving { ProgressView().tint(.white) }
                        Text("Save")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isSaving || storeUnavailable)
            }
            .padding(.bottom, 12)
        }
        .padding(.horizontal, 24)
        .padding(.top, 4)
        .presentationDetents([.height(360), .medium])
    }

    private func save() {
        // PIN-6: both calls below read `draft.instance` — the one snapshot
        // this sheet was constructed with — never `TapInstanceManager` again.
        guard let record = PinFactory.makeRecord(from: draft.instance,
                                                 geometry: draft.geometry,
                                                 tag: tag.isEmpty ? nil : tag,
                                                 note: nil) else {
            saveErrorMessage = "Could not build a Pin record (mask data incomplete)"
            return
        }
        let alpha = PinFactory.maskAlpha(from: draft.instance)

        isSaving = true
        saveErrorMessage = nil
        let savedInstanceID = draft.instance.id
        let savedTag = record.tag
        store.create(record, maskAlpha: alpha) { result in
            isSaving = false
            switch result {
            case .success:
                onSaved(savedInstanceID, savedTag)
                onDismiss()
            case .failure(let err):
                saveErrorMessage = "Save failed: \(err)"
            }
        }
    }
}
