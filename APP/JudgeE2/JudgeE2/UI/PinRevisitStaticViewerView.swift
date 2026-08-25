//
//  PinRevisitStaticViewerView.swift
//  JudgeE2
//
//  Phase 4B — Day 6 (Builder)
//
//  The static fallback view for the two revisit outcomes that do not produce
//  a live mask (architect_output.md §19.4.4):
//
//    R-B (revisit refused, G1 failed) — shows `masks/<id>.png` plus the
//        explicit reason text; must never silently degrade to an unexplained
//        image.
//    R-C (revisit failed — timeout / no camera frame / empty decode) — the
//        existing Requirement C failure banner already covers the "why"; this
//        view is the "static fallback entry" §19.4.4 additionally requires.
//
//  PIN-2's third clause / §19.4.4: the static fallback must be visually
//  distinguishable from an active mask, not composited to look the same.
//  Active masks render in the cyan/aqua/spring-cyan palette (§3.3.1) with a
//  live camera feed behind them; this view renders the stored mask
//  desaturated, on a solid background, with an explicit "STATIC SNAPSHOT"
//  badge — nothing here could be mistaken for a live composite.
//

import SwiftUI

struct PinRevisitStaticViewerView: View {
    let pin: Pin
    @ObservedObject var store: FilePinStore
    /// nil for a plain "open the snapshot" request (e.g. tapped straight from
    /// PinList); non-nil for R-B/R-C, carrying the reason/failure text.
    let reason: String?
    let onDismiss: () -> Void

    @State private var image: UIImage?
    @State private var loadFailed = false

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                Label("STATIC SNAPSHOT — not live", systemImage: "photo.badge.exclamationmark")
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.orange)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Color.orange.opacity(0.15))
                    .cornerRadius(8)

                Group {
                    if let image = image {
                        Image(uiImage: image)
                            .resizable()
                            .interpolation(.none)
                            .scaledToFit()
                            // Desaturated + dashed border: deliberately distinct
                            // from the live cyan/aqua/spring-cyan mask palette
                            // (§3.3.1) so this can never read as an active mask.
                            .saturation(0)
                    } else if loadFailed {
                        VStack(spacing: 8) {
                            Image(systemName: "photo.slash")
                                .font(.system(size: 32))
                                .foregroundColor(.secondary)
                            Text("No snapshot available for this Pin")
                                .font(.footnote)
                                .foregroundColor(.secondary)
                        }
                    } else {
                        ProgressView()
                    }
                }
                .frame(maxWidth: .infinity)
                .frame(height: 260)
                .background(Color.gray.opacity(0.25))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                        .foregroundColor(.orange.opacity(0.6))
                )
                .cornerRadius(10)
                .padding(.horizontal)

                if let reason = reason {
                    Text(reason)
                        .font(.subheadline)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 24)
                }

                if let tag = pin.tag, !tag.isEmpty {
                    Text(tag)
                        .font(.headline)
                }
                if let note = pin.note, !note.isEmpty {
                    Text(note)
                        .font(.footnote)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 24)
                }

                Spacer()
            }
            .padding(.top, 12)
            .navigationTitle("Saved Snapshot")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Close") { onDismiss() }
                }
            }
            .onAppear(perform: loadImage)
        }
    }

    private func loadImage() {
        guard pin.maskFile != nil else {
            loadFailed = true
            return
        }
        store.loadMaskImage(id: pin.id) { result in
            switch result {
            case .success(let alpha):
                if let png = try? MaskPNGCodec.encode(alpha), let img = UIImage(data: png) {
                    image = img
                } else {
                    loadFailed = true
                }
            case .failure:
                loadFailed = true
            }
        }
    }
}
