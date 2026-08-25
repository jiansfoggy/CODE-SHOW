//
//  PinListView.swift
//  JudgeE2
//
//  Phase 4B — Day 6 (Builder)
//
//  Lists every saved Pin (thumbnail + tag + time), sortable by time and
//  filterable by tag. Per architect_output.md §19.1.2, sort/filter are
//  trivial operations on the already-in-memory `@Published pins` array
//  (sort ≈0.1 ms, filter microsecond-scale on N ≤ 1,000) — no new query
//  infrastructure.
//
//  UX split (Day 6 interpretation, §19.4 does not dictate this): tapping a
//  row's body revisits the Pin (§19.4 G1–G4 → R-A/R-B/R-C, via
//  `CameraManager.handleTap(fromPin:store:)`) and dismisses this list so the
//  camera view is visible; a separate trailing "Edit" (pencil) button opens
//  AnnotationView without leaving the list. This keeps the common action
//  (look at it again) one tap away while the editing action stays a
//  deliberate, visually distinct affordance.
//

import SwiftUI

struct PinListView: View {
    @ObservedObject var store: FilePinStore
    let onDismiss: () -> Void
    /// Called after a revisit is *issued* (accepted or rejected — the caller
    /// dismisses either way; outcome banners live in ContentView, driven by
    /// `cameraManager.pinRevisitEvent`).
    let onRevisit: (Pin) -> Void

    private enum SortOrder: String, CaseIterable, Identifiable {
        case newestFirst = "Newest"
        case oldestFirst = "Oldest"
        var id: String { rawValue }
    }

    @State private var sortOrder: SortOrder = .newestFirst
    @State private var tagFilter: String = ""

    /// Which Pin's editor is presented, plus the snapshot `AnnotationView` uses
    /// to seed its editing buffers.
    ///
    /// ⛔ This value is frozen the moment the swipe action fires and is never
    /// refreshed while the sheet is up — nothing writes to it but the swipe
    /// action and `onDismiss`.  That is fine *because* `AnnotationView` now
    /// displays `store.fetch(id:)` rather than what it was handed
    /// (2026-08-23 fix); it would not be fine again if any displayed value
    /// were ever read back off this property.  Freshness on reopen comes from
    /// `filteredPins`, which recomputes from the live `store.pins`.
    @State private var editingPin: Pin?

    /// Mask blobs are immutable (PIN-2), so a cached thumbnail cannot go stale
    /// for a Pin that still exists — no invalidation needed on edit.
    @State private var thumbnails: [UUID: UIImage] = [:]

    private var availableTags: [String] {
        let tags = Set(store.pins.compactMap { $0.tag }.filter { !$0.isEmpty })
        return tags.sorted()
    }

    private var filteredPins: [Pin] {
        var pins = store.pins
        if !tagFilter.isEmpty {
            pins = pins.filter { ($0.tag ?? "").localizedCaseInsensitiveContains(tagFilter) }
        }
        switch sortOrder {
        case .newestFirst: return pins.sorted { $0.createdAt > $1.createdAt }
        case .oldestFirst: return pins.sorted { $0.createdAt < $1.createdAt }
        }
    }

    var body: some View {
        NavigationView {
            Group {
                // §19.3.2 — while not loaded, show a loading state, never
                // "no pins yet"; the two must not be visually interchangeable.
                if !store.isLoaded {
                    ProgressView("Loading Pins…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if store.pins.isEmpty {
                    ContentUnavailableViewCompat(
                        title: "No Pins yet",
                        message: "Tap something in the camera to select it, then press and hold the selection to save it here.")
                } else {
                    List {
                        if !availableTags.isEmpty {
                            Section {
                                Picker("Filter by tag", selection: $tagFilter) {
                                    Text("All tags").tag("")
                                    ForEach(availableTags, id: \.self) { t in
                                        Text(t).tag(t)
                                    }
                                }
                            }
                        }
                        Section {
                            ForEach(filteredPins) { pin in
                                PinRow(pin: pin, thumbnail: thumbnails[pin.id])
                                    .contentShape(Rectangle())
                                    .onTapGesture { revisit(pin) }
                                    .swipeActions(edge: .trailing) {
                                        Button {
                                            editingPin = pin
                                        } label: {
                                            Label("Edit", systemImage: "pencil")
                                        }
                                        .tint(.blue)
                                    }
                                    .onAppear { loadThumbnail(for: pin) }
                            }
                        }
                    }
                    // Phase 5 — the long-press → sheet → list handoff ends on
                    // this list.  Animating on the row count (not on `pins`,
                    // which republishes on every edit too) makes a new Pin
                    // slide in instead of blinking into place, and a delete
                    // close the gap it left.
                    .animation(.spring(response: 0.32, dampingFraction: 0.85),
                               value: filteredPins.count)
                }
            }
            .navigationTitle("Pins")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") { onDismiss() }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Picker("Sort", selection: $sortOrder) {
                        ForEach(SortOrder.allCases) { order in
                            Text(order.rawValue).tag(order)
                        }
                    }
                    .pickerStyle(.menu)
                }
            }
            .sheet(item: $editingPin) { pin in
                AnnotationView(pin: pin, store: store) { editingPin = nil }
            }
        }
    }

    private func revisit(_ pin: Pin) {
        onRevisit(pin)
        onDismiss()
    }

    private func loadThumbnail(for pin: Pin) {
        guard thumbnails[pin.id] == nil, pin.maskFile != nil else { return }
        store.loadMaskImage(id: pin.id) { result in
            guard case .success(let alpha) = result,
                  let png = try? MaskPNGCodec.encode(alpha),
                  let img = UIImage(data: png) else { return }
            thumbnails[pin.id] = img
        }
    }
}

private struct PinRow: View {
    let pin: Pin
    let thumbnail: UIImage?

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .short
        f.timeStyle = .short
        return f
    }()

    var body: some View {
        HStack(spacing: 12) {
            Group {
                if let thumbnail = thumbnail {
                    Image(uiImage: thumbnail)
                        .resizable()
                        .interpolation(.none)
                        .scaledToFit()
                } else {
                    ProgressView()
                }
            }
            .frame(width: 44, height: 44)
            .background(Color.black.opacity(0.85))
            .cornerRadius(6)

            VStack(alignment: .leading, spacing: 2) {
                Text((pin.tag?.isEmpty == false) ? pin.tag! : "Untitled Pin")
                    .font(.body.weight(.medium))
                // Labelled "Created" on purpose (2026-08-23).  The row shows
                // `createdAt` — a PIN-1 frozen field that never changes — and
                // the list sorts by it too (`sortedFiltered`).  An unlabelled
                // timestamp here reads as "last modified" and made an edit
                // look like it had not been saved, when in fact `updatedAt`
                // had advanced correctly on disk.  `updatedAt` stays in
                // AnnotationView's "Modified" row; showing it here as well
                // would disagree with the Newest/Oldest ordering.
                Text("Created \(Self.dateFormatter.string(from: pin.createdAt))")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Spacer()
            if pin.isDegraded {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundColor(.orange)
                    .help("Mask blob is degraded — metadata only")
            }
        }
        .padding(.vertical, 2)
    }
}

/// `ContentUnavailableView` requires iOS 17; the deployment target is 17.0
/// per §19.5.5's Application Support note, so this thin wrapper exists only
/// to keep the call site readable — not a compatibility shim.
private struct ContentUnavailableViewCompat: View {
    let title: String
    let message: String

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "mappin.slash")
                .font(.system(size: 40))
                .foregroundColor(.secondary)
            Text(title)
                .font(.headline)
            Text(message)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
