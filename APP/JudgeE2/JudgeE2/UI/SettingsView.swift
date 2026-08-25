//
//  SettingsView.swift
//  JudgeE2
//
//  Phase 5 — Settings, split by responsibility (architect_output.md §27.2,
//  last row: "产品化的 Settings 面板…只暴露用户需要的项；Compute Units / Encoder
//  Res AB / Force Slow Path / P-9 自检…保留但收敛到一个明确标记为「开发者/调试」
//  的分区或条件编译").
//
//  What moved here, and from where:
//
//    user-facing        —  About, version, Delete All Pins (new)
//    #if DEBUG          —  App Mode picker          (was ContentView:256-261)
//                          Compute Units picker     (was ContentView:263-268)
//                          Encoder Res A/B picker   (was ContentView:272-276)
//                          Perf Quiet Log toggle    (was ContentView:280-284)
//                          Force Slow Path toggle   (was ContentView:296-300)
//                          P-9 round-trip check     (was ContentView:339-353)
//
//  Nothing was deleted; every control keeps the binding it had, so in a Debug
//  build each one is behaviourally identical to its old self.  In Release the
//  whole Developer section is compiled out, which is what makes the acceptance
//  criterion ("Release 全程不出现模式切换或调试选项") true by construction
//  rather than by discipline.
//
//  ⛔ No frozen surface (§26.5) is touched.  The Developer section binds to
//  `CameraManager`'s existing published properties and calls its existing
//  public methods — the same call sites `ContentView` already had.
//

import SwiftUI

struct SettingsView: View {
    @ObservedObject var store: FilePinStore
    let onDismiss: () -> Void

    // Developer-section bindings.  Declared unconditionally so the caller has
    // one initialiser in both configurations; only read inside `#if DEBUG`.
    @Binding var mode: AppMode
    @Binding var backend: InferenceBackend
    @Binding var encoderRes: SAMConfiguration.EncoderResolution
    @Binding var quietLog: Bool
    @ObservedObject var cameraManager: CameraManager

    @State private var confirmDeleteAll = false
    @State private var isDeletingAll = false
    @State private var deleteAllError: String?
    #if DEBUG
    @State private var p9CheckJustRan = false
    #endif

    private var pinCount: Int { store.pins.count }

    private var versionText: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "—"
        let build = info?["CFBundleVersion"] as? String ?? "—"
        return "\(short) (\(build))"
    }

    /// B-30 (§22.1.2 point 3): a store that has gone unavailable must disable
    /// every write-triggering control, not fail silently when one is pressed.
    private var storeUnavailable: Bool {
        if case .unavailable = store.lastWriteError { return true }
        return false
    }

    var body: some View {
        NavigationView {
            Form {
                Section {
                    BrandLockup()
                        .padding(.vertical, 6)
                    Text("Tap anything in the camera to select it. Press and hold a selection to save it as a Pin, with a label and notes you can come back to.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section {
                    LabeledContent("Saved") {
                        Text("\(pinCount)")
                            .monospacedDigit()
                    }
                    Button(role: .destructive) {
                        confirmDeleteAll = true
                    } label: {
                        if isDeletingAll {
                            HStack(spacing: 8) {
                                ProgressView()
                                Text("Deleting…")
                            }
                        } else {
                            Text("Delete All Pins")
                        }
                    }
                    .disabled(pinCount == 0 || isDeletingAll || storeUnavailable)

                    if storeUnavailable {
                        Label("Pin storage is unavailable right now.",
                              systemImage: "externaldrive.badge.exclamationmark")
                            .font(.footnote)
                            .foregroundStyle(.orange)
                    }
                    if let deleteAllError {
                        Label(deleteAllError, systemImage: "exclamationmark.triangle")
                            .font(.footnote)
                            .foregroundStyle(.red)
                    }
                } header: {
                    Text("Pins")
                } footer: {
                    Text("Deleting a Pin removes its saved outline and everything you wrote about it. This cannot be undone.")
                }

                Section("About") {
                    LabeledContent("Version") { Text(versionText).monospacedDigit() }
                }

                #if DEBUG
                developerSection
                #endif
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { onDismiss() }
                }
            }
            .confirmationDialog("Delete all \(pinCount) Pins?",
                                isPresented: $confirmDeleteAll,
                                titleVisibility: .visible) {
                Button("Delete All Pins", role: .destructive) { deleteAllPins() }
                Button("Cancel", role: .cancel) { }
            } message: {
                Text("This permanently removes every saved Pin and its notes.")
            }
        }
    }

    // MARK: - Delete all

    /// Bulk delete, built strictly out of the store's existing per-record
    /// `delete(id:completion:)` — no new persistence code, no new queue, no
    /// second write path.
    ///
    /// ⚠️ R34 (§26.2, P0, root cause not located: PinStore counts have twice
    /// come back wrong after a real kill+relaunch) says any *new* bulk-Pin
    /// entry point widens that open defect's blast radius, and §27.4 asks for
    /// care here specifically.  Three things keep this one narrow: the ids are
    /// snapshotted before the first call so a concurrent publish cannot make
    /// the loop chase a moving array; every delete goes through the one
    /// audited code path, which means every record produces its own
    /// `[PIN] delete … ok pins=N` line for exactly the kind of count forensics
    /// R34 needs; and the deletes coalesce into a single manifest write, so
    /// this is one atomic rename, not N of them.
    private func deleteAllPins() {
        let ids = store.pins.map(\.id)
        guard !ids.isEmpty else { return }
        deleteAllError = nil
        isDeletingAll = true

        var remaining = ids.count
        var firstFailure: PinStoreError?
        for id in ids {
            store.delete(id: id) { result in
                // `FilePinStore` routes every completion to the main queue, so
                // these counters need no synchronisation of their own.
                if case .failure(let err) = result, firstFailure == nil {
                    firstFailure = err
                }
                remaining -= 1
                guard remaining == 0 else { return }
                isDeletingAll = false
                if let firstFailure {
                    deleteAllError = "Some Pins could not be deleted (\(firstFailure))."
                }
            }
        }
    }

    // MARK: - Developer

    #if DEBUG
    /// Everything that used to live in the always-on gear panel.  Preserved
    /// verbatim in behaviour — same bindings, same call sites, same wording on
    /// the controls whose wording was load-bearing for a test protocol.
    @ViewBuilder
    private var developerSection: some View {
        Section {
            Picker("App Mode", selection: $mode) {
                ForEach(AppMode.allCases) { item in
                    Text(item.displayName).tag(item)
                }
            }
            // The Detection and Segmentation modes are no longer product
            // features (2026-08-24 direction).  The `AppMode` enum still has
            // all three cases and `CameraManager`'s routing is untouched —
            // this picker is the only remaining way to reach the other two,
            // and it exists only in a Debug build.
            .pickerStyle(.menu)

            Picker("Compute Units", selection: $backend) {
                ForEach(InferenceBackend.allCases) { item in
                    Text(item.displayName).tag(item)
                }
            }
            .pickerStyle(.menu)

            // Phase 3 Day 4 A/B test: encoder input resolution.
            // 1024 = quality baseline (default, C-1); 768 = speed variant.
            Picker("Encoder Res", selection: $encoderRes) {
                Text("1024").tag(SAMConfiguration.EncoderResolution.res1024)
                Text("768 (AB)").tag(SAMConfiguration.EncoderResolution.res768)
            }
            .pickerStyle(.menu)

            // Perf-session log switch: keeps measurement lines, drops
            // per-frame diagnostics (see PerfLogging.swift).
            Toggle("Perf Quiet Log", isOn: $quietLog)

            // Force slow path: clears the embedding cache and suspends
            // background refresh, so every tap takes the full encode+decode
            // path — used to collect slow-path latency samples.
            Toggle("Force Slow Path (testing)", isOn: $cameraManager.forceSlowPath)
                .tint(.orange)

            // P-9 self-check (§23.2.8): one-shot canonical↔view round-trip
            // measurement over a 9-point grid, against whatever geometry is
            // live right now.  Results go to the console as `[P9]` lines only.
            // Press once per configuration; P-9 needs all four
            // (portrait/landscape × back/front).
            Button {
                cameraManager.runCanonicalRoundTripSelfCheck()
                p9CheckJustRan = true
            } label: {
                Text(p9CheckJustRan ? "✓ Ran — check console"
                                    : "Run Geometry Round-Trip Check")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .foregroundStyle(.orange)
            .task(id: p9CheckJustRan) {
                guard p9CheckJustRan else { return }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard !Task.isCancelled else { return }
                p9CheckJustRan = false
            }
        } header: {
            Label("Developer", systemImage: "wrench.and.screwdriver")
        } footer: {
            Text("Debug builds only. These controls are engineering instrumentation and are compiled out of Release.")
        }
    }
    #endif
}
