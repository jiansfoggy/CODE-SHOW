import SwiftUI
import UIKit
import QuartzCore
import Combine

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var fpsMonitor = FPSMonitor()
    // Phase 4B Day 5 — single app-wide PinStore instance, owned by JudgeE2App.
    @EnvironmentObject private var pinStore: FilePinStore
    @State private var backend: InferenceBackend = .all
    // Phase 5 — the app has one product mode.  `.tapToSegment` is no longer a
    // choice the user makes: it is where the app starts and, in a Release
    // build, the only value this ever holds.  The other two `AppMode` cases
    // and every downstream branch in `CameraManager` are untouched (§26.5) —
    // only the UI that could *reach* them moved, into `SettingsView`'s
    // `#if DEBUG` Developer section.
    @State private var mode: AppMode = .tapToSegment
    // Phase 5 — the product Settings sheet (replaces the old always-on gear
    // panel; see `SettingsView` for what moved where).
    @State private var showSettings: Bool = false
    // Phase 3 Day 4 — encoder resolution AB test (default 1024 per Architect C-1).
    @State private var encoderRes: SAMConfiguration.EncoderResolution = .res1024
    // Phase 3 Day 4 — performance-session log switch (default off = current
    // behaviour).  On-screen so a perf run can be started on an installed
    // device build without a rebuild.
    @State private var quietLog: Bool = PerfLogging.quietMode
    // Day 6 — tap ripple: set to the viewPoint on each new accepted tap,
    // then cleared after the ripple animation completes.
    @State private var ripplePoint: CGPoint? = nil
    @State private var rippleTrigger: Int = 0

    // Phase 4B Day 6 — Pin List entry point + the sheets it can open.
    @State private var showPinList = false
    @State private var staticViewerPin: Pin?
    @State private var staticViewerReason: String?

    // User-instructed UI change — "tap a pin shows its label".
    //
    // Pure presentation state: which pinned marker is currently showing its
    // tag, and a monotonic token used to time the auto-hide.  Nothing here
    // feeds the tap pipeline; it is *driven by* state the pipeline already
    // publishes (`tapAnchorMarkers[].isPrimary` and `lastTapIndex`), so the
    // §3.2 / §22.2.2 tap contract — promote-to-primary, no re-decode — is
    // untouched.  See `revealTag(for:)` for the reveal policy.
    @State private var revealedTagMarkerID: UUID? = nil
    @State private var tagRevealToken: Int = 0

    // Phase 5 — first-run coach mark for the long-press → Pin gesture.
    //
    // Tapping is discoverable (the whole preview is the target and a ripple
    // answers immediately); pressing and holding is not, and it is the only
    // way anything ever reaches the Pin list.  Shown once, the first time a
    // selection actually exists to press on — a hint about a gesture the user
    // cannot yet perform is noise — then never again.  `@AppStorage` because
    // "once" has to survive relaunch; it feeds nothing but this banner.
    @AppStorage("ui.hasSeenSaveHint") private var hasSeenSaveHint: Bool = false
    @State private var showSaveHint: Bool = false

    // MARK: - B-39 — "Pin List" count badge (§23.1.8)
    //
    // Phase 5 moved the badge itself onto `BottomControlBar`; the reasoning
    // below is why it is a count on an entry point at all, and still governs.
    //
    // WHY A BADGE AND NOT MARKERS ON THE PREVIEW.
    // After a relaunch the camera view carries no 📌 at all.  That is correct
    // by design — §22.2.4 makes the marker a decoration on a `TapInstance`,
    // not on the stored record, so an empty instance pool means no markers —
    // but a user read that absence as data loss.  §23.1.8 rules that the fix
    // is *entry-point visibility*, and ⛔ explicitly forbids the alternative
    // of overlaying every saved Pin on the preview: a stored `canonicalPoint`
    // means nothing against the current frame until G1 has passed, so drawing
    // it would be a geometric claim the app cannot back.  A count depends only
    // on how many records are on disk (PIN-7 ①), which is always true.

    /// Records on disk.  `FilePinStore.pins` is `@Published`, so this recomputes
    /// and redraws on its own — no timer, no notification, no cached mirror.
    private var pinCount: Int { pinStore.pins.count }

    /// True while at least one live selection is on screen — drives the
    /// bottom bar's Clear button.
    private var hasActiveSelection: Bool { !cameraManager.tapAnchorMarkers.isEmpty }

    var body: some View {
        ZStack(alignment: .topLeading) {
            CameraPreview(session: cameraManager.session,
                          boxes: cameraManager.boxes,
                          maskImage: cameraManager.maskImage,
                          maskOutlines: cameraManager.maskOutlines,
                          maskRotationAngle: cameraManager.maskRotationAngle,
                          maskMirrored: cameraManager.maskMirrored,
                          cameraManager: cameraManager)
                .overlay {
                    // Overlay coordinates match CameraPreview's bounds, which is
                    // the same view space TouchHandler reported the tap in.
                    //
                    // Phase 3 Day 5 — Requirement C: the failure marker takes
                    // precedence over the pulse.  Architect §10.4 C forbids
                    // clearing the indicator with nothing in its place; a tap
                    // that produced no mask must say why, briefly, right where
                    // the user tapped.
                    ZStack {
                        if let failure = cameraManager.tapFailure {
                            TapFailureIndicator(message: failure.message)
                                .position(failure.viewPoint ?? cameraManager.lastTapViewPoint
                                          ?? CGPoint(x: 200, y: 200))
                        } else if cameraManager.tapProcessing,
                                  let p = cameraManager.lastTapViewPoint {
                            TapLoadingIndicator()
                                .position(p)
                        }
                    }
                    .allowsHitTesting(false)   // must not steal taps from TouchHandler
                }
                .ignoresSafeArea()

            // Cold-start banner (Requirement C): while the SAM encoder warms up
            // the user gets an explicit "initialising" state instead of tapping
            // into an invisible 1.3–8.6 s wait.
            if mode == .tapToSegment && cameraManager.samWarmingUp {
                VStack {
                    HStack {
                        Spacer()
                        Label("Preparing segmentation…", systemImage: "hourglass")
                            .font(.footnote)
                            .foregroundColor(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 7)
                            .background(.black.opacity(0.65))
                            .cornerRadius(8)
                        Spacer()
                    }
                    .padding(.top, 70)
                    Spacer()
                }
                .allowsHitTesting(false)
            }

            // Day 6 — Tap anchor markers and ripple overlay.
            // Rendered above everything; allowsHitTesting false so TouchHandler
            // still receives all taps.
            if mode == .tapToSegment {
                ZStack {
                    // Anchor markers: numbered discs (1/2/3), primary drawn heavier.
                    // Phase 4B Day 5 — a saved instance switches to 📌 (§22.2.4:
                    // decorates the instance, not the persisted record).
                    ForEach(cameraManager.tapAnchorMarkers) { marker in
                        TapAnchorMarkerView(slotIndex: marker.slotIndex,
                                            isPrimary: marker.isPrimary,
                                            isPinned: marker.isPinned,
                                            tag: marker.tag,
                                            showsTag: marker.id == revealedTagMarkerID,
                                            isRevisitOrigin: marker.isRevisitOrigin,
                                            revisitPinTag: marker.revisitPinTag)
                            .position(marker.viewPoint)
                    }
                    // One-shot ripple at last accepted tap point.
                    if let rp = ripplePoint {
                        TapRippleEffect(trigger: rippleTrigger)
                            .position(rp)
                    }
                }
                .ignoresSafeArea()
                .allowsHitTesting(false)
            }

            // Day 6 — "Tap to segment" hint text after double-tap clear.
            if cameraManager.showSegmentHint {
                VStack {
                    Spacer()
                    Text("Tap to segment")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 10)
                        .background(.black.opacity(0.72))
                        .cornerRadius(10)
                    Spacer()
                        .frame(height: 120)
                }
                .frame(maxWidth: .infinity)
                .transition(.opacity)
                .allowsHitTesting(false)
            }

            // Phase 5 — first-run coach mark for the save gesture.  Appears
            // once, the first time a selection exists to press on, and never
            // competes with the post-clear "Tap to segment" hint.
            if showSaveHint {
                VStack {
                    Spacer()
                    Label("Press and hold a selection to save it as a Pin",
                          systemImage: "hand.point.up.left")
                        .font(.footnote.weight(.semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 9)
                        .background(.black.opacity(0.78))
                        .cornerRadius(10)
                        .padding(.bottom, 92)
                }
                .frame(maxWidth: .infinity)
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .allowsHitTesting(false)
            }

            // Phase 4B Day 5 (B-30, §22.1.2 point 3) — persistent banner the
            // moment the store becomes unavailable, not only after a failed
            // save.  Sits above the bottom control bar so it never blocks the
            // preview or steals taps.
            if case .unavailable = pinStore.lastWriteError {
                VStack {
                    Spacer()
                    Label("Pin storage unavailable — new Pins cannot be saved", systemImage: "externaldrive.badge.exclamationmark")
                        .font(.footnote.weight(.semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(Color.red.opacity(0.85))
                        .cornerRadius(10)
                        .padding(.bottom, 90)
                }
                .allowsHitTesting(false)
            }

            // Debug instrumentation column (debug builds only), stacked
            // bottom-trailing above the control bar.
            //
            // `#N` burns the current tap index into a screen recording so each
            // frame maps to its `[TAP#N]` log chain without timestamp
            // guessing; `FPS` is the main-thread CADisplayLink cadence, so UI
            // frame drops show up immediately during tap profiling.  Both are
            // measurement scaffolding, not product UI — Phase 5 moved them
            // behind `#if DEBUG` with the rest of the instrumentation, so a
            // Release build's preview carries nothing but the camera, the
            // selections and the bar.
            #if DEBUG
            VStack {
                Spacer()
                HStack {
                    Spacer()
                    VStack(alignment: .trailing, spacing: 6) {
                        if mode == .tapToSegment {
                            Text("#\(cameraManager.lastTapIndex)")
                                .font(.system(size: 26, weight: .bold, design: .monospaced))
                                .foregroundColor(.white)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 5)
                                .background(.black.opacity(0.6))
                                .cornerRadius(8)
                        }
                        Text("FPS: \(fpsMonitor.fps)")
                            .font(.system(size: 13, design: .monospaced))
                            .foregroundColor(
                                fpsMonitor.fps >= 50 ? .white :
                                fpsMonitor.fps >= 30 ? .yellow : .red
                            )
                            .padding(6)
                            .background(.black.opacity(0.6))
                            .cornerRadius(6)
                    }
                }
            }
            .padding(.trailing, 14)
            .padding(.bottom, 88)
            .allowsHitTesting(false)
            #endif

            // Phase 5 — the app's one persistent control surface (D-27.1).
            // Last child of the ZStack so it draws above the overlays; it is
            // the only region of the screen that deliberately does NOT reach
            // TouchHandler, and it sizes to its content, so every tap outside
            // the capsule still segments.
            VStack {
                Spacer()
                BottomControlBar(
                    mode: mode,
                    pinCount: pinCount,
                    hasActiveSelection: hasActiveSelection,
                    onOpenPinList: { showPinList = true },
                    onClearSelections: {
                        // Same entry point the double-tap gesture uses
                        // (§3.2), with the same haptic TouchHandler fires —
                        // a button press and a double-tap must not feel like
                        // two different features.
                        let haptic = UIImpactFeedbackGenerator(style: .heavy)
                        haptic.prepare()
                        haptic.impactOccurred()
                        cameraManager.handleClearAllTapMasks()
                    },
                    onFlipCamera: { cameraManager.toggleCamera() },
                    onOpenSettings: { showSettings = true }
                )
                .padding(.horizontal, 18)
                .padding(.bottom, 6)
            }
            .frame(maxWidth: .infinity)
        }
        .animation(.easeInOut(duration: 0.2), value: cameraManager.showSegmentHint)
        .animation(.easeInOut(duration: 0.25), value: showSaveHint)
        .onAppear {
            cameraManager.start()
            cameraManager.setBackend(backend)
            cameraManager.setMode(mode)
        }
        .onChange(of: backend) { _, newValue in
            cameraManager.setBackend(newValue)
        }
        .onChange(of: mode) { _, newValue in
            cameraManager.setMode(newValue)
        }
        .onChange(of: encoderRes) { _, newValue in
            cameraManager.setEncoderResolution(newValue)
        }
        .onChange(of: quietLog) { _, newValue in
            // Deliberately NOT routed through CameraManager: unlike backend /
            // mode / resolution this has no pipeline side effect (no cache to
            // drop, no model to rebuild), so a sessionQueue hop would only
            // delay the switch taking effect.  See PerfLogging.quietMode for
            // the threading rationale.
            PerfLogging.quietMode = newValue
        }
        // Day 6 — ripple: fire when a new tap is accepted (lastTapIndex increments).
        // lastTapViewPoint is the view-space position of the most recent accepted tap.
        .onChange(of: cameraManager.lastTapIndex) { _, _ in
            if let vp = cameraManager.lastTapViewPoint {
                ripplePoint = vp
                rippleTrigger &+= 1
            }
            // Re-tapping the pin that is *already* primary changes no marker
            // state at all, so `primaryMarkerID` below would not fire.  This
            // hook covers that case (and re-arms the timer on a repeat tap).
            revealTag(for: primaryMarkerID)
        }
        // A tap that lands inside a *different* instance's mask promotes it;
        // CameraManager republishes `tapAnchorMarkers` with the new primary a
        // few milliseconds later, which is the moment we know which label to
        // show.  Reveal is single-valued, so the previously revealed pin's
        // label hides as this one appears.
        .onChange(of: primaryMarkerID) { _, newValue in
            revealTag(for: newValue)
        }
        // B-37 (§23.1.4 展示时机) — the provenance label auto-shows ONCE, at
        // the moment the revisit product's mask first appears.  That is
        // exactly when the user used to conclude "the pin is gone"; showing
        // the `From Pin "…"` connection right then is the whole point of the
        // decoration.  Afterwards it follows the same tap-to-reveal policy as
        // the pinned tag.
        // Phase 5 — arm the save coach mark the first time a selection
        // exists to press on.  Watching `hasActiveSelection` (a Bool) rather
        // than the marker array keeps this to one comparison per publish.
        .onChange(of: hasActiveSelection) { _, active in
            guard active, !hasSeenSaveHint, !showSaveHint else { return }
            showSaveHint = true
        }
        // Retire it the moment the gesture is actually performed — the hint
        // has done its job and holding it on screen would only argue with the
        // sheet that just opened.
        .onChange(of: cameraManager.pinCreationDraft != nil) { _, opened in
            guard opened else { return }
            hasSeenSaveHint = true
            showSaveHint = false
        }
        .task(id: showSaveHint) {
            guard showSaveHint else { return }
            try? await Task.sleep(nanoseconds: 4_500_000_000)
            guard !Task.isCancelled else { return }
            hasSeenSaveHint = true
            showSaveHint = false
        }
        .onChange(of: revisitOriginMarkerIDs) { oldIDs, newIDs in
            if let newborn = newIDs.first(where: { !oldIDs.contains($0) }) {
                revealTag(for: newborn)
            }
        }
        // Auto-hide.  `.task(id:)` cancels and restarts whenever the token is
        // bumped, so a fresh reveal always gets a full window — no timer
        // bookkeeping, no queue, no lock.
        .task(id: tagRevealToken) {
            guard revealedTagMarkerID != nil else { return }
            try? await Task.sleep(nanoseconds: 2_600_000_000)
            guard !Task.isCancelled else { return }
            withAnimation(.easeIn(duration: 0.25)) { revealedTagMarkerID = nil }
        }
        .onDisappear {
            cameraManager.stop()
        }
        // Phase 4B Day 5 (§22.2.2) — long-press-on-existing-mask opens this.
        // `item:` binding: presented iff `pinCreationDraft != nil`, dismissed
        // by setting it back to nil.
        .sheet(item: $cameraManager.pinCreationDraft) { draft in
            PinCreationSheet(
                draft: draft,
                store: pinStore,
                onSaved: { instanceID, tag in
                    cameraManager.markInstancePinned(id: instanceID, tag: tag)
                    // Save confirmation: the label the user just typed appears
                    // once beside the marker it now belongs to, then fades —
                    // the same reveal a later tap gives it.  A long-press does
                    // not promote (§22.2.2), so this is the one reveal that is
                    // not driven by a primary change.
                    revealTag(for: instanceID)
                },
                onDismiss: {
                    cameraManager.pinCreationDraft = nil
                }
            )
        }
        // Phase 5 — product Settings.  Same `.sheet()` presentation the Pin
        // surfaces already use (§27.2's "复用现有 .sheet() 呈现模式").  The
        // Developer bindings are handed over unconditionally; `SettingsView`
        // compiles the section that reads them out of Release.
        .sheet(isPresented: $showSettings) {
            SettingsView(store: pinStore,
                         onDismiss: { showSettings = false },
                         mode: $mode,
                         backend: $backend,
                         encoderRes: $encoderRes,
                         quietLog: $quietLog,
                         cameraManager: cameraManager)
        }
        // Phase 4B Day 6 — Pin List entry point.
        .sheet(isPresented: $showPinList) {
            PinListView(
                store: pinStore,
                onDismiss: { showPinList = false },
                onRevisit: { pin in
                    cameraManager.handleTap(fromPin: pin, store: pinStore)
                }
            )
        }
        // Phase 4B Day 6 (§19.4.4 R-B / R-C) — the static fallback viewer.
        .sheet(item: $staticViewerPin) { pin in
            PinRevisitStaticViewerView(
                pin: pin,
                store: pinStore,
                reason: staticViewerReason,
                onDismiss: {
                    staticViewerPin = nil
                    staticViewerReason = nil
                }
            )
        }
        // Phase 4B Day 6 (§19.4.4) — revisit outcome banner. R-A needs no
        // banner (an ordinary active instance appears, exactly as an
        // ordinary tap would); R-B/R-C both surface here with an explicit
        // reason and an entry into the static viewer.
        .overlay(alignment: .bottom) {
            if let event = cameraManager.pinRevisitEvent {
                PinRevisitBanner(event: event) {
                    staticViewerReason = revisitReasonText(for: event)
                    staticViewerPin = pinStore.fetch(id: event.pinID)
                } onDismiss: {
                    cameraManager.pinRevisitEvent = nil
                }
                .padding(.bottom, 90)
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.2), value: cameraManager.pinRevisitEvent)
        // Phase 4B Day 6 (§19.4.4 R-A, wording per §23.1.4's permitted list) —
        // in-flight revisit copy, reusing the existing tapProcessing pulse
        // rather than introducing a second waiting UI.  Never shown on the
        // promote outcome R-D: no decode runs there, `lastTapWasRevisit`
        // stays false (B-40), and this copy would be a decidable falsehood
        // (PIN-7).
        .overlay(alignment: .top) {
            if cameraManager.lastTapWasRevisit && cameraManager.tapProcessing {
                Text("Re-segmenting at this Pin's saved point…")
                    .font(.footnote.weight(.semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(.black.opacity(0.7))
                    .cornerRadius(8)
                    .padding(.top, 110)
                    .allowsHitTesting(false)
            }
        }
    }

    // MARK: - Pin label reveal (user-instructed UI change)

    /// The marker currently drawn as primary, if any.
    ///
    /// This is the only signal the reveal needs, and it costs nothing beyond a
    /// scan of at most three published markers.
    private var primaryMarkerID: UUID? {
        cameraManager.tapAnchorMarkers.first(where: { $0.isPrimary })?.id
    }

    /// B-37 — ids of markers carrying the ↻ revisit decoration, in publish
    /// order.  Watched by `onChange` above to fire the one automatic
    /// provenance-label reveal when a revisit product first lands.
    private var revisitOriginMarkerIDs: [UUID] {
        cameraManager.tapAnchorMarkers.filter(\.isRevisitOrigin).map(\.id)
    }

    /// Reveal `id`'s tag label for a few seconds.
    ///
    /// Why "reveal temporarily" rather than "toggle": a toggle needs a second
    /// tap on the *same* pin to mean "hide", but a tap on an already-primary
    /// mask produces no observable state change in the pipeline (§3.2 promote
    /// is a no-op when the instance is already primary), so the view layer
    /// cannot see it without adding a marker-specific hit region — which would
    /// fragment the single tap contract §22.2.2 froze.  A timed reveal needs
    /// no such signal.
    ///
    /// Why it composes with promote rather than duplicating it: promote
    /// already says *which* instance is now the assertion (bigger disc, white
    /// ring); the label answers the next question — *what* it is — and then
    /// gets out of the way so three pinned masks do not permanently carry
    /// three floating text chips across the preview.
    private func revealTag(for id: UUID?) {
        guard let id = id else { return }
        withAnimation(.easeOut(duration: 0.18)) {
            revealedTagMarkerID = id
        }
        tagRevealToken &+= 1
    }

    private func revisitReasonText(for event: CameraManager.PinRevisitEvent) -> String? {
        switch event.kind {
        case .refused(let reason): return reason
        case .failed: return "The last revisit attempt did not produce a mask. Showing the saved snapshot instead."
        // R-D (§23.1.9): the banner has no snapshot entry for a promote, so
        // this is never reached from it; nil keeps the viewer reason empty
        // should a future path get here.
        case .promoted: return nil
        }
    }
}

/// Phase 4B Day 6 (§19.4.4 R-B / R-C) — bottom card shown on a revisit
/// rejection/failure, with an explicit reason and an entry into the static
/// fallback viewer (PIN-2's "never silently degrade" clause).
private struct PinRevisitBanner: View {
    let event: CameraManager.PinRevisitEvent
    let onViewSnapshot: () -> Void
    let onDismiss: () -> Void

    private var message: String {
        switch event.kind {
        case .refused(let reason): return reason
        case .failed: return "Revisit failed — the tap did not produce a mask."
        // R-D copy (§23.1.9's outcome table) — ⛔ no "Re-segment(ing)"
        // wording on this path (nothing was decoded).  The optional
        // provenance suffix is the ONLY place the tag may appear, and only
        // inside the quoted `From Pin "…"` phrase (§23.1.4 / PIN-7).
        case .promoted(let fromPinTag):
            let base = "That spot is already covered by a selection on screen."
            if let tag = fromPinTag, !tag.isEmpty {
                return base + " From Pin “\(tag)”."
            }
            return base
        }
    }

    /// R-D is informational, not an error, and has no static-snapshot detour:
    /// the live instance the user asked about is right there on screen.
    private var isPromoted: Bool {
        if case .promoted = event.kind { return true }
        return false
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(message,
                  systemImage: isPromoted ? "info.circle.fill"
                                          : "exclamationmark.triangle.fill")
                .font(.footnote)
                .foregroundColor(.white)
            HStack {
                if !isPromoted {
                    Button("View Saved Snapshot", action: onViewSnapshot)
                        .font(.footnote.weight(.semibold))
                        .buttonStyle(.borderedProminent)
                        .tint(.orange)
                }
                Button("Dismiss", action: onDismiss)
                    .font(.footnote)
                    .buttonStyle(.bordered)
                    .tint(.white)
            }
        }
        .padding(12)
        .background(Color.black.opacity(0.85))
        .cornerRadius(12)
        .padding(.horizontal, 24)
    }
}

/// Pulsing ring shown at the tap point while SAM inference is in flight.
struct TapLoadingIndicator: View {
    @State private var pulsing = false

    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.cyan, lineWidth: 3)
                .frame(width: 44, height: 44)
                .scaleEffect(pulsing ? 1.4 : 0.8)
                .opacity(pulsing ? 0.2 : 0.9)
            Circle()
                .fill(Color.cyan)
                .frame(width: 10, height: 10)
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.7).repeatForever(autoreverses: false)) {
                pulsing = true
            }
        }
        .allowsHitTesting(false)
    }
}

/// Shown at the tap point when a tap produced no mask (Requirement C).
///
/// Deliberately NOT a progress affordance: Architect D1/D2 rule out anything
/// that implies duration.  This is a terminal state — the ring stops, turns
/// red, states the reason, and CameraManager clears it after ~1.6 s.
struct TapFailureIndicator: View {
    let message: String

    var body: some View {
        VStack(spacing: 6) {
            ZStack {
                Circle()
                    .stroke(Color.red.opacity(0.9), lineWidth: 3)
                    .frame(width: 44, height: 44)
                Image(systemName: "xmark")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(.red)
            }
            Text(message)
                .font(.caption2)
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 190)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.black.opacity(0.7))
                .cornerRadius(6)
        }
        .transition(.opacity)
        .allowsHitTesting(false)
    }
}

// MARK: - Day 6 UI Components

/// One-shot circular ripple at the accepted tap point.
///
/// Triggered by a change in `trigger` (wrapping-add counter).  The outer ring
/// expands from 0→80pt and fades out over 0.4 s; the inner dot stays for 0.4 s
/// so the user has a stable target anchor.  After the animation the view stays
/// in place until `ripplePoint` is updated — it is invisible (opacity 0) and
/// allowsHitTesting(false).
struct TapRippleEffect: View {
    let trigger: Int

    @State private var scale: CGFloat = 0.1
    @State private var opacity: Double = 0.85

    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.cyan, lineWidth: 2.5)
                .frame(width: 80, height: 80)
                .scaleEffect(scale)
                .opacity(opacity)
            Circle()
                .fill(Color.cyan.opacity(0.55))
                .frame(width: 14, height: 14)
                .opacity(opacity)
        }
        .onChange(of: trigger) { _, _ in
            scale = 0.1
            opacity = 0.85
            withAnimation(.easeOut(duration: 0.4)) {
                scale = 1.0
                opacity = 0.0
            }
        }
        .onAppear {
            // Fire the animation immediately on first appearance.
            withAnimation(.easeOut(duration: 0.4)) {
                scale = 1.0
                opacity = 0.0
            }
        }
        .allowsHitTesting(false)
    }
}

/// Tap anchor marker: a filled disc carrying the instance's number (1, 2 or 3)
/// in the slot's assigned colour.  Persists at the tap's view-space anchor
/// until the instance is evicted from the pool.  Phase 4 Day 1 (L3 carrier,
/// W-1 shortfall): the number is the *identity* channel — it tells the user
/// which of the three on-screen masks this anchor owns — and it is derived from
/// the SAME slot index as the fill colour, so number N and hue N always agree
/// and an evicted instance releases its number together with its hue.
///
/// Slot colours (architect §3.3.1 palette — FINAL, do not re-tune here):
///   0 → cyan        rgb(0, 217, 255)   1 → aqua rgb(0, 255, 242)
///   2 → spring cyan rgb(0, 255, 170)
///
/// Legibility over arbitrary camera content is carried by three stacked layers,
/// none of which depends on the scene behind it: an opaque black backing disc,
/// the saturated slot fill, and a black glyph on that fill (all three palette
/// hues have relative luminance Y ≥ 0.568, so black type on them clears 4.5:1).
///
/// Primary vs secondary (§3.4 role split, mirroring the 0.60 / 0.40 mask fill):
/// the primary — the newest tap — is drawn larger, with a heavier glyph and a
/// white ring; secondaries keep full colour identity at reduced weight so they
/// read as "this was selected", not as the current assertion.
struct TapAnchorMarkerView: View {
    let slotIndex: Int
    /// Newest live instance.  Defaulted so existing call sites and previews that
    /// do not know the role still compile and render as a secondary marker.
    var isPrimary: Bool = false
    /// Phase 4B Day 5 (§22.2.4, tasks.md Day 5) — true once this instance has
    /// a saved Pin.  Swaps the numbered glyph for 📌, distinguishing a
    /// transient tap marker from a persisted one, exactly as tasks.md
    /// specifies ("图标变为图钉📌样式").  Defaulted so existing callers/previews
    /// still compile.
    var isPinned: Bool = false
    /// Phase 4B Day 6 — the Pin's tag, rendered as a small label beside the
    /// icon (task instruction: put the label next to the pin). nil for an
    /// unpinned marker, or a pinned one saved with no tag.
    var tag: String? = nil
    /// User-instructed UI change — the label is now shown *on tap* instead of
    /// permanently.  ContentView sets this true for the marker that a tap just
    /// made primary, and clears it a few seconds later.  Defaulted to false so
    /// existing call sites and previews still compile.
    var showsTag: Bool = false
    /// B-37 (§23.1.4) — this instance's mask was decoded by a Pin revisit.
    /// Glyph becomes ↻ (SF Symbol `arrow.clockwise`; ⛔ never 📌 — this mask
    /// has no disk record, §23.1.5) on the same chip geometry, slot colour
    /// and primary-ring rules.  `isPinned` wins when both are set: a later
    /// long-press save writes a NEW record, at which point 📌 is true and
    /// replaces ↻ (§23.1.6-2).
    var isRevisitOrigin: Bool = false
    /// The origin Pin's tag; rendered ONLY inside the quoted
    /// `From Pin "<tag>"` label (§23.1.4's construction rule — a tag outside
    /// that prefix is a bare identity claim and is forbidden).  nil / empty ⇒
    /// `From an untagged Pin`.
    var revisitPinTag: String? = nil

    private var color: Color {
        switch slotIndex {
        case 0:  return Color(red: 0/255, green: 217/255, blue: 255/255)
        case 1:  return Color(red: 0/255, green: 255/255, blue: 242/255)
        default: return Color(red: 0/255, green: 255/255, blue: 170/255)
        }
    }

    private var diameter: CGFloat { isPrimary ? 28 : 22 }
    private var fontSize: CGFloat { isPrimary ? 16 : 12 }

    /// §23.1.4's permitted provenance phrases — the only two forms this label
    /// may take, and the only context a tag may appear in.
    private var provenanceLabel: String {
        if let tag = revisitPinTag, !tag.isEmpty {
            return "From Pin “\(tag)”"
        }
        return "From an untagged Pin"
    }

    var body: some View {
        ZStack {
            // Backing chip: opaque, so the glyph never has to survive whatever
            // the camera happens to be pointing at.
            Circle()
                .fill(Color.black.opacity(0.85))
                .frame(width: diameter + 4, height: diameter + 4)
            Circle()
                .fill(color)
                .frame(width: diameter, height: diameter)
            // Primary ring: a second, achromatic difference channel, so the
            // role stays readable for viewers who cannot separate the hues.
            if isPrimary {
                Circle()
                    .strokeBorder(Color.white, lineWidth: 2)
                    .frame(width: diameter + 4, height: diameter + 4)
            }
            if isPinned {
                Text("📌")
                    .font(.system(size: fontSize + 2))
            } else if isRevisitOrigin {
                // B-37: ↻ — provenance, not identity (PIN-7).  Black on the
                // slot fill, same as the number glyph, so it keeps the chip's
                // contrast guarantees.
                Image(systemName: "arrow.clockwise")
                    .font(.system(size: fontSize, weight: isPrimary ? .heavy : .bold))
                    .foregroundColor(.black)
            } else {
                Text("\(slotIndex + 1)")
                    .font(.system(size: fontSize,
                                  weight: isPrimary ? .heavy : .bold,
                                  design: .rounded))
                    .foregroundColor(.black)
            }
        }
        // Day 6 — the Pin's tag, rendered beside the icon. An `.overlay`
        // (not a sibling in an HStack) so the icon itself stays anchored
        // exactly at `marker.viewPoint`; only the label is offset outward.
        //
        // Gated on `showsTag` (user-instructed change): revealed by the tap
        // that promotes this instance to primary, then hidden again, so a
        // preview holding three pinned masks is not permanently covered by
        // three text chips.  The scale transition grows from the marker's own
        // edge, so the label reads as coming *out of* the pin.
        .overlay(alignment: .trailing) {
            if showsTag, isPinned, let tag = tag, !tag.isEmpty {
                Text(tag)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.black.opacity(0.75))
                    .cornerRadius(6)
                    .fixedSize()
                    .offset(x: diameter + 10)
                    .transition(.opacity.combined(
                        with: .scale(scale: 0.7, anchor: .leading)))
            } else if showsTag, isRevisitOrigin {
                // B-37 provenance label (§23.1.4 permitted list).  Same
                // reveal policy as the pinned tag, plus one automatic reveal
                // when the revisit mask first appears (driven in
                // ContentView).  The tag appears ONLY inside the quoted
                // `From Pin "…"` phrase — never bare (PIN-7).
                Text(provenanceLabel)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.black.opacity(0.75))
                    .cornerRadius(6)
                    .fixedSize()
                    .offset(x: diameter + 10)
                    .transition(.opacity.combined(
                        with: .scale(scale: 0.7, anchor: .leading)))
            }
        }
        .opacity(isPrimary ? 1.0 : 0.85)
        .shadow(color: .black.opacity(0.5), radius: 2, x: 0, y: 1)
        .allowsHitTesting(false)
    }
}

// MARK: - FPS Monitor

/// Measures main-thread frame cadence via CADisplayLink and publishes the
/// rolling FPS value every 0.5 s.  Used by the debug FPS badge in ContentView.
///
/// Threading: the displayLink fires on the main run-loop (.common mode), so
/// all mutations happen on the main thread — no locking needed.
final class FPSMonitor: ObservableObject {
    @Published var fps: Int = 0

    private var displayLink: CADisplayLink?
    private var frameCount: Int = 0
    private var lastTimestamp: CFTimeInterval = 0

    init() {
        displayLink = CADisplayLink(target: self, selector: #selector(tick(_:)))
        displayLink?.add(to: .main, forMode: .common)
    }

    deinit {
        displayLink?.invalidate()
        displayLink = nil
    }

    @objc private func tick(_ link: CADisplayLink) {
        // Skip the very first callback: we have no reference timestamp yet.
        if lastTimestamp == 0 {
            lastTimestamp = link.timestamp
            return
        }
        frameCount += 1
        let elapsed = link.timestamp - lastTimestamp
        // Publish every 0.5 s so the badge is readable without flickering.
        if elapsed >= 0.5 {
            fps = Int(Double(frameCount) / elapsed + 0.5)   // round-nearest
            frameCount = 0
            lastTimestamp = link.timestamp
        }
    }
}

#Preview {
    ContentView()
}
