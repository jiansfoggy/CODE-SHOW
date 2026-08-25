//
//  BottomControlBar.swift
//  JudgeE2
//
//  Phase 5 — the app's one persistent control surface (architect_output.md
//  §27.5 / D-27.1, as amended by the 2026-08-24 single-function direction).
//
//  D-27.1 originally scoped this file as "merge the App Mode segmented
//  picker (`ContentView.swift:256-261`) and the floating mode quick-toggle
//  (`:414-441`) into one bottom bar bound to the same `$mode`".  The product
//  direction that followed removes the *user-facing* half of that: Detection
//  and Segmentation are no longer product features, so a three-state switcher
//  has nothing left to switch and the bar carries the Tap-to-Segment + Pin
//  workflow instead.  Both old entry points are still deleted from
//  `ContentView`, which is the part of D-27.1 that still binds — there is
//  exactly one place `mode` can be changed, and in a Release build there is
//  none.
//
//  ⛔ Scope: this view owns no state.  Every control is a closure the caller
//  already had a call site for; `mode` arrives as a plain value (not a
//  Binding) because nothing in this bar writes it.  No frozen surface
//  (§26.5) is touched from here.
//

import SwiftUI

struct BottomControlBar: View {
    /// Read-only.  In Release this is always `.tapToSegment`; in a Debug build
    /// the hidden Developer section can put the app into `.detectionOnly` /
    /// `.segmentation`, and the workflow buttons below dim accordingly rather
    /// than pretending to act.
    let mode: AppMode

    /// Records on disk, for the Pin List badge.  Passed in rather than read
    /// from the store here so this view stays free of dependencies.
    let pinCount: Int

    /// True when there is at least one live selection to clear.
    let hasActiveSelection: Bool

    let onOpenPinList: () -> Void
    let onClearSelections: () -> Void
    let onFlipCamera: () -> Void
    let onOpenSettings: () -> Void

    private var isWorkflowActive: Bool { mode == .tapToSegment }

    /// ⛔ §23.1.8 bans the object word family ("7 objects" / "7 tracked") —
    /// naming what was segmented is an identity claim (PIN-7).  Counting
    /// *Pins* names the app's own record type, which stays on the provenance
    /// side of that line.  Kept verbatim from the button's previous home in
    /// `ContentView`.
    private var pinListAccessibilityLabel: String {
        switch pinCount {
        case 0:  return "Pin list, empty"
        case 1:  return "Pin list, 1 pin"
        default: return "Pin list, \(pinCount) pins"
        }
    }

    /// Bounded so a three-digit library cannot widen the badge past the button
    /// it sits inside.
    private var badgeText: String { pinCount > 99 ? "99+" : "\(pinCount)" }

    var body: some View {
        HStack(spacing: 0) {
            BarButton(systemImage: "mappin.and.ellipse",
                      title: "Pins",
                      accessibilityLabel: pinListAccessibilityLabel,
                      action: onOpenPinList) {
                if pinCount > 0 {
                    Text(badgeText)
                        .font(.system(size: 10, weight: .semibold))
                        .monospacedDigit()
                        .foregroundColor(.white)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(BrandPalette.accent))
                        // Decoration: VoiceOver reads the count off the
                        // button's own label, so exposing it twice is noise.
                        .accessibilityHidden(true)
                        // Capped on purpose — at Accessibility XXXL an
                        // uncapped badge overflows the button and becomes
                        // unreadable, which is worse than not scaling.  The
                        // count still reaches VoiceOver via the label.
                        .dynamicTypeSize(...DynamicTypeSize.large)
                        .offset(x: 13, y: -11)
                        // The badge is the only thing on this bar that changes
                        // on its own.  A short spring on the count makes a
                        // just-saved Pin visibly land somewhere, which is the
                        // long-press → sheet → list handoff's last beat.
                        .transition(.scale.combined(with: .opacity))
                        .id(pinCount)
                }
            }

            BarButton(systemImage: "xmark.circle",
                      title: "Clear",
                      accessibilityLabel: "Clear all selections",
                      action: onClearSelections)
                .disabled(!isWorkflowActive || !hasActiveSelection)

            BarButton(systemImage: "arrow.triangle.2.circlepath.camera",
                      title: "Flip",
                      accessibilityLabel: "Switch camera",
                      action: onFlipCamera)

            BarButton(systemImage: "gearshape",
                      title: "Settings",
                      accessibilityLabel: "Settings",
                      action: onOpenSettings)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 6)
        // ⛔ Deliberately an opaque scrim, not `.ultraThinMaterial`.  A
        // material samples whatever is behind it, and behind this bar is a
        // live camera feed — point the phone at a white wall and a vibrancy
        // background turns pale, taking the white glyphs with it.  That is the
        // same failure mode §3.3.1 C-4 rules out for the mask itself ("L1 must
        // not depend on what the user points the camera at"); the rule is not
        // weaker for the controls.  0.62 black is the value the app's other
        // chrome already uses.
        .background(
            Capsule(style: .continuous)
                .fill(Color.black.opacity(0.62))
        )
        .overlay(
            Capsule(style: .continuous)
                .strokeBorder(Color.white.opacity(0.14), lineWidth: 0.5)
        )
        .shadow(color: .black.opacity(0.35), radius: 12, y: 4)
        .animation(.spring(response: 0.34, dampingFraction: 0.7), value: pinCount)
        .animation(.easeInOut(duration: 0.18), value: hasActiveSelection)
    }
}

/// One control in the bar: glyph over caption, in a hit target that fills its
/// share of the bar's width.
///
/// The explicit `.contentShape(Rectangle())` is not decorative — a bare
/// `Button` label hit-tests only its own glyph/text boxes, which is exactly
/// the gap that made the P-9 button in the old panel unreachable on device
/// (2026-08-23).  Every control in this file sets one.
private struct BarButton<Badge: View>: View {
    let systemImage: String
    let title: String
    let accessibilityLabel: String
    let action: () -> Void
    @ViewBuilder var badge: () -> Badge

    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Image(systemName: systemImage)
                    .font(.system(size: 19, weight: .medium))
                    .overlay(alignment: .topTrailing) { badge() }
                Text(title)
                    .font(.system(size: 10, weight: .medium))
                    .dynamicTypeSize(...DynamicTypeSize.large)
            }
            .foregroundColor(.white.opacity(isEnabled ? 1.0 : 0.38))
            .frame(maxWidth: .infinity)
            .frame(height: 46)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
    }
}

extension BarButton where Badge == EmptyView {
    init(systemImage: String,
         title: String,
         accessibilityLabel: String,
         action: @escaping () -> Void)
    {
        self.init(systemImage: systemImage,
                  title: title,
                  accessibilityLabel: accessibilityLabel,
                  action: action,
                  badge: { EmptyView() })
    }
}

#Preview {
    ZStack {
        Color.gray
        VStack {
            Spacer()
            BottomControlBar(mode: .tapToSegment,
                             pinCount: 7,
                             hasActiveSelection: true,
                             onOpenPinList: {},
                             onClearSelections: {},
                             onFlipCamera: {},
                             onOpenSettings: {})
                .padding(.horizontal, 18)
        }
    }
    .ignoresSafeArea()
}
