//
//  PerfLogging.swift
//  JudgeE2
//
//  Phase 3 Day 4 — "Perf Quiet Log": a log-volume switch for on-device
//  performance sessions.  No algorithm, threshold, queue or data-flow change —
//  this file only decides *which* existing `print` lines reach the console.
//
//  WHY THIS EXISTS
//  The Day 4 performance protocol (shared/phase3_day4_test_checklist.md §2)
//  compares YOLO latency across detectionOnly / segmentation / tapToSegment.
//  Measured on device, the three modes emit wildly different log volumes
//  (~515 vs ~1039 lines per 100 frames), and `print` is a synchronous,
//  lock-taking write.  The per-frame diagnostics therefore land *inside* the
//  `Post=` timing window and become a confound of the very number under test —
//  besides scrolling the console faster than a human can read it.
//
//  DESIGN — classify at the call site, not with `if` at the call site.
//  Every existing `print` is rewritten to exactly one of four helpers, chosen
//  once by what the line *is*.  Nothing else at the call site changes, so with
//  `quietMode == false` the console output is byte-identical to before this
//  file existed (the only deliberate additions are the O1 mode-switch banner
//  and the quiet-only aggregate lines, see below).
//
//  AMENDED, Phase 4 §18.4.3 (B-11): every line now carries a `[t=…]` monotonic
//  prefix.  Output is therefore no longer byte-identical — but it is
//  *prefix*-identical, which is the property the §16.9.3 / §33 extraction
//  commands actually depend on.  See `PerfLogging.lineStampMs`.
//
//    perfLog(_:)         measurement data — ALWAYS printed.
//                        Window stats, Pre/Infer/Post, FPS/Memory, model-load
//                        and warmup lines, mode-switch banner.
//
//    diagLog(_:)         per-frame / per-tap diagnostics — printed only when
//                        quiet mode is OFF.  This is the ~90 % of log volume
//                        that pollutes the measurement.
//
//    faultLog(_:)        errors, aborts, numeric-sentinel rejections — ALWAYS
//                        printed.  A silent failure path would make a quiet
//                        session unreadable and is never worth the lines it
//                        saves; anomaly paths are never suppressed.
//
//    quietSummaryLog(_:) aggregate stand-ins printed ONLY when quiet mode is
//                        ON, replacing per-frame detail that quiet suppressed
//                        (currently: the detection-count window mean, which
//                        checklist §2.6 uses to confirm the four scenes were
//                        shot with comparable content).  Keeping these OFF in
//                        the default path is what preserves byte-identical
//                        non-quiet output.
//
//  `perfLog` and `faultLog` have the same body today.  They are deliberately
//  kept apart: the split records the measurement-vs-anomaly classification in
//  the source, so a future "even quieter" mode can drop one without anyone
//  having to re-derive which of ~110 call sites is which.
//
//  The message is an @autoclosure: when a line is suppressed its
//  `String(format:)` / interpolation is never evaluated, so quiet mode removes
//  the formatting cost from the hot path as well as the write.  That is the
//  whole point — a suppressed line must cost nothing inside `Post=`.

import Foundation

/// Global log-verbosity switch for on-device performance sessions.
enum PerfLogging {

    /// `true` → suppress per-frame / per-tap diagnostics, keep measurement and
    /// error lines.  Default `false` = exactly the pre-existing behaviour.
    ///
    /// Toggled from the Settings panel (ContentView, "Perf Quiet Log") so a
    /// performance session can be run on a device build without recompiling.
    ///
    /// THREADING: written on the main thread by the UI toggle, read on
    /// videoQueue / encoderQueue / decoderQueue / sessionQueue.  A `Bool` is a
    /// single aligned word on every architecture we ship, and the worst case
    /// of a torn read is that one log line lands on the wrong side of a toggle
    /// the user just flipped — harmless, and not worth a lock in a path that
    /// runs per frame.  Same pattern (and same rationale) as
    /// `SAMConfiguration.encoderResolution`.
    static var quietMode: Bool = false

    // MARK: - Monotonic line timestamp (§18.4.3 / B-11)
    //
    // WHY.  Every one of the 2338 lines in the Day 2–3 device session was
    // untimestamped, so a log could only ever be read in *line order* — and line
    // order is not time order across four queues.  §34.1.5 recorded that gap and
    // §18.4.2 escalated it: ISSUE-P4-DECODE's hypothesis (c) is judged by
    // "temporal correlation", which was simply not computable from the data,
    // so the experiment could not fail — it could not run.  A prefix costs one
    // `CACurrentMediaTime()` and closes that whole class of undecidability for
    // every future analysis, not just this one issue.
    //
    // ⚠️ THE PREFIX MUST STAY AT THE START OF THE LINE.  §16.9.3 and §33 extract
    // fields positionally (`grep -o 'qwait: [0-9.]*'`, the `[REANCHOR][inst#N]`
    // chains, the `Pre=…|Mem=` series).  A prefix in front of the whole line
    // leaves every one of those intact; a timestamp inserted between a tag and
    // its fields would break all of them at once.
    //
    // Monotonic, not wall clock: `CACurrentMediaTime` does not jump when the
    // system clock is adjusted, and the numbers must be subtractable.  Rebased
    // to process start so a session reads from ≈0 instead of from device uptime.

    /// `PerfLogger.nowMs()` at the first log line of the process.
    private static let processStartMs: Double = PerfLogger.nowMs()

    /// Milliseconds since process start, monotonic.
    @inline(__always)
    static func lineStampMs() -> Double { PerfLogger.nowMs() - processStartMs }
}

/// The `[t=…]` line-start prefix shared by every log helper (B-11).
@inline(__always)
private func stamped(_ message: String) -> String {
    String(format: "[t=%.1f] ", PerfLogging.lineStampMs()) + message
}

/// Measurement data — always printed, in both modes.
@inline(__always)
func perfLog(_ message: @autoclosure () -> String) {
    print(stamped(message()))
}

/// Per-frame / per-tap diagnostics — suppressed while quiet mode is on.
@inline(__always)
func diagLog(_ message: @autoclosure () -> String) {
    guard !PerfLogging.quietMode else { return }
    print(stamped(message()))
}

/// Errors, aborts and numeric-sentinel rejections — always printed.
/// Anomaly paths are never silenced, regardless of quiet mode.
@inline(__always)
func faultLog(_ message: @autoclosure () -> String) {
    print(stamped(message()))
}

/// Aggregate stand-in for detail that quiet mode suppressed — printed ONLY
/// while quiet mode is on, so the default (non-quiet) output is unchanged.
@inline(__always)
func quietSummaryLog(_ message: @autoclosure () -> String) {
    guard PerfLogging.quietMode else { return }
    print(stamped(message()))
}
