//
//  JudgeE2App.swift
//  JudgeE2
//
//  Created by Jian Sun on 2/26/26.
//
//  Phase 4B — Day 5 (Builder).  Owns the single `FilePinStore` instance and
//  the two forced-flush call sites B-21 left open on Day 4 (§22.1.11 —
//  "already-implemented `flush(waitUntilDone:)` must be wired to
//  `scenePhase → .background` and `willTerminate`; the third site, coalescing-
//  window expiry, has been live since Day 4").  `willTerminate` has no SwiftUI
//  scene-phase equivalent, so it is picked up via `NotificationCenter` — no
//  `UIApplicationDelegateAdaptor` needed for one notification.
//

import SwiftUI
import UIKit

@main
struct JudgeE2App: App {
    /// Single store instance for the whole app (§19.1: one `FilePinStore`).
    /// `@StateObject` here, not in `ContentView`, so its identity survives
    /// scene-phase transitions and the willTerminate observer below can close
    /// over the same instance the UI is bound to.
    @StateObject private var pinStore = FilePinStore()

    @Environment(\.scenePhase) private var scenePhase

    /// Retains the `willTerminate` observer for the process lifetime; removed
    /// only implicitly at process exit.  Kept off the stack so ARC does not
    /// collect it early — `NotificationCenter.addObserver` does not retain it.
    @State private var willTerminateObserver: NSObjectProtocol?

    init() {
        ModelLoader.testLoad(backend: .all)
        ModelLoader.testMobileSAMLoad()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(pinStore)
                .onAppear {
                    pinStore.load()
                    #if DEBUG
                    PinDebugFixture.runFromLaunchArgumentsIfPresent(store: pinStore)
                    #endif
                    if willTerminateObserver == nil {
                        willTerminateObserver = NotificationCenter.default.addObserver(
                            forName: UIApplication.willTerminateNotification,
                            object: nil,
                            queue: nil
                        ) { [pinStore] _ in
                            // §19.3.3 / §22.1.11: at willTerminate no real-time
                            // inference is running, so blocking on pin.store.io
                            // here costs nothing that matters, and it is the
                            // last chance any code gets to run at all.
                            pinStore.flush(waitUntilDone: true)
                        }
                    }
                }
        }
        // §19.3.3 / §22.1.11: the second of the three forced-flush sites.
        // `.background` (not `.inactive`) — matches the P-3(i) acceptance
        // protocol ("send to background, then kill from the app switcher").
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .background {
                pinStore.flush(waitUntilDone: true)
            }
        }
    }
}
