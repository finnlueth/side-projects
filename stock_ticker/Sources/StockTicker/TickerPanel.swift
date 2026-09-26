import AppKit
import SwiftUI

/// Borderless, transparent panel so the SwiftUI glass shape *is* the window chrome.
final class GlassPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

/// Shows/hides the ticker editor anchored beneath the status item, popover-style.
@MainActor
final class TickerPanelController {
    private let store: TickerStore
    private var panel: GlassPanel?
    private weak var anchorButton: NSStatusBarButton?
    private var eventMonitors: [Any] = []
    private var lastHiddenAt = Date.distantPast
    private var dismissObservers: [NSObjectProtocol] = []
    private var resizeObserver: NSObjectProtocol?

    init(store: TickerStore) {
        self.store = store
    }

    var isVisible: Bool { panel?.isVisible ?? false }

    func toggle(relativeTo button: NSStatusBarButton?) {
        if isVisible {
            hide()
        } else if Date().timeIntervalSince(lastHiddenAt) > 0.3 {
            // A status item click can dismiss the panel on mouse-down (via the monitors below) before the
            // button action arrives on mouse-up; don't let that same click reopen what it just closed.
            show(relativeTo: button)
        }
    }

    func show(relativeTo button: NSStatusBarButton?) {
        anchorButton = button
        let panel = self.panel ?? makePanel()
        self.panel = panel
        panel.layoutIfNeeded()
        reposition()
        NSApp.activate()
        panel.makeKeyAndOrderFront(nil)
        installMonitors()
    }

    func hide() {
        guard isVisible else { return }
        removeMonitors()
        ColorPanelBridge.shared.close()
        panel?.orderOut(nil)
        lastHiddenAt = Date()
    }

    // MARK: Window

    private func makePanel() -> GlassPanel {
        let host = NSHostingController(rootView: TickerPanelView(store: store))
        host.sizingOptions = [.preferredContentSize]

        let panel = GlassPanel(
            contentRect: NSRect(x: 0, y: 0, width: 340, height: 300),
            styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        panel.contentViewController = host
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false // the glass shape draws its own edge; a window shadow would outline the rect frame
        panel.level = .floating // same tier as the colour panel, so the key one wins
        panel.hidesOnDeactivate = false
        panel.animationBehavior = .utilityWindow
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
        panel.isReleasedWhenClosed = false

        // Keep the top edge glued to the menu bar as the content grows/shrinks.
        resizeObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.didResizeNotification, object: panel, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.reposition() }
        }
        return panel
    }

    private func reposition() {
        guard let panel else { return }
        let size = panel.frame.size
        let screen = anchorButton?.window?.screen ?? NSScreen.main
        guard let visible = screen?.visibleFrame else { return }

        // Anchor under the status item; if it has no on-screen frame (e.g. tucked away by a menu bar
        // manager like Bartender/Ice) fall back to the top centre of the screen.
        var anchor = NSRect(x: visible.midX, y: visible.maxY, width: 0, height: 0)
        if let button = anchorButton, let buttonWindow = button.window, buttonWindow.frame.height > 0 {
            anchor = buttonWindow.convertToScreen(button.convert(button.bounds, to: nil))
        }

        var x = anchor.midX - size.width / 2
        x = min(max(x, visible.minX + 8), visible.maxX - size.width - 8)
        let y = anchor.minY - size.height - 6
        panel.setFrameOrigin(NSPoint(x: x, y: y))
        panel.invalidateShadow()
    }

    // MARK: Dismissal

    private func installMonitors() {
        removeMonitors()

        // Clicks in other apps.
        let globalMonitor = NSEvent.addGlobalMonitorForEvents(
            matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown],
            handler: { [weak self] _ in
                Task { @MainActor in self?.hide() }
            }
        )
        if let globalMonitor { eventMonitors.append(globalMonitor) }

        // Escape. Clicks inside our own app (the panel, the status item, the colour panel and picker
        // popovers) never dismiss; the status item's own action handles the toggle.
        let localMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [.keyDown],
            handler: { [weak self] event in
                guard let self, event.keyCode == 53 else { return event } // Esc
                self.hide()
                return nil
            }
        )
        if let localMonitor { eventMonitors.append(localMonitor) }

        dismissObservers.append(NotificationCenter.default.addObserver(
            forName: NSApplication.didResignActiveNotification, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.hide() }
        })
    }

    private func removeMonitors() {
        eventMonitors.forEach { NSEvent.removeMonitor($0) }
        eventMonitors.removeAll()
        dismissObservers.forEach { NotificationCenter.default.removeObserver($0) }
        dismissObservers.removeAll()
    }
}
