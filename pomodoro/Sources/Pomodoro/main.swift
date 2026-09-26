import AppKit
import Combine
import SwiftUI

/// Borderless window that can become key (for the Space shortcut) and can be dragged from anywhere,
/// including on top of buttons. A press that moves less than a few points is still a normal click.
final class GlassWindow: NSWindow {
    override var canBecomeKey: Bool { true }

    /// AppKit keeps windows below the menu bar, but our transparent shadow padding may overlap it.
    override func constrainFrameRect(_ frameRect: NSRect, to screen: NSScreen?) -> NSRect { frameRect }

    var onDragEnded: (() -> Void)?
    private var dragStart: (mouse: NSPoint, origin: NSPoint)?
    private var isDragging = false

    override func sendEvent(_ event: NSEvent) {
        switch event.type {
        case .leftMouseDown:
            dragStart = (NSEvent.mouseLocation, frame.origin)
            isDragging = false
        case .leftMouseDragged:
            guard let start = dragStart else { break }
            let mouse = NSEvent.mouseLocation
            let dx = mouse.x - start.mouse.x, dy = mouse.y - start.mouse.y
            if !isDragging && hypot(dx, dy) > 4 {
                isDragging = true
                cancelPendingClick(event)
            }
            if isDragging {
                setFrameOrigin(NSPoint(x: start.origin.x + dx, y: start.origin.y + dy))
                return
            }
        case .leftMouseUp:
            dragStart = nil
            if isDragging {
                isDragging = false
                onDragEnded?()
                return
            }
        default:
            break
        }
        super.sendEvent(event)
    }

    /// Releases the press far outside the window so a button under the cursor doesn't fire or stay highlighted.
    private func cancelPendingClick(_ event: NSEvent) {
        guard let up = NSEvent.mouseEvent(
            with: .leftMouseUp, location: NSPoint(x: -10_000, y: -10_000), modifierFlags: [],
            timestamp: event.timestamp, windowNumber: windowNumber, context: nil,
            eventNumber: 0, clickCount: 1, pressure: 0
        ) else { return }
        super.sendEvent(up)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let timer = PomodoroTimer()
    private let windowState = WindowState()
    private let hover = HoverState()
    /// Gap between the pill and the edge of the usable screen area.
    private let margin: CGFloat = 12
    /// Tighter against the menu bar, which already reads as a boundary.
    private let topMargin: CGFloat = 5
    private var window: GlassWindow!
    private var cancellables = Set<AnyCancellable>()

    func applicationDidFinishLaunching(_ notification: Notification) {
        let hosting = NSHostingView(rootView: ContentView(timer: timer, windowState: windowState, hover: hover))
        hosting.sizingOptions = [.intrinsicContentSize]

        window = GlassWindow(
            contentRect: NSRect(origin: .zero, size: hosting.fittingSize),
            styleMask: [.borderless, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.contentView = hosting
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false  // SwiftUI draws a shadow matching the rounded glass
        window.isReleasedWhenClosed = false
        window.onDragEnded = { [weak self] in self?.snapToNearestAnchor() }
        window.setFrame(targetFrame(for: windowState.anchor, on: NSScreen.main), display: false)

        NotificationCenter.default.publisher(for: NSApplication.didChangeScreenParametersNotification)
            .sink { [weak self] _ in
                guard let self else { return }
                window.setFrame(targetFrame(for: windowState.anchor, on: window.screen ?? NSScreen.main), display: true)
            }
            .store(in: &cancellables)

        windowState.$isOverlay
            .sink { [weak self] in self?.applyOverlay($0) }
            .store(in: &cancellables)

        // Show the countdown on the Dock icon so it's visible even when the window is covered.
        timer.objectWillChange
            .receive(on: RunLoop.main)
            .sink { [weak self] in self?.updateDockBadge() }
            .store(in: &cancellables)

        buildMainMenu()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        window.makeKeyAndOrderFront(nil)
        return true
    }

    private func applyOverlay(_ overlay: Bool) {
        window.level = overlay ? .floating : .normal
        window.collectionBehavior = overlay
            ? [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
            : [.managed]
        window.hidesOnDeactivate = false
    }

    private func updateDockBadge() {
        // objectWillChange fires before the new value lands, so read on the next runloop turn.
        DispatchQueue.main.async { [timer] in
            NSApp.dockTile.badgeLabel = timer.isRunning ? timer.formattedRemaining : nil
        }
    }

    // MARK: Snapping

    /// Horizontal offset of the pill inside the window for a given alignment.
    private func pillOffset(_ h: Anchor.Horizontal) -> CGFloat {
        let slack = Layout.windowContentWidth - windowState.pillWidth
        let inset: CGFloat = switch h {
        case .left: 0
        case .center: slack / 2
        case .right: slack
        }
        return Layout.shadowPadding + inset
    }

    /// Window frame that puts the pill at `anchor`; the transparent shadow padding may hang off-screen.
    private func targetFrame(for anchor: Anchor, on screen: NSScreen?) -> NSRect {
        let size = window.frame.size
        guard let vf = screen?.visibleFrame else { return window.frame }
        let pad = Layout.shadowPadding
        let x: CGFloat = switch anchor.horizontal {
        case .left: vf.minX + margin - pad
        case .center: vf.midX - size.width / 2
        case .right: vf.maxX - margin + pad - size.width
        }
        let y: CGFloat = switch anchor.vertical {
        case .top: vf.maxY - topMargin + pad - size.height
        case .middle: vf.midY - size.height / 2
        case .bottom: vf.minY + margin - pad
        }
        return NSRect(origin: NSPoint(x: x, y: y), size: size)
    }

    private func snapToNearestAnchor() {
        let old = windowState.anchor
        let pillCenter = NSPoint(
            x: window.frame.minX + pillOffset(old.horizontal) + windowState.pillWidth / 2,
            y: window.frame.midY
        )
        let screen = NSScreen.screens.first { $0.frame.contains(pillCenter) } ?? window.screen ?? NSScreen.main
        guard let vf = screen?.visibleFrame else { return }

        // Split the usable area into thirds along each axis.
        let fx = (pillCenter.x - vf.minX) / vf.width
        let fy = (pillCenter.y - vf.minY) / vf.height
        let new = Anchor(
            horizontal: fx < 1 / 3 ? .left : fx > 2 / 3 ? .right : .center,
            vertical: fy > 2 / 3 ? .top : fy < 1 / 3 ? .bottom : .middle
        )

        if new.horizontal != old.horizontal {
            // The pill's alignment inside the window changes with the anchor; shift the window to compensate
            // so the pill doesn't jump before the snap animation starts.
            windowState.anchor = new
            window.contentView?.layoutSubtreeIfNeeded()
            window.setFrameOrigin(NSPoint(
                x: window.frame.minX + pillOffset(old.horizontal) - pillOffset(new.horizontal),
                y: window.frame.minY
            ))
        } else {
            windowState.anchor = new
        }

        let target = targetFrame(for: new, on: screen)
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.3
            ctx.timingFunction = CAMediaTimingFunction(controlPoints: 0.2, 0.9, 0.3, 1.0)
            window.animator().setFrame(target, display: true)
        }
    }

    private func buildMainMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem()
        main.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Hide Pomodoro", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Pomodoro", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        NSApp.mainMenu = main
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
