import AppKit

/// Drives the shared NSColorPanel directly (target/action) instead of through a colour well, so it works
/// from the non-activating glass panel: an accessory app can't always activate itself, and a colour panel
/// left with the default `hidesOnDeactivate` would never show.
@MainActor
final class ColorPanelBridge: NSObject {
    static let shared = ColorPanelBridge()

    private var onChange: ((NSColor) -> Void)?

    func open(with color: NSColor, onChange: @escaping (NSColor) -> Void) {
        self.onChange = onChange
        let panel = NSColorPanel.shared
        panel.showsAlpha = false
        panel.isContinuous = true
        panel.hidesOnDeactivate = false
        panel.color = color
        panel.setTarget(self)
        panel.setAction(#selector(colorDidChange(_:)))
        NSApp.activate()
        panel.makeKeyAndOrderFront(nil)
    }

    func close() {
        guard NSColorPanel.sharedColorPanelExists else { return }
        NSColorPanel.shared.setTarget(nil)
        NSColorPanel.shared.orderOut(nil)
        onChange = nil
    }

    @objc private func colorDidChange(_ sender: NSColorPanel) {
        onChange?(sender.color)
    }
}
