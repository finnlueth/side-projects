import AppKit
import Combine

/// Owns the NSStatusItem, keeps the bar view in sync with the store, and routes clicks.
@MainActor
final class StatusItemController: NSObject {
    private let store: TickerStore
    private let statusItem: NSStatusItem
    private let barView = TickerBarView()
    private let panel: TickerPanelController
    private var cancellable: AnyCancellable?

    init(store: TickerStore) {
        self.store = store
        self.panel = TickerPanelController(store: store)
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        super.init()

        if let button = statusItem.button {
            barView.frame = button.bounds
            barView.autoresizingMask = [.width, .height]
            button.addSubview(barView)
            button.target = self
            button.action = #selector(statusItemClicked(_:))
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
        }
        barView.onHeightChange = { [weak self] in self?.updateLength() }

        // objectWillChange fires before the property mutates; hop to the next run loop turn to read the new values.
        cancellable = store.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.update() }
        update()
    }

    private func update() {
        barView.gainColor = store.gainColor
        barView.lossColor = store.lossColor
        barView.showsPercentChange = store.showPercentChange
        barView.showsDollarChange = store.showDollarChange
        barView.compactLargeAmounts = store.compactLargeAmounts
        barView.entries = store.tickers.isEmpty
            ? [TickerBarView.Entry(ticker: .stock("Add ticker"), quote: nil)] // something to click on
            : store.tickers.map { TickerBarView.Entry(ticker: $0, quote: store.quotes[$0.id]) }
        updateLength()
        statusItem.button?.toolTip = tooltip
    }

    /// Width follows the text, which follows the font size, which follows the bar height.
    private func updateLength() {
        statusItem.length = barView.preferredWidth
    }

    private var tooltip: String {
        var lines = store.tickers.map { ticker -> String in
            let label = ticker.name.map { "\(ticker.symbol) (\($0))" } ?? ticker.symbol
            guard let quote = store.quotes[ticker.id] else {
                return "\(label): " + (store.errors[ticker.id] ?? "loading…")
            }
            let sign = quote.isPositive ? "+" : "-"
            let compact = store.compactLargeAmounts
            return "\(label)  \(quote.priceText(compact: compact))  \(sign)\(Quote.money(abs(quote.change), compact: compact))  (\(quote.percentText))"
        }
        if let updated = store.lastUpdated {
            lines.append("Updated " + updated.formatted(date: .omitted, time: .shortened))
        }
        return lines.joined(separator: "\n")
    }

    // MARK: Actions

    @objc private func statusItemClicked(_ sender: Any?) {
        if NSApp.currentEvent?.type == .rightMouseUp {
            panel.hide()
            showContextMenu()
        } else {
            panel.toggle(relativeTo: statusItem.button)
        }
    }

    private func showContextMenu() {
        let menu = NSMenu()
        menu.addItem(withTitle: "Refresh Now", action: #selector(refreshNow), keyEquivalent: "r").target = self
        menu.addItem(withTitle: "Edit Tickers…", action: #selector(openPanel), keyEquivalent: ",").target = self
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit Stock Ticker", action: #selector(quit), keyEquivalent: "q").target = self

        // Attach the menu only for the duration of the click so left clicks keep toggling the panel.
        statusItem.menu = menu
        statusItem.button?.performClick(nil)
        statusItem.menu = nil
    }

    @objc private func refreshNow() { store.refresh() }
    @objc private func openPanel() { panel.show(relativeTo: statusItem.button) }
    @objc private func quit() { NSApp.terminate(nil) }
}
