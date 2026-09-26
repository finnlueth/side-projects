import SwiftUI

/// The ticker editor. The outer container is Liquid Glass on macOS 26+, falling back to a material sheet.
struct TickerPanelView: View {
    @ObservedObject var store: TickerStore
    @ObservedObject private var stockDirectory: SymbolDirectory
    @ObservedObject private var coinDirectory: CoinDirectory
    @ObservedObject private var currencyDirectory: CurrencyDirectory
    // `@State` is macro-backed in the macOS 26+ SDK and the plugin only ships with Xcode, not the Command Line
    // Tools, so transient view state lives in a small observable object instead.
    @StateObject private var draft = PanelState()
    @FocusState private var focusedField: Ticker.Kind?

    init(store: TickerStore) {
        self.store = store
        self.stockDirectory = store.stockDirectory
        self.coinDirectory = store.coinDirectory
        self.currencyDirectory = store.currencyDirectory
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            symbols
            addRows
            Divider()
            settings
            footer
        }
        .padding(20)
        .frame(width: 360)
        .glassBackground(cornerRadius: 26)
        // The bubble that follows the cursor while reordering. It lives on the root view so its offset is
        // in the same (hosting view, top-left) space as the measured chip frames.
        .overlay(alignment: .topLeading) {
            if let ticker = draft.dragging {
                chip(ticker)
                    .scaleEffect(1.05)
                    .shadow(color: .black.opacity(0.3), radius: 8, y: 4)
                    .offset(x: draft.dragOrigin.x, y: draft.dragOrigin.y)
                    .allowsHitTesting(false)
            }
        }
        .onAppear {
            focusedField = .stock
            store.stockDirectory.loadIfNeeded(keyID: store.alpacaKeyID, secretKey: store.alpacaSecretKey)
            store.coinDirectory.loadIfNeeded(apiKey: store.coinMarketCapKey)
            store.currencyDirectory.loadIfNeeded()
        }
    }

    // MARK: Sections

    private var header: some View {
        HStack(spacing: 10) {
            Image(nsImage: NSApp.applicationIconImage)
                .resizable()
                .frame(width: 26, height: 26)
            Text("Stock Ticker")
                .font(.headline)
            Spacer()
            if store.isRefreshing {
                ProgressView().controlSize(.small)
            } else {
                Button {
                    store.refresh()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .help("Refresh now")
            }
        }
    }

    @ViewBuilder
    private var symbols: some View {
        if store.tickers.isEmpty {
            Text("Nothing yet — add a stock, coin or currency pair below.")
                .font(.callout)
                .foregroundStyle(.secondary)
        } else {
            VStack(alignment: .leading, spacing: 6) {
                FlowLayout(spacing: 8) {
                    ForEach(store.tickers) { ticker in
                        chip(ticker)
                            // Report the layout slot; the chip itself never moves, a floating copy does.
                            .background(GeometryReader { geometry in
                                Color.clear.preference(key: ChipFramesKey.self, value: [ticker.id: geometry.frame(in: .global)])
                            })
                            .opacity(draft.dragging == ticker ? 0 : 1)
                            .gesture(reorderGesture(for: ticker))
                    }
                }
                .onPreferenceChange(ChipFramesKey.self) { draft.chipFrames = $0 }
                if store.tickers.count > 1 {
                    Text("Drag to reorder")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    private var addRows: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Ticker.Kind.allCases, id: \.self) { kind in
                addRow(for: kind)
            }
        }
    }

    private func addRow(for kind: Ticker.Kind) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(kind.title)
                    .font(.callout)
                    .frame(width: 52, alignment: .leading)
                TextField(placeholder(for: kind), text: draft.binding(for: kind))
                    .textFieldStyle(.roundedBorder)
                    .focused($focusedField, equals: kind)
                    .onSubmit { add(kind) }
                    .onKeyPress(.downArrow) { moveHighlight(1, in: kind) }
                    .onKeyPress(.upArrow) { moveHighlight(-1, in: kind) }
                Button("Add") { add(kind) }
                    .disabled(draft.text(for: kind).trimmingCharacters(in: .whitespaces).isEmpty)
            }
            if focusedField == kind {
                let rows = suggestions(for: kind)
                if !rows.isEmpty {
                    suggestionList(rows, kind: kind)
                }
            }
        }
    }

    private func placeholder(for kind: Ticker.Kind) -> String {
        switch kind {
        case .stock: return "Add symbol, e.g. AAPL, MSFT"
        case .crypto: return "Add coin, e.g. BTC or Ethereum"
        case .forex: return "Add pair, e.g. EUR/USD"
        }
    }

    private struct Suggestion {
        let title: String
        let subtitle: String
        let trailing: String
        let add: () -> Void
    }

    private func suggestions(for kind: Ticker.Kind) -> [Suggestion] {
        let query = draft.text(for: kind)
        switch kind {
        case .stock:
            let held = Set(store.tickers.filter { $0.kind == .stock }.map(\.symbol))
            return stockDirectory.suggestions(for: query, excluding: held).map { entry in
                Suggestion(title: entry.symbol, subtitle: entry.name, trailing: entry.exchange) {
                    add(.stock) { store.addStocks(entry.symbol) }
                }
            }
        case .crypto:
            let held = Set(store.tickers.compactMap(\.coinID))
            return coinDirectory.suggestions(for: query, excluding: held).map { coin in
                Suggestion(title: coin.symbol, subtitle: coin.name, trailing: "#\(coin.rank)") {
                    add(.crypto) { store.add(coin: coin) }
                }
            }
        case .forex:
            let held = Set(store.tickers.filter { $0.kind == .forex }.map(\.symbol))
            return currencyDirectory.suggestions(for: query, excluding: held).map { pair in
                Suggestion(title: pair.symbol, subtitle: currencyDirectory.name(for: pair) ?? "", trailing: "ECB") {
                    add(.forex) { store.add(pair: pair) }
                }
            }
        }
    }

    private func suggestionList(_ rows: [Suggestion], kind: Ticker.Kind) -> some View {
        VStack(spacing: 1) {
            ForEach(0..<rows.count, id: \.self) { index in
                let row = rows[index]
                Button(action: row.add) {
                    HStack(spacing: 8) {
                        Text(row.title)
                            .font(.system(size: 12, weight: .bold, design: .rounded))
                            .frame(width: 68, alignment: .leading)
                        Text(row.subtitle)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                        Spacer(minLength: 4)
                        Text(row.trailing)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(
                        Color.accentColor.opacity(index == draft.highlighted ? 0.3 : 0),
                        in: RoundedRectangle(cornerRadius: 6, style: .continuous)
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(4)
        .background(.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .padding(.leading, 60)
    }

    private func moveHighlight(_ delta: Int, in kind: Ticker.Kind) -> KeyPress.Result {
        let count = suggestions(for: kind).count
        guard count > 0 else { return .ignored }
        draft.highlighted = (draft.highlighted + delta + count) % count
        return .handled
    }

    /// Enter takes the highlighted suggestion when there is one; otherwise whatever was typed.
    private func add(_ kind: Ticker.Kind) {
        let rows = suggestions(for: kind)
        if rows.indices.contains(draft.highlighted) {
            rows[draft.highlighted].add()
            return
        }
        let text = draft.text(for: kind)
        add(kind) {
            switch kind {
            case .stock: store.addStocks(text)
            case .crypto: store.addCrypto(text)
            case .forex: store.addForex(text)
            }
        }
    }

    private func add(_ kind: Ticker.Kind, _ action: () -> Void) {
        action()
        draft.setText("", for: kind)
        focusedField = kind
    }

    private var settings: some View {
        VStack(alignment: .leading, spacing: 10) {
            settingRow("Alpaca key ID") {
                TextField("Alpaca API key ID", text: $store.alpacaKeyID)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
            }
            settingRow("Secret") {
                SecureField("Alpaca secret key", text: $store.alpacaSecretKey)
                    .textFieldStyle(.roundedBorder)
            }
            settingRow("CoinMarketCap") {
                TextField("CoinMarketCap API key", text: $store.coinMarketCapKey)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
            }

            settingRow("Refresh every") {
                Picker("Refresh every", selection: $store.refreshMinutes) {
                    ForEach(TickerStore.refreshChoices, id: \.self) { minutes in
                        Text("\(minutes) min").tag(minutes)
                    }
                }
                .labelsHidden()
                .frame(width: 100)
            }

            Toggle("Only during US market hours", isOn: $store.marketHoursOnly)
                .toggleStyle(.switch)
                .controlSize(.small)

            settingRow("Show") {
                HStack(spacing: 12) {
                    Toggle("% change", isOn: $store.showPercentChange)
                    Toggle("$ change", isOn: $store.showDollarChange)
                }
                .toggleStyle(.checkbox)
            }

            Toggle("Hide cents from $10,000 up", isOn: $store.compactLargeAmounts)
                .toggleStyle(.switch)
                .controlSize(.small)

            settingRow("Colors") {
                HStack(spacing: 8) {
                    ForEach(ColorRole.allCases, id: \.self) { role in
                        colorButton(role)
                    }
                }
            }
            if let role = draft.editingColor {
                swatchStrip(for: role)
            }
        }
        .font(.callout)
    }

    /// Swatches live inline rather than in a popover or the system colour panel, which can't be relied on
    /// from an accessory app's non-activating panel.
    private func swatchStrip(for role: ColorRole) -> some View {
        let choice = store.colorChoice(for: role)
        let current = choice.color(for: role)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                modeButton("System", role: role, choice: .system, help: "Apple's system green / red")
                modeButton("Dynamic", role: role, choice: .dynamic,
                           help: "Vivid on dark menu bars, deep on light ones — follows the wallpaper")
                Spacer()
                Button("Custom…") {
                    ColorPanelBridge.shared.open(with: current) { store.setColorChoice(.custom($0), for: role) }
                }
                Button("Done") { draft.editingColor = nil }
            }
            .controlSize(.small)
            FlowLayout(spacing: 6) {
                ForEach(0..<Self.swatches.count, id: \.self) { index in
                    let swatch = Self.swatches[index]
                    Button {
                        store.setColorChoice(.custom(swatch), for: role)
                    } label: {
                        Circle()
                            .fill(Color(nsColor: swatch))
                            .frame(width: 22, height: 22)
                            .overlay(Circle().strokeBorder(.primary.opacity(0.2), lineWidth: 1))
                            .overlay {
                                if choice.isCustom, swatch.matches(current) {
                                    Image(systemName: "checkmark")
                                        .font(.system(size: 11, weight: .bold))
                                        .foregroundStyle(swatch.isBright ? .black : .white)
                                }
                            }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding(10)
        .background(.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    @ViewBuilder
    private func modeButton(_ title: String, role: ColorRole, choice: ColorChoice, help: String) -> some View {
        let selected = store.colorChoice(for: role) == choice
        if selected {
            Button(title) {}.buttonStyle(.borderedProminent).help(help)
        } else {
            Button(title) { store.setColorChoice(choice, for: role) }.buttonStyle(.bordered).help(help)
        }
    }

    private static let swatches: [NSColor] = [
        .systemGreen, .systemMint, .systemTeal, .systemCyan, .systemBlue, .systemIndigo, .systemPurple,
        .systemPink, .systemRed, .systemOrange, .systemYellow, .systemBrown, .systemGray, .white,
    ]

    private var footer: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let error = store.firstError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .lineLimit(2)
            }
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(lastUpdatedText)
                    Text(TickerStore.quotaNote)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                Spacer()
                Button("Quit") { NSApp.terminate(nil) }
                    .controlSize(.small)
            }
        }
    }

    // MARK: Reordering

    private func chip(_ ticker: Ticker) -> some View {
        SymbolChip(
            ticker: ticker,
            quote: store.quotes[ticker.id],
            compact: store.compactLargeAmounts,
            gainColor: Color(nsColor: store.gainColor),
            lossColor: Color(nsColor: store.lossColor),
            onRemove: { store.remove(ticker) }
        )
    }

    /// The gesture only supplies timing; positions come from AppKit. On macOS the gesture's locations in a
    /// named/global space are vertically flipped relative to GeometryReader frames, so mixing them puts the
    /// floating bubble in the wrong place.
    private func reorderGesture(for ticker: Ticker) -> some Gesture {
        DragGesture(minimumDistance: 3)
            .onChanged { _ in
                guard let pointer = pointerInHostingView() else { return }
                if draft.dragging != ticker {
                    let frame = draft.chipFrames[ticker.id] ?? .zero
                    draft.grabOffset = CGSize(width: pointer.x - frame.minX, height: pointer.y - frame.minY)
                    draft.dragging = ticker
                }
                draft.dragOrigin = CGPoint(x: pointer.x - draft.grabOffset.width,
                                           y: pointer.y - draft.grabOffset.height)
                if let target = slotIndex(for: ticker, at: pointer) {
                    withAnimation(.snappy(duration: 0.25)) { store.move(ticker, to: target) }
                }
            }
            .onEnded { _ in
                withAnimation(.snappy(duration: 0.2)) { draft.dragging = nil }
            }
    }

    /// Mouse position in the hosting view's coordinate space (top-left origin), matching `frame(in: .global)`.
    private func pointerInHostingView() -> CGPoint? {
        guard let window = NSApp.windows.first(where: { $0 is GlassPanel }),
              let contentView = window.contentView else { return nil }
        let inWindow = window.convertPoint(fromScreen: NSEvent.mouseLocation) // bottom-left origin
        return CGPoint(x: inWindow.x, y: contentView.bounds.height - inWindow.y)
    }

    /// The dragged chip takes the slot of whichever chip the pointer is over, once the pointer has passed
    /// that chip's midpoint in the direction of travel. Waiting for the midpoint keeps chips of different
    /// widths from ping-ponging after they swap.
    private func slotIndex(for ticker: Ticker, at pointer: CGPoint) -> Int? {
        guard let from = store.tickers.firstIndex(of: ticker),
              let dragged = draft.chipFrames[ticker.id] else { return nil }
        for (index, other) in store.tickers.enumerated() where other != ticker {
            guard let frame = draft.chipFrames[other.id], frame.insetBy(dx: -4, dy: -4).contains(pointer) else { continue }
            let sameRow = abs(frame.midY - dragged.midY) < frame.height / 2
            let forward = index > from
            let pastMidpoint = sameRow
                ? (forward ? pointer.x > frame.midX : pointer.x < frame.midX)
                : (forward ? pointer.y > frame.midY : pointer.y < frame.midY)
            return pastMidpoint ? index : nil
        }
        return nil
    }

    // MARK: Helpers

    private func settingRow<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        HStack {
            Text(title)
            Spacer()
            content().frame(maxWidth: 200, alignment: .trailing)
        }
    }

    private func colorButton(_ role: ColorRole) -> some View {
        let editing = draft.editingColor == role
        return Button {
            draft.editingColor = editing ? nil : role
        } label: {
            HStack(spacing: 5) {
                Circle()
                    .fill(Color(nsColor: store.colorChoice(for: role).color(for: role)))
                    .frame(width: 12, height: 12)
                    .overlay(Circle().strokeBorder(.primary.opacity(0.25), lineWidth: 1))
                Text(role.title)
                Image(systemName: editing ? "chevron.up" : "chevron.down")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(.secondary)
            }
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
    }

    private var lastUpdatedText: String {
        guard let date = store.lastUpdated else { return "Not updated yet" }
        return "Updated " + date.formatted(date: .omitted, time: .shortened)
    }

}

private final class PanelState: ObservableObject {
    @Published var stockText = "" { didSet { highlighted = 0 } }
    @Published var cryptoText = "" { didSet { highlighted = 0 } }
    @Published var forexText = "" { didSet { highlighted = 0 } }
    @Published var highlighted = 0
    @Published var editingColor: ColorRole?
    /// Chip currently being dragged, and where its top-left corner should be drawn (hosting view space).
    @Published var dragging: Ticker?
    @Published var dragOrigin: CGPoint = .zero
    var grabOffset: CGSize = .zero
    /// Latest layout frame of every chip in the hosting view's space (top-left origin). Not published: only
    /// the gesture reads it.
    var chipFrames: [String: CGRect] = [:]

    func text(for kind: Ticker.Kind) -> String {
        switch kind {
        case .stock: return stockText
        case .crypto: return cryptoText
        case .forex: return forexText
        }
    }

    func setText(_ text: String, for kind: Ticker.Kind) {
        switch kind {
        case .stock: stockText = text
        case .crypto: cryptoText = text
        case .forex: forexText = text
        }
    }

    func binding(for kind: Ticker.Kind) -> Binding<String> {
        Binding(get: { self.text(for: kind) }, set: { self.setText($0, for: kind) })
    }
}

private struct ChipFramesKey: PreferenceKey {
    static let defaultValue: [String: CGRect] = [:]
    static func reduce(value: inout [String: CGRect], nextValue: () -> [String: CGRect]) {
        value.merge(nextValue()) { $1 }
    }
}

// MARK: - Chip

private struct SymbolChip: View {
    let ticker: Ticker
    let quote: Quote?
    let compact: Bool
    let gainColor: Color
    let lossColor: Color
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            if ticker.kind != .stock {
                Image(systemName: ticker.kind == .crypto ? "bitcoinsign" : "arrow.left.arrow.right")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
            Text(ticker.symbol)
                .font(.system(size: 12, weight: .bold, design: .rounded))
            if let quote {
                Text(quote.priceText(compact: compact))
                    .font(.system(size: 11, weight: .medium, design: .rounded).monospacedDigit())
                    .foregroundStyle(.secondary)
                Text(quote.percentText)
                    .font(.system(size: 11, weight: .semibold, design: .rounded).monospacedDigit())
                    .foregroundStyle(quote.isPositive ? gainColor : lossColor)
            } else {
                Text("—").foregroundStyle(.secondary)
            }
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Remove \(ticker.symbol)")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.primary.opacity(0.08), in: Capsule())
        .contentShape(Capsule())
    }
}

private extension NSColor {
    var sRGB: NSColor? { usingColorSpace(.sRGB) }

    /// Loose equality so a swatch shows its checkmark after a round trip through UserDefaults.
    func matches(_ other: NSColor) -> Bool {
        guard let a = sRGB, let b = other.sRGB else { return self == other }
        return abs(a.redComponent - b.redComponent) < 0.02
            && abs(a.greenComponent - b.greenComponent) < 0.02
            && abs(a.blueComponent - b.blueComponent) < 0.02
    }

    var isBright: Bool {
        guard let c = sRGB else { return false }
        return 0.299 * c.redComponent + 0.587 * c.greenComponent + 0.114 * c.blueComponent > 0.65
    }
}

// MARK: - Glass background

private extension View {
    @ViewBuilder
    func glassBackground(cornerRadius: CGFloat) -> some View {
        if #available(macOS 26, *) {
            self.glassEffect(.regular, in: .rect(cornerRadius: cornerRadius))
        } else {
            self
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .strokeBorder(.white.opacity(0.18), lineWidth: 1)
                )
        }
    }
}

// MARK: - Flow layout

/// Wraps children onto new lines when they run out of horizontal room.
struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        arrange(proposal: proposal, subviews: subviews).size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let positions = arrange(proposal: proposal, subviews: subviews).positions
        for (subview, position) in zip(subviews, positions) {
            subview.place(
                at: CGPoint(x: bounds.minX + position.x, y: bounds.minY + position.y),
                proposal: .unspecified
            )
        }
    }

    private func arrange(proposal: ProposedViewSize, subviews: Subviews) -> (size: CGSize, positions: [CGPoint]) {
        let maxWidth = proposal.width ?? .infinity
        var positions: [CGPoint] = []
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0, totalWidth: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0 && x + size.width > maxWidth {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            positions.append(CGPoint(x: x, y: y))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
            totalWidth = max(totalWidth, x - spacing)
        }
        return (CGSize(width: totalWidth, height: y + rowHeight), positions)
    }
}
