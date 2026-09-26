import AppKit

/// Draws the two-row ticker inside the status item button.
///
///   SPCX 2.60%   SPY 1.13%
///   $154.81 $3.93  $762.60 $8.55
///
/// Text colours resolve through the button's effective appearance, so they flip between dark and light
/// automatically as the menu bar changes with the wallpaper / system appearance.
final class TickerBarView: NSView {
    struct Entry {
        let ticker: Ticker
        let quote: Quote?
    }

    var entries: [Entry] = [] { didSet { needsDisplay = true } }
    /// System colours by default, which also adapt to the bar's light/dark appearance on their own.
    var gainColor: NSColor = .systemGreen { didSet { needsDisplay = true } }
    var lossColor: NSColor = .systemRed { didSet { needsDisplay = true } }
    var showsPercentChange = true { didSet { needsDisplay = true } }
    var showsDollarChange = true { didSet { needsDisplay = true } }
    var compactLargeAmounts = true { didSet { needsDisplay = true } }
    /// Called when the bar height changes (e.g. 22pt → 38pt once a menu bar manager reveals the item on a
    /// notch Mac) so the owner can re-measure `preferredWidth`, which depends on the font size.
    var onHeightChange: (() -> Void)?

    /// The real menu bar height: the button autoresizes with its window, so the bounds tell the truth
    /// (38pt on notch Macs, 22pt otherwise). Drives font size and row placement.
    private var barHeight: CGFloat {
        bounds.height > 0 ? bounds.height : NSStatusBar.system.thickness
    }

    private enum Metrics {
        static let horizontalPadding: CGFloat = 5
        static let segmentGap: CGFloat = 4
        static let blockSpacing: CGFloat = 9
        static let rowCenters: (top: CGFloat, bottom: CGFloat) = (0.26, 0.74)
    }

    override var isFlipped: Bool { true }

    /// Let every click fall through to the status bar button so highlighting and the action still work.
    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        needsDisplay = true
    }

    override func setFrameSize(_ newSize: NSSize) {
        let heightChanged = newSize.height != frame.height
        super.setFrameSize(newSize)
        if heightChanged {
            needsDisplay = true
            onHeightChange?()
        }
    }

    // MARK: Layout

    private struct Line {
        let segments: [NSAttributedString]
        var width: CGFloat {
            segments.reduce(0) { $0 + $1.size().width } + Metrics.segmentGap * CGFloat(max(0, segments.count - 1))
        }
    }

    private struct Block {
        let top: Line
        let bottom: Line
        var width: CGFloat { max(top.width, bottom.width) }
    }

    private var fontSize: CGFloat {
        // ≈12.5pt on a 38pt bar, clamped so a 22pt bar still fits two rows.
        min(13, max(9, (barHeight / 2) * 0.66))
    }

    private var font: NSFont { .systemFont(ofSize: fontSize, weight: .semibold) }

    var preferredWidth: CGFloat {
        let blocks = makeBlocks()
        guard !blocks.isEmpty else { return 0 }
        let content = blocks.reduce(0) { $0 + $1.width } + Metrics.blockSpacing * CGFloat(blocks.count - 1)
        return ceil(content + Metrics.horizontalPadding * 2)
    }

    private func makeBlocks() -> [Block] {
        let font = self.font
        func text(_ string: String, _ color: NSColor) -> NSAttributedString {
            NSAttributedString(string: string, attributes: [.font: font, .foregroundColor: color])
        }
        return entries.map { entry in
            guard let quote = entry.quote else {
                let placeholder = text("—", .secondaryLabelColor)
                return Block(
                    top: Line(segments: [text(entry.ticker.symbol, .labelColor)] + (showsPercentChange ? [placeholder] : [])),
                    bottom: Line(segments: [placeholder])
                )
            }
            let tint = quote.isPositive ? gainColor : lossColor
            return Block(
                top: Line(segments: [text(entry.ticker.symbol, .labelColor)]
                    + (showsPercentChange ? [text(quote.percentText, tint)] : [])),
                bottom: Line(segments: [text(quote.priceText(compact: compactLargeAmounts), .labelColor)]
                    + (showsDollarChange ? [text(quote.changeText(compact: compactLargeAmounts), tint)] : []))
            )
        }
    }

    // MARK: Drawing

    override func draw(_ dirtyRect: NSRect) {
        let blocks = makeBlocks()
        var x = Metrics.horizontalPadding
        for block in blocks {
            draw(block.top, x: x, centerY: barHeight * Metrics.rowCenters.top)
            draw(block.bottom, x: x, centerY: barHeight * Metrics.rowCenters.bottom)
            x += block.width + Metrics.blockSpacing
        }
    }

    /// Places each segment so its cap height is centred on `centerY` (digits and capitals share that box).
    private func draw(_ line: Line, x: CGFloat, centerY: CGFloat) {
        var cursor = x
        for segment in line.segments {
            let size = segment.size()
            let baseline = centerY + font.capHeight / 2
            segment.draw(at: NSPoint(x: cursor, y: baseline - font.ascender))
            cursor += size.width + Metrics.segmentGap
        }
    }
}
