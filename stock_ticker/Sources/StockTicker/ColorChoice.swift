import AppKit

enum ColorRole: CaseIterable {
    case gain, loss

    var title: String { self == .gain ? "Gain" : "Loss" }
}

/// How a gain/loss tint is chosen.
enum ColorChoice: Equatable, Codable {
    /// `NSColor.systemGreen` / `.systemRed`.
    case system
    /// A high-contrast pair that flips with the menu bar's appearance: vivid on dark bars, deep on light ones.
    /// The menu bar picks its appearance from the wallpaper behind it, so this tracks the actual background.
    case dynamic
    /// A fixed sRGB colour.
    case custom(red: Double, green: Double, blue: Double)

    static func custom(_ color: NSColor) -> ColorChoice {
        let c = color.usingColorSpace(.sRGB) ?? color
        return .custom(red: c.redComponent, green: c.greenComponent, blue: c.blueComponent)
    }

    func color(for role: ColorRole) -> NSColor {
        switch self {
        case .system:
            return role == .gain ? .systemGreen : .systemRed
        case .dynamic:
            return NSColor(name: nil) { appearance in
                let dark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
                switch role {
                case .gain:
                    return dark
                        ? NSColor(srgbRed: 0.46, green: 0.84, blue: 0.29, alpha: 1)
                        : NSColor(srgbRed: 0.09, green: 0.52, blue: 0.14, alpha: 1)
                case .loss:
                    return dark
                        ? NSColor(srgbRed: 1.00, green: 0.43, blue: 0.38, alpha: 1)
                        : NSColor(srgbRed: 0.82, green: 0.14, blue: 0.14, alpha: 1)
                }
            }
        case .custom(let red, let green, let blue):
            return NSColor(srgbRed: red, green: green, blue: blue, alpha: 1)
        }
    }

    var isCustom: Bool {
        if case .custom = self { return true }
        return false
    }
}
