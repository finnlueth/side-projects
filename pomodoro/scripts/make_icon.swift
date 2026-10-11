// Renders the app icon (a glass timer dial with a tomato stem on a red squircle) to Resources/AppIcon.icns.
// Run via scripts/make_icon.sh. Drawn in SwiftUI so the squircle uses Apple's continuous corner curve.
import AppKit
import SwiftUI

// MARK: - Design

private let canvas: CGFloat = 1024          // macOS icon grid
private let tile: CGFloat = 824             // the squircle itself; the margin holds the drop shadow
private let tileCornerRadius: CGFloat = 185.4   // Apple's ratio for an 824pt tile

private let tomatoTop = Color(red: 1.00, green: 0.42, blue: 0.36)
private let tomatoBottom = Color(red: 0.82, green: 0.15, blue: 0.20)
private let leaf = Color(red: 0.40, green: 0.80, blue: 0.36)

private let dial: CGFloat = 560
private let ringWidth: CGFloat = 64
private let progress: CGFloat = 0.68

private struct Squircle: InsettableShape {
    var insetAmount: CGFloat = 0

    func inset(by amount: CGFloat) -> Squircle { Squircle(insetAmount: insetAmount + amount) }

    func path(in rect: CGRect) -> Path {
        RoundedRectangle(cornerRadius: tileCornerRadius - insetAmount, style: .continuous)
            .path(in: rect.insetBy(dx: insetAmount, dy: insetAmount))
    }
}

/// A pointed leaf, tip at the top of its rect.
private struct Leaf: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: rect.midX, y: rect.minY))
        p.addQuadCurve(to: CGPoint(x: rect.midX, y: rect.maxY), control: CGPoint(x: rect.maxX, y: rect.midY))
        p.addQuadCurve(to: CGPoint(x: rect.midX, y: rect.minY), control: CGPoint(x: rect.minX, y: rect.midY))
        return p
    }
}

private struct Stem: View {
    var body: some View {
        ZStack {
            // Calyx: leaves splay sideways and droop over the dial, like the top of a tomato.
            ForEach([-118.0, -72, 72, 118], id: \.self) { angle in
                Leaf()
                    .fill(LinearGradient(colors: [leaf, leaf.opacity(0.85)], startPoint: .top, endPoint: .bottom))
                    .frame(width: 56, height: 118)
                    .offset(y: -50)
                    .rotationEffect(.degrees(angle))
            }
            // Short stalk
            Capsule()
                .fill(leaf)
                .frame(width: 30, height: 84)
                .offset(y: -38)
                .rotationEffect(.degrees(12), anchor: .bottom)
            Circle().fill(leaf).frame(width: 44, height: 44)
        }
        .shadow(color: .black.opacity(0.25), radius: 10, y: 6)
    }
}

private struct IconView: View {
    var body: some View {
        ZStack {
            // Tile
            Squircle()
                .fill(LinearGradient(colors: [tomatoTop, tomatoBottom], startPoint: .top, endPoint: .bottom))
                .overlay(
                    Squircle().fill(
                        RadialGradient(colors: [.white.opacity(0.22), .clear],
                                       center: UnitPoint(x: 0.5, y: -0.1), startRadius: 0, endRadius: 720)
                    )
                )
                .overlay(Squircle().strokeBorder(.white.opacity(0.18), lineWidth: 5))
                .shadow(color: .black.opacity(0.3), radius: 16, y: 14)

            // Frosted glass dial
            Circle()
                .fill(LinearGradient(colors: [.white.opacity(0.30), .white.opacity(0.10)],
                                     startPoint: .topLeading, endPoint: .bottomTrailing))
                .overlay(
                    Circle().strokeBorder(
                        LinearGradient(colors: [.white.opacity(0.75), .white.opacity(0.10), .white.opacity(0.35)],
                                       startPoint: .topLeading, endPoint: .bottomTrailing),
                        lineWidth: 6)
                )
                .shadow(color: Color(red: 0.45, green: 0.02, blue: 0.08).opacity(0.35), radius: 30, y: 18)
                .frame(width: dial, height: dial)

            // Progress ring
            let ringDiameter = dial - 150
            Circle()
                .stroke(.white.opacity(0.22), lineWidth: ringWidth)
                .frame(width: ringDiameter, height: ringDiameter)
            Circle()
                .trim(from: 0, to: progress)
                .stroke(.white, style: StrokeStyle(lineWidth: ringWidth, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .frame(width: ringDiameter, height: ringDiameter)
                .shadow(color: .white.opacity(0.5), radius: 18)

            // Tomato stem sitting on top of the dial
            Stem().offset(y: -dial / 2 - 6)
        }
        .frame(width: tile, height: tile)
        .frame(width: canvas, height: canvas)
    }
}

// MARK: - Export

@MainActor
func render(pixels: Int) -> CGImage {
    let renderer = ImageRenderer(content: IconView())
    renderer.scale = CGFloat(pixels) / canvas
    renderer.isOpaque = false
    guard let image = renderer.cgImage else { fatalError("render failed at \(pixels)px") }
    return image
}

func writePNG(_ image: CGImage, to url: URL) throws {
    let rep = NSBitmapImageRep(cgImage: image)
    guard let data = rep.representation(using: .png, properties: [:]) else { fatalError("png encode failed") }
    try data.write(to: url)
}

@main
struct MakeIcon {
    @MainActor
    static func main() throws {
        let outputDir = URL(fileURLWithPath: CommandLine.arguments.dropFirst().first ?? ".")
        let iconset = outputDir.appendingPathComponent("AppIcon.iconset")
        try? FileManager.default.removeItem(at: iconset)
        try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)

        _ = NSApplication.shared
        for base in [16, 32, 128, 256, 512] {
            try writePNG(render(pixels: base), to: iconset.appendingPathComponent("icon_\(base)x\(base).png"))
            try writePNG(render(pixels: base * 2), to: iconset.appendingPathComponent("icon_\(base)x\(base)@2x.png"))
        }
        // Full-size preview for the README / eyeballing
        try writePNG(render(pixels: 1024), to: outputDir.appendingPathComponent("AppIcon-preview.png"))
        print("wrote \(iconset.path)")
    }
}
