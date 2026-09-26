// Renders the app icon (a rising ticker line on a dark squircle) to Resources/AppIcon.icns.
// Run via scripts/make_icon.sh. Drawn in SwiftUI so the squircle uses Apple's continuous corner curve.
import AppKit
import SwiftUI

// MARK: - Design

private let canvas: CGFloat = 1024          // macOS icon grid
private let tile: CGFloat = 824             // the squircle itself; the margin holds the drop shadow
private let tileCornerRadius: CGFloat = 185.4   // Apple's ratio for an 824pt tile

private let lime = Color(red: 0.46, green: 0.84, blue: 0.29)
private let navyTop = Color(red: 0.13, green: 0.19, blue: 0.33)
private let navyBottom = Color(red: 0.04, green: 0.06, blue: 0.13)

/// Chart control points in unit space (y grows downward), a plausible "up and to the right" run.
private let points: [CGPoint] = [
    CGPoint(x: 0.09, y: 0.74),
    CGPoint(x: 0.24, y: 0.58),
    CGPoint(x: 0.36, y: 0.66),
    CGPoint(x: 0.50, y: 0.40),
    CGPoint(x: 0.62, y: 0.49),
    CGPoint(x: 0.76, y: 0.26),
    CGPoint(x: 0.91, y: 0.19),
]

private struct Squircle: InsettableShape {
    var insetAmount: CGFloat = 0

    func inset(by amount: CGFloat) -> Squircle { Squircle(insetAmount: insetAmount + amount) }

    func path(in rect: CGRect) -> Path {
        RoundedRectangle(cornerRadius: tileCornerRadius - insetAmount, style: .continuous)
            .path(in: rect.insetBy(dx: insetAmount, dy: insetAmount))
    }
}

private struct ChartLine: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        for (index, point) in points.enumerated() {
            let p = CGPoint(x: rect.minX + point.x * rect.width, y: rect.minY + point.y * rect.height)
            index == 0 ? path.move(to: p) : path.addLine(to: p)
        }
        return path
    }
}

private struct ChartArea: Shape {
    func path(in rect: CGRect) -> Path {
        var path = ChartLine().path(in: rect)
        let last = points.last!, first = points.first!
        path.addLine(to: CGPoint(x: rect.minX + last.x * rect.width, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX + first.x * rect.width, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}

private struct IconView: View {
    var body: some View {
        ZStack {
            // Tile
            Squircle()
                .fill(LinearGradient(colors: [navyTop, navyBottom], startPoint: .top, endPoint: .bottom))
                .overlay(
                    Squircle().fill(
                        RadialGradient(colors: [.white.opacity(0.14), .clear],
                                       center: UnitPoint(x: 0.5, y: -0.15), startRadius: 0, endRadius: 760)
                    )
                )
                .shadow(color: .black.opacity(0.35), radius: 16, y: 14)

            // Chart, clipped to the tile
            ZStack {
                // Faint grid
                VStack(spacing: 0) {
                    ForEach(0..<5) { _ in
                        Spacer()
                        Rectangle().fill(.white.opacity(0.07)).frame(height: 5)
                    }
                    Spacer()
                }
                .padding(.vertical, 40)

                ChartArea()
                    .fill(LinearGradient(colors: [lime.opacity(0.45), lime.opacity(0.0)],
                                         startPoint: .top, endPoint: .bottom))
                    // Soften the vertical edges where the area starts/ends.
                    .mask(
                        LinearGradient(stops: [
                            .init(color: .clear, location: points.first!.x),
                            .init(color: .black, location: points.first!.x + 0.14),
                            .init(color: .black, location: points.last!.x - 0.14),
                            .init(color: .clear, location: points.last!.x),
                        ], startPoint: .leading, endPoint: .trailing)
                    )

                ChartLine()
                    .stroke(lime, style: StrokeStyle(lineWidth: 58, lineCap: .round, lineJoin: .round))
                    .shadow(color: lime.opacity(0.55), radius: 26)

                // Latest print
                let end = points.last!
                Circle()
                    .fill(.white)
                    .frame(width: 92, height: 92)
                    .overlay(Circle().stroke(lime, lineWidth: 22))
                    .shadow(color: lime.opacity(0.7), radius: 30)
                    .position(x: end.x * tile, y: end.y * tile)
            }
            .frame(width: tile, height: tile)
            .clipShape(Squircle())
            .overlay(Squircle().strokeBorder(.white.opacity(0.10), lineWidth: 6))
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
