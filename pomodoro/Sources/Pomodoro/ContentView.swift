import SwiftUI

/// One of nine snap positions: top/middle/bottom × left/center/right.
struct Anchor: Equatable {
    enum Horizontal: String, CaseIterable { case left, center, right }
    enum Vertical: String, CaseIterable { case top, middle, bottom }

    var horizontal: Horizontal
    var vertical: Vertical

    var rawValue: String { "\(vertical.rawValue)-\(horizontal.rawValue)" }

    init(horizontal: Horizontal, vertical: Vertical) {
        self.horizontal = horizontal
        self.vertical = vertical
    }

    init?(rawValue: String) {
        let parts = rawValue.split(separator: "-").map(String.init)
        guard parts.count == 2,
              let v = Vertical(rawValue: parts[0]),
              let h = Horizontal(rawValue: parts[1]) else { return nil }
        self.init(horizontal: h, vertical: v)
    }

    /// Where the pill sits inside the (wider) transparent window, so hover-expansion grows away from the screen edge.
    var alignment: Alignment {
        switch horizontal {
        case .left: .leading
        case .center: .center
        case .right: .trailing
        }
    }
}

final class WindowState: ObservableObject {
    @Published var isOverlay: Bool {
        didSet { UserDefaults.standard.set(isOverlay, forKey: "isOverlay") }
    }
    @Published var anchor: Anchor {
        didSet { UserDefaults.standard.set(anchor.rawValue, forKey: "anchor") }
    }
    @Published var alwaysShowControls: Bool {
        didSet { UserDefaults.standard.set(alwaysShowControls, forKey: "alwaysShowControls") }
    }
    @Published var showCycleDots: Bool {
        didSet { UserDefaults.standard.set(showCycleDots, forKey: "showCycleDots") }
    }
    @Published var showDockBadge: Bool {
        didSet { UserDefaults.standard.set(showDockBadge, forKey: "showDockBadge") }
    }
    @Published var showInDock: Bool {
        didSet { UserDefaults.standard.set(showInDock, forKey: "showInDock") }
    }
    /// Measured width of the pill, needed to keep it visually still when the anchor (and so its alignment) changes.
    @Published var pillWidth: CGFloat = 0

    init() {
        isOverlay = UserDefaults.standard.object(forKey: "isOverlay") as? Bool ?? true
        anchor = UserDefaults.standard.string(forKey: "anchor").flatMap(Anchor.init) ?? Anchor(horizontal: .right, vertical: .top)
        alwaysShowControls = UserDefaults.standard.object(forKey: "alwaysShowControls") as? Bool ?? false
        showCycleDots = UserDefaults.standard.object(forKey: "showCycleDots") as? Bool ?? true
        showDockBadge = UserDefaults.standard.object(forKey: "showDockBadge") as? Bool ?? true
        showInDock = UserDefaults.standard.object(forKey: "showInDock") as? Bool ?? true
    }
}

enum Layout {
    /// Transparent margin around the pill so its shadow isn't clipped by the window edge.
    static let shadowPadding: CGFloat = 20
    /// Wide enough for the expanded pill, including the maximum of 8 cycle dots.
    static let windowContentWidth: CGFloat = 330
}

private struct PillWidthKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

struct ContentView: View {
    @ObservedObject var timer: PomodoroTimer
    @ObservedObject var windowState: WindowState
    @ObservedObject var hover: HoverState

    private var tint: Color {
        switch timer.phase {
        case .focus: .red
        case .shortBreak: .green
        case .longBreak: .blue
        }
    }

    var body: some View {
        pill
            .frame(width: Layout.windowContentWidth, alignment: windowState.anchor.alignment)
            .padding(Layout.shadowPadding)
            .background(spaceShortcut)
    }

    private var showsControls: Bool { hover.isHovering || windowState.alwaysShowControls }

    private var pill: some View {
        HStack(spacing: 8) {
            progressRing
            Text(timer.formattedRemaining)
                .font(.system(size: 15, weight: .semibold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(timer.isRunning ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                .contentTransition(.numericText(countsDown: true))
                .animation(.snappy, value: timer.formattedRemaining)
            if showsControls {
                if windowState.showCycleDots {
                    cycleDots
                        .transition(.opacity)
                }
                controls
                    .transition(.opacity.combined(with: .scale(scale: 0.8, anchor: .leading)))
            }
        }
        .padding(.leading, 11)
        .padding(.trailing, showsControls ? 6 : 13)
        .frame(height: 34)
        // Controls are laid out at their final position while the capsule's width animates;
        // clip so they never show outside the glass mid-expand/collapse.
        .clipShape(.capsule)
        .glassEffect(.regular.tint(tint.opacity(0.08)), in: .capsule)
        .shadow(color: .black.opacity(0.15), radius: 6, y: 2)
        .background(GeometryReader { Color.clear.preference(key: PillWidthKey.self, value: $0.size.width) })
        .onPreferenceChange(PillWidthKey.self) { [windowState] in windowState.pillWidth = $0 }
        .onHover { [hover] in hover.isHovering = $0 }
        .contextMenu { settingsMenu }
        .animation(.smooth(duration: 0.25), value: showsControls)
        .animation(.smooth, value: timer.phase)
        .help("\(timer.phase.title) · session \(min(timer.completedSessions + 1, timer.sessionsPerCycle)) of \(timer.sessionsPerCycle)")
    }

    /// One dot per focus session in the cycle: done, current (while focusing), upcoming.
    private var cycleDots: some View {
        HStack(spacing: 4) {
            ForEach(0..<timer.sessionsPerCycle, id: \.self) { i in
                let isDone = i < timer.completedSessions
                let isCurrent = i == timer.completedSessions && timer.phase == .focus
                Circle()
                    .fill(isDone ? AnyShapeStyle(tint) : isCurrent ? AnyShapeStyle(tint.opacity(0.45)) : AnyShapeStyle(.quaternary))
                    .frame(width: 5, height: 5)
            }
        }
        .padding(.leading, 2)
    }

    private var progressRing: some View {
        ZStack {
            Circle().stroke(.quaternary, lineWidth: 2.5)
            Circle()
                .trim(from: 0, to: timer.progress)
                .stroke(tint, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.linear(duration: 0.25), value: timer.progress)
        }
        .frame(width: 13, height: 13)
    }

    private var controls: some View {
        HStack(spacing: 0) {
            iconButton(timer.isRunning ? "pause.fill" : "play.fill", help: timer.isRunning ? "Pause" : "Start",
                       color: tint, action: timer.toggle)
            iconButton("arrow.counterclockwise", help: "Reset", action: timer.reset)
            iconButton("forward.end.fill", help: "Skip to next phase", action: timer.skip)
            iconButton(windowState.isOverlay ? "pin.fill" : "pin",
                       help: windowState.isOverlay ? "Stop floating above other windows" : "Float above other windows",
                       color: windowState.isOverlay ? tint : nil) {
                windowState.isOverlay.toggle()
            }
            iconButton("gearshape.fill", help: "Settings", action: AppDelegate.showSettings)
        }
    }

    private func iconButton(_ symbol: String, help: String, color: Color? = nil, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(color.map(AnyShapeStyle.init) ?? AnyShapeStyle(.secondary))
                .frame(width: 24, height: 24)
                .contentShape(.circle)
        }
        .buttonStyle(.plain)
        .help(help)
    }

    /// Invisible button so Space toggles the timer even while the controls are collapsed.
    private var spaceShortcut: some View {
        Button("", action: timer.toggle)
            .keyboardShortcut(.space, modifiers: [])
            .opacity(0)
            .allowsHitTesting(false)
    }

    @ViewBuilder private var settingsMenu: some View {
        Button("Settings…", action: AppDelegate.showSettings)
            .keyboardShortcut(",")
        Divider()
        Toggle("Float Above Other Windows", isOn: $windowState.isOverlay)
        Toggle("Play Sounds", isOn: $timer.soundsEnabled)
        Divider()
        Button("Quit Pomodoro") { NSApp.terminate(nil) }
    }
}

final class HoverState: ObservableObject {
    @Published var isHovering = false
}
