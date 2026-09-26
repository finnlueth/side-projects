import AppKit
import Combine

enum Phase: String {
    case focus, shortBreak, longBreak

    var title: String {
        switch self {
        case .focus: "Focus"
        case .shortBreak: "Short Break"
        case .longBreak: "Long Break"
        }
    }
}

final class PomodoroTimer: ObservableObject {
    static let sessionsPerCycle = 4

    @Published private(set) var phase: Phase = .focus
    @Published private(set) var remaining: TimeInterval
    @Published private(set) var isRunning = false
    /// Focus sessions completed in the current cycle (0..<sessionsPerCycle).
    @Published private(set) var completedSessions = 0

    @Published var focusMinutes: Int { didSet { settingsChanged(\.focusMinutes, .focus) } }
    @Published var shortBreakMinutes: Int { didSet { settingsChanged(\.shortBreakMinutes, .shortBreak) } }
    @Published var longBreakMinutes: Int { didSet { settingsChanged(\.longBreakMinutes, .longBreak) } }

    // Counting down against an end date keeps the timer accurate across sleep and App Nap.
    private var endDate: Date?
    private var ticker: Timer?
    private let defaults = UserDefaults.standard

    init() {
        focusMinutes = defaults.object(forKey: "focusMinutes") as? Int ?? 25
        shortBreakMinutes = defaults.object(forKey: "shortBreakMinutes") as? Int ?? 5
        longBreakMinutes = defaults.object(forKey: "longBreakMinutes") as? Int ?? 15
        remaining = TimeInterval((defaults.object(forKey: "focusMinutes") as? Int ?? 25) * 60)
    }

    var duration: TimeInterval {
        let minutes = switch phase {
        case .focus: focusMinutes
        case .shortBreak: shortBreakMinutes
        case .longBreak: longBreakMinutes
        }
        return TimeInterval(minutes * 60)
    }

    var progress: Double { duration > 0 ? 1 - remaining / duration : 0 }

    var formattedRemaining: String {
        let seconds = Int(remaining.rounded(.up))
        return String(format: "%02d:%02d", seconds / 60, seconds % 60)
    }

    func toggle() { isRunning ? pause() : start() }

    func start() {
        guard !isRunning else { return }
        endDate = Date().addingTimeInterval(remaining)
        isRunning = true
        let timer = Timer(timeInterval: 0.25, repeats: true) { [weak self] _ in self?.tick() }
        RunLoop.main.add(timer, forMode: .common)
        ticker = timer
    }

    func pause() {
        guard isRunning else { return }
        tick()
        stopTicker()
    }

    func reset() {
        stopTicker()
        remaining = duration
    }

    /// Jumps to the next phase without playing the completion chime.
    func skip() { advance(chime: false) }

    private func tick() {
        guard let endDate else { return }
        remaining = max(0, endDate.timeIntervalSinceNow)
        if remaining == 0 { advance(chime: true) }
    }

    private func advance(chime: Bool) {
        let wasRunning = isRunning
        stopTicker()
        if phase == .focus {
            completedSessions += 1
            if completedSessions >= Self.sessionsPerCycle {
                completedSessions = 0
                phase = .longBreak
            } else {
                phase = .shortBreak
            }
        } else {
            phase = .focus
        }
        remaining = duration
        if chime {
            NSSound(named: "Glass")?.play()
            NSApp.requestUserAttention(.informationalRequest)
        }
        // Keep the flow going after a natural finish; a manual skip keeps the previous run state.
        if chime || wasRunning { start() }
    }

    private func stopTicker() {
        ticker?.invalidate()
        ticker = nil
        endDate = nil
        isRunning = false
    }

    private func settingsChanged(_ key: KeyPath<PomodoroTimer, Int>, _ affected: Phase) {
        let name = switch affected {
        case .focus: "focusMinutes"
        case .shortBreak: "shortBreakMinutes"
        case .longBreak: "longBreakMinutes"
        }
        defaults.set(self[keyPath: key], forKey: name)
        if phase == affected && !isRunning { remaining = duration }
    }
}
