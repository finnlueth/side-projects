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
    @Published private(set) var phase: Phase = .focus
    @Published private(set) var remaining: TimeInterval
    @Published private(set) var isRunning = false
    /// Focus sessions completed in the current cycle (0..<sessionsPerCycle).
    @Published private(set) var completedSessions = 0

    @Published var focusMinutes: Int { didSet { settingsChanged(\.focusMinutes, .focus) } }
    @Published var shortBreakMinutes: Int { didSet { settingsChanged(\.shortBreakMinutes, .shortBreak) } }
    @Published var longBreakMinutes: Int { didSet { settingsChanged(\.longBreakMinutes, .longBreak) } }
    /// Focus sessions before a long break.
    @Published var sessionsPerCycle: Int {
        didSet {
            defaults.set(sessionsPerCycle, forKey: "sessionsPerCycle")
            completedSessions = min(completedSessions, sessionsPerCycle - 1)
        }
    }
    @Published var autoStartBreaks: Bool { didSet { defaults.set(autoStartBreaks, forKey: "autoStartBreaks") } }
    @Published var autoStartFocus: Bool { didSet { defaults.set(autoStartFocus, forKey: "autoStartFocus") } }
    @Published var soundsEnabled: Bool { didSet { defaults.set(soundsEnabled, forKey: "soundsEnabled") } }
    @Published var soundVolume: Double { didSet { defaults.set(soundVolume, forKey: "soundVolume") } }

    // Counting down against an end date keeps the timer accurate across sleep and App Nap.
    private var endDate: Date?
    private var ticker: Timer?
    private let defaults = UserDefaults.standard

    init() {
        focusMinutes = defaults.object(forKey: "focusMinutes") as? Int ?? 25
        shortBreakMinutes = defaults.object(forKey: "shortBreakMinutes") as? Int ?? 5
        longBreakMinutes = defaults.object(forKey: "longBreakMinutes") as? Int ?? 15
        sessionsPerCycle = defaults.object(forKey: "sessionsPerCycle") as? Int ?? 4
        autoStartBreaks = defaults.object(forKey: "autoStartBreaks") as? Bool ?? true
        autoStartFocus = defaults.object(forKey: "autoStartFocus") as? Bool ?? true
        soundsEnabled = defaults.object(forKey: "soundsEnabled") as? Bool ?? true
        soundVolume = defaults.object(forKey: "soundVolume") as? Double ?? 0.8
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

    /// Starts the cycle over: back to the first focus session, stopped.
    func resetCycle() {
        stopTicker()
        completedSessions = 0
        phase = .focus
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
            if completedSessions >= sessionsPerCycle {
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
            if soundsEnabled {
                switch phase {
                case .shortBreak: Chime.focusComplete.play(volume: soundVolume)
                case .longBreak: Chime.cycleComplete.play(volume: soundVolume)
                case .focus: Chime.breakComplete.play(volume: soundVolume)
                }
            }
            NSApp.requestUserAttention(.informationalRequest)
        }
        // After a natural finish, continue only if auto-start is on for the next phase;
        // a manual skip keeps the previous run state.
        let autoStart = phase == .focus ? autoStartFocus : autoStartBreaks
        if chime ? autoStart : wasRunning { start() }
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

/// Phase-completion sounds bundled in Contents/Resources/Sounds (generated by scripts/make_sounds.py).
enum Chime: String, CaseIterable {
    case focusComplete = "focus-complete"
    case breakComplete = "break-complete"
    case cycleComplete = "cycle-complete"

    private static var cache: [Chime: NSSound] = [:]

    var title: String {
        switch self {
        case .focusComplete: "Focus complete"
        case .breakComplete: "Break complete"
        case .cycleComplete: "Cycle complete"
        }
    }

    func play(volume: Double) {
        let sound = Self.cache[self] ?? {
            // Falls back to a system sound when run outside the .app bundle (e.g. `swift run`).
            let url = Bundle.main.url(forResource: rawValue, withExtension: "wav", subdirectory: "Sounds")
            let sound = url.flatMap { NSSound(contentsOf: $0, byReference: true) } ?? NSSound(named: "Glass")!
            Self.cache[self] = sound
            return sound
        }()
        sound.stop()
        sound.volume = Float(volume)
        sound.play()
    }
}
