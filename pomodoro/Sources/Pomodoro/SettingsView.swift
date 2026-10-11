import SwiftUI

struct SettingsView: View {
    @ObservedObject var timer: PomodoroTimer
    @ObservedObject var windowState: WindowState

    var body: some View {
        Form {
            Section("Durations") {
                minutesStepper("Focus", value: $timer.focusMinutes, range: 1...120)
                minutesStepper("Short break", value: $timer.shortBreakMinutes, range: 1...60)
                minutesStepper("Long break", value: $timer.longBreakMinutes, range: 1...60)
                stepper("Sessions before long break", value: $timer.sessionsPerCycle, range: 2...8, unit: "")
            }

            Section {
                Toggle("Start breaks automatically", isOn: $timer.autoStartBreaks)
                Toggle("Start focus automatically after a break", isOn: $timer.autoStartFocus)
                LabeledContent("Current cycle") {
                    HStack {
                        Text("Session \(min(timer.completedSessions + 1, timer.sessionsPerCycle)) of \(timer.sessionsPerCycle)")
                            .foregroundStyle(.secondary)
                        Button("Restart Cycle", action: timer.resetCycle)
                    }
                }
            } header: {
                Text("Flow")
            } footer: {
                Text("When auto-start is off, the next phase waits for you to press play.")
                    .foregroundStyle(.secondary)
            }

            Section("Sounds") {
                Toggle("Play sounds when a phase ends", isOn: $timer.soundsEnabled)
                LabeledContent("Volume") {
                    HStack {
                        Image(systemName: "speaker.fill").foregroundStyle(.secondary)
                        Slider(value: $timer.soundVolume, in: 0...1)
                        Image(systemName: "speaker.wave.3.fill").foregroundStyle(.secondary)
                    }
                    .frame(width: 200)
                }
                .disabled(!timer.soundsEnabled)
                ForEach(Chime.allCases, id: \.self) { chime in
                    LabeledContent(chime.title) {
                        Button {
                            chime.play(volume: timer.soundVolume)
                        } label: {
                            Label("Preview", systemImage: "play.fill")
                        }
                    }
                }
            }

            Section("Window") {
                Toggle("Float above other windows", isOn: $windowState.isOverlay)
                Toggle("Always show controls", isOn: $windowState.alwaysShowControls)
                Toggle("Show cycle progress on hover", isOn: $windowState.showCycleDots)
                Toggle("Show countdown on Dock icon", isOn: $windowState.showDockBadge)
                    .disabled(!windowState.showInDock)
                Toggle("Show in Dock", isOn: $windowState.showInDock)
            }
        }
        .formStyle(.grouped)
        .frame(width: 440)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func minutesStepper(_ title: String, value: Binding<Int>, range: ClosedRange<Int>) -> some View {
        stepper(title, value: value, range: range, unit: "min")
    }

    /// Label on the left; an editable number field and stepper arrows on the right.
    private func stepper(_ title: String, value: Binding<Int>, range: ClosedRange<Int>, unit: String) -> some View {
        // Out-of-range typing is clamped rather than rejected.
        let clamped = Binding<Int>(
            get: { value.wrappedValue },
            set: { value.wrappedValue = min(max($0, range.lowerBound), range.upperBound) }
        )
        return LabeledContent(title) {
            HStack(spacing: 6) {
                TextField(title, value: clamped, format: .number)
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .monospacedDigit()
                    .frame(width: 48)
                // Unitless rows keep the space so all fields line up.
                Text(unit.isEmpty ? "min" : unit).foregroundStyle(.secondary).opacity(unit.isEmpty ? 0 : 1)
                Stepper(title, value: clamped, in: range).labelsHidden()
            }
        }
    }
}
