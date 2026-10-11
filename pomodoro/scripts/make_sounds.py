#!/usr/bin/env python3
"""Synthesizes the phase-completion chimes into Resources/Sounds/ (stdlib only).

Each note is a small additive "glass bell": a few partials, including a slightly
inharmonic one for shimmer, with a soft attack and exponential decay.
"""
import math
import os
import struct
import wave

RATE = 44100
OUT = os.path.join(os.path.dirname(__file__), "..", "Resources", "Sounds")

# (frequency ratio, relative amplitude, decay seconds) — the 2.76 partial gives the glassy bell colour.
PARTIALS = [(1.0, 1.0, 1.1), (2.0, 0.35, 0.6), (2.76, 0.18, 0.35), (4.07, 0.06, 0.2)]


def note_freq(name):
    names = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}
    semis = names[name[0]] + (1 if "#" in name else 0) + (int(name[-1]) - 4) * 12
    return 440.0 * 2 ** (semis / 12)


def render(notes, length, gain=0.5):
    """notes: (start seconds, note name, velocity). Returns stereo float frames."""
    n = int(length * RATE)
    left, right = [0.0] * n, [0.0] * n
    for i, (start, name, vel) in enumerate(notes):
        f = note_freq(name)
        pan = 0.5 + 0.18 * math.sin(i * 1.7)  # spread successive notes slightly
        s0 = int(start * RATE)
        for k in range(s0, n):
            t = (k - s0) / RATE
            attack = min(1.0, t / 0.006)
            v = sum(a * math.exp(-t / d) * math.sin(2 * math.pi * f * r * t) for r, a, d in PARTIALS)
            v *= attack * vel
            left[k] += v * (1 - pan) * 2
            right[k] += v * pan * 2
    peak = max(max(map(abs, left)), max(map(abs, right))) or 1
    scale = gain / peak
    # Short fade-out so the file never ends on a click.
    fade = int(0.08 * RATE)
    for k in range(n - fade, n):
        w = (n - k) / fade
        left[k] *= w
        right[k] *= w
    return [(l * scale, r * scale) for l, r in zip(left, right)]


def write(name, frames):
    path = os.path.join(OUT, name)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<hh", int(l * 32767), int(r * 32767)) for l, r in frames))
    print("wrote", os.path.normpath(path))


def main():
    os.makedirs(OUT, exist_ok=True)
    # Focus finished → short break: bright, rising, a little lift.
    write("focus-complete.wav", render([(0.0, "E5", 0.8), (0.12, "G#5", 0.85), (0.24, "B5", 1.0)], 1.6))
    # Break finished → back to focus: calm two-note "ding-dong", lower and gentler.
    write("break-complete.wav", render([(0.0, "G5", 0.9), (0.28, "C5", 1.0)], 1.8, gain=0.45))
    # Fourth focus finished → long break: celebratory arpeggio ending on a held chord.
    write("cycle-complete.wav", render(
        [(0.0, "C5", 0.7), (0.1, "E5", 0.75), (0.2, "G5", 0.8), (0.3, "C6", 0.9),
         (0.5, "E6", 0.6), (0.5, "G5", 0.5), (0.5, "C6", 0.7)], 2.4))


if __name__ == "__main__":
    main()
