"""Render the downloaded Bad Apple MIDI as a monophonic piezo-buzzer demo.

Source MIDI: github.com/Handhule90/badapple-midi, community transcription,
original composition ZUN / Touhou Project. Only the Synth1 melody track is
used; overlapping tracks and retriggered notes caused the previous versions
to stack notes on top of each other.
"""

from __future__ import annotations

import wave
from pathlib import Path

import mido
import numpy as np

RATE = 48_000
START_SECONDS = 27.0
MAX_SECONDS = 218.0 - START_SECONDS
TRACK_NAME = "Synth1"
MIDI = Path(__file__).with_name("BadApple.mid")
OUT = Path(__file__).with_name("badapple_buzzer.wav")


def note_freq(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def events(path: Path) -> list[tuple[float, int, bool]]:
    mid = mido.MidiFile(str(path))
    tempo = 500_000
    ppq = mid.ticks_per_beat
    for track in mid.tracks:
        for message in track:
            if message.type == "set_tempo":
                tempo = message.tempo
    result: list[tuple[float, int, bool]] = []
    for track in mid.tracks:
        if TRACK_NAME not in track.name:
            continue
        tick = 0
        for message in track:
            tick += message.time
            if message.type in ("note_on", "note_off"):
                seconds = tick * tempo / ppq / 1_000_000.0
                result.append((seconds, message.note, message.type == "note_on" and message.velocity > 0))
    return sorted(result)


def render() -> np.ndarray:
    out = np.zeros(int(MAX_SECONDS * RATE), dtype=np.float32)
    active: dict[int, float] = {}
    max_poly = 0

    def emit(start: float, end: float, note: int) -> None:
        duration = end - start
        if duration <= 0.0 or start >= MAX_SECONDS:
            return
        index = int(start * RATE)
        count = min(int(duration * RATE), len(out) - index)
        if count <= 0:
            return
        t = np.arange(count, dtype=np.float32) / RATE
        phase = np.mod(t * note_freq(note), 1.0)
        wave_form = np.where(phase < 0.28, 1.0, -1.0)
        env = np.minimum(1.0, t * 90.0) * np.minimum(1.0, (duration - t) * 90.0)
        out[index : index + count] += (0.34 * env * wave_form).astype(np.float32)

    for raw_time, note, is_on in events(MIDI):
        time = raw_time - START_SECONDS
        if time >= MAX_SECONDS:
            break
        if is_on:
            # Force monophony: close every previous note before opening a new one.
            for old_note, start in list(active.items()):
                emit(start, time, old_note)
                active.pop(old_note)
            if time >= 0.0:
                active[note] = max(time, 0.0)
            max_poly = max(max_poly, len(active))
        elif note in active:
            start = active.pop(note)
            emit(start, time, note)

    for note, start in active.items():
        emit(start, MAX_SECONDS, note)

    print(f"notes from {TRACK_NAME}, max polyphony {max_poly}")
    return np.clip(0.85 * out, -1.0, 1.0)


def main() -> None:
    audio = render()
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(OUT), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm.tobytes())
    print(f"wrote {OUT} ({len(audio) / RATE:.1f} s)")


if __name__ == "__main__":
    main()
