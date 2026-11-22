"""Render the downloaded Bad Apple MIDI as a piezo-buzzer demo.

Source MIDI: github.com/Handhule90/badapple-midi (tempo 138, public community
transcription). Original composition: ZUN / Touhou Project.
"""

from __future__ import annotations

import wave
from pathlib import Path

import mido
import numpy as np

RATE = 48_000
MAX_SECONDS = 24.0
MIDI = Path(__file__).with_name("BadApple.mid")
OUT = Path(__file__).with_name("badapple_buzzer.wav")


def note_freq(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def events(path: Path) -> list[tuple[float, int, bool]]:
    mid = mido.MidiFile(str(path))
    tempo = 500_000
    ppq = mid.ticks_per_beat
    result: list[tuple[float, int, bool]] = []
    for track in mid.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                tempo = message.tempo
            if message.type in ("note_on", "note_off"):
                seconds = tick * tempo / ppq / 1_000_000.0
                is_on = message.type == "note_on" and message.velocity > 0
                result.append((seconds, message.note, is_on))
    return sorted(result)


def render() -> np.ndarray:
    notes = events(MIDI)
    active: dict[int, tuple[float, float]] = {}
    out = np.zeros(int((MAX_SECONDS + 1.0) * RATE), dtype=np.float32)
    for time, note, is_on in notes:
        index = int(time * RATE)
        if index >= len(out):
            break
        if is_on:
            active[note] = (time, note_freq(note))
        elif note in active:
            start, freq = active.pop(note)
            count = int((time - start) * RATE)
            if count <= 0:
                continue
            t = np.arange(count, dtype=np.float32) / RATE
            phase = np.mod(t * freq, 1.0)
            wave_form = np.where(phase < 0.28, 1.0, -1.0)
            env = np.minimum(1.0, t * 90.0) * np.minimum(1.0, (time - start - t) * 90.0)
            end = min(index, len(out)) + count
            out[index : min(end, len(out))] += (0.12 * env * wave_form).astype(np.float32)

    return np.clip(0.6 * out[: int(MAX_SECONDS * RATE)], -1.0, 1.0)


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
