"""Render the downloaded Bad Apple MIDI as a piezo-buzzer demo (full length).

Source MIDI: github.com/Handhule90/badapple-midi, community transcription,
original composition ZUN / Touhou Project. Two melodic tracks are rendered:
Synth1 and Vocals. Drums/bass/percussion are intentionally omitted because
stacking all 13 tracks on one buzzers produced an unreadable result.
"""

from __future__ import annotations

import wave
from pathlib import Path

import mido
import numpy as np

RATE = 48_000
START_SECONDS = 27.0
MAX_SECONDS = 218.0 - START_SECONDS
TRACKS = ("Synth1", "Vocals")
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
        if not any(name in track.name for name in TRACKS):
            continue
        tick = 0
        for message in track:
            tick += message.time
            if message.type in ("note_on", "note_off"):
                seconds = tick * tempo / ppq / 1_000_000.0
                is_on = message.type == "note_on" and message.velocity > 0
                result.append((seconds, message.note, is_on))
    return sorted(result)


def render() -> np.ndarray:
    out = np.zeros(int(MAX_SECONDS * RATE), dtype=np.float32)
    active: dict[int, tuple[float, float]] = {}

    for raw_time, note, is_on in events(MIDI):
        time = raw_time - START_SECONDS
        if time < 0.0:
            continue
        if time >= MAX_SECONDS:
            break
        index = int(time * RATE)
        if is_on:
            active[note] = (time, note_freq(note))
            continue
        if note not in active:
            continue
        start, freq = active.pop(note)
        count = int((time - start) * RATE)
        if count <= 0 or index >= len(out):
            continue
        t = np.arange(count, dtype=np.float32) / RATE
        phase = np.mod(t * freq, 1.0)
        wave_form = np.where(phase < 0.28, 1.0, -1.0)
        env = np.minimum(1.0, t * 90.0) * np.minimum(1.0, (time - start - t) * 90.0)
        end = min(index + count, len(out))
        out[index:end] += (0.22 * env * wave_form[: end - index]).astype(np.float32)

    audio = 0.85 * out[: int(MAX_SECONDS * RATE)]
    return np.clip(audio, -1.0, 1.0)


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
