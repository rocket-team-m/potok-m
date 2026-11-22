"""Render Bad Apple MIDI as a multi-track piezo-buzzer demo.

Source MIDI: github.com/Handhule90/badapple-midi, community transcription,
original composition ZUN / Touhou Project. Every track is rendered, but each
track keeps its own note table, so a note-off on one instrument can no longer
cut a different instrument's note or stack onto it.
"""

from __future__ import annotations

import wave
from pathlib import Path

import mido
import numpy as np

RATE = 48_000
MAX_SECONDS = 218.0
MIDI = Path(__file__).with_name("BadApple.mid")
OUT = Path(__file__).with_name("badapple_buzzer.wav")
RNG = np.random.default_rng(7)


def note_freq(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def tempo_of(mid: mido.MidiFile) -> int:
    tempo = 500_000
    for track in mid.tracks:
        for message in track:
            if message.type == "set_tempo":
                tempo = message.tempo
    return tempo


def track_events(track: mido.MidiTrack, tempo: int, ppq: int):
    tick = 0
    for message in track:
        tick += message.time
        if message.type in ("note_on", "note_off"):
            seconds = tick * tempo / ppq / 1_000_000.0
            yield seconds, message.note, message.type == "note_on" and message.velocity > 0


def voice(name: str) -> tuple[str, float]:
    if "Drums" in name or "Perc" in name:
        return "noise", 0.10
    if "Bass" in name or "Sub" in name:
        return "tri", 0.06
    if "Vocals" in name or "Synth1" in name:
        return "square", 0.07
    return "square", 0.045


def wave_form(kind: str, t: np.ndarray, freq: float) -> np.ndarray:
    phase = np.mod(t * freq, 1.0)
    if kind == "tri":
        return 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    return np.where(phase < 0.28, 1.0, -1.0)


def emit_note(out: np.ndarray, start: float, end: float, note: int, kind: str, level: float) -> None:
    duration = end - start
    if duration <= 0.0 or start >= MAX_SECONDS:
        return
    index = int(start * RATE)
    count = min(int(duration * RATE), len(out) - index)
    if count <= 0:
        return
    t = np.arange(count, dtype=np.float32) / RATE
    env = np.minimum(1.0, t * 90.0) * np.minimum(1.0, (duration - t) * 90.0)
    sample = wave_form(kind, t, note_freq(note)) if kind != "noise" else RNG.normal(0, 1, count)
    out[index : index + count] += (level * env * sample).astype(np.float32)


def emit_drum(out: np.ndarray, start: float, note: int, level: float) -> None:
    if start >= MAX_SECONDS:
        return
    index = int(start * RATE)
    count = min(int(0.08 * RATE), len(out) - index)
    if count <= 0:
        return
    t = np.arange(count, dtype=np.float32) / RATE
    env = np.exp(-t * 45.0)
    pitch = 70.0 if note > 70 else 120.0
    sample = 0.6 * np.sin(2 * np.pi * pitch * t) + 0.4 * RNG.normal(0, 1, count)
    out[index : index + count] += (level * env * sample).astype(np.float32)


def render() -> np.ndarray:
    mid = mido.MidiFile(str(MIDI))
    tempo = tempo_of(mid)
    out = np.zeros(int(MAX_SECONDS * RATE), dtype=np.float32)

    for track in mid.tracks:
        kind, level = voice(track.name)
        active: dict[int, float] = {}
        for time, note, is_on in track_events(track, tempo, mid.ticks_per_beat):
            if time >= MAX_SECONDS:
                break
            if kind == "noise":
                if is_on:
                    emit_drum(out, time, note, level)
                continue
            if is_on:
                if note in active:
                    start = active.pop(note)
                    emit_note(out, start, time, note, kind, level)
                active[note] = time
            elif note in active:
                start = active.pop(note)
                emit_note(out, start, time, note, kind, level)
        for note, start in active.items():
            emit_note(out, start, MAX_SECONDS, note, kind, level)

    peak = float(np.max(np.abs(out)))
    if peak > 0.0:
        out *= 0.8 / peak
    return out


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
