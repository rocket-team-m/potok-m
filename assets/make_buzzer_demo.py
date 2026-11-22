"""Synthesize the post-landing buzzer demo (search beacon + chiptune tail)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

RATE = 48_000
OUT = Path(__file__).with_name("buzzer_demo.wav")


def tone(t: np.ndarray, freq: float, duty: float = 0.28) -> np.ndarray:
    phase = np.mod(t * freq, 1.0)
    wave = np.where(phase < duty, 1.0, -1.0)
    fade = np.minimum(1.0, t * 80.0)
    return (0.22 * fade * wave).astype(np.float32)


def render() -> np.ndarray:
    pieces: list[np.ndarray] = []

    # Search beacon: three short chirps, then a pause.
    for _ in range(3):
        t = np.arange(int(0.16 * RATE), dtype=np.float32) / RATE
        pieces.append(tone(t, 2350.0))
        pieces.append(np.zeros(int(0.34 * RATE), dtype=np.float32))
    pieces.append(np.zeros(int(0.8 * RATE), dtype=np.float32))

    # Short excitation melody on the same physical source.
    notes = [
        (329.63, 0.22), (392.00, 0.22), (440.00, 0.22), (392.00, 0.22),
        (329.63, 0.22), (293.66, 0.22), (261.63, 0.44), (0.0, 0.18),
        (329.63, 0.22), (392.00, 0.22), (493.88, 0.22), (587.33, 0.33),
        (493.88, 0.22), (440.00, 0.22), (392.00, 0.44), (0.0, 0.30),
    ]
    for freq, duration in notes:
        count = int(duration * RATE)
        if freq == 0.0:
            pieces.append(np.zeros(count, dtype=np.float32))
            continue
        t = np.arange(count, dtype=np.float32) / RATE
        env = np.minimum(1.0, t * 60.0) * np.minimum(1.0, (duration - t) * 60.0)
        pieces.append(tone(t, freq) * env)

    # Final beacon so the demo ends the way the real device would.
    t = np.arange(int(0.30 * RATE), dtype=np.float32) / RATE
    pieces.append(tone(t, 2350.0))

    audio = np.concatenate(pieces)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


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
