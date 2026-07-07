"""Generate original layered formula-engine WAV assets."""

from __future__ import annotations

import argparse
import io
import math
import random
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 22_050
LOOP_SECONDS = 1.20
LOOP_SAMPLES = int(SAMPLE_RATE * LOOP_SECONDS)
MAX_PCM16 = 32_767
RANDOM_SEED = 438_911
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "src" / "racing" / "assets" / "audio"
FORMULA_ENGINE_FILENAMES = ("formula_engine_body_loop.wav",)


def render_formula_engine_assets() -> dict[str, bytes]:
    """Render deterministic formula-engine layer WAVs."""
    rng = random.Random(RANDOM_SEED)
    body = _looped(_engine_body_samples(rng), peak=0.62)
    return {
        "formula_engine_body_loop.wav": body,
    }


def write_formula_engine_assets(output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[Path, ...]:
    """Write generated formula-engine assets and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_formula_engine_assets()
    paths: list[Path] = []
    for filename in FORMULA_ENGINE_FILENAMES:
        path = output_dir / filename
        path.write_bytes(rendered[filename])
        paths.append(path)
    return tuple(paths)


def _engine_body_samples(rng: random.Random) -> list[float]:
    phases = [rng.random() * math.tau for _ in range(10)]
    samples: list[float] = []
    for index in range(LOOP_SAMPLES):
        seconds = index / SAMPLE_RATE
        cycle = seconds / LOOP_SECONDS
        pulse = _pulse_train(cycle, lobes=18, sharpness=3.2)
        harmonic = (
            math.sin(math.tau * 90.0 * seconds + phases[0]) * 0.34
            + math.sin(math.tau * 180.0 * seconds + phases[1]) * 0.22
            + math.sin(math.tau * 270.0 * seconds + phases[2]) * 0.16
            + math.sin(math.tau * 450.0 * seconds + phases[3]) * 0.08
        )
        burble = math.sin(math.tau * 28.0 * seconds + phases[4]) * 0.05
        samples.append((harmonic + burble + pulse * 0.30) * (0.92 + 0.08 * math.sin(math.tau * cycle * 6.0)))
    return samples


def _looped(samples: list[float], *, peak: float) -> bytes:
    _remove_dc(samples)
    _crossfade_loop(samples, int(SAMPLE_RATE * 0.10))
    _match_loop_seam(samples, int(SAMPLE_RATE * 0.018))
    return _wav_bytes(_normalized(samples, peak=peak))


def _pulse_train(cycle: float, *, lobes: int, sharpness: float) -> float:
    position = (cycle * lobes) % 1.0
    distance = min(position, 1.0 - position)
    return max(0.0, 1.0 - distance * 2.0) ** sharpness * 2.0 - 0.18


def _remove_dc(samples: list[float]) -> None:
    if not samples:
        return
    average = sum(samples) / len(samples)
    for index, sample in enumerate(samples):
        samples[index] = sample - average


def _normalized(samples: list[float], *, peak: float) -> list[float]:
    current_peak = max((abs(sample) for sample in samples), default=1.0)
    if current_peak <= 0.0:
        return samples
    scale = peak / current_peak
    return [sample * scale for sample in samples]


def _crossfade_loop(samples: list[float], length: int) -> None:
    fade_length = min(length, len(samples) // 2)
    for index in range(fade_length):
        amount = index / fade_length
        tail_index = len(samples) - fade_length + index
        samples[tail_index] = samples[tail_index] * (1.0 - amount) + samples[index] * amount


def _match_loop_seam(samples: list[float], length: int) -> None:
    fade_length = min(length, len(samples) // 2)
    if fade_length <= 0:
        return
    seam_delta = samples[0] - samples[-1]
    for index in range(fade_length):
        amount = (index + 1) / fade_length
        eased = amount * amount * (3.0 - 2.0 * amount)
        tail_index = len(samples) - fade_length + index
        samples[tail_index] += seam_delta * eased


def _wav_bytes(samples: list[float]) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(
            b"".join(struct.pack("<h", int(max(-1.0, min(1.0, sample)) * MAX_PCM16)) for sample in samples)
        )
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate formula engine synth layer WAV assets.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for path in write_formula_engine_assets(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
