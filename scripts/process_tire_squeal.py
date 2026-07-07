"""Create bundled tire-squeal loops from a decoded source WAV."""

from __future__ import annotations

import argparse
import io
import math
import struct
import wave
from pathlib import Path
from typing import NamedTuple

SAMPLE_RATE = 22_050
SQUEAL_SECONDS = 0.85
SQUEAL_SAMPLES = int(SAMPLE_RATE * SQUEAL_SECONDS)
SQUEAL_CROSSFADE_SAMPLES = int(SAMPLE_RATE * 0.10)
SQUEAL_SEAM_MATCH_SAMPLES = int(SAMPLE_RATE * 0.018)
SQUEAL_CLIP_COUNT = 3
MAX_PCM16 = 32_767
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "src" / "racing" / "assets" / "audio"
SQUEAL_FILENAMES = tuple(f"tire_squeal_{index}.wav" for index in range(1, SQUEAL_CLIP_COUNT + 1))


class SquealCandidate(NamedTuple):
    """One candidate source window for a tire-squeal loop."""

    start: int
    score: float


def render_tire_squeal_loops_from_wav(source_path: Path) -> tuple[bytes, ...]:
    """Render squeal loops from a decoded source WAV."""
    samples, sample_rate = _read_wav_mono(source_path)
    if sample_rate != SAMPLE_RATE:
        samples = _resampled(samples, source_rate=sample_rate, target_rate=SAMPLE_RATE)
    _remove_dc(samples)
    candidates = _squeal_candidates(samples)
    loops: list[bytes] = []
    for candidate in candidates[:SQUEAL_CLIP_COUNT]:
        loop = samples[candidate.start : candidate.start + SQUEAL_SAMPLES]
        if len(loop) < SQUEAL_SAMPLES:
            loop = [*loop, *([0.0] * (SQUEAL_SAMPLES - len(loop)))]
        _remove_dc(loop)
        _crossfade_loop(loop, SQUEAL_CROSSFADE_SAMPLES)
        _match_loop_seam(loop, SQUEAL_SEAM_MATCH_SAMPLES)
        loops.append(_wav_bytes(_normalized(loop, peak=0.72)))
    if len(loops) < SQUEAL_CLIP_COUNT:
        raise ValueError("source audio did not contain enough usable tire squeal regions")
    return tuple(loops)


def write_tire_squeal_loops_from_wav(source_path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[Path, ...]:
    """Write processed tire-squeal loops and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_tire_squeal_loops_from_wav(source_path)
    paths: list[Path] = []
    for filename, data in zip(SQUEAL_FILENAMES, rendered, strict=True):
        path = output_dir / filename
        path.write_bytes(data)
        paths.append(path)
    return tuple(paths)


def _squeal_candidates(samples: list[float]) -> tuple[SquealCandidate, ...]:
    if len(samples) < SQUEAL_SAMPLES:
        raise ValueError("source audio is shorter than one tire squeal loop")
    hop = int(SAMPLE_RATE * 0.08)
    min_gap = int(SAMPLE_RATE * 1.10)
    raw: list[SquealCandidate] = []
    for start in range(0, len(samples) - SQUEAL_SAMPLES, hop):
        window = samples[start : start + SQUEAL_SAMPLES]
        rms = _rms(window)
        zcr = _zero_crossing_rate(window)
        stability = 1.0 - min(1.0, _chunk_rms_spread(window) * 2.5)
        edge_stability = 1.0 - min(1.0, abs(_rms(window[:1024]) - _rms(window[-1024:])) * 8.0)
        score = rms * (1.0 + zcr * 3.4) + stability * 0.10 + edge_stability * 0.04
        raw.append(SquealCandidate(start=start, score=score))
    selected: list[SquealCandidate] = []
    for candidate in sorted(raw, key=lambda item: item.score, reverse=True):
        if all(abs(candidate.start - existing.start) >= min_gap for existing in selected):
            selected.append(candidate)
        if len(selected) >= SQUEAL_CLIP_COUNT:
            break
    return tuple(sorted(selected, key=lambda item: item.start))


def _read_wav_mono(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError("source WAV must be 16-bit PCM")
    unpacked = tuple(sample[0] / MAX_PCM16 for sample in struct.iter_unpack("<h", frames))
    if channels == 1:
        return list(unpacked), sample_rate
    if channels < 1:
        raise ValueError("source WAV must have at least one channel")
    mono: list[float] = []
    for index in range(0, len(unpacked), channels):
        mono.append(sum(unpacked[index : index + channels]) / channels)
    return mono, sample_rate


def _resampled(samples: list[float], *, source_rate: int, target_rate: int) -> list[float]:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    output_length = int(len(samples) * target_rate / source_rate)
    output: list[float] = []
    for index in range(output_length):
        source_position = index * source_rate / target_rate
        left = min(int(source_position), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        amount = source_position - left
        output.append(samples[left] * (1.0 - amount) + samples[right] * amount)
    return output


def _remove_dc(samples: list[float]) -> None:
    if not samples:
        return
    average = sum(samples) / len(samples)
    for index, sample in enumerate(samples):
        samples[index] = sample - average


def _rms(samples: list[float]) -> float:
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _chunk_rms_spread(samples: list[float], chunks: int = 5) -> float:
    chunk_length = max(1, len(samples) // chunks)
    values = [_rms(samples[index : index + chunk_length]) for index in range(0, len(samples), chunk_length)]
    if len(values) < 2:
        return 0.0
    average = sum(values) / len(values)
    if average <= 0.0:
        return 0.0
    variance = sum((value - average) * (value - average) for value in values) / len(values)
    return math.sqrt(variance) / average


def _zero_crossing_rate(samples: list[float]) -> float:
    crossings = 0
    previous_positive = samples[0] >= 0.0
    for sample in samples[1:]:
        current_positive = sample >= 0.0
        if current_positive != previous_positive:
            crossings += 1
        previous_positive = current_positive
    return crossings / max(1, len(samples) - 1)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Process decoded tire squeal source audio into loop clips.")
    parser.add_argument("source_wav", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for path in write_tire_squeal_loops_from_wav(args.source_wav, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
