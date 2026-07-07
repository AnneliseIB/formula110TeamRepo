"""Create the bundled engine loop from the Pixabay Freesound Community F1 MP3.

The source MP3 is Pixabay sound effect 32824, `f1`, by freesound_community. One
reproducible source-prep flow after downloading the MP3 is:

    uv run --with soundfile --with numpy python scripts/decode_preview_audio.py \
        src/racing/assets/audio/freesound_community-f1-32824.mp3 \
        /tmp/freesound_community-f1-32824.wav
    uv run python scripts/process_freesound_engine.py /tmp/freesound_community-f1-32824.wav
"""

from __future__ import annotations

import argparse
import io
import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 22_050
ENGINE_SECONDS = 3.2
ENGINE_LOOP_SAMPLES = int(SAMPLE_RATE * ENGINE_SECONDS)
ENGINE_LOOP_CROSSFADE_SAMPLES = int(SAMPLE_RATE * 0.28)
ENGINE_LOOP_SEAM_MATCH_SAMPLES = int(SAMPLE_RATE * 0.024)
MAX_PCM16 = 32_767
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "src" / "racing" / "assets" / "audio" / "f1_engine_loop.wav"


def render_engine_loop_from_wav(source_path: Path) -> bytes:
    """Render the bundled engine loop from a decoded source WAV."""
    samples, sample_rate = _read_wav_mono(source_path)
    if sample_rate != SAMPLE_RATE:
        samples = _resampled(samples, source_rate=sample_rate, target_rate=SAMPLE_RATE)
    start = _selected_loop_start(samples)
    loop = samples[start : start + ENGINE_LOOP_SAMPLES]
    _remove_dc(loop)
    _smooth_spikes(loop, threshold=0.20)
    _low_pass_blend(loop, amount=0.16)
    _stabilize_envelope(loop, window_samples=int(SAMPLE_RATE * 0.045), strength=0.42)
    _crossfade_loop(loop, ENGINE_LOOP_CROSSFADE_SAMPLES)
    _match_loop_seam(loop, ENGINE_LOOP_SEAM_MATCH_SAMPLES)
    return _wav_bytes(_soft_limited(_normalized(loop, peak=0.68), threshold=0.74))


def write_engine_loop_from_wav(source_path: Path, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Write the processed engine loop and return the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(render_engine_loop_from_wav(source_path))
    return output_path


def _selected_loop_start(samples: list[float]) -> int:
    if len(samples) < ENGINE_LOOP_SAMPLES:
        raise ValueError("source audio is shorter than the engine loop")
    hop = int(SAMPLE_RATE * 0.04)
    end = len(samples) - ENGINE_LOOP_SAMPLES
    first_candidate = min(end, max(0, int(len(samples) * 0.12)))
    last_candidate = max(first_candidate + 1, min(end, int(len(samples) * 0.78)))
    best_start = first_candidate
    best_score = float("-inf")
    for start in range(first_candidate, last_candidate, hop):
        window = samples[start : start + ENGINE_LOOP_SAMPLES]
        rms = _rms(window)
        zcr = _zero_crossing_rate(window)
        continuity = 1.0 - min(1.0, abs(window[0] - window[-1]) * 8.0)
        energy_stability = 1.0 - min(1.0, _chunk_rms_spread(window) * 4.0)
        edge_stability = 1.0 - min(1.0, abs(_rms(window[:1024]) - _rms(window[-1024:])) * 6.0)
        score = rms * (1.0 + zcr * 1.2) + energy_stability * 0.16 + edge_stability * 0.10 + continuity * 0.04
        if score > best_score:
            best_score = score
            best_start = start
    return best_start


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


def _chunk_rms_spread(samples: list[float], chunks: int = 8) -> float:
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


def _smooth_spikes(samples: list[float], *, threshold: float) -> None:
    for index in range(1, len(samples) - 1):
        neighbor_average = (samples[index - 1] + samples[index + 1]) * 0.5
        delta = samples[index] - neighbor_average
        if abs(delta) > threshold:
            samples[index] = neighbor_average + threshold * (1.0 if delta > 0.0 else -1.0)


def _low_pass_blend(samples: list[float], *, amount: float) -> None:
    if len(samples) < 3:
        return
    original = samples[:]
    blend = max(0.0, min(1.0, amount))
    for index in range(1, len(samples) - 1):
        smoothed = original[index - 1] * 0.25 + original[index] * 0.50 + original[index + 1] * 0.25
        samples[index] = original[index] * (1.0 - blend) + smoothed * blend


def _stabilize_envelope(samples: list[float], *, window_samples: int, strength: float) -> None:
    if len(samples) < 2 or window_samples <= 1:
        return
    half_window = max(1, window_samples // 2)
    squared_prefix = [0.0]
    for sample in samples:
        squared_prefix.append(squared_prefix[-1] + sample * sample)
    local_rms: list[float] = []
    for index in range(len(samples)):
        start = max(0, index - half_window)
        end = min(len(samples), index + half_window)
        count = max(1, end - start)
        local_rms.append(math.sqrt((squared_prefix[end] - squared_prefix[start]) / count))
    target_rms = sorted(local_rms)[len(local_rms) // 2]
    if target_rms <= 0.0:
        return
    amount = max(0.0, min(1.0, strength))
    for index, rms in enumerate(local_rms):
        if rms <= 0.0001:
            continue
        gain = max(0.72, min(1.32, target_rms / rms))
        samples[index] *= 1.0 + (gain - 1.0) * amount


def _soft_limited(samples: list[float], *, threshold: float) -> list[float]:
    limited: list[float] = []
    for sample in samples:
        sign = 1.0 if sample >= 0.0 else -1.0
        magnitude = abs(sample)
        if magnitude <= threshold:
            limited.append(sample)
            continue
        excess = magnitude - threshold
        limited.append(sign * (threshold + excess / (1.0 + excess * 5.0)))
    return limited


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
    parser = argparse.ArgumentParser(
        description="Process the Pixabay/Freesound Community F1 source into the engine loop."
    )
    parser.add_argument("source_wav", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    print(write_engine_loop_from_wav(args.source_wav, args.output))


if __name__ == "__main__":
    main()
