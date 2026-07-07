"""Create the bundled music loop from Freesound sound 564665.

The original WAV download requires a Freesound login, but the public preview is
usable as source material. One reproducible source-prep flow is:

    curl -L "https://cdn.freesound.org/previews/564/564665_6738752-hq.mp3" -o /tmp/564665.mp3
    uv run --with soundfile --with numpy python scripts/decode_preview_audio.py /tmp/564665.mp3 /tmp/564665.wav
    uv run python scripts/process_freesound_music.py /tmp/564665.wav
"""

from __future__ import annotations

import argparse
import io
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 22_050
MUSIC_LOOP_CROSSFADE_SAMPLES = int(SAMPLE_RATE * 0.75)
MUSIC_ONSET_WINDOW_SAMPLES = int(SAMPLE_RATE * 0.05)
MUSIC_ONSET_RELATIVE_THRESHOLD = 0.08
MUSIC_ONSET_ABSOLUTE_THRESHOLD = 0.006
MUSIC_FADE_IN_SAMPLES = int(SAMPLE_RATE * 5.0)
MAX_PCM16 = 32_767
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "racing" / "assets" / "audio" / "berlin_town_music.wav"
)


def render_music_loop_from_wav(source_path: Path) -> bytes:
    """Render the bundled music loop from a decoded Freesound WAV source."""
    samples, sample_rate = _read_wav_mono(source_path)
    if sample_rate != SAMPLE_RATE:
        samples = _resampled(samples, source_rate=sample_rate, target_rate=SAMPLE_RATE)
    _remove_dc(samples)
    samples = _trim_leading_quiet(samples)
    _fade_in(samples, MUSIC_FADE_IN_SAMPLES)
    _crossfade_loop(samples, MUSIC_LOOP_CROSSFADE_SAMPLES)
    return _wav_bytes(_normalized(samples, peak=0.82))


def write_music_loop_from_wav(source_path: Path, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Write the processed music loop and return the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(render_music_loop_from_wav(source_path))
    return output_path


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


def _trim_leading_quiet(samples: list[float]) -> list[float]:
    if not samples:
        return samples
    window_length = max(1, MUSIC_ONSET_WINDOW_SAMPLES)
    rms_windows: list[tuple[int, float]] = []
    for start in range(0, len(samples), window_length):
        window = samples[start : start + window_length]
        if not window:
            continue
        rms = (sum(sample * sample for sample in window) / len(window)) ** 0.5
        rms_windows.append((start, rms))
    max_rms = max((rms for _, rms in rms_windows), default=0.0)
    if max_rms <= 0.0:
        return samples
    threshold = max(MUSIC_ONSET_ABSOLUTE_THRESHOLD, max_rms * MUSIC_ONSET_RELATIVE_THRESHOLD)
    for start, rms in rms_windows:
        if rms >= threshold:
            return samples[start:]
    return samples


def _fade_in(samples: list[float], length: int) -> None:
    fade_length = min(length, len(samples))
    if fade_length <= 1:
        return
    for index in range(fade_length):
        amount = index / (fade_length - 1)
        gain = amount * amount * (3.0 - 2.0 * amount)
        samples[index] *= gain


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Process Freesound 564665 into the bundled music loop.")
    parser.add_argument("source_wav", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    print(write_music_loop_from_wav(args.source_wav, args.output))


if __name__ == "__main__":
    main()
