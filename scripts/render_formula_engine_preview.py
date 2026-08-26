"""Render an offline preview of the formula engine layer model."""

from __future__ import annotations

import argparse
import struct
import wave
from pathlib import Path

from racing.game.config import RacingAudioConfig
from racing.sound.audio import (
    FORMULA_ENGINE_AUDIO_FILENAMES,
    FormulaEngineRuntimeState,
    formula_engine_audio_state_for_robot,
)
from racing.student.api import RobotCommand

SAMPLE_RATE = 22_050
MAX_PCM16 = 32_767
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "formula_engine_preview.wav"
ASSET_DIR = Path(__file__).resolve().parents[1] / "src" / "racing" / "assets" / "audio"


def render_formula_engine_preview(output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Render a deterministic preview WAV for manual auditioning."""
    layers = tuple(_read_wav_mono(ASSET_DIR / filename) for filename in FORMULA_ENGINE_AUDIO_FILENAMES)
    config = RacingAudioConfig(music_enabled=False)
    runtime_state = FormulaEngineRuntimeState()
    layer_positions = [0.0 for _ in layers]
    samples: list[float] = []
    delta_seconds = 1.0 / 60.0
    frame_samples = int(SAMPLE_RATE * delta_seconds)
    total_frames = int(14.0 / delta_seconds)
    for frame in range(total_frames):
        seconds = frame * delta_seconds
        speed_mps, command = _preview_motion(seconds)
        state = formula_engine_audio_state_for_robot(
            speed_mps=speed_mps,
            command=command,
            previous_state=runtime_state,
            delta_seconds=delta_seconds,
            eliminated=False,
            config=config,
            muted=False,
        )
        runtime_state = state.runtime_state
        for _ in range(frame_samples):
            value = 0.0
            for index, layer in enumerate(layers):
                position = int(layer_positions[index]) % len(layer)
                value += layer[position] * state.layer_volumes[index] * 0.56
                layer_positions[index] = (layer_positions[index] + state.layer_play_rates[index]) % len(layer)
            samples.append(value * 0.72)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(output_path, samples)
    return output_path


def _preview_motion(seconds: float) -> tuple[float, RobotCommand]:
    if seconds < 1.0:
        return 0.0, RobotCommand(throttle=0.10)
    if seconds < 8.0:
        amount = (seconds - 1.0) / 7.0
        return amount * 48.0, RobotCommand(throttle=1.0)
    if seconds < 10.0:
        amount = (seconds - 8.0) / 2.0
        return 48.0 + amount * 6.0, RobotCommand(throttle=0.20)
    if seconds < 12.0:
        amount = (seconds - 10.0) / 2.0
        return 54.0 - amount * 24.0, RobotCommand(throttle=-0.85)
    amount = (seconds - 12.0) / 2.0
    return 30.0 - amount * 10.0, RobotCommand(throttle=0.55)


def _read_wav_mono(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError(f"{path} must be a mono 16-bit PCM WAV")
        frames = wav_file.readframes(wav_file.getnframes())
    return [sample[0] / MAX_PCM16 for sample in struct.iter_unpack("<h", frames)]


def _write_wav(path: Path, samples: list[float]) -> None:
    peak = max((abs(sample) for sample in samples), default=1.0)
    scale = min(1.0, 0.90 / peak) if peak > 0.0 else 1.0
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(
            b"".join(struct.pack("<h", int(max(-1.0, min(1.0, sample * scale)) * MAX_PCM16)) for sample in samples)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a formula engine audio preview WAV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    print(render_formula_engine_preview(args.output))


if __name__ == "__main__":
    main()
