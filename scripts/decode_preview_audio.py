"""Decode a public Freesound preview into mono WAV for asset processing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast


def decode_preview_audio(source_path: Path, output_path: Path) -> Path:
    """Decode an MP3/OGG preview to mono 16-bit PCM WAV."""
    soundfile = cast(Any, _soundfile_module())
    data, sample_rate = soundfile.read(str(source_path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    soundfile.write(str(output_path), mono, sample_rate, subtype="PCM_16")
    return output_path


def _soundfile_module() -> object:
    try:
        import soundfile
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is required for preview decoding; run with "
            "`uv run --with soundfile --with numpy python scripts/decode_preview_audio.py ...`"
        ) from exc
    return soundfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode a compressed audio preview into mono WAV.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(decode_preview_audio(args.source, args.output))


if __name__ == "__main__":
    main()
