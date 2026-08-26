from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from racing.game.app import create_head_to_head_viewer_app
from racing.game.config import CameraView, HeadToHeadViewerConfig
from racing.student.api import default_student_controller


def parse_size(value: str) -> tuple[int, int]:
    width_text, height_text = value.lower().split("x", 1)
    return int(width_text), int(height_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a deterministic screenshot of the head-to-head viewer.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--size", type=parse_size, default=(1280, 720))
    parser.add_argument("--camera", choices=tuple(view.value for view in CameraView), default=CameraView.DRONE.value)
    parser.add_argument("--round-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    app = cast(
        Any,
        create_head_to_head_viewer_app(
            HeadToHeadViewerConfig(
                size=args.size,
                camera_view=CameraView(str(args.camera)),
                challenger_name="Candidate",
                incumbent_name="Baseline",
                challenger_controller=default_student_controller,
                incumbent_controller=default_student_controller,
                round_seconds=args.round_seconds,
                window_type="offscreen",
                development_mode=True,
                vsync=False,
            )
        ),
    )
    try:
        for _ in range(args.frames):
            app.step()
        result = app.screenshot(namePrefix=str(args.output.resolve()), defaultFilename=False)
        if result is None:
            raise RuntimeError("Panda3D did not write a screenshot")
        print(result)
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
