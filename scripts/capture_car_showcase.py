from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from racing.main import (
    DEFAULT_FORMULA_TEAM_COLOR,
    CarShowcaseConfig,
    CarShowcaseView,
    create_car_showcase_app,
    parse_color_rgba,
)


def parse_size(value: str) -> tuple[int, int]:
    width_text, height_text = value.lower().split("x", 1)
    return int(width_text), int(height_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture deterministic close-up screenshots of the car art.")
    parser.add_argument("--view", choices=tuple(view.value for view in CarShowcaseView), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=parse_size, default=(1280, 720))
    parser.add_argument("--team-color", type=parse_color_rgba, default=DEFAULT_FORMULA_TEAM_COLOR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    app = cast(
        Any,
        create_car_showcase_app(
            CarShowcaseConfig(
                view=CarShowcaseView(args.view),
                size=args.size,
                team_color=args.team_color,
            )
        ),
    )
    try:
        for _ in range(args.frames):
            app.step()
        result = app.screenshot(namePrefix=str(output.resolve()), defaultFilename=False)
        if result is None:
            raise RuntimeError("Panda3D did not write a screenshot")
        print(result)
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
