from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from racing.main import DEFAULT_FORMULA_TEAM_COLOR, CameraView, GameConfig, create_app, parse_color_rgba
from racing.graphics.track_rendering import bind_nearest_track_spotlights_to_node


def parse_size(value: str) -> tuple[int, int]:
    width_text, height_text = value.lower().split("x", 1)
    return int(width_text), int(height_text)


def parse_xz(value: str) -> tuple[float, float]:
    x_text, z_text = value.split(",", 1)
    return float(x_text), float(z_text)


def parse_xyz(value: str) -> tuple[float, float, float]:
    x_text, y_text, z_text = value.split(",", 2)
    return float(x_text), float(y_text), float(z_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture deterministic screenshots of the full racing scene.")
    parser.add_argument("--camera", choices=tuple(view.value for view in CameraView), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--size", type=parse_size, default=(1280, 720))
    parser.add_argument("--team-color", type=parse_color_rgba, default=DEFAULT_FORMULA_TEAM_COLOR)
    parser.add_argument("--top-down-center", type=parse_xz)
    parser.add_argument("--top-down-height", type=float, default=2.5)
    parser.add_argument("--top-down-fov", type=float, default=8.0)
    parser.add_argument("--car-position", type=parse_xz)
    parser.add_argument("--car-heading", type=float)
    parser.add_argument("--camera-position", type=parse_xyz)
    parser.add_argument("--camera-look-at", type=parse_xyz)
    parser.add_argument("--camera-fov", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    app = cast(
        Any,
        create_app(
            GameConfig(
                size=args.size,
                camera_view=CameraView(str(args.camera)),
                window_type="offscreen",
                development_mode=True,
                vsync=False,
                team_color=args.team_color,
            )
        ),
    )
    try:
        ursina = cast(Any, import_module("ursina"))
        for _ in range(args.frames):
            app.step()
        if args.car_position is not None:
            robot = app.racing_robot
            x, z = args.car_position
            _, current_y, _ = robot.chassis_np.getPos(ursina.scene)
            robot.chassis_np.setPos(x, current_y, z)
            if args.car_heading is not None:
                robot.chassis_np.setHpr(args.car_heading, 0.0, 0.0)
            bind_nearest_track_spotlights_to_node(ursina=ursina, node=robot.chassis_np)
            app.graphicsEngine.renderFrame()
        if args.top_down_center is not None:
            x, z = args.top_down_center
            ursina.camera.orthographic = True
            ursina.camera.position = (x, args.top_down_height, z)
            ursina.camera.setHpr(0.0, -90.0, 0.0)
            ursina.camera.fov = args.top_down_fov
            app.graphicsEngine.renderFrame()
        if args.camera_position is not None and args.camera_look_at is not None:
            ursina.camera.orthographic = False
            ursina.camera.position = args.camera_position
            ursina.camera.look_at(ursina.Vec3(*args.camera_look_at))
            if args.camera_fov is not None:
                ursina.camera.fov = args.camera_fov
            app.graphicsEngine.renderFrame()
        result = app.screenshot(namePrefix=str(output.resolve()), defaultFilename=False)
        if result is None:
            raise RuntimeError("Panda3D did not write a screenshot")
        print(result)
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
