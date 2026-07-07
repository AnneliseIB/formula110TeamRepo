from __future__ import annotations

from pathlib import Path

import pytest

from racing.game import cli
from racing.game.config import CameraView, GameConfig, HeadToHeadViewerConfig
from racing.game.cli import build_argument_parser
from racing.graphics.colors import (
    DEFAULT_CHALLENGER_TEAM_COLOR,
    DEFAULT_FORMULA_TEAM_COLOR,
    DEFAULT_INCUMBENT_TEAM_COLOR,
)


class _FakeApp:
    def run(self) -> None:
        pass


def test_parser_has_student_controller_playable_mode() -> None:
    args = build_argument_parser().parse_args(["--student-module", "student_driver.py", "--camera", "follow"])

    assert args.student_module == "student_driver.py"
    assert args.camera == "follow"
    assert args.command is None


def test_parser_uses_unc_default_formula_color() -> None:
    args = build_argument_parser().parse_args([])

    assert args.team_color == DEFAULT_FORMULA_TEAM_COLOR


def test_parser_defaults_to_drone_camera() -> None:
    parser = build_argument_parser()
    playable_args = parser.parse_args([])
    head_to_head_args = parser.parse_args(["h2h"])

    assert playable_args.camera == CameraView.DRONE.value
    assert head_to_head_args.camera == CameraView.DRONE.value


def test_parser_rejects_removed_vehicle_flag() -> None:
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(["--vehicle", "formula"])


def test_parser_accepts_student_head_to_head_without_vehicle_flag() -> None:
    args = build_argument_parser().parse_args(
        [
            "h2h",
            "--challenger-student-module",
            "driver_a.py",
            "--incumbent-student-module",
            "driver_b.py",
            "--races",
            "3",
        ]
    )

    assert args.command == "h2h"
    assert args.challenger_student_module == "driver_a.py"
    assert args.incumbent_student_module == "driver_b.py"
    assert args.races == 3
    assert args.challenger_team_color == DEFAULT_CHALLENGER_TEAM_COLOR
    assert args.incumbent_team_color == DEFAULT_INCUMBENT_TEAM_COLOR


def test_playable_student_color_overrides_cli_team_color(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "orange_driver.py"
    module_path.write_text(
        "from racing import RobotCommand, RobotSensors\n"
        "\n"
        "RACING_COLOR = '#ff8000'\n"
        "\n"
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        "    return RobotCommand(throttle=0.4)\n",
        encoding="utf-8",
    )
    captured_config: GameConfig | None = None

    def fake_create_app(config: GameConfig) -> _FakeApp:
        nonlocal captured_config
        captured_config = config
        return _FakeApp()

    monkeypatch.setattr(cli, "create_app", fake_create_app)

    cli.main(["--student-module", str(module_path), "--team-color", "#0000ff"])

    assert captured_config is not None
    assert captured_config.team_color == (1.0, 128 / 255, 0.0, 1.0)


def test_watched_head_to_head_student_colors_override_team_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    challenger_path = tmp_path / "orange_challenger.py"
    challenger_path.write_text(
        "from racing import RobotCommand, RobotSensors\n"
        "\n"
        "RACING_COLOR = '#ff8000'\n"
        "\n"
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        "    return RobotCommand(throttle=0.4)\n",
        encoding="utf-8",
    )
    incumbent_path = tmp_path / "green_incumbent.py"
    incumbent_path.write_text(
        "from racing import RobotCommand, RobotSensors\n"
        "\n"
        "RACING_COLOR = (0.0, 1.0, 0.0)\n"
        "\n"
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        "    return RobotCommand(throttle=0.2)\n",
        encoding="utf-8",
    )
    captured_config: HeadToHeadViewerConfig | None = None

    def fake_create_head_to_head_viewer_app(config: HeadToHeadViewerConfig) -> _FakeApp:
        nonlocal captured_config
        captured_config = config
        return _FakeApp()

    monkeypatch.setattr(cli, "create_head_to_head_viewer_app", fake_create_head_to_head_viewer_app)

    cli.main(
        [
            "h2h",
            "--watch",
            "--challenger-student-module",
            str(challenger_path),
            "--incumbent-student-module",
            str(incumbent_path),
            "--challenger-team-color",
            "#0000ff",
            "--incumbent-team-color",
            "#ff0000",
        ]
    )

    assert captured_config is not None
    assert captured_config.challenger_team_color == (1.0, 128 / 255, 0.0, 1.0)
    assert captured_config.incumbent_team_color == (0.0, 1.0, 0.0, 1.0)
