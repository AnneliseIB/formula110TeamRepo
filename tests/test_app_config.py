from __future__ import annotations

import argparse

import pytest

from racing.game.config import (
    CameraView,
    CarShowcaseConfig,
    GameConfig,
    HeadToHeadViewerConfig,
    parse_color_rgba,
    parse_window_size,
    positive_float,
)
from racing.graphics.colors import (
    DEFAULT_CHALLENGER_TEAM_COLOR,
    DEFAULT_FORMULA_TEAM_COLOR,
    DEFAULT_INCUMBENT_TEAM_COLOR,
    UNC_CAROLINA_BLUE,
    UNC_FORDHAM_FOUNTAIN,
)


def test_parse_window_size_accepts_width_by_height() -> None:
    assert parse_window_size("1440x900") == (1440, 900)


def test_parse_color_rgba_accepts_hex_and_decimal_channels() -> None:
    assert parse_color_rgba("#ff8000") == (1.0, 128 / 255, 0.0, 1.0)
    assert parse_color_rgba("0.1, 0.2, 0.3, 0.4") == (0.1, 0.2, 0.3, 0.4)


def test_parse_color_rgba_rejects_out_of_range_channels() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_color_rgba("1.2, 0, 0")


def test_positive_float_rejects_zero() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("0")


def test_game_config_has_no_vehicle_selector() -> None:
    assert not hasattr(GameConfig(), "vehicle_model")


def test_racing_views_default_to_drone_camera() -> None:
    assert GameConfig().camera_view is CameraView.DRONE
    assert HeadToHeadViewerConfig().camera_view is CameraView.DRONE


def test_default_formula_car_uses_fordham_fountain() -> None:
    assert DEFAULT_FORMULA_TEAM_COLOR == UNC_FORDHAM_FOUNTAIN
    assert GameConfig().team_color == DEFAULT_FORMULA_TEAM_COLOR
    assert CarShowcaseConfig().team_color == DEFAULT_FORMULA_TEAM_COLOR


def test_default_head_to_head_colors_use_fordham_fountain_versus_carolina_blue() -> None:
    assert DEFAULT_CHALLENGER_TEAM_COLOR == UNC_FORDHAM_FOUNTAIN
    assert DEFAULT_INCUMBENT_TEAM_COLOR == UNC_CAROLINA_BLUE
    assert HeadToHeadViewerConfig().challenger_team_color == DEFAULT_CHALLENGER_TEAM_COLOR
    assert HeadToHeadViewerConfig().incumbent_team_color == DEFAULT_INCUMBENT_TEAM_COLOR
