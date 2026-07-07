"""Shared formula car appearance values."""

from __future__ import annotations

ColorRGBA = tuple[float, float, float, float]

UNC_CAROLINA_BLUE: ColorRGBA = (75 / 255, 156 / 255, 211 / 255, 1.0)
UNC_FORDHAM_FOUNTAIN: ColorRGBA = (183 / 255, 215 / 255, 237 / 255, 1.0)
UNC_BOLIN_CREEK: ColorRGBA = (44 / 255, 80 / 255, 128 / 255, 1.0)
UNC_CORNERSTONE: ColorRGBA = (207 / 255, 211 / 255, 213 / 255, 1.0)

DEFAULT_FORMULA_TEAM_COLOR: ColorRGBA = UNC_FORDHAM_FOUNTAIN
DEFAULT_CHALLENGER_TEAM_COLOR: ColorRGBA = UNC_FORDHAM_FOUNTAIN
DEFAULT_INCUMBENT_TEAM_COLOR: ColorRGBA = UNC_CAROLINA_BLUE

# Historical public name kept for callers that imported the original default.
FORMULA_TEAM_RED: ColorRGBA = DEFAULT_FORMULA_TEAM_COLOR
