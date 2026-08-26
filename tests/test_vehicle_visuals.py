from __future__ import annotations

import pytest

from racing.graphics.vehicle_visuals import (
    COCKPIT_INSTRUMENT_BASE_POSITION,
    COCKPIT_INSTRUMENT_BASE_SCALE,
    COCKPIT_INSTRUMENT_MAST_POSITION,
    COCKPIT_INSTRUMENT_MAST_SCALE,
    vehicle_visual_scale,
    wheel_visual_diameter,
)
from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG


def test_formula_visual_scale_is_positive() -> None:
    assert vehicle_visual_scale(FORMULA_VEHICLE_PHYSICS_CONFIG) > 0.0


def test_wheel_visual_diameter_uses_physics_radius() -> None:
    assert wheel_visual_diameter(FORMULA_VEHICLE_PHYSICS_CONFIG) == pytest.approx(
        FORMULA_VEHICLE_PHYSICS_CONFIG.wheel_radius * 2
    )


def test_cockpit_instrument_mast_meets_its_base() -> None:
    base_top = COCKPIT_INSTRUMENT_BASE_POSITION[1] + COCKPIT_INSTRUMENT_BASE_SCALE[1] / 2
    mast_bottom = COCKPIT_INSTRUMENT_MAST_POSITION[1] - COCKPIT_INSTRUMENT_MAST_SCALE[1] / 2

    assert mast_bottom == pytest.approx(base_top)
