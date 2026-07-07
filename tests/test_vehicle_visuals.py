from __future__ import annotations

import pytest

from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG
from racing.graphics.vehicle_visuals import vehicle_visual_scale, wheel_visual_diameter


def test_formula_visual_scale_is_positive() -> None:
    assert vehicle_visual_scale(FORMULA_VEHICLE_PHYSICS_CONFIG) > 0.0


def test_wheel_visual_diameter_uses_physics_radius() -> None:
    assert wheel_visual_diameter(FORMULA_VEHICLE_PHYSICS_CONFIG) == pytest.approx(
        FORMULA_VEHICLE_PHYSICS_CONFIG.wheel_radius * 2
    )
