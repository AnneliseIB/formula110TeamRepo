from __future__ import annotations

from math import tanh

import pytest

from controllers.smol_brain import clamp_and_normalize, control
from racing import CameraSensors, OdometrySensors, RobotSensors


def _sensors(*, speed_mps: float = 0.0, heading_error_degrees: float = 0.0) -> RobotSensors:
    return RobotSensors(
        odometry=OdometrySensors(speed_mps=speed_mps),
        camera=CameraSensors(heading_error_degrees=heading_error_degrees),
    )


def test_accelerates_from_a_stop() -> None:
    command = control(_sensors(speed_mps=0.0, heading_error_degrees=0.0))

    assert command.throttle == pytest.approx(tanh(1.0))
    assert command.steer == 0.0


def test_throttle_decreases_as_speed_increases() -> None:
    slow = control(_sensors(speed_mps=0.0))
    fast = control(_sensors(speed_mps=20.0))

    assert fast.throttle < slow.throttle


def test_brakes_once_above_the_governed_speed() -> None:
    command = control(_sensors(speed_mps=40.0))

    assert command.throttle < 0.0


def test_steer_saturates_toward_correcting_even_a_modest_heading_error() -> None:
    command = control(_sensors(speed_mps=5.0, heading_error_degrees=10.0))

    assert command.steer > 0.99


def test_steer_sign_matches_heading_error_sign() -> None:
    positive = control(_sensors(speed_mps=5.0, heading_error_degrees=15.0))
    negative = control(_sensors(speed_mps=5.0, heading_error_degrees=-15.0))

    assert positive.steer > 0.0
    assert negative.steer < 0.0


def test_outputs_stay_within_actuator_bounds_for_extreme_inputs() -> None:
    command = control(_sensors(speed_mps=500.0, heading_error_degrees=999.0))

    assert -1.0 <= command.throttle <= 1.0
    assert -1.0 <= command.steer <= 1.0


def test_clamp_and_normalize_maps_range_to_unit_interval() -> None:
    assert clamp_and_normalize(0.0, -50.0, 50.0) == 0.0
    assert clamp_and_normalize(50.0, -50.0, 50.0) == 1.0
    assert clamp_and_normalize(-50.0, -50.0, 50.0) == -1.0


def test_clamp_and_normalize_clamps_out_of_range_values() -> None:
    assert clamp_and_normalize(1000.0, -50.0, 50.0) == 1.0
    assert clamp_and_normalize(-1000.0, -50.0, 50.0) == -1.0


def test_clamp_and_normalize_rejects_invalid_range() -> None:
    with pytest.raises(ValueError):
        clamp_and_normalize(0.0, 10.0, 10.0)
