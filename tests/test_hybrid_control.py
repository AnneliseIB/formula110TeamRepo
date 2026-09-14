from dataclasses import replace
from math import isfinite
from unittest.mock import patch

from controllers.hybrid_control import create_controller
from controllers.preview_control import PreviewController
from racing import CameraSensors, OdometrySensors, RobotCommand, RobotSensors


def test_preview_brakes_and_turns_before_local_heading_changes() -> None:
    preview = PreviewController()
    straight = RobotSensors(odometry=OdometrySensors(speed_mps=28.0))
    bend = replace(straight, camera=CameraSensors(lookahead_offsets_m=(0.0, 4.0, 12.0)))
    assert preview(straight).throttle > 0.0
    assert preview(bend).throttle < 0.0
    assert preview(bend).steer > 0.0


def test_hybrid_uses_learned_planner_for_slow_recovery_only() -> None:
    controller = create_controller()
    lost_heading = RobotSensors(
        camera=CameraSensors(heading_error_degrees=60.0, center_offset_m=2.0),
        odometry=OdometrySensors(speed_mps=2.0),
    )
    with patch("controllers.hybrid_control.LearnedDynamicsMpcController") as planner:
        planner.return_value.return_value = RobotCommand(0.2, 0.25)
        assert controller(lost_heading) == RobotCommand(0.2, 0.25)
        planner.assert_called_once()
    with patch("controllers.hybrid_control.LearnedDynamicsMpcController") as planner:
        controller(replace(lost_heading, odometry=OdometrySensors(speed_mps=25.0)))
        planner.assert_not_called()


def test_copies_do_not_inherit_a_braking_transition() -> None:
    controller = create_controller()
    bend = RobotSensors(
        camera=CameraSensors(lookahead_offsets_m=(0.0, 4.0, 12.0)),
        odometry=OdometrySensors(speed_mps=28.0),
    )
    assert controller(bend).throttle < 0.0
    other = controller.copy_for_car()
    straight = RobotSensors(odometry=OdometrySensors(speed_mps=20.0))
    assert other(straight).throttle > 0.0
    assert controller(straight).throttle == 0.0
    assert controller(straight).throttle > 0.0


def test_missing_or_invalid_preview_returns_finite_coasting_command() -> None:
    controller = create_controller()
    for camera in (
        CameraSensors(visible=False),
        CameraSensors(lookahead_offsets_m=(), lookahead_distances_m=()),
        CameraSensors(heading_error_degrees=float("nan")),
    ):
        command = controller(RobotSensors(camera=camera, odometry=OdometrySensors(speed_mps=25.0)))
        assert command.throttle == 0.0
        assert isfinite(command.steer)
        assert -1.0 <= command.steer <= 1.0
