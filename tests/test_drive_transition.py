from controllers.drive_transition import DriveTransitionGuard
from racing import RobotCommand
from racing.physics import resolve_vehicle_actuator_command


def test_braking_can_resume_forward_drive_without_stopping() -> None:
    guard = DriveTransitionGuard()
    pending_direction = 0
    actuators = []
    for throttle in (-0.8, -0.4, 0.7, 0.7):
        command = guard(RobotCommand(throttle, 0.2), speed_mps=20.0)
        assert command.steer == 0.2
        actuator = resolve_vehicle_actuator_command(
            command=command, current_speed_kmh=72.0, pending_drive_direction=pending_direction
        )
        pending_direction = actuator.next_pending_drive_direction
        actuators.append(actuator)
    assert actuators[0].brake_force > 0.0
    assert actuators[1].brake_force > 0.0
    assert actuators[2].engine_force == actuators[2].brake_force == 0.0
    assert actuators[3].engine_force > 0.0
    assert actuators[3].brake_force == 0.0


def test_reverse_braking_can_resume_reverse_drive() -> None:
    guard = DriveTransitionGuard()
    assert guard(RobotCommand(0.8, -0.3), speed_mps=-4.0).throttle == 0.8
    assert guard(RobotCommand(-0.6, -0.3), speed_mps=-4.0) == RobotCommand(0.0, -0.3)
    assert guard(RobotCommand(-0.6, -0.3), speed_mps=-4.0).throttle == -0.6


def test_neutral_and_standstill_clear_braking_state() -> None:
    for speed, throttle in ((10.0, 0.0), (0.0, -0.5)):
        guard = DriveTransitionGuard()
        guard(RobotCommand(-0.5), speed_mps=10.0)
        assert guard(RobotCommand(throttle), speed_mps=speed).throttle == throttle
        assert guard(RobotCommand(0.5), speed_mps=10.0).throttle == 0.5
