"""Release direction-change braking before requesting drive in the current direction."""

from __future__ import annotations

from racing import RobotCommand

# The simulator allows a direction change below 1 km/h.
DIRECTION_CHANGE_SPEED_MPS = 1.0 / 3.6


class DriveTransitionGuard:
    """Insert one neutral tick when acceleration should resume after braking.

    Negative throttle while moving forward starts a latched direction change.
    Sending positive throttle immediately afterwards otherwise continues braking
    to a standstill. A neutral tick releases that latch without changing steer.
    The same transition applies when backing up and braking with positive throttle.
    """

    def __init__(self) -> None:
        self._braking = False

    def __call__(self, command: RobotCommand, *, speed_mps: float) -> RobotCommand:
        if command.throttle == 0.0 or abs(speed_mps) <= DIRECTION_CHANGE_SPEED_MPS:
            self._braking = False
            return command
        opposes_motion = command.throttle * speed_mps < 0.0
        if self._braking and not opposes_motion:
            self._braking = False
            return RobotCommand(throttle=0.0, steer=command.steer)
        self._braking = opposes_motion
        return command
