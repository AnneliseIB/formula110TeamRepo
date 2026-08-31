"""Reactive, self-tuning controller for the Formula 110 track.

The controller does not use track coordinates or lap progress.  It follows the
camera's centre-line observations and uses the wall LiDAR as a last safety
check.  A small coordinate-search optimiser continuously tunes the two most
important policy parameters: steering response and throttle response.  The
search is deliberately bounded and slow so that a bad trial cannot turn into a
large, race-ending control change.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, isfinite, tanh

from racing import RobotCommand, RobotSensors

RACING_NAME = "Reactive Optimizer"
RACING_COLOR = "#22b8cf"


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _finite(value: float, fallback: float = 0.0) -> float:
    """Use a harmless fallback for unusual manually-created sensor values."""
    return value if isfinite(value) else fallback


@dataclass(slots=True)
class _GainSearch:
    """Bounded coordinate search using recent driving quality as its score."""

    steer_gain: float = 1.0
    throttle_gain: float = 0.040
    direction: float = 1.0
    parameter: int = 0
    elapsed_s: float = 0.0
    score_sum: float = 0.0
    samples: int = 0
    best_score: float = float("-inf")
    step: float = 0.006

    def observe(self, *, dt_s: float, score: float) -> None:
        """Evaluate one perturbation every few seconds and retain improvements."""
        # Optimisation starts after enough samples to make a comparison useful.
        self.elapsed_s += _clip(dt_s, 0.0, 0.1)
        self.score_sum += score
        self.samples += 1
        if self.elapsed_s < 2.5 or self.samples < 30:
            return

        mean_score = self.score_sum / self.samples
        if mean_score >= self.best_score:
            self.best_score = mean_score
            # Keep moving in a successful direction, but become less twitchy.
            self.step = max(0.002, self.step * 0.92)
        else:
            # Undo the failed probe and test the opposite direction next.
            self._adjust(-self.direction * self.step)
            self.direction *= -1.0
            self.step = max(0.002, self.step * 0.97)

        self._adjust(self.direction * self.step)
        self.parameter = 1 - self.parameter
        self.elapsed_s = 0.0
        self.score_sum = 0.0
        self.samples = 0

    def _adjust(self, amount: float) -> None:
        if self.parameter == 0:
            self.steer_gain = _clip(self.steer_gain + amount, 0.96, 1.04)
        else:
            self.throttle_gain = _clip(self.throttle_gain + amount, 0.035, 0.045)


class ReactiveController:
    """Drive quickly while minimizing the camera-derived path error."""

    def __init__(self) -> None:
        self.gains = _GainSearch()
        self.previous_path_error = 0.0
        self.recovery_s = 0.0

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        dt_s = _clip(_finite(sensors.dt_s), 0.0, 0.1)
        speed = _finite(sensors.odometry.speed_mps)

        if sensors.contact.wall > 0.0 or sensors.contact.any_contact > 0.35:
            self.recovery_s = 0.55
        self.recovery_s = max(0.0, self.recovery_s - dt_s)
        if self.recovery_s > 0.0:
            # Back away while pointing towards the side with more open space.
            side_clearance = _finite(sensors.wall_lidar.right_m, 3.0) - _finite(sensors.wall_lidar.left_m, 3.0)
            turn = _clip(side_clearance / 4.0, -1.0, 1.0)
            return RobotCommand(throttle=-0.65, steer=turn)

        camera = sensors.camera
        heading_error = _finite(camera.heading_error_degrees) if camera.visible else 0.0
        center_offset = _finite(camera.center_offset_m) if camera.visible else 0.0
        offsets = camera.lookahead_offsets_m if camera.visible else ()
        distances = camera.lookahead_distances_m if camera.visible else ()

        # Angles to upcoming centre-line points give both a steering target and
        # a curvature estimate; far lookahead makes the controller anticipate
        # bends instead of reacting after the car has already drifted outward.
        preview_angles = [
            degrees(atan2(_finite(offset), max(0.5, _finite(distance, 0.5))))
            for offset, distance in zip(offsets, distances, strict=True)
        ]
        near_preview = preview_angles[0] if preview_angles else 0.0
        far_preview = preview_angles[-1] if preview_angles else 0.0
        # Heading is the most reliable immediate steering signal.  Preview is
        # kept deliberately light: at racing speed it is useful as a small
        # turn-in hint, but must not fight the local track tangent.
        path_error = heading_error / 1.8 + 0.015 * near_preview + 0.005 * far_preview + center_offset / 20.0
        error_rate = (path_error - self.previous_path_error) / max(dt_s, 1 / 60)
        self.previous_path_error = path_error
        steer = tanh(
            self.gains.steer_gain * path_error
            - 0.0005 * _finite(sensors.imu.yaw_rate_degrees_per_s)
            - 0.0002 * error_rate
        )

        # Let tire grip and steering scrub speed in corners.  The former
        # curvature speed cap caused braking while the car was safely following
        # the centreline, so throttle now resembles a cruise-speed controller.
        # A negative command is reserved for a genuinely imminent wall.
        throttle = tanh(1.06 - self.gains.throttle_gain * speed)
        front_wall = _finite(sensors.wall_lidar.front_m, 100.0)
        if front_wall < 1.5 and speed > 5.0:
            throttle = min(throttle, _clip(-0.35 * (speed - 4.0), -1.0, 0.0))

        # Fast, centred, low-error driving is the optimisation objective.  A
        # collision is heavily penalised, so unsafe gain probes are rejected.
        quality = speed - 0.35 * abs(path_error) - 0.002 * abs(error_rate)
        if sensors.contact.any_contact > 0.0:
            quality -= 25.0
        self.gains.observe(dt_s=dt_s, score=quality)
        return RobotCommand(throttle=throttle, steer=steer)


def create_controller() -> ReactiveController:
    """Give every car/race its own optimiser and controller state."""
    return ReactiveController()


def control(sensors: RobotSensors) -> RobotCommand:
    """Compatibility entry point for tools that load a plain function."""
    return ReactiveController()(sensors)
