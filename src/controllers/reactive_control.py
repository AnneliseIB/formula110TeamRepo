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


def _preview_angle(offset_m: float, distance_m: float) -> float:
    """Return the direction, in degrees, to one visible centre-line point."""
    return degrees(atan2(_finite(offset_m), max(0.5, _finite(distance_m, 0.5))))


@dataclass(slots=True)
class _GainSearch:
    """Bounded coordinate search using recent driving quality as its score."""

    steer_gain: float = 0.95
    throttle_gain: float = 0.035
    direction: float = -1.0
    parameter: int = 0
    elapsed_s: float = 0.0
    score_sum: float = 0.0
    samples: int = 0
    best_score: float = float("-inf")
    steer_step: float = 0.08
    throttle_step: float = 0.004

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
            # Keep moving in a successful direction, but shorten future probes.
            self._scale_step(0.92)
        else:
            # Undo the failed probe and test the opposite direction next.
            self._adjust(-self.direction)
            self.direction *= -1.0
            self._scale_step(0.97)

        self._adjust(self.direction)
        self.parameter = 1 - self.parameter
        self.elapsed_s = 0.0
        self.score_sum = 0.0
        self.samples = 0

    def _adjust(self, direction: float) -> None:
        if self.parameter == 0:
            self.steer_gain = _clip(self.steer_gain + direction * self.steer_step, 0.70, 1.25)
        else:
            self.throttle_gain = _clip(self.throttle_gain + direction * self.throttle_step, 0.025, 0.060)

    def _scale_step(self, scale: float) -> None:
        self.steer_step = max(0.02, self.steer_step * scale)
        self.throttle_step = max(0.001, self.throttle_step * scale)


class ReactiveController:
    """Drive quickly while minimizing the camera-derived path error."""

    def __init__(self) -> None:
        self.gains = _GainSearch()
        self.previous_path_error = 0.0
        self.recovery_s = 0.0

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        dt_s = _clip(_finite(sensors.dt_s), 0.0, 0.1)
        speed = _finite(sensors.odometry.speed_mps)
        walls = sensors.wall_lidar
        contact = sensors.contact

        if contact.wall > 0.0 or contact.any_contact > 0.35:
            self.recovery_s = 0.55
        self.recovery_s = max(0.0, self.recovery_s - dt_s)
        if self.recovery_s > 0.0:
            # Back away while pointing towards the side with more open space.
            side_clearance = _finite(walls.right_m, 3.0) - _finite(walls.left_m, 3.0)
            turn = _clip(side_clearance / 4.0, -1.0, 1.0)
            return RobotCommand(throttle=-0.65, steer=turn)

        camera = sensors.camera
        heading_error = center_offset = near_preview = far_preview = 0.0
        if camera.visible:
            heading_error = _finite(camera.heading_error_degrees)
            center_offset = _finite(camera.center_offset_m)
            if camera.lookahead_offsets_m:
                near_preview = _preview_angle(
                    camera.lookahead_offsets_m[0], camera.lookahead_distances_m[0]
                )
                far_preview = _preview_angle(
                    camera.lookahead_offsets_m[-1], camera.lookahead_distances_m[-1]
                )

        # Local heading drives immediate steering. Lookahead and centre offset
        # begin the turn earlier, which avoids a late left/right correction.
        path_error = heading_error / 2.1 + near_preview / 25.0 + far_preview / 80.0 + center_offset / 4.5
        error_rate = (path_error - self.previous_path_error) / max(dt_s, 1 / 60)
        self.previous_path_error = path_error
        steer = tanh(
            self.gains.steer_gain * path_error
            - 0.0005 * _finite(sensors.imu.yaw_rate_degrees_per_s)
            - 0.0002 * error_rate
        )

        # A weighted left/right LiDAR difference is a continuous wall-avoidance
        # correction. It is intentionally small while clear, then becomes
        # decisive as a side wall enters the near field.
        left_wall = _finite(walls.left_m, 100.0)
        right_wall = _finite(walls.right_m, 100.0)
        side_pressure = (1.0 / max(0.25, left_wall)) - (1.0 / max(0.25, right_wall))
        steer = _clip(steer + 0.22 * side_pressure, -1.0, 1.0)

        # Ignore tiny path changes on a straight to suppress left/right chatter.
        if abs(path_error) < 0.08 and abs(side_pressure) < 0.03:
            steer = 0.0

        # Pass a slower car instead of sitting in its wake. The camera angle
        # tells which side it occupies; a centred car is passed on the side
        # with more LiDAR clearance.
        nearest_ahead = min(
            (
                competitor
                for competitor in camera.competitors
                if competitor.distance_m < 7.0 and abs(competitor.angle_degrees) < 30.0
            ),
            key=lambda competitor: competitor.distance_m,
            default=None,
        )
        if nearest_ahead is not None:
            if abs(nearest_ahead.angle_degrees) < 3.0:
                pass_direction = 1.0 if right_wall > left_wall else -1.0
            else:
                pass_direction = -1.0 if nearest_ahead.angle_degrees > 0.0 else 1.0
            pass_urgency = _clip((7.0 - nearest_ahead.distance_m) / 7.0, 0.0, 1.0)
            steer = _clip(steer + 0.38 * pass_direction * pass_urgency, -1.0, 1.0)

        front_wall = _finite(walls.front_m, 100.0)
        front_object = _finite(sensors.lidar.front_m, 100.0)
        forward_clearance = min(front_wall, front_object)

        # Previewed curvature and forward clearance form a feed-forward speed
        # signal: accelerate harder only when the visible track is straight and
        # neither a wall nor another object is in front, then ease off early.
        preview_turn = _clip((abs(near_preview) + 0.5 * abs(far_preview)) / 35.0, 0.0, 1.0)
        open_track = _clip((forward_clearance - 2.0) / 5.0, 0.0, 1.0)
        straight_boost = 0.46 * (1.0 - preview_turn) * open_track if camera.visible else 0.0
        sharp_turn = _clip((preview_turn - 0.35) / 0.65, 0.0, 1.0)
        throttle = tanh(
            1.15
            + straight_boost
            - self.gains.throttle_gain * speed
            - 0.08 * abs(steer)
            - 0.35 * sharp_turn
        )
        brake_distance = 1.3 + 0.15 * max(speed, 0.0)
        if front_wall < brake_distance and speed > 5.0:
            throttle = min(throttle, _clip(-0.14 * (speed - 3.0), -1.0, 0.0))

        # Fast, centred, low-error driving is the optimisation objective.  A
        # collision is heavily penalised, so unsafe gain probes are rejected.
        quality = speed - 0.35 * abs(path_error) - 0.002 * abs(error_rate)
        if contact.any_contact > 0.0:
            quality -= 25.0
        self.gains.observe(dt_s=dt_s, score=quality)
        return RobotCommand(throttle=throttle, steer=steer)


def create_controller() -> ReactiveController:
    """Give every car/race its own optimiser and controller state."""
    return ReactiveController()


_fallback_controller = ReactiveController()


def control(sensors: RobotSensors) -> RobotCommand:
    """Compatibility entry point for tools that load a plain function."""
    return _fallback_controller(sensors)
