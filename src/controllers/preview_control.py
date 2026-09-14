"""Camera-preview feedback for fast driving within visible track geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import tanh

from racing import RobotCommand, RobotSensors


@dataclass(frozen=True, slots=True)
class PreviewGains:
    """Feedback gains tuned in physics trials, without position or lap inputs."""

    heading: float = -0.011102112821549196
    center: float = -0.20864621004237865
    near: float = 0.4059745002774199
    far: float = 0.2551542909821989
    yaw_rate: float = -0.0031272543528565616
    target_speed_mps: float = 32.29395368210786
    speed_response: float = 0.7665646987760519
    heading_speed_penalty: float = 0.014128298287824637
    near_speed_penalty: float = 0.4174753391585728
    far_speed_penalty: float = 1.8013652980845143


DEFAULT_GAINS = PreviewGains()


class PreviewController:
    """Turn toward the visible path and brake before entering a bend.

    Lookahead offsets already contain heading and lateral displacement. Local
    heading and center gains therefore correct the preview signal rather than
    duplicating it with a strong, competing heading-only steering command.
    """

    def __init__(self, *, gains: PreviewGains = DEFAULT_GAINS) -> None:
        self.gains = gains

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        camera = sensors.camera
        near = camera.lookahead_offsets_m[0]
        far = camera.lookahead_offsets_m[-1]
        heading = camera.heading_error_degrees
        gains = self.gains
        steer = tanh(
            gains.heading * heading
            + gains.center * camera.center_offset_m
            + gains.near * near
            + gains.far * far
            + gains.yaw_rate * sensors.imu.yaw_rate_degrees_per_s
        )
        target_speed = (
            gains.target_speed_mps
            - gains.heading_speed_penalty * abs(heading)
            - gains.near_speed_penalty * abs(near)
            - gains.far_speed_penalty * abs(far)
        )
        throttle = tanh(gains.speed_response * (target_speed - sensors.odometry.speed_mps))
        return RobotCommand(throttle=throttle, steer=steer)
