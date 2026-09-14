"""Fast camera-preview driving with learned-dynamics MPC for slow recovery.

The learned model is unreliable when extrapolated far beyond its training
speeds. Preview feedback handles racing pace; MPC corrects large path errors
at low speed, and reactive control backs away after contact.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import numpy as np

from controllers.drive_transition import DriveTransitionGuard
from controllers.dynamics_model import DynamicsModel
from controllers.learned_dynamics_mpc import (
    DEFAULT_MODEL_PATH,
    LearnedDynamicsMpcController,
)
from controllers.preview_control import PreviewController
from controllers.reactive_control import ReactiveController
from racing import RobotCommand, RobotSensors

RACING_NAME = "Hybrid Reactive MPC"
RACING_COLOR = "#F1C40F"

RECOVERY_HEADING_ERROR_DEGREES = 35.0
RECOVERY_CENTER_OFFSET_M = 2.4
RECOVERY_SPEED_MPS = 8.0
MAX_CRUISE_SPEED_MPS = 40.0


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _finite(value: float, fallback: float) -> float:
    return value if isfinite(value) else fallback


class HybridController:
    """Use preview feedback at racing speed and MPC for low-speed recovery."""

    def __init__(self, *, model: DynamicsModel, random_seed: int = 110) -> None:
        self._model = model
        self._rng = np.random.default_rng(random_seed)
        self._reactive = ReactiveController()
        self._preview = PreviewController()
        self._drive_transition = DriveTransitionGuard()
        self._recovering = False
        self._mpc = LearnedDynamicsMpcController(model=model, random_seed=int(self._rng.integers(0, 2**31 - 1)))

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        reactive = self._reactive(sensors)
        speed = _finite(sensors.odometry.speed_mps, 0.0)
        heading_error = abs(_finite(sensors.camera.heading_error_degrees, 180.0))
        center_offset = abs(_finite(sensors.camera.center_offset_m, 10.0))
        preview_available = (
            sensors.camera.visible
            and len(sensors.camera.lookahead_offsets_m) >= 2
            and all(
                isfinite(value)
                for value in (
                    sensors.camera.heading_error_degrees,
                    sensors.camera.center_offset_m,
                    *sensors.camera.lookahead_offsets_m,
                    sensors.imu.yaw_rate_degrees_per_s,
                    sensors.odometry.speed_mps,
                )
            )
        )
        needs_recovery = (
            preview_available
            and abs(speed) < RECOVERY_SPEED_MPS
            and (heading_error > RECOVERY_HEADING_ERROR_DEGREES or center_offset > RECOVERY_CENTER_OFFSET_M)
        )
        # Ordinary braking must not be mistaken for a post-contact reverse.
        if self._reactive.recovery_s > 0.0 or sensors.contact.wall > 0.0:
            command = reactive
            needs_recovery = False
        elif not preview_available:
            command = RobotCommand(throttle=0.0, steer=reactive.steer)
        elif needs_recovery:
            if not self._recovering:
                # A plan from a previous recovery is no longer a useful warm start.
                self._mpc = LearnedDynamicsMpcController(
                    model=self._model,
                    random_seed=int(self._rng.integers(0, 2**31 - 1)),
                )
            command = self._mpc(sensors)
        else:
            command = self._preview(sensors)
        self._recovering = needs_recovery
        throttle = min(command.throttle, 0.0) if speed >= MAX_CRUISE_SPEED_MPS else command.throttle
        command = RobotCommand(
            throttle=_clip(_finite(throttle, 0.0), -1.0, 1.0),
            steer=_clip(_finite(command.steer, 0.0), -1.0, 1.0),
        )
        return self._drive_transition(command, speed_mps=speed)

    def copy_for_car(self) -> HybridController:
        return HybridController(model=self._model, random_seed=int(self._rng.integers(0, 2**31 - 1)))


def create_controller(*, model_path: Path = DEFAULT_MODEL_PATH) -> HybridController:
    return HybridController(model=DynamicsModel.load(model_path))


_fallback_controller: HybridController | None = None


def control(sensors: RobotSensors) -> RobotCommand:
    """Compatibility entry point for tools that load a plain function."""
    global _fallback_controller
    if _fallback_controller is None:
        _fallback_controller = create_controller()
    return _fallback_controller(sensors)


__all__ = ["HybridController", "control", "create_controller"]
