"""Pathway 5 (Planning/MPC) minimum experiment: plan against a hand-written kinematic model.

Every tick, this controller samples candidate throttle/steer sequences, rolls
each through `kinematic_predict` (a simple physically-motivated guess, not the
real simulator), scores them with `mpc_lib.rollout_cost`, and drives with the
first action of the best sequence. No training or data collection involved --
this is the cheapest possible planning baseline pathway 7's learned model
(`controllers.learned_dynamics_mpc`) is meant to improve on.
"""

from __future__ import annotations

import numpy as np

from controllers.mpc_lib import LocalState, local_state_from_sensors, plan_action_sequence
from racing import RobotCommand, RobotSensors

RACING_NAME = "MPC Baseline"
RACING_COLOR = "#2ECC71"

PLANNING_HORIZON_STEPS = 12
PLANNING_SAMPLES = 128
PLANNING_DT_S = 1 / 20
PLANNING_RANDOM_SEED = 110

# First-order speed response: how quickly speed moves toward throttle * TARGET_SPEED_MPS.
TARGET_SPEED_MPS = 6.0
SPEED_RESPONSE_RATE_PER_S = 2.5
# Rough bicycle-model yaw response: steer=1.0 -> this many rad/s at TARGET_SPEED_MPS.
MAX_YAW_RATE_RAD_PER_S = 1.6
LOOKAHEAD_DISTANCES_M = (4.0, 9.0, 16.0)


def kinematic_predict(states: np.ndarray, actions: np.ndarray, dt_s: float) -> np.ndarray:
    """Hand-written next-state guess: `(samples, STATE_DIM), (samples, ACTION_DIM) -> (samples, STATE_DIM)`.

    Uses the nearest lookahead offset as a fixed estimate of local track
    curvature for the whole horizon (the sensor snapshot isn't re-queried
    mid-rollout, so it can't be updated as the plan advances). Known to be
    wrong on curves at higher speeds (see the lab notebook), but restored
    here as the last version confirmed to drive safely without stranger
    oscillation -- see the entry documenting why the "fix" for this was
    reverted.
    """
    speed_mps = states[:, 0]
    heading_error_rad = states[:, 1]
    center_offset_m = states[:, 2]
    lookahead_offsets_m = states[:, 3:6]
    throttle = actions[:, 0]
    steer = actions[:, 1]

    next_speed_mps = speed_mps + SPEED_RESPONSE_RATE_PER_S * (throttle * TARGET_SPEED_MPS - speed_mps) * dt_s

    curvature_turn_rad_per_s = (lookahead_offsets_m[:, 0] / LOOKAHEAD_DISTANCES_M[0]) * (speed_mps / TARGET_SPEED_MPS)
    yaw_rate_rad_per_s = steer * MAX_YAW_RATE_RAD_PER_S - curvature_turn_rad_per_s
    next_heading_error_rad = heading_error_rad + yaw_rate_rad_per_s * dt_s
    next_center_offset_m = center_offset_m + speed_mps * np.sin(heading_error_rad) * dt_s

    return np.stack(
        (next_speed_mps, next_heading_error_rad, next_center_offset_m, *lookahead_offsets_m.T),
        axis=1,
    )


class MpcBaselineController:
    """Stateful controller holding its own RNG so plans are reproducible."""

    def __init__(self, *, random_seed: int = PLANNING_RANDOM_SEED) -> None:
        self._rng = np.random.default_rng(random_seed)

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        state = local_state_from_sensors(sensors)
        plan = plan_action_sequence(
            state,
            kinematic_predict,
            horizon=PLANNING_HORIZON_STEPS,
            samples=PLANNING_SAMPLES,
            dt_s=PLANNING_DT_S,
            rng=self._rng,
        )
        return plan[0]

    def copy_for_car(self) -> MpcBaselineController:
        """Give every car/race its own RNG stream, matching the module-level `LocalState` type."""
        return MpcBaselineController(random_seed=int(self._rng.integers(0, 2**31 - 1)))


def create_controller() -> MpcBaselineController:
    return MpcBaselineController()


__all__ = ["LocalState", "MpcBaselineController", "create_controller", "kinematic_predict"]
