"""Shared model-predictive-control planner used by mpc_baseline and learned_dynamics_mpc.

Both controllers plan over the same compact, track-relative state and the same
random-shooting search; they differ only in the `predict_fn` passed to
`plan_action_sequence` (a hand-written kinematic model vs. a learned one). A
controller only ever receives one `RobotSensors` snapshot per tick and cannot
query the simulator directly, so `predict_fn` is this planner's own internal
guess at "what happens if I do this," not a call into the real physics world.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import radians
from typing import Any

import numpy as np

from racing import TRACK_WIDTH, RobotCommand, RobotSensors

STATE_DIM = 6
ACTION_DIM = 2
OFF_TRACK_PENALTY = 50.0
WALL_COLLISION_PENALTY = 25.0
# How much the planner values going fast vs. staying near the track center.
# 1.0 = the two matter about equally (this was the original, cautious/slow behavior).
SPEED_REWARD_WEIGHT = 20.0
# Tiny tie-breaker toward less steering. Steer doesn't affect the speed model at
# all, so many candidate steer values score identically on tracking once close to
# centered -- without this, which one "wins" is arbitrary and flickers tick to
# tick. Deliberately much smaller than OFF_TRACK_PENALTY/WALL_COLLISION_PENALTY so
# it only breaks near-ties and never overrides an actually necessary swerve.
STEERING_EFFORT_WEIGHT = 2.0

# A numpy array of floats. Spelled out (rather than bare `np.ndarray`) because
# strict type checking requires numpy's generic type parameters to be filled in.
FloatArray = np.ndarray[Any, np.dtype[np.float64]]

# predict_fn(states, actions, dt_s) -> next_states
# states/next_states: float array shaped (samples, STATE_DIM); actions: (samples, ACTION_DIM)
PredictFn = Callable[[FloatArray, FloatArray, float], FloatArray]


@dataclass(frozen=True, slots=True)
class LocalState:
    """Compact track-relative state both dynamics models plan over.

    Attributes:
        speed_mps: Signed forward speed.
        heading_error_rad: Car heading minus local track heading, in radians.
        center_offset_m: Signed lateral offset from the track centerline.
        lookahead_offsets_m: Lateral offsets at the camera's lookahead distances.
        front_wall_m: Nearest forward wall-lidar reading, used to penalize plans
            that would drive into a wall within the planning horizon.
    """

    speed_mps: float
    heading_error_rad: float
    center_offset_m: float
    lookahead_offsets_m: tuple[float, float, float]
    front_wall_m: float

    def to_array(self) -> FloatArray:
        """Flatten to the fixed-width vector `predict_fn` operates on."""
        return np.array(
            (self.speed_mps, self.heading_error_rad, self.center_offset_m, *self.lookahead_offsets_m),
            dtype=np.float64,
        )

    @staticmethod
    def from_array(values: FloatArray, *, front_wall_m: float) -> LocalState:
        """Rebuild a `LocalState` from a predicted state vector."""
        speed_mps, heading_error_rad, center_offset_m, y1, y2, y3 = (float(value) for value in values)
        return LocalState(
            speed_mps=speed_mps,
            heading_error_rad=heading_error_rad,
            center_offset_m=center_offset_m,
            lookahead_offsets_m=(y1, y2, y3),
            front_wall_m=front_wall_m,
        )


def local_state_from_sensors(sensors: RobotSensors) -> LocalState:
    """Build a `LocalState` from a real `RobotSensors` snapshot."""
    offsets = sensors.camera.lookahead_offsets_m
    padded: tuple[float, float, float] = (
        offsets[0] if len(offsets) > 0 else 0.0,
        offsets[1] if len(offsets) > 1 else 0.0,
        offsets[2] if len(offsets) > 2 else 0.0,
    )
    return LocalState(
        speed_mps=sensors.odometry.speed_mps,
        heading_error_rad=radians(sensors.camera.heading_error_degrees),
        center_offset_m=sensors.camera.center_offset_m,
        lookahead_offsets_m=padded,
        front_wall_m=sensors.wall_lidar.front_m,
    )


def rollout_cost(states: FloatArray, actions: FloatArray, *, dt_s: float) -> FloatArray:
    """Score each candidate rollout; lower is better.

    Args:
        states: Predicted states for each sample/step, shaped
            `(samples, horizon, STATE_DIM)`, produced by repeatedly calling
            `predict_fn` starting from the current state.
        actions: The sampled actions that produced `states`, shaped
            `(samples, horizon, ACTION_DIM)`.
        dt_s: Seconds represented by each planning step.

    Returns:
        One cost per sample, shaped `(samples,)`.
    """
    speed_mps = states[:, :, 0]
    heading_error_rad = states[:, :, 1]
    center_offset_m = states[:, :, 2]

    steer = actions[:, :, 1]

    progress_reward = np.sum(speed_mps * dt_s, axis=1)
    tracking_cost = np.sum(center_offset_m**2 + heading_error_rad**2, axis=1)
    steering_effort_cost = STEERING_EFFORT_WEIGHT * np.sum(steer**2, axis=1)
    off_track = np.any(np.abs(center_offset_m) > TRACK_WIDTH / 2.0, axis=1)

    return (
        -SPEED_REWARD_WEIGHT * progress_reward
        + tracking_cost
        + steering_effort_cost
        + OFF_TRACK_PENALTY * off_track.astype(np.float64)
    )


def wall_collision_cost(*, front_wall_m: float, states: FloatArray, dt_s: float) -> FloatArray:
    """Penalize rollouts predicted to travel past the current front wall reading.

    `front_wall_m` only reflects the wall directly ahead at plan time (the
    sensor snapshot doesn't update mid-rollout), so this is a coarse
    straight-ahead check, not a real collision model.
    """
    if not np.isfinite(front_wall_m):
        zero_cost: FloatArray = np.zeros(states.shape[0], dtype=np.float64)
        return zero_cost
    speed_mps = np.maximum(states[:, :, 0], 0.0)
    predicted_travel_m = np.cumsum(speed_mps * dt_s, axis=1)
    exceeds_wall = np.any(predicted_travel_m > front_wall_m, axis=1)
    return WALL_COLLISION_PENALTY * exceeds_wall.astype(np.float64)


def shift_plan_for_warm_start(plan: tuple[RobotCommand, ...]) -> FloatArray:
    """Turn last tick's chosen plan into a starting point for this tick's search.

    Drops the action that already ran and repeats the final action once to
    keep the same length, since the planner has no better guess for what
    happens beyond its old horizon.
    """
    array = np.array([[command.throttle, command.steer] for command in plan], dtype=np.float64)
    return np.concatenate([array[1:], array[-1:]], axis=0)


def plan_action_sequence(
    state: LocalState,
    predict_fn: PredictFn,
    *,
    horizon: int = 12,
    samples: int = 128,
    dt_s: float = 1 / 20,
    rng: np.random.Generator,
    warm_start: FloatArray | None = None,
    exploration_std: float = 0.4,
) -> tuple[RobotCommand, ...]:
    """Random-shooting MPC: sample action sequences, roll out, keep the best.

    Only the first action of the winning sequence is meant to be executed;
    the controller should call this again next tick (receding-horizon
    control) rather than executing the whole sequence open-loop.

    Without `warm_start`, every candidate sequence is drawn independently
    from scratch across the whole action range, which makes consecutive
    ticks' chosen actions unrelated to each other and can look jittery. Pass
    `warm_start` (shaped `(horizon, ACTION_DIM)`, e.g. from
    `shift_plan_for_warm_start` on the plan this function returned last
    tick) to instead sample small random nudges around it, so this tick's
    search starts from what was already decided rather than from nothing.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one")
    if samples < 1:
        raise ValueError("samples must be at least one")
    if warm_start is not None and warm_start.shape != (horizon, ACTION_DIM):
        raise ValueError("warm_start must be shaped (horizon, ACTION_DIM)")

    if warm_start is None:
        actions = rng.uniform(-1.0, 1.0, size=(samples, horizon, ACTION_DIM))
    else:
        noise = rng.normal(0.0, exploration_std, size=(samples, horizon, ACTION_DIM))
        actions = np.clip(warm_start[None, :, :] + noise, -1.0, 1.0)
    states = np.empty((samples, horizon, STATE_DIM), dtype=np.float64)

    current = np.tile(state.to_array(), (samples, 1))
    for step in range(horizon):
        current = predict_fn(current, actions[:, step, :], dt_s)
        states[:, step, :] = current

    cost = rollout_cost(states, actions, dt_s=dt_s) + wall_collision_cost(
        front_wall_m=state.front_wall_m, states=states, dt_s=dt_s
    )
    best_index = int(np.argmin(cost))
    return tuple(RobotCommand(throttle=float(throttle), steer=float(steer)) for throttle, steer in actions[best_index])
