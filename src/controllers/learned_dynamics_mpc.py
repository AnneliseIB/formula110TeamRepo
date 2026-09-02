"""Pathway 7 (Learned Dynamics) controller: plan against a trained dynamics model.

Uses the same cost function as `mpc_baseline.py` (`mpc_lib.rollout_cost` /
`wall_collision_cost`) and a `predict_fn` that wraps a `DynamicsModel` trained
on logged driving data instead of the hand-written `kinematic_predict`. It
does *not*, however, reuse `mpc_lib.plan_action_sequence`'s sampling -- see
`_plan_with_correlated_noise` below for why -- so this file implements its
own local search on top of the shared cost functions instead. `mpc_baseline`
is untouched either way.

Requires a trained model at `DEFAULT_MODEL_PATH` (see
`scripts/collect_dynamics_dataset.py` and `scripts/train_dynamics_model.py`).
"""

from __future__ import annotations

from math import radians
from pathlib import Path
from typing import Any

import numpy as np

from controllers.dynamics_model import DynamicsModel
from controllers.mpc_lib import (
    ACTION_DIM,
    STATE_DIM,
    LocalState,
    local_state_from_sensors,
    rollout_cost,
    wall_collision_cost,
)
from racing import RobotCommand, RobotSensors

RACING_NAME = "Learned Dynamics MPC"
RACING_COLOR = "#9B59B6"

DEFAULT_MODEL_PATH = Path("artifacts/dynamics_model.npz")
PLANNING_SAMPLES = 128
PLANNING_RANDOM_SEED = 110

# `mpc_lib.plan_action_sequence` perturbs every one of the `horizon` planned
# steps independently. For a normal, near-track state that's fine, but once
# badly off track it's not: with a large heading/center error, the model's
# own *deterministic* cost curve (steer held constant across the horizon)
# has one clear minimum -- but sampling 36-90 *independent* per-step values
# means a candidate sequence's total cost is dominated by noise from the
# other steps, not by whether its first action is actually right. Verified
# directly: identical state, identical warm start, different random draws
# alone swung the chosen steer between -0.99 and +0.92. So instead of
# reusing `plan_action_sequence`, this file samples ONE (throttle, steer)
# offset per candidate and holds it constant across the whole horizon --
# confirmed to produce a consistent, correctly-signed choice across repeated
# random seeds where the independent-per-step version did not.
FloatArray = np.ndarray[Any, np.dtype[np.float64]]

NORMAL_LOOKAHEAD_SECONDS = 0.6
RECOVERY_LOOKAHEAD_SECONDS = 1.5  # longer: a large error takes more than 0.6s to correct,
# and the model needs to be able to "see" that far to justify committing to it now.
RECOVERY_HEADING_ERROR_RAD = radians(25.0)  # matches the 25-degree threshold `default_student_controller` uses
RECOVERY_CENTER_OFFSET_M = 0.5
NORMAL_EXPLORATION_STD = 0.15

# Recovery mode originally dropped warm-starting entirely (searched fresh from scratch every
# tick) to avoid getting stuck near a bad anchor. Real telemetry showed the cost of that: with no
# memory of what it was just doing, the search can flip between similarly-scoring strategies
# (e.g. "creep forward" vs. "reverse a little") from one real tick to the next based on tiny state
# differences, executing throttle that alternates +0.50/-1.00/+0.50/-0.97 while heading sits
# frozen and speed never builds enough for steering to do anything (yaw rate scales with speed).
# So recovery now warm-starts too, just with much wider exploration than normal driving -- enough
# to still escape a genuinely bad anchor, but keeping continuity with what it was just doing
# instead of re-deciding from nothing every tick.
RECOVERY_EXPLORATION_STD = 0.6

# A now-decisive planner can competently chase `rollout_cost`'s speed reward
# for the first time (the old chattering accidentally masked this) -- and a
# planner that trusts its model beyond what it actually learned from will
# exploit that trust and crash. A flat throttle cap alone doesn't prevent
# this either: holding *any* moderate throttle continuously for many seconds
# (which this planner now does) still reaches the same danger zone eventually
# over a long enough straight, just later. So the cap is speed-*dependent*
# instead: freely accelerate below SAFE_SPEED_MPS, but force throttle to stop
# adding more speed once at or above it, regardless of the horizon. This is
# done here rather than raising rollout_cost's speed weight or adding a new
# penalty there, since that file (mpc_lib.py) is also used by mpc_baseline.
#
# SAFE_SPEED_MPS is set from the training dataset's actual coverage, not
# guessed: `scripts/collect_dynamics_dataset.py`'s original short (~0.13s)
# action holds meant speed rarely built past a few m/s, forcing a 6 m/s cap
# that made this controller much slower than the simple reactive `smol_brain`
# baseline despite the vehicle safely handling far higher speeds under simple
# control. Holding actions for ~0.75s and biasing exploration toward forward
# throttle (see that script) gave the dataset dense coverage up to its
# 99th-percentile speed of ~18.6 m/s, so the cap can move up to match.
MAX_THROTTLE_COMMAND = 1.0
SAFE_SPEED_MPS = 15.0

# Separately: real telemetry (2026-09-01 lab notebook, head-to-head vs. `smol_brain`) showed the
# planner accelerating hard (throttle 0.46-0.98) while *already* 42-56 degrees off heading and
# closing fast on a wall -- i.e. already past `RECOVERY_HEADING_ERROR_RAD`/`RECOVERY_CENTER_OFFSET_M`
# below -- then crashing and ending up stuck at an even worse ~80-degree misalignment. Recovery
# mode widens the *search* (longer horizon, no warm start) but never stopped acceleration outright,
# so a wide-enough search could still rate "floor it" as best if the model's multi-step rollout
# didn't foresee the collision clearly enough. Capping throttle during `needs_recovery` is a direct
# fix for that -- but capping it all the way to zero was tried first and made things worse: every
# race starts with a modest `center_offset` from the spawn point alone, which is enough to trigger
# `needs_recovery` on tick zero, and a car that isn't allowed to move at all can never steer its way
# out (steering does nothing at zero speed -- real yaw rate scales with speed). A modest cap still
# lets it move and correct, just not floor it while already meaningfully off track.
RECOVERY_MAX_THROTTLE_COMMAND = 0.5

# Real telemetry (seed 2027) found a further, more subtle failure: the model's predicted
# steering response is highly state-dependent, and at high speed (~15 m/s) combined with a large
# heading error, it predicts strong self-correction even at near-zero steer -- a combination that
# real driving rarely produces (a well-behaving car rarely reaches high speed while badly
# misaligned), so the model is extrapolating unreliably in exactly this regime. The real car did
# not self-correct there; heading error got worse while cruising at the speed cap. So once
# `needs_recovery` and already going fast, stop allowing further acceleration -- get back toward
# a speed regime the model reasons about reliably before trusting its steering judgment.
#
# Forcing hard *negative* throttle here (tried: -0.4) was also tested -- it fixed this specific
# seed's crash but was a net regression across the full 5-seed suite (719m -> 482m total
# distance), evidently over-triggering on routine cornering rather than just genuine crash risk.
# Capping at 0.0 (stop accelerating, don't force active braking) tested as a net improvement
# instead (719m -> 781m) -- less decisive on this one scenario, but better overall.
RECOVERY_BRAKE_SPEED_THRESHOLD_MPS = 8.0
RECOVERY_BRAKE_MAX_THROTTLE = 0.0

# A curve-aware braking cost (estimating curvature from the camera's lookahead offsets and
# penalizing predicted speed beyond a curve-dependent safe speed) was tried here on 2026-09-01
# to address remaining wall contact after the recovery throttle cap above. Verified against the
# same head-to-head race it was meant to fix: total distance collapsed from 257.94m to 38.60m and
# marshal resets (full stuck-and-reset events) rose from 1 to 2 -- clearly worse, not better, so
# it was reverted rather than shipped. See the 2026-09-01 lab notebook entry for the full record.


def _plan_with_correlated_noise(
    state: LocalState,
    predict_fn: Any,
    *,
    horizon: int,
    samples: int,
    dt_s: float,
    rng: np.random.Generator,
    warm_start: FloatArray | None,
    exploration_std: float,
    max_throttle: float,
) -> tuple[RobotCommand, ...]:
    """Random-shooting MPC with one noise draw per candidate, not per step.

    Without `warm_start`, each candidate holds ONE randomly sampled
    (throttle, steer) pair constant across the whole horizon -- a coherent
    "commit to this" trajectory, not `horizon` independent random values.
    With `warm_start` (shaped `(horizon, ACTION_DIM)`), each candidate is the
    warm start plus one random offset, again held constant across all steps,
    so the search nudges the previous plan's shape rather than reintroducing
    high-dimensional independent noise every tick.

    `max_throttle` is decided by the caller (based on current speed *and*
    tracking error -- see `LearnedDynamicsMpcController.__call__`), not
    computed here, since it depends on more than just this rollout.
    """
    if warm_start is None:
        base_action = rng.uniform(-1.0, 1.0, size=(samples, 1, ACTION_DIM))
        actions = np.tile(base_action, (1, horizon, 1))
    else:
        offset = rng.normal(0.0, exploration_std, size=(samples, 1, ACTION_DIM))
        actions = warm_start[None, :, :] + offset

    actions[..., 0] = np.clip(actions[..., 0], -1.0, max_throttle)
    actions[..., 1] = np.clip(actions[..., 1], -1.0, 1.0)

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


class LearnedDynamicsMpcController:
    """Stateful controller: loaded model, own RNG, and last tick's chosen action for warm-starting.

    Every plan this controller produces holds one (throttle, steer) pair constant across its
    whole horizon (see `_plan_with_correlated_noise`), so the plan's first action *is* the whole
    plan -- tracking just that one action, rather than a full horizon-length sequence, is enough
    to warm-start the next tick's search regardless of whether the horizon length changes between
    normal and recovery mode.
    """

    def __init__(self, *, model: DynamicsModel, random_seed: int = PLANNING_RANDOM_SEED) -> None:
        self._model = model
        self._rng = np.random.default_rng(random_seed)
        self._normal_horizon = max(1, round(NORMAL_LOOKAHEAD_SECONDS / model.training_dt_s))
        self._recovery_horizon = max(1, round(RECOVERY_LOOKAHEAD_SECONDS / model.training_dt_s))
        self._previous_action: RobotCommand | None = None

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        state = local_state_from_sensors(sensors)
        needs_recovery = (
            abs(state.heading_error_rad) > RECOVERY_HEADING_ERROR_RAD
            or abs(state.center_offset_m) > RECOVERY_CENTER_OFFSET_M
        )
        horizon = self._recovery_horizon if needs_recovery else self._normal_horizon

        warm_start: FloatArray | None = None
        if self._previous_action is not None:
            warm_start = np.tile(
                np.array([[self._previous_action.throttle, self._previous_action.steer]], dtype=np.float64),
                (horizon, 1),
            )

        exploration_std = RECOVERY_EXPLORATION_STD if needs_recovery else NORMAL_EXPLORATION_STD
        max_throttle = RECOVERY_MAX_THROTTLE_COMMAND if needs_recovery else MAX_THROTTLE_COMMAND
        if needs_recovery and state.speed_mps > RECOVERY_BRAKE_SPEED_THRESHOLD_MPS:
            max_throttle = min(max_throttle, RECOVERY_BRAKE_MAX_THROTTLE)
        if state.speed_mps >= SAFE_SPEED_MPS:
            max_throttle = min(max_throttle, 0.0)

        plan = _plan_with_correlated_noise(
            state,
            self._model.predict_fn,
            horizon=horizon,
            samples=PLANNING_SAMPLES,
            dt_s=self._model.training_dt_s,
            rng=self._rng,
            warm_start=warm_start,
            exploration_std=exploration_std,
            max_throttle=max_throttle,
        )
        self._previous_action = plan[0]
        return plan[0]

    def copy_for_car(self) -> LearnedDynamicsMpcController:
        """Give every car/race its own RNG stream, sharing the same loaded model."""
        return LearnedDynamicsMpcController(model=self._model, random_seed=int(self._rng.integers(0, 2**31 - 1)))


def create_controller(*, model_path: Path = DEFAULT_MODEL_PATH) -> LearnedDynamicsMpcController:
    model = DynamicsModel.load(model_path)
    return LearnedDynamicsMpcController(model=model)


__all__ = ["LearnedDynamicsMpcController", "LocalState", "create_controller"]
