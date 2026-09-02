from __future__ import annotations

import numpy as np

from controllers.learned_dynamics_mpc import (
    NORMAL_LOOKAHEAD_SECONDS,
    RECOVERY_BRAKE_MAX_THROTTLE,
    RECOVERY_BRAKE_SPEED_THRESHOLD_MPS,
    RECOVERY_LOOKAHEAD_SECONDS,
    RECOVERY_MAX_THROTTLE_COMMAND,
    LearnedDynamicsMpcController,
    _plan_with_correlated_noise,
)
from controllers.mpc_baseline import kinematic_predict
from controllers.mpc_lib import LocalState
from racing import CameraSensors, OdometrySensors, RobotSensors


class _ConstantDtDummyModel:
    """Minimal stand-in for `DynamicsModel`: only `training_dt_s` and `predict_fn` are used."""

    def __init__(self, training_dt_s: float) -> None:
        self.training_dt_s = training_dt_s

    def predict_fn(self, states: np.ndarray, actions: np.ndarray, dt_s: float) -> np.ndarray:
        return kinematic_predict(states, actions, dt_s)


def test_plan_with_correlated_noise_returns_one_command_per_horizon_step() -> None:
    state = LocalState(
        speed_mps=2.0, heading_error_rad=0.0, center_offset_m=0.0,
        lookahead_offsets_m=(0.0, 0.0, 0.0), front_wall_m=float("inf"),
    )
    plan = _plan_with_correlated_noise(
        state, kinematic_predict, horizon=10, samples=32, dt_s=0.05,
        rng=np.random.default_rng(0), warm_start=None, exploration_std=0.4, max_throttle=1.0,
    )
    assert len(plan) == 10
    assert all(-1.0 <= command.throttle <= 1.0 and -1.0 <= command.steer <= 1.0 for command in plan)


def test_plan_with_correlated_noise_holds_one_action_across_the_horizon_when_cold_started() -> None:
    state = LocalState(
        speed_mps=2.0, heading_error_rad=0.0, center_offset_m=0.0,
        lookahead_offsets_m=(0.0, 0.0, 0.0), front_wall_m=float("inf"),
    )
    plan = _plan_with_correlated_noise(
        state, kinematic_predict, horizon=10, samples=32, dt_s=0.05,
        rng=np.random.default_rng(0), warm_start=None, exploration_std=0.4, max_throttle=1.0,
    )
    assert all(command == plan[0] for command in plan)


def test_plan_with_correlated_noise_consistently_picks_the_correct_steer_direction() -> None:
    """The property this function exists for: a large, clear heading error should

    reliably produce a correctly-signed correction across independent random
    draws, unlike `mpc_lib.plan_action_sequence`'s independent per-step noise
    (see the module docstring for the direct comparison that motivated this).
    """
    state = LocalState(
        speed_mps=3.0, heading_error_rad=np.radians(60.0), center_offset_m=0.0,
        lookahead_offsets_m=(0.0, 0.0, 0.0), front_wall_m=float("inf"),
    )
    steers = [
        _plan_with_correlated_noise(
            state, kinematic_predict, horizon=60, samples=128, dt_s=1 / 60,
            rng=np.random.default_rng(seed), warm_start=None, exploration_std=0.4, max_throttle=1.0,
        )[0].steer
        for seed in range(10)
    ]
    # kinematic_predict: positive steer increases heading_error, so correcting
    # a large positive heading_error needs negative steer.
    assert all(steer < 0.0 for steer in steers)


def test_plan_with_correlated_noise_with_warm_start_stays_close_to_it() -> None:
    state = LocalState(
        speed_mps=2.0, heading_error_rad=0.0, center_offset_m=0.0,
        lookahead_offsets_m=(0.0, 0.0, 0.0), front_wall_m=float("inf"),
    )
    warm_start = np.full((5, 2), 0.5)
    plan = _plan_with_correlated_noise(
        state, kinematic_predict, horizon=5, samples=64, dt_s=0.05,
        rng=np.random.default_rng(7), warm_start=warm_start, exploration_std=0.05, max_throttle=1.0,
    )
    assert all(0.3 <= command.throttle <= 0.7 for command in plan)


def test_controller_uses_longer_horizon_when_off_track() -> None:
    controller = LearnedDynamicsMpcController(model=_ConstantDtDummyModel(training_dt_s=1 / 60))
    assert controller._normal_horizon == round(NORMAL_LOOKAHEAD_SECONDS / (1 / 60))
    assert controller._recovery_horizon == round(RECOVERY_LOOKAHEAD_SECONDS / (1 / 60))
    assert controller._recovery_horizon > controller._normal_horizon


def test_controller_returns_valid_command_when_badly_off_track() -> None:
    controller = LearnedDynamicsMpcController(model=_ConstantDtDummyModel(training_dt_s=1 / 60))
    sensors = RobotSensors(
        dt_s=1 / 60,
        odometry=OdometrySensors(speed_mps=1.0),
        camera=CameraSensors(center_offset_m=1.0, heading_error_degrees=70.0),
    )
    command = controller(sensors)
    assert -1.0 <= command.throttle <= 1.0
    assert -1.0 <= command.steer <= 1.0


def test_controller_does_not_accelerate_while_badly_off_track() -> None:
    """Regression test for the exact bug found via real telemetry: the planner

    accelerating hard (throttle 0.46-0.98) while already 42-56 degrees off
    heading and closing on a wall, before crashing (2026-09-01 lab notebook).
    """
    controller = LearnedDynamicsMpcController(model=_ConstantDtDummyModel(training_dt_s=1 / 60))
    sensors = RobotSensors(
        dt_s=1 / 60,
        odometry=OdometrySensors(speed_mps=8.0),
        camera=CameraSensors(center_offset_m=0.0, heading_error_degrees=50.0),
    )
    command = controller(sensors)
    assert command.throttle <= RECOVERY_MAX_THROTTLE_COMMAND


def test_controller_can_still_move_from_a_stop_with_a_modest_starting_offset() -> None:
    """Regression test: capping recovery throttle all the way to zero (tried first) meant a car

    spawning with a modest, ordinary starting `center_offset` (enough alone to trigger
    `needs_recovery`) could never move at all, since steering does nothing at zero speed.
    """
    controller = LearnedDynamicsMpcController(model=_ConstantDtDummyModel(training_dt_s=1 / 60))
    sensors = RobotSensors(
        dt_s=1 / 60,
        odometry=OdometrySensors(speed_mps=0.0),
        camera=CameraSensors(center_offset_m=-1.16, heading_error_degrees=0.0),
    )
    command = controller(sensors)
    assert command.throttle > 0.0


def test_controller_stops_accelerating_when_badly_off_track_at_speed() -> None:
    """Regression test: real telemetry (seed 2027) found the model's predicted steering response

    is unreliable at high speed combined with a large heading error (a combination rarely seen
    in training data). Forcing active (negative) braking there was tried and was a net
    regression across the full seed suite; capping further acceleration (not forcing braking)
    tested as a net improvement instead -- see the constant's comment for the full comparison.
    """
    controller = LearnedDynamicsMpcController(model=_ConstantDtDummyModel(training_dt_s=1 / 60))
    sensors = RobotSensors(
        dt_s=1 / 60,
        odometry=OdometrySensors(speed_mps=RECOVERY_BRAKE_SPEED_THRESHOLD_MPS + 1.0),
        camera=CameraSensors(center_offset_m=0.0, heading_error_degrees=60.0),
    )
    command = controller(sensors)
    assert command.throttle <= RECOVERY_BRAKE_MAX_THROTTLE
