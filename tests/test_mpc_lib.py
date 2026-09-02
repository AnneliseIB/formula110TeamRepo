from __future__ import annotations

import numpy as np

from controllers.mpc_baseline import kinematic_predict
from controllers.mpc_lib import (
    LocalState,
    local_state_from_sensors,
    plan_action_sequence,
    rollout_cost,
    shift_plan_for_warm_start,
)
from racing import CameraSensors, LidarSensors, OdometrySensors, RobotCommand, RobotSensors


def test_local_state_from_sensors_reads_camera_and_odometry_fields() -> None:
    sensors = RobotSensors(
        odometry=OdometrySensors(speed_mps=3.5),
        camera=CameraSensors(center_offset_m=0.4, heading_error_degrees=90.0, lookahead_offsets_m=(0.1, 0.2, 0.3)),
        wall_lidar=LidarSensors(distances_m=(5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5)),
    )

    state = local_state_from_sensors(sensors)

    assert state.speed_mps == 3.5
    assert state.heading_error_rad == np.pi / 2
    assert state.center_offset_m == 0.4
    assert state.lookahead_offsets_m == (0.1, 0.2, 0.3)
    assert state.front_wall_m == sensors.wall_lidar.front_m


def test_local_state_from_sensors_pads_short_lookahead_tuples() -> None:
    sensors = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(), lookahead_distances_m=()))

    state = local_state_from_sensors(sensors)

    assert state.lookahead_offsets_m == (0.0, 0.0, 0.0)


def test_kinematic_predict_increases_speed_toward_full_throttle() -> None:
    states = np.zeros((1, 6), dtype=np.float64)
    actions = np.array([[1.0, 0.0]], dtype=np.float64)

    next_states = kinematic_predict(states, actions, dt_s=0.1)

    assert next_states[0, 0] > 0.0


def test_kinematic_predict_brakes_toward_zero_from_positive_speed() -> None:
    states = np.array([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    actions = np.array([[0.0, 0.0]], dtype=np.float64)

    next_states = kinematic_predict(states, actions, dt_s=0.1)

    assert next_states[0, 0] < 10.0


def test_kinematic_predict_positive_steer_turns_heading_right() -> None:
    states = np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    actions = np.array([[0.0, 1.0]], dtype=np.float64)

    next_states = kinematic_predict(states, actions, dt_s=0.1)

    assert next_states[0, 1] > 0.0


def test_rollout_cost_penalizes_off_track_excursion() -> None:
    on_track = np.zeros((1, 3, 6), dtype=np.float64)
    on_track[:, :, 0] = 5.0

    off_track = on_track.copy()
    off_track[:, :, 2] = 100.0

    actions = np.zeros((1, 3, 2), dtype=np.float64)

    on_track_cost = rollout_cost(on_track, actions, dt_s=0.1)
    off_track_cost = rollout_cost(off_track, actions, dt_s=0.1)

    assert off_track_cost[0] > on_track_cost[0]


def test_rollout_cost_rewards_forward_progress() -> None:
    fast = np.zeros((1, 3, 6), dtype=np.float64)
    fast[:, :, 0] = 10.0

    slow = np.zeros((1, 3, 6), dtype=np.float64)
    slow[:, :, 0] = 1.0

    actions = np.zeros((1, 3, 2), dtype=np.float64)

    assert rollout_cost(fast, actions, dt_s=0.1)[0] < rollout_cost(slow, actions, dt_s=0.1)[0]


def test_plan_action_sequence_is_deterministic_for_a_seeded_generator() -> None:
    state = LocalState(
        speed_mps=2.0,
        heading_error_rad=0.0,
        center_offset_m=0.0,
        lookahead_offsets_m=(0.0, 0.0, 0.0),
        front_wall_m=float("inf"),
    )

    first_plan = plan_action_sequence(
        state, kinematic_predict, horizon=4, samples=16, dt_s=0.05, rng=np.random.default_rng(42)
    )
    second_plan = plan_action_sequence(
        state, kinematic_predict, horizon=4, samples=16, dt_s=0.05, rng=np.random.default_rng(42)
    )

    assert first_plan == second_plan


def test_plan_action_sequence_returns_one_command_per_horizon_step() -> None:
    state = LocalState(
        speed_mps=0.0,
        heading_error_rad=0.0,
        center_offset_m=0.0,
        lookahead_offsets_m=(0.0, 0.0, 0.0),
        front_wall_m=float("inf"),
    )

    plan = plan_action_sequence(state, kinematic_predict, horizon=6, samples=8, dt_s=0.05, rng=np.random.default_rng(1))

    assert len(plan) == 6
    assert all(-1.0 <= command.throttle <= 1.0 and -1.0 <= command.steer <= 1.0 for command in plan)


def test_shift_plan_for_warm_start_drops_first_action_and_repeats_last() -> None:
    plan = (
        RobotCommand(throttle=0.1, steer=0.0),
        RobotCommand(throttle=0.2, steer=0.0),
        RobotCommand(throttle=0.3, steer=0.0),
    )

    warm_start = shift_plan_for_warm_start(plan)

    np.testing.assert_array_equal(warm_start, [[0.2, 0.0], [0.3, 0.0], [0.3, 0.0]])


def test_plan_action_sequence_with_warm_start_stays_close_to_it() -> None:
    state = LocalState(
        speed_mps=2.0,
        heading_error_rad=0.0,
        center_offset_m=0.0,
        lookahead_offsets_m=(0.0, 0.0, 0.0),
        front_wall_m=float("inf"),
    )
    warm_start = np.full((5, 2), 0.5)

    plan = plan_action_sequence(
        state,
        kinematic_predict,
        horizon=5,
        samples=64,
        dt_s=0.05,
        rng=np.random.default_rng(7),
        warm_start=warm_start,
        exploration_std=0.05,
    )

    # With tiny exploration noise around a 0.5 warm start, the chosen plan should
    # land close to 0.5 rather than anywhere in the full [-1, 1] range.
    assert all(0.3 <= command.throttle <= 0.7 for command in plan)
