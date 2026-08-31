from __future__ import annotations

import math
from pathlib import Path

import torch

from controllers.mlp_rl import OBSERVATION_SIZE, Controller, RacingPolicy, observation_values
from controllers.train_mlp_rl import (
    EpisodeMetrics,
    _promotable_damage_policy_score,  # pyright: ignore[reportPrivateUsage]
    _promotable_policy_score,  # pyright: ignore[reportPrivateUsage]
)
from racing import (
    CameraSensors,
    LidarSensors,
    OdometrySensors,
    RobotCommand,
    RobotSensors,
    load_student_submission,
)


def test_mlp_observation_is_fixed_size_and_finite() -> None:
    sensors = RobotSensors(
        odometry=OdometrySensors(speed_mps=7.5),
        wall_lidar=LidarSensors(angles_degrees=(0.0, 90.0), distances_m=(float("inf"), 3.0)),
        camera=CameraSensors(
            center_offset_m=1.0,
            heading_error_degrees=-15.0,
            lookahead_offsets_m=(0.5,),
            lookahead_distances_m=(4.0,),
        ),
    )

    values = observation_values(sensors)

    assert len(values) == OBSERVATION_SIZE
    assert all(math.isfinite(value) for value in values)


def test_mlp_controller_loads_cpu_weights_and_returns_bounded_command(tmp_path: Path) -> None:
    output = tmp_path / "weights.pt"
    torch.save(RacingPolicy().state_dict(), output)
    controller = Controller(output)

    command = controller(RobotSensors())

    assert isinstance(command, RobotCommand)
    assert 0.0 <= command.throttle <= 1.0
    assert -1.0 <= command.steer <= 1.0
    assert next(controller.model.parameters()).device.type == "cpu"


def test_throttle_only_policy_never_brakes_or_drives_against_reverse_motion() -> None:
    model = RacingPolicy()
    observation = torch.zeros(OBSERVATION_SIZE)
    observation[0] = -0.1

    action = model.actions_from_indices(
        torch.tensor(2),
        torch.tensor(20),
        observations=observation,
        no_brakes=True,
    )

    assert action[0].item() == 0.0


def test_bundled_mlp_checkpoint_loads_through_student_factory() -> None:
    submission = load_student_submission("controllers.mlp_rl")

    command = submission.controller(RobotSensors())

    assert isinstance(command, RobotCommand)
    assert submission.display_name == "Torch MLP RL"


def test_training_score_balances_speed_drift_and_survival_after_distance_floor() -> None:
    def result(
        *,
        distance_m: float,
        damage: float,
        drift_distance_m: float = 0.0,
        eliminated: bool = False,
    ) -> EpisodeMetrics:
        return EpisodeMetrics(
            seed=110,
            elapsed_seconds=30.0,
            distance_m=distance_m,
            laps=3,
            damage=damage,
            eliminated=eliminated,
            wall_contact_seconds=0.0,
            marshal_count=0,
            max_speed_mps=30.0,
            drift_distance_m=drift_distance_m,
            fastest_lap_seconds=9.0,
        )

    fast_and_drifting = _promotable_policy_score(
        (result(distance_m=650.0, damage=0.6, drift_distance_m=30.0),),
        minimum_distance_m=500.0,
    )
    slower_and_safer = _promotable_policy_score(
        (result(distance_m=550.0, damage=0.4),),
        minimum_distance_m=500.0,
    )
    below_distance_floor = _promotable_policy_score(
        (result(distance_m=499.0, damage=0.0),),
        minimum_distance_m=500.0,
    )
    crashed_out = _promotable_policy_score(
        (result(distance_m=700.0, damage=1.0, drift_distance_m=50.0, eliminated=True),),
        minimum_distance_m=500.0,
    )

    assert fast_and_drifting is not None
    assert slower_and_safer is not None
    assert fast_and_drifting > slower_and_safer
    assert below_distance_floor is None
    assert crashed_out is None


def test_damage_training_score_maximizes_completed_lap_damage_but_rejects_crashes() -> None:
    def result(
        *,
        lap_damage: float,
        damage: float,
        laps: int = 3,
        eliminated: bool = False,
        brake_applied: bool = False,
    ) -> EpisodeMetrics:
        return EpisodeMetrics(
            seed=110,
            elapsed_seconds=30.0,
            distance_m=600.0,
            laps=laps,
            damage=damage,
            eliminated=eliminated,
            wall_contact_seconds=0.0,
            marshal_count=0,
            max_speed_mps=30.0,
            maximum_lap_damage=lap_damage,
            brake_applied=brake_applied,
        )

    cooked = _promotable_damage_policy_score((result(lap_damage=0.45, damage=0.9),), minimum_laps=3)
    timid = _promotable_damage_policy_score((result(lap_damage=0.10, damage=0.2),), minimum_laps=3)
    too_few_laps = _promotable_damage_policy_score(
        (result(lap_damage=0.8, damage=0.9, laps=2),),
        minimum_laps=3,
    )
    crashed = _promotable_damage_policy_score(
        (result(lap_damage=0.8, damage=1.0, eliminated=True),),
        minimum_laps=3,
    )
    braking = _promotable_damage_policy_score(
        (result(lap_damage=0.8, damage=0.9, brake_applied=True),),
        minimum_laps=3,
        require_brake_free=True,
    )

    assert cooked is not None
    assert timid is not None
    assert cooked > timid
    assert too_few_laps is None
    assert crashed is None
    assert braking is None
