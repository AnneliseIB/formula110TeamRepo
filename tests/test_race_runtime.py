from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG, vehicle_collision_bounds
from racing.race.progress import (
    LapProgressTracker,
    TrackProjection,
    build_track_progress_model,
    track_pose_at_distance,
)
from racing.race.runtime import (
    RaceCarRuntime,
    RaceContactState,
    race_scored_distance_m,
    race_spawn_poses,
    seeded_race_start_finish_pose,
    start_finish_pose_for_progress,
    update_race_runtime_after_step,
)
from racing.track.spatial import track_left_vector
from racing.track.world import TrackPoint


def square_track_model_points() -> tuple[TrackPoint, ...]:
    return (
        TrackPoint(0.0, 0.0),
        TrackPoint(0.0, 10.0),
        TrackPoint(10.0, 10.0),
        TrackPoint(10.0, 0.0),
    )


def test_start_finish_pose_is_two_car_lengths_ahead_of_start_progress() -> None:
    model = build_track_progress_model(square_track_model_points())
    car_length_m = vehicle_collision_bounds(FORMULA_VEHICLE_PHYSICS_CONFIG).half_length * 2.0

    pose = start_finish_pose_for_progress(
        model=model,
        start_progress_distance_m=3.0,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
    )

    assert pose.progress_distance_m == pytest.approx(3.0 + car_length_m * 2.0)
    assert pose.position.x == pytest.approx(0.0)
    assert pose.position.z == pytest.approx(7.6)
    assert pose.heading_degrees == pytest.approx(0.0)


def test_single_car_spawn_position_is_deterministic_for_seed() -> None:
    model = build_track_progress_model(square_track_model_points())

    first = race_spawn_poses(1, model=model, random_seed=271, race_index=1)[0]
    repeated = race_spawn_poses(1, model=model, random_seed=271, race_index=1)[0]
    different = race_spawn_poses(1, model=model, random_seed=272, race_index=1)[0]

    assert first == repeated
    assert first.progress_distance_m != different.progress_distance_m


def test_seeded_start_finish_pose_matches_front_spawn_progress() -> None:
    model = build_track_progress_model(square_track_model_points())
    spawn_pose = race_spawn_poses(
        1,
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=110,
        race_index=2,
    )[0]
    car_length_m = vehicle_collision_bounds(FORMULA_VEHICLE_PHYSICS_CONFIG).half_length * 2.0
    expected_pose = track_pose_at_distance(model, spawn_pose.progress_distance_m + car_length_m * 2.0)

    start_finish_pose = seeded_race_start_finish_pose(
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=110,
        race_index=2,
    )

    assert start_finish_pose.progress_distance_m == pytest.approx(expected_pose.progress_distance_m)
    assert start_finish_pose.position.x == pytest.approx(expected_pose.position.x)
    assert start_finish_pose.position.z == pytest.approx(expected_pose.position.z)
    assert start_finish_pose.heading_degrees == pytest.approx(expected_pose.heading_degrees)


def test_seeded_start_finish_pose_uses_front_slot_for_shuffled_grid() -> None:
    model = build_track_progress_model(square_track_model_points())
    spawn_poses = race_spawn_poses(
        4,
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=110,
        race_index=2,
    )
    car_length_m = vehicle_collision_bounds(FORMULA_VEHICLE_PHYSICS_CONFIG).half_length * 2.0

    start_finish_pose = seeded_race_start_finish_pose(
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=110,
        race_index=2,
    )
    distances_ahead = tuple(
        (start_finish_pose.progress_distance_m - spawn_pose.progress_distance_m) % model.total_length_m
        for spawn_pose in spawn_poses
    )

    assert min(distances_ahead) == pytest.approx(car_length_m * 2.0)


def test_two_car_grid_starts_inside_car_behind_outside_car() -> None:
    model = build_track_progress_model(square_track_model_points())
    spawn_poses = race_spawn_poses(
        2,
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=110,
        race_index=2,
    )
    start_finish_pose = seeded_race_start_finish_pose(
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=110,
        race_index=2,
    )

    def lateral_offset(spawn_pose_index: int) -> float:
        spawn_pose = spawn_poses[spawn_pose_index]
        center_pose = track_pose_at_distance(model, spawn_pose.progress_distance_m)
        left_x, left_z = track_left_vector(spawn_pose.heading_degrees)
        return (
            (spawn_pose.position[0] - center_pose.position.x) * left_x
            + (spawn_pose.position[2] - center_pose.position.z) * left_z
        )

    inside_index = min(range(2), key=lateral_offset)
    outside_index = max(range(2), key=lateral_offset)
    distances_to_finish = tuple(
        (start_finish_pose.progress_distance_m - spawn_pose.progress_distance_m) % model.total_length_m
        for spawn_pose in spawn_poses
    )
    car_length_m = vehicle_collision_bounds(FORMULA_VEHICLE_PHYSICS_CONFIG).half_length * 2.0

    assert lateral_offset(inside_index) < 0.0
    assert lateral_offset(outside_index) > 0.0
    assert distances_to_finish[inside_index] - distances_to_finish[outside_index] == pytest.approx(
        car_length_m * 2.25
    )


def test_race_scored_distance_ignores_damage_and_applies_marshal_penalty() -> None:
    runtime = RaceCarRuntime(
        robot=cast(Any, SimpleNamespace(damage=1.0, eliminated=True)),
        tracker=LapProgressTracker(total_length_m=100.0, best_distance_m=42.0),
        marshal_penalty_m=5.0,
    )

    assert race_scored_distance_m(runtime) == 37.0


def test_race_progress_counts_while_car_is_touching_wall_and_another_car() -> None:
    runtime = RaceCarRuntime(
        robot=cast(
            Any,
            SimpleNamespace(
                vehicle=SimpleNamespace(getCurrentSpeedKmHour=lambda: 18.0),
                damage=0.0,
                eliminated=False,
            ),
        ),
        tracker=LapProgressTracker(total_length_m=100.0, starting_progress_distance_m=10.0),
    )
    projection = TrackProjection(
        position=TrackPoint(0.0, 15.0),
        nearest_center=TrackPoint(0.0, 15.0),
        progress_distance_m=15.0,
        lap_progress=0.15,
        signed_distance_to_center_m=0.0,
        heading_degrees=0.0,
    )

    update_race_runtime_after_step(
        runtime=runtime,
        projection=projection,
        contact_state=RaceContactState(wall_contact=1.0, car_contact=1.0),
        elapsed_seconds=1.0,
        delta_seconds=1.0,
    )

    assert runtime.tracker.best_distance_m == 5.0
    assert runtime.tracker.last_counted_progress_delta_m == 5.0
    assert runtime.tracker.penalized_distance_m == 0.0
    assert runtime.tracker.wall_contact_seconds == 1.0
    assert runtime.tracker.car_contact_seconds == 1.0
