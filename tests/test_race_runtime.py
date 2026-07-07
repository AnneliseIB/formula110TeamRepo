from __future__ import annotations

import pytest

from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG, vehicle_collision_bounds
from racing.race.progress import build_track_progress_model, track_pose_at_distance
from racing.race.runtime import race_spawn_poses, seeded_race_start_finish_pose, start_finish_pose_for_progress
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
