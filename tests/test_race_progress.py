from __future__ import annotations

import pytest

from racing.race.progress import (
    LapProgressTracker,
    build_track_progress_model,
    damage_adjusted_score,
    heading_error_degrees,
    project_track_position,
    track_pose_at_distance,
)
from racing.track.world import TrackPoint


def square_track_model_points() -> tuple[TrackPoint, ...]:
    return (
        TrackPoint(0.0, 0.0),
        TrackPoint(0.0, 10.0),
        TrackPoint(10.0, 10.0),
        TrackPoint(10.0, 0.0),
    )


def test_project_track_position_returns_nearest_centerline_progress() -> None:
    model = build_track_progress_model(square_track_model_points())

    projection = project_track_position(model, TrackPoint(2.0, 5.0))

    assert projection.nearest_center == TrackPoint(0.0, 5.0)
    assert projection.progress_distance_m == pytest.approx(5.0)
    assert projection.lap_progress == pytest.approx(5.0 / 40.0)
    assert projection.heading_degrees == pytest.approx(0.0)


def test_track_pose_wraps_distance_around_closed_loop() -> None:
    model = build_track_progress_model(square_track_model_points())

    pose = track_pose_at_distance(model, 45.0)

    assert pose.position == TrackPoint(0.0, 5.0)
    assert pose.progress_distance_m == pytest.approx(5.0)


def test_lap_progress_tracker_counts_wrapped_forward_progress() -> None:
    tracker = LapProgressTracker(total_length_m=100.0, starting_progress_distance_m=90.0)

    tracker.update(95.0, 1.0)
    tracker.update(5.0, 2.0)
    tracker.update(50.0, 3.0)
    tracker.update(91.0, 4.0)

    assert tracker.best_distance_m == pytest.approx(101.0)
    assert tracker.completed_lap
    assert tracker.lap_count == 1
    assert tracker.lap_time_seconds == 4.0


def test_heading_error_wraps_to_signed_shortest_turn() -> None:
    assert heading_error_degrees(current_heading_degrees=350.0, target_heading_degrees=10.0) == pytest.approx(20.0)
    assert heading_error_degrees(current_heading_degrees=10.0, target_heading_degrees=350.0) == pytest.approx(-20.0)


def test_damage_adjusted_score_scales_by_remaining_health() -> None:
    assert damage_adjusted_score(100.0, damage=0.25) == pytest.approx(75.0)
    assert damage_adjusted_score(100.0, damage=2.0) == 0.0
