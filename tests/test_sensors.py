from __future__ import annotations

import pytest

from racing.race.progress import build_track_progress_model
from racing.race.sensors import camera_competitor_readings, camera_sensors_from_track
from racing.track.world import TrackPoint


class FakeNodePath:
    def __init__(self, position: tuple[float, float, float], heading_degrees: float = 0.0) -> None:
        self._position = position
        self._heading_degrees = heading_degrees

    def getPos(self) -> tuple[float, float, float]:
        return self._position

    def getH(self) -> float:
        return self._heading_degrees


def square_track_model_points() -> tuple[TrackPoint, ...]:
    return (
        TrackPoint(0.0, 0.0),
        TrackPoint(0.0, 10.0),
        TrackPoint(10.0, 10.0),
        TrackPoint(10.0, 0.0),
    )


def test_camera_sensors_report_center_and_heading_error() -> None:
    model = build_track_progress_model(square_track_model_points())

    sensors = camera_sensors_from_track(model=model, position=TrackPoint(2.0, 5.0), heading_degrees=0.0)

    assert sensors.visible
    assert sensors.center_offset_m == pytest.approx(-2.0)
    assert sensors.heading_error_degrees == pytest.approx(0.0)
    assert len(sensors.lookahead_offsets_m) == len(sensors.lookahead_distances_m)


def test_camera_competitor_readings_are_relative_and_sorted() -> None:
    readings = camera_competitor_readings(
        position=(0.0, 0.0, 0.0),
        heading_degrees=0.0,
        speed_mps=10.0,
        competitor_chassis_nps=(
            FakeNodePath((3.0, 0.0, 4.0), heading_degrees=45.0),
            FakeNodePath((0.0, 0.0, 2.0), heading_degrees=0.0),
        ),
        competitor_speeds_mps=(4.0, 7.0),
    )

    assert tuple(round(reading.distance_m, 3) for reading in readings) == (2.0, 5.0)
    assert readings[0].angle_degrees == pytest.approx(0.0)
    assert readings[0].closing_speed_mps == pytest.approx(3.0)
    assert readings[1].relative_heading_degrees == pytest.approx(45.0)


def test_camera_competitor_readings_honor_max_competitors() -> None:
    readings = camera_competitor_readings(
        position=(0.0, 0.0, 0.0),
        heading_degrees=0.0,
        competitor_chassis_nps=(
            FakeNodePath((0.0, 0.0, 4.0)),
            FakeNodePath((0.0, 0.0, 2.0)),
            FakeNodePath((0.0, 0.0, 3.0)),
        ),
        max_competitors=2,
    )

    assert tuple(reading.distance_m for reading in readings) == (2.0, 3.0)
