"""Pure track geometry, layout constants, and centerline sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from math import hypot


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """Two-dimensional point on the racing track layout."""

    x: float
    z: float
    label: str = ""


@dataclass(frozen=True, slots=True)
class TrackSegment:
    """Line segment between two track centerline points."""

    start: TrackPoint
    end: TrackPoint

    @property
    def length(self) -> float:
        """Measure this centerline segment in world units."""
        return distance_between(self.start, self.end)


@dataclass(frozen=True, slots=True)
class TrackBounds:
    """Axis-aligned bounds of a track layout in X/Z space."""

    min_x: float
    max_x: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        """Measure the X span of these bounds."""
        return self.max_x - self.min_x

    @property
    def length(self) -> float:
        """Measure the Z span of these bounds."""
        return self.max_z - self.min_z


@dataclass(frozen=True, slots=True)
class TrackLayout:
    """Named pure centerline layout used by racing scenes."""

    track_id: str
    points: tuple[TrackPoint, ...]
    start_position: TrackPoint


TRACK_SCALE = 2.0
TRACK_LAYOUT_SCALE = 2.8
NOMINAL_CAR_WIDTH = 1.0 * TRACK_SCALE
TRACK_WIDTH = (2.3 * TRACK_SCALE) + NOMINAL_CAR_WIDTH
TRACK_ID_MUGELLO_SHORT = "mugello-short"
TRACK_ID_MUGELLO_SHORT_WIDE = "mugello-short-wide"
TRACK_ID_MUGELLO_SHORT_LONG = "mugello-short-long"


def _scaled_track_point(x: float, z: float, label: str = "") -> TrackPoint:
    # Mirror the traced map points so clockwise laps leave the start line toward San Donato.
    return TrackPoint(-x * TRACK_LAYOUT_SCALE, z * TRACK_LAYOUT_SCALE, label)


MAIN_STRAIGHT_ENTRY = _scaled_track_point(5.0, -6.2, "Main straight entry")
MAIN_STRAIGHT = _scaled_track_point(1.0, -6.1, "Main straight")
START_STRAIGHT = _scaled_track_point(-5.8, -5.9, "Start straight")
START_POSITION = TrackPoint(
    x=(MAIN_STRAIGHT.x + START_STRAIGHT.x) / 2,
    z=(MAIN_STRAIGHT.z + START_STRAIGHT.z) / 2,
    label="Start grid",
)

MUGELLO_SHORT_LAYOUT: tuple[TrackPoint, ...] = (
    MAIN_STRAIGHT_ENTRY,
    MAIN_STRAIGHT,
    START_STRAIGHT,
    _scaled_track_point(-12.4, -5.8, "San Donato approach"),
    _scaled_track_point(-16.2, -5.2, "San Donato braking"),
    _scaled_track_point(-17.6, -3.8, "San Donato"),
    _scaled_track_point(-16.8, -2.2, "San Donato exit"),
    _scaled_track_point(-14.5, -1.6, "Luco"),
    _scaled_track_point(-13.0, -0.6, "Luco climb"),
    _scaled_track_point(-12.7, 2.3, "Poggiosecco"),
    _scaled_track_point(-10.8, 3.8, "Poggiosecco exit"),
    _scaled_track_point(-6.2, 3.0, "Materassi"),
    _scaled_track_point(-2.2, 2.3, "Materassi exit"),
    _scaled_track_point(-0.2, 4.1, "Borgo San Lorenzo"),
    _scaled_track_point(1.6, 4.3, "Borgo crest"),
    _scaled_track_point(4.9, 3.4, "Casanova approach"),
    _scaled_track_point(7.5, 2.3, "Casanova"),
    _scaled_track_point(8.8, -0.6, "Casanova exit"),
    _scaled_track_point(8.4, -3.5, "Return bend"),
    _scaled_track_point(7.0, -5.5, "Final return"),
)


def _scaled_layout_about_start(
    points: tuple[TrackPoint, ...],
    *,
    x_scale: float,
    z_scale: float,
) -> tuple[TrackPoint, ...]:
    return tuple(
        TrackPoint(
            x=START_POSITION.x + (point.x - START_POSITION.x) * x_scale,
            z=START_POSITION.z + (point.z - START_POSITION.z) * z_scale,
            label=point.label,
        )
        for point in points
    )


MUGELLO_SHORT_WIDE_LAYOUT = _scaled_layout_about_start(
    MUGELLO_SHORT_LAYOUT,
    x_scale=1.06,
    z_scale=0.96,
)
MUGELLO_SHORT_LONG_LAYOUT = _scaled_layout_about_start(
    MUGELLO_SHORT_LAYOUT,
    x_scale=0.94,
    z_scale=1.06,
)
TRACK_LAYOUTS = (
    TrackLayout(
        track_id=TRACK_ID_MUGELLO_SHORT,
        points=MUGELLO_SHORT_LAYOUT,
        start_position=START_POSITION,
    ),
    TrackLayout(
        track_id=TRACK_ID_MUGELLO_SHORT_WIDE,
        points=MUGELLO_SHORT_WIDE_LAYOUT,
        start_position=START_POSITION,
    ),
    TrackLayout(
        track_id=TRACK_ID_MUGELLO_SHORT_LONG,
        points=MUGELLO_SHORT_LONG_LAYOUT,
        start_position=START_POSITION,
    ),
)


def track_layout_ids() -> tuple[str, ...]:
    """Return supported track layout ids."""
    return tuple(layout.track_id for layout in TRACK_LAYOUTS)


def track_layout_by_id(track_id: str) -> TrackLayout:
    """Return a named pure track layout."""
    for layout in TRACK_LAYOUTS:
        if layout.track_id == track_id:
            return layout
    valid_ids = ", ".join(track_layout_ids())
    raise ValueError(f"unknown track layout: {track_id}; expected one of {valid_ids}")


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to an inclusive numeric range."""
    return min(max(value, low), high)


def distance_between(start: TrackPoint, end: TrackPoint) -> float:
    """Return the X/Z distance between two track points."""
    return hypot(end.x - start.x, end.z - start.z)


def closed_track_points(points: tuple[TrackPoint, ...] = MUGELLO_SHORT_LAYOUT) -> tuple[TrackPoint, ...]:
    """Return centerline points with the starting point repeated at the end."""
    if len(points) < 3:
        raise ValueError("a racing track needs at least three centerline points")
    return (*points, points[0])


def track_segments(points: tuple[TrackPoint, ...] = MUGELLO_SHORT_LAYOUT) -> tuple[TrackSegment, ...]:
    """Return centerline segments, including the closing segment."""
    closed_points = closed_track_points(points)
    return tuple(TrackSegment(start=start, end=end) for start, end in pairwise(closed_points))


def total_track_length(points: tuple[TrackPoint, ...] = MUGELLO_SHORT_LAYOUT) -> float:
    """Return the full closed-loop centerline length."""
    return sum(segment.length for segment in track_segments(points))


@lru_cache(maxsize=16)
def sampled_track_centerline(
    points: tuple[TrackPoint, ...] = MUGELLO_SHORT_LAYOUT,
    *,
    samples_per_segment: int = 8,
) -> tuple[TrackPoint, ...]:
    """Return smooth Catmull-Rom samples around the track centerline."""
    if samples_per_segment < 1:
        raise ValueError("samples_per_segment must be at least one")
    if len(points) < 4:
        raise ValueError("a smooth racing track needs at least four centerline points")

    samples: list[TrackPoint] = []
    for index, point in enumerate(points):
        previous_point = points[index - 1]
        next_point = points[(index + 1) % len(points)]
        after_next_point = points[(index + 2) % len(points)]

        for sample_index in range(samples_per_segment):
            t = sample_index / samples_per_segment
            samples.append(
                TrackPoint(
                    x=_catmull_rom(previous_point.x, point.x, next_point.x, after_next_point.x, t),
                    z=_catmull_rom(previous_point.z, point.z, next_point.z, after_next_point.z, t),
                    label=point.label if sample_index == 0 else "",
                )
            )
    return tuple(samples)


def track_bounds(
    points: tuple[TrackPoint, ...] = MUGELLO_SHORT_LAYOUT,
    *,
    margin: float = TRACK_WIDTH,
) -> TrackBounds:
    """Return X/Z bounds for a set of track points plus margin."""
    xs = tuple(point.x for point in points)
    zs = tuple(point.z for point in points)
    return TrackBounds(
        min_x=min(xs) - margin,
        max_x=max(xs) + margin,
        min_z=min(zs) - margin,
        max_z=max(zs) + margin,
    )


def _catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
