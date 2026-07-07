"""Track path, wall, and ribbon mesh construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from itertools import combinations, pairwise
from math import atan2, degrees, hypot
from typing import Any, cast

from racing.track.world import TrackPoint


@dataclass(frozen=True, slots=True)
class RaisedKerbCollisionSegment:
    """Oriented box collider dimensions for one visible kerb slab."""

    center_x: float
    center_y: float
    center_z: float
    length: float
    width: float
    height: float
    heading_degrees: float
    slab_index: int


@dataclass(frozen=True, slots=True)
class WallPaintOffsets:
    """Offset coordinates for visual paint bands attached to a rendered wall."""

    inner_face_offset: float
    top_inner_offset: float
    top_outer_offset: float


@dataclass(frozen=True, slots=True)
class _RaisedKerbSlabPiece:
    slab_index: int
    inner_start: tuple[float, float, float]
    outer_start: tuple[float, float, float]
    outer_end: tuple[float, float, float]
    inner_end: tuple[float, float, float]


def _tuple_segment_pose(
    segment: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float, float, float]:
    start, end = segment
    dx = end[0] - start[0]
    dz = end[2] - start[2]
    center_x = (start[0] + end[0]) / 2
    center_z = (start[2] + end[2]) / 2
    heading = degrees(atan2(dz, dx))
    return center_x, center_z, hypot(dx, dz), heading


def _tuple_point_segments(
    points: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    return tuple((point, points[(index + 1) % len(points)]) for index, point in enumerate(points))


def ribbon_mesh(
    ursina: Any,
    samples: tuple[TrackPoint, ...],
    *,
    inner_offset: float,
    outer_offset: float,
    y: float,
    uv_scale: float,
) -> Any:
    """Build one flat strip mesh between two offset copies of the track path."""
    inner_points = clean_offset_path(samples, inner_offset, y)
    outer_points = clean_offset_path(samples, outer_offset, y)
    return _flat_ring_mesh(ursina, inner_points, outer_points, uv_scale=uv_scale)


def segmented_ribbon_mesh(
    ursina: Any,
    samples: tuple[TrackPoint, ...],
    *,
    inner_offset: float,
    outer_offset: float,
    y: float,
    uv_scale: float,
) -> Any:
    """Build a track strip as separate quads so lighting has local normals."""
    inner_points = _offset_path(samples, inner_offset, y)
    outer_points = _offset_path(samples, outer_offset, y)
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []

    for index, inner_point in enumerate(inner_points):
        next_index = (index + 1) % len(inner_points)
        segment_vertices = (
            inner_point,
            outer_points[index],
            outer_points[next_index],
            inner_points[next_index],
        )
        vertex_start = len(vertices)
        vertices.extend(segment_vertices)
        triangles.append((vertex_start, vertex_start + 1, vertex_start + 2))
        triangles.append((vertex_start, vertex_start + 2, vertex_start + 3))
        uvs.extend((point[0] / uv_scale, point[2] / uv_scale) for point in segment_vertices)
        normals.extend((0.0, 1.0, 0.0) for _ in segment_vertices)

    return ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True)


def ribbon_chunk_meshes(
    ursina: Any,
    samples: tuple[TrackPoint, ...],
    *,
    inner_offset: float,
    outer_offset: float,
    y: float,
    uv_scale: float,
    segments_per_chunk: int,
) -> tuple[Any, ...]:
    """Split a long track strip into smaller meshes for better lighting."""
    if segments_per_chunk < 1:
        raise ValueError("segments_per_chunk must be at least one")

    inner_points = _offset_path(samples, inner_offset, y)
    outer_points = _offset_path(samples, outer_offset, y)
    meshes: list[Any] = []
    for start_index in range(0, len(inner_points), segments_per_chunk):
        vertices: list[tuple[float, float, float]] = []
        triangles: list[tuple[int, int, int]] = []
        uvs: list[tuple[float, float]] = []
        normals: list[tuple[float, float, float]] = []
        for offset in range(segments_per_chunk):
            index = (start_index + offset) % len(inner_points)
            if start_index + offset >= len(inner_points):
                break
            next_index = (index + 1) % len(inner_points)
            segment_vertices = (
                inner_points[index],
                outer_points[index],
                outer_points[next_index],
                inner_points[next_index],
            )
            vertex_start = len(vertices)
            vertices.extend(segment_vertices)
            triangles.append((vertex_start, vertex_start + 1, vertex_start + 2))
            triangles.append((vertex_start, vertex_start + 2, vertex_start + 3))
            uvs.extend((point[0] / uv_scale, point[2] / uv_scale) for point in segment_vertices)
            normals.extend((0.0, 1.0, 0.0) for _ in segment_vertices)
        meshes.append(ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True))

    return tuple(meshes)


def raised_kerb_chunk_meshes(
    ursina: Any,
    samples: tuple[TrackPoint, ...],
    *,
    inner_offset: float,
    outer_offset: float,
    base_y: float,
    low_height: float,
    ridge_height: float,
    uv_scale: float,
    slab_length: float,
) -> tuple[Any, ...]:
    """Return raised, alternating kerb slab meshes split by travelled distance."""
    _validate_raised_kerb_profile(low_height=low_height, ridge_height=ridge_height)

    pieces = _raised_kerb_slab_pieces(
        samples,
        inner_offset=inner_offset,
        outer_offset=outer_offset,
        base_y=base_y,
        slab_length=slab_length,
    )
    meshes: list[Any] = []
    for slab_index, slab_pieces in _group_kerb_pieces_by_slab(pieces):
        vertices: list[tuple[float, float, float]] = []
        triangles: list[tuple[int, int, int]] = []
        uvs: list[tuple[float, float]] = []
        normals: list[tuple[float, float, float]] = []
        segment_height = raised_kerb_slab_height(slab_index, low_height=low_height, ridge_height=ridge_height)
        for piece in slab_pieces:
            _append_kerb_slab(
                vertices,
                triangles,
                uvs,
                normals,
                inner_start=piece.inner_start,
                outer_start=piece.outer_start,
                outer_end=piece.outer_end,
                inner_end=piece.inner_end,
                top_y=base_y + segment_height,
                uv_scale=uv_scale,
            )
        meshes.append(ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True))

    return tuple(meshes)


def raised_kerb_collision_segments(
    samples: tuple[TrackPoint, ...],
    *,
    inner_offset: float,
    outer_offset: float,
    base_y: float,
    low_height: float,
    ridge_height: float,
    slab_length: float,
) -> tuple[RaisedKerbCollisionSegment, ...]:
    """Return flat oriented boxes covering each distance-based kerb slab."""
    _validate_raised_kerb_profile(low_height=low_height, ridge_height=ridge_height)

    pieces = _raised_kerb_slab_pieces(
        samples,
        inner_offset=inner_offset,
        outer_offset=outer_offset,
        base_y=base_y,
        slab_length=slab_length,
    )
    segments: list[RaisedKerbCollisionSegment] = []
    for piece in pieces:
        center_start = _midpoint(piece.inner_start, piece.outer_start)
        center_end = _midpoint(piece.inner_end, piece.outer_end)
        dx = center_end[0] - center_start[0]
        dz = center_end[2] - center_start[2]
        length = hypot(dx, dz)
        width = abs(outer_offset - inner_offset)
        if length <= 0 or width <= 0:
            raise ValueError("kerb collision segments must have positive length and width")

        height = low_height
        segments.append(
            RaisedKerbCollisionSegment(
                center_x=(center_start[0] + center_end[0]) / 2,
                center_y=base_y + height / 2,
                center_z=(center_start[2] + center_end[2]) / 2,
                length=length,
                width=width,
                height=height,
                heading_degrees=degrees(atan2(dz, dx)),
                slab_index=piece.slab_index,
            )
        )
    return tuple(segments)


def raised_kerb_slab_height(index: int, *, low_height: float, ridge_height: float) -> float:
    """Choose the low or raised height for an alternating kerb block."""
    _validate_raised_kerb_profile(low_height=low_height, ridge_height=ridge_height)
    return ridge_height if index % 2 == 0 else low_height


def _validate_raised_kerb_profile(*, low_height: float, ridge_height: float) -> None:
    if low_height <= 0 or ridge_height <= 0:
        raise ValueError("kerb heights must be positive")
    if ridge_height < low_height:
        raise ValueError("ridge_height must be at least low_height")


def wall_offsets_for_side(side: int, *, inside_distance: float, thickness: float) -> tuple[float, float, float]:
    """Compute inner, outer, and center offsets for one track wall."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    inside_offset = side * inside_distance
    outside_offset = side * (inside_distance + thickness)
    center_offset = side * (inside_distance + thickness / 2)
    return inside_offset, outside_offset, center_offset


def wall_collision_thickness(*, visual_thickness: float, extra_thickness: float) -> float:
    """Make the physical wall a little thicker than the visible wall."""
    if visual_thickness <= 0 or extra_thickness < 0:
        raise ValueError("wall thickness values must be positive")
    return visual_thickness + extra_thickness


def wall_collision_offsets_for_side(
    side: int,
    *,
    inside_distance: float,
    visual_thickness: float,
    extra_thickness: float,
) -> tuple[float, float, float]:
    """Compute wall offsets for the invisible collision boxes."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    thickness = wall_collision_thickness(visual_thickness=visual_thickness, extra_thickness=extra_thickness)
    inside_offset = side * inside_distance
    outside_offset = side * (inside_distance + thickness)
    center_offset = side * (inside_distance + thickness / 2)
    return inside_offset, outside_offset, center_offset


def wall_collision_segment_pose(
    segment: tuple[tuple[float, float, float], tuple[float, float, float]],
    *,
    side: int,
    thickness: float,
) -> tuple[float, float, float, float]:
    """Find the center, length, and heading for one wall collision box."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    if thickness <= 0:
        raise ValueError("thickness must be positive")
    start, end = segment
    dx = end[0] - start[0]
    dz = end[2] - start[2]
    length = hypot(dx, dz)
    if length == 0:
        raise ValueError("wall collision segments must have positive length")

    center_x, center_z, _, heading = _tuple_segment_pose(segment)
    normal_x = -dz / length
    normal_z = dx / length
    return (
        center_x + normal_x * side * thickness / 2,
        center_z + normal_z * side * thickness / 2,
        length,
        heading,
    )


@lru_cache(maxsize=32)
def wall_inner_segments_for_side(
    samples: tuple[TrackPoint, ...],
    *,
    side: int,
    inside_distance: float,
    visual_thickness: float,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    """Build line segments along the inside edge of one wall."""
    inside_offset, _, _ = wall_offsets_for_side(
        side,
        inside_distance=inside_distance,
        thickness=visual_thickness,
    )
    rendered_inner_wall_path = clean_offset_path(samples, inside_offset, 0)
    return _tuple_point_segments(rendered_inner_wall_path)


@lru_cache(maxsize=32)
def wall_collision_segments_for_side(
    samples: tuple[TrackPoint, ...],
    *,
    side: int,
    inside_distance: float,
    visual_thickness: float,
    extra_thickness: float,
) -> tuple[tuple[float, float, float, float], ...]:
    """Build all oriented collision boxes for one side of the track wall."""
    collision_thickness = wall_collision_thickness(
        visual_thickness=visual_thickness,
        extra_thickness=extra_thickness,
    )
    return tuple(
        wall_collision_segment_pose(segment, side=side, thickness=collision_thickness)
        for segment in wall_inner_segments_for_side(
            samples,
            side=side,
            inside_distance=inside_distance,
            visual_thickness=visual_thickness,
        )
    )


def wall_paint_offsets_for_side(
    side: int,
    *,
    inside_offset: float,
    outside_offset: float,
    band_width: float,
    extrusion: float,
) -> WallPaintOffsets:
    """Return render-only paint offsets for one wall side."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    if band_width <= 0 or extrusion <= 0:
        raise ValueError("paint band width and extrusion must be positive")
    if (outside_offset - inside_offset) * side <= 0:
        raise ValueError("wall offsets must be ordered from inside to outside for side")

    return WallPaintOffsets(
        inner_face_offset=inside_offset - side * extrusion,
        top_inner_offset=inside_offset,
        top_outer_offset=outside_offset,
    )


def wall_face_band_chunk_meshes(
    ursina: Any,
    samples: tuple[TrackPoint, ...],
    *,
    face_offset: float,
    bottom_y: float,
    top_y: float,
    uv_scale: float,
    flip_normal: bool,
    segments_per_chunk: int,
) -> tuple[Any, ...]:
    """Return vertical wall-face band meshes split into local chunks."""
    if top_y <= bottom_y:
        raise ValueError("top_y must be greater than bottom_y")
    if uv_scale <= 0:
        raise ValueError("uv_scale must be positive")
    if segments_per_chunk < 1:
        raise ValueError("segments_per_chunk must be at least one")

    bottom_points = clean_offset_path(samples, face_offset, bottom_y)
    top_points = tuple((x, top_y, z) for x, _, z in bottom_points)
    path_lengths = _path_lengths(bottom_points)
    path_normals = _path_normals(bottom_points, flip=flip_normal)
    wall_height = top_y - bottom_y
    meshes: list[Any] = []
    for start_index in range(0, len(bottom_points), segments_per_chunk):
        vertices: list[tuple[float, float, float]] = []
        triangles: list[tuple[int, int, int]] = []
        uvs: list[tuple[float, float]] = []
        normals: list[tuple[float, float, float]] = []
        for offset in range(segments_per_chunk):
            index = (start_index + offset) % len(bottom_points)
            if start_index + offset >= len(bottom_points):
                break
            next_index = (index + 1) % len(bottom_points)
            _append_vertical_band_segment(
                vertices,
                triangles,
                uvs,
                normals,
                bottom_start=bottom_points[index],
                top_start=top_points[index],
                bottom_end=bottom_points[next_index],
                top_end=top_points[next_index],
                start_u=path_lengths[index] / uv_scale,
                end_u=(path_lengths[index] + _xz_distance(bottom_points[index], bottom_points[next_index])) / uv_scale,
                v=wall_height / uv_scale,
                start_normal=path_normals[index],
                end_normal=path_normals[next_index],
                flip=flip_normal,
            )
        meshes.append(ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True))

    return tuple(meshes)


def wall_face_band_mesh(
    ursina: Any,
    samples: tuple[TrackPoint, ...],
    *,
    face_offset: float,
    bottom_y: float,
    top_y: float,
    uv_scale: float,
    flip_normal: bool,
) -> Any:
    """Return one continuous vertical wall-face band mesh."""
    if top_y <= bottom_y:
        raise ValueError("top_y must be greater than bottom_y")
    if uv_scale <= 0:
        raise ValueError("uv_scale must be positive")

    bottom_points = clean_offset_path(samples, face_offset, bottom_y)
    top_points = tuple((x, top_y, z) for x, _, z in bottom_points)
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    _append_vertical_loop(
        vertices,
        triangles,
        uvs,
        normals,
        bottom_points,
        top_points,
        flip=flip_normal,
        uv_scale=uv_scale,
    )
    return ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True)


def wall_top_band_chunk_meshes(
    ursina: Any,
    samples: tuple[TrackPoint, ...],
    *,
    edge_offset: float,
    side: int,
    band_width: float,
    y: float,
    uv_scale: float,
    extends_outside: bool,
    segments_per_chunk: int,
) -> tuple[Any, ...]:
    """Return top-surface wall paint band meshes split into local chunks."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    if band_width <= 0:
        raise ValueError("band_width must be positive")
    if uv_scale <= 0:
        raise ValueError("uv_scale must be positive")
    if segments_per_chunk < 1:
        raise ValueError("segments_per_chunk must be at least one")

    edge_points = clean_offset_path(samples, edge_offset, y)
    across_sign = side if extends_outside else -side
    across_normals = _path_normals(edge_points, flip=across_sign < 0)
    across_points = tuple(
        (point[0] + normal[0] * band_width, point[1], point[2] + normal[2] * band_width)
        for point, normal in zip(edge_points, across_normals, strict=True)
    )
    path_lengths = _path_lengths(edge_points)
    meshes: list[Any] = []
    for start_index in range(0, len(edge_points), segments_per_chunk):
        vertices: list[tuple[float, float, float]] = []
        triangles: list[tuple[int, int, int]] = []
        uvs: list[tuple[float, float]] = []
        normals: list[tuple[float, float, float]] = []
        for offset in range(segments_per_chunk):
            index = (start_index + offset) % len(edge_points)
            if start_index + offset >= len(edge_points):
                break
            next_index = (index + 1) % len(edge_points)
            start = len(vertices)
            segment_length = _xz_distance(edge_points[index], edge_points[next_index])
            start_u = path_lengths[index] / uv_scale
            end_u = (path_lengths[index] + segment_length) / uv_scale
            vertices.extend(
                (edge_points[index], across_points[index], across_points[next_index], edge_points[next_index])
            )
            triangles.append((start, start + 1, start + 2))
            triangles.append((start, start + 2, start + 3))
            uvs.extend(((start_u, 0.0), (start_u, band_width / uv_scale), (end_u, band_width / uv_scale), (end_u, 0.0)))
            normals.extend((0.0, 1.0, 0.0) for _ in range(4))
        meshes.append(ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True))

    return tuple(meshes)


def wall_top_band_mesh(
    ursina: Any,
    samples: tuple[TrackPoint, ...],
    *,
    edge_offset: float,
    side: int,
    band_width: float,
    y: float,
    uv_scale: float,
    extends_outside: bool,
) -> Any:
    """Return one continuous top-surface wall paint mesh made from local quads."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    if band_width <= 0:
        raise ValueError("band_width must be positive")
    if uv_scale <= 0:
        raise ValueError("uv_scale must be positive")

    edge_points = clean_offset_path(samples, edge_offset, y)
    across_sign = side if extends_outside else -side
    across_normals = _path_normals(edge_points, flip=across_sign < 0)
    across_points = tuple(
        (point[0] + normal[0] * band_width, point[1], point[2] + normal[2] * band_width)
        for point, normal in zip(edge_points, across_normals, strict=True)
    )
    path_lengths = _path_lengths(edge_points)
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    for index, edge_point in enumerate(edge_points):
        next_index = (index + 1) % len(edge_points)
        start = len(vertices)
        segment_length = _xz_distance(edge_point, edge_points[next_index])
        start_u = path_lengths[index] / uv_scale
        end_u = (path_lengths[index] + segment_length) / uv_scale
        vertices.extend((edge_point, across_points[index], across_points[next_index], edge_points[next_index]))
        triangles.append((start, start + 1, start + 2))
        triangles.append((start, start + 2, start + 3))
        uvs.extend(((start_u, 0.0), (start_u, band_width / uv_scale), (end_u, band_width / uv_scale), (end_u, 0.0)))
        normals.extend((0.0, 1.0, 0.0) for _ in range(4))

    return ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True)


def wall_mesh(
    ursina: Any,
    samples: tuple[TrackPoint, ...],
    *,
    inside_offset: float,
    outside_offset: float,
    base_y: float,
    height: float,
    uv_scale: float,
) -> Any:
    """Build the visible concrete wall mesh between two offset paths."""
    inner_points_bottom = clean_offset_path(samples, inside_offset, base_y)
    outer_points_bottom = clean_offset_path(samples, outside_offset, base_y)
    inner_points_top = tuple((x, base_y + height, z) for x, _, z in inner_points_bottom)
    outer_points_top = tuple((x, base_y + height, z) for x, _, z in outer_points_bottom)
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []

    _append_vertical_loop(
        vertices,
        triangles,
        uvs,
        normals,
        inner_points_bottom,
        inner_points_top,
        flip=False,
        uv_scale=uv_scale,
    )
    _append_vertical_loop(
        vertices,
        triangles,
        uvs,
        normals,
        outer_points_bottom,
        outer_points_top,
        flip=True,
        uv_scale=uv_scale,
    )

    return ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True)


def _flat_ring_mesh(
    ursina: Any,
    first_points: tuple[tuple[float, float, float], ...],
    second_points: tuple[tuple[float, float, float], ...],
    *,
    uv_scale: float,
) -> Any:
    core = cast(Any, import_module("panda3d.core"))
    triangulator = core.Triangulator()
    outer_points, hole_points = sorted(
        (first_points, second_points),
        key=lambda points: abs(_polygon_area(points)),
        reverse=True,
    )
    vertices: list[tuple[float, float, float]] = []

    for point in _oriented_contour(outer_points, clockwise=False):
        vertices.append(point)
        triangulator.addPolygonVertex(triangulator.addVertex(point[0], point[2]))

    triangulator.beginHole()
    for point in _oriented_contour(hole_points, clockwise=True):
        vertices.append(point)
        triangulator.addHoleVertex(triangulator.addVertex(point[0], point[2]))

    triangulator.triangulate()
    triangles = [
        (
            int(triangulator.getTriangleV0(index)),
            int(triangulator.getTriangleV1(index)),
            int(triangulator.getTriangleV2(index)),
        )
        for index in range(int(triangulator.getNumTriangles()))
    ]
    uvs = tuple((point[0] / uv_scale, point[2] / uv_scale) for point in vertices)
    normals = tuple((0.0, 1.0, 0.0) for _ in vertices)
    return ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True)


def _offset_path(samples: tuple[TrackPoint, ...], offset: float, y: float) -> tuple[tuple[float, float, float], ...]:
    points: list[tuple[float, float, float]] = []
    for index, sample in enumerate(samples):
        previous_sample = samples[index - 1]
        next_sample = samples[(index + 1) % len(samples)]
        tangent_x = next_sample.x - previous_sample.x
        tangent_z = next_sample.z - previous_sample.z
        tangent_length = hypot(tangent_x, tangent_z)
        if tangent_length == 0:
            points.append((sample.x, y, sample.z))
            continue

        normal_x = -tangent_z / tangent_length
        normal_z = tangent_x / tangent_length
        points.append((sample.x + normal_x * offset, y, sample.z + normal_z * offset))
    return tuple(points)


def offset_path(samples: tuple[TrackPoint, ...], offset: float, y: float) -> tuple[tuple[float, float, float], ...]:
    """Move the centerline sideways by a fixed amount at each sample."""
    return _offset_path(samples, offset, y)


def _raised_kerb_slab_pieces(
    samples: tuple[TrackPoint, ...],
    *,
    inner_offset: float,
    outer_offset: float,
    base_y: float,
    slab_length: float,
) -> tuple[_RaisedKerbSlabPiece, ...]:
    if slab_length <= 0:
        raise ValueError("slab_length must be positive")

    inner_points = _offset_path(samples, inner_offset, base_y)
    outer_points = _offset_path(samples, outer_offset, base_y)
    center_points = tuple(_midpoint(inner, outer) for inner, outer in zip(inner_points, outer_points, strict=True))

    pieces: list[_RaisedKerbSlabPiece] = []
    distance_at_segment_start = 0.0
    epsilon = 1e-9
    for index, center_start in enumerate(center_points):
        next_index = (index + 1) % len(center_points)
        center_end = center_points[next_index]
        segment_length = _xz_distance(center_start, center_end)
        if segment_length <= epsilon:
            continue

        current_distance = distance_at_segment_start
        segment_end_distance = distance_at_segment_start + segment_length
        while current_distance < segment_end_distance - epsilon:
            slab_index = int((current_distance + epsilon) // slab_length)
            next_slab_distance = (slab_index + 1) * slab_length
            cut_distance = min(segment_end_distance, next_slab_distance)
            if cut_distance <= current_distance + epsilon:
                cut_distance = min(segment_end_distance, current_distance + epsilon)

            start_fraction = (current_distance - distance_at_segment_start) / segment_length
            end_fraction = (cut_distance - distance_at_segment_start) / segment_length
            pieces.append(
                _RaisedKerbSlabPiece(
                    slab_index=slab_index,
                    inner_start=_lerp_point(inner_points[index], inner_points[next_index], start_fraction),
                    outer_start=_lerp_point(outer_points[index], outer_points[next_index], start_fraction),
                    outer_end=_lerp_point(outer_points[index], outer_points[next_index], end_fraction),
                    inner_end=_lerp_point(inner_points[index], inner_points[next_index], end_fraction),
                )
            )
            current_distance = cut_distance

        distance_at_segment_start = segment_end_distance

    return tuple(pieces)


def _group_kerb_pieces_by_slab(
    pieces: tuple[_RaisedKerbSlabPiece, ...],
) -> tuple[tuple[int, tuple[_RaisedKerbSlabPiece, ...]], ...]:
    grouped: list[tuple[int, tuple[_RaisedKerbSlabPiece, ...]]] = []
    current_slab_index: int | None = None
    current_pieces: list[_RaisedKerbSlabPiece] = []

    for piece in pieces:
        if current_slab_index is None:
            current_slab_index = piece.slab_index
        elif piece.slab_index != current_slab_index:
            grouped.append((current_slab_index, tuple(current_pieces)))
            current_slab_index = piece.slab_index
            current_pieces = []
        current_pieces.append(piece)

    if current_slab_index is not None:
        grouped.append((current_slab_index, tuple(current_pieces)))

    return tuple(grouped)


def _lerp_point(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    fraction: float,
) -> tuple[float, float, float]:
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction,
        start[2] + (end[2] - start[2]) * fraction,
    )


def _midpoint(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2, (first[2] + second[2]) / 2)


def _xz_distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return hypot(second[0] - first[0], second[2] - first[2])


def _append_kerb_slab(
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    uvs: list[tuple[float, float]],
    normals: list[tuple[float, float, float]],
    *,
    inner_start: tuple[float, float, float],
    outer_start: tuple[float, float, float],
    outer_end: tuple[float, float, float],
    inner_end: tuple[float, float, float],
    top_y: float,
    uv_scale: float,
) -> None:
    inner_start_top = (inner_start[0], top_y, inner_start[2])
    outer_start_top = (outer_start[0], top_y, outer_start[2])
    outer_end_top = (outer_end[0], top_y, outer_end[2])
    inner_end_top = (inner_end[0], top_y, inner_end[2])

    _append_face(
        vertices,
        triangles,
        uvs,
        normals,
        (inner_start_top, outer_start_top, outer_end_top, inner_end_top),
        normal=(0.0, 1.0, 0.0),
        uv_scale=uv_scale,
    )
    _append_face(
        vertices,
        triangles,
        uvs,
        normals,
        (inner_start, inner_end, outer_end, outer_start),
        normal=(0.0, -1.0, 0.0),
        uv_scale=uv_scale,
    )
    _append_face_with_computed_normal(
        vertices,
        triangles,
        uvs,
        normals,
        (inner_start, inner_start_top, inner_end_top, inner_end),
        uv_scale=uv_scale,
    )
    _append_face_with_computed_normal(
        vertices,
        triangles,
        uvs,
        normals,
        (outer_start, outer_end, outer_end_top, outer_start_top),
        uv_scale=uv_scale,
    )
    _append_face_with_computed_normal(
        vertices,
        triangles,
        uvs,
        normals,
        (inner_start, outer_start, outer_start_top, inner_start_top),
        uv_scale=uv_scale,
    )
    _append_face_with_computed_normal(
        vertices,
        triangles,
        uvs,
        normals,
        (inner_end, inner_end_top, outer_end_top, outer_end),
        uv_scale=uv_scale,
    )


def _append_face_with_computed_normal(
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    uvs: list[tuple[float, float]],
    normals: list[tuple[float, float, float]],
    face: tuple[tuple[float, float, float], ...],
    *,
    uv_scale: float,
) -> None:
    _append_face(
        vertices,
        triangles,
        uvs,
        normals,
        face,
        normal=_face_normal(face[0], face[1], face[2]),
        uv_scale=uv_scale,
    )


def _append_face(
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    uvs: list[tuple[float, float]],
    normals: list[tuple[float, float, float]],
    face: tuple[tuple[float, float, float], ...],
    *,
    normal: tuple[float, float, float],
    uv_scale: float,
) -> None:
    start = len(vertices)
    vertices.extend(face)
    normals.extend(normal for _ in face)
    uvs.extend((point[0] / uv_scale, point[2] / uv_scale) for point in face)
    for index in range(1, len(face) - 1):
        triangles.append((start, start + index, start + index + 1))


def _face_normal(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
) -> tuple[float, float, float]:
    first_edge = (second[0] - first[0], second[1] - first[1], second[2] - first[2])
    second_edge = (third[0] - first[0], third[1] - first[1], third[2] - first[2])
    normal = (
        first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
    )
    length = hypot(normal[0], hypot(normal[1], normal[2]))
    if length == 0:
        return (0.0, 1.0, 0.0)
    return (normal[0] / length, normal[1] / length, normal[2] / length)


@lru_cache(maxsize=64)
def clean_offset_path(
    samples: tuple[TrackPoint, ...], offset: float, y: float
) -> tuple[tuple[float, float, float], ...]:
    """Move the centerline sideways and trim simple self-intersections."""
    return _trim_self_intersections(_offset_path(samples, offset, y))


def _trim_self_intersections(points: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    cleaned = list(points)
    max_passes = len(cleaned)
    for _ in range(max_passes):
        intersection = _first_path_intersection(cleaned)
        if intersection is None:
            break
        start_index, end_index, point = intersection
        forward_count = end_index - start_index
        backward_count = len(cleaned) - forward_count
        if forward_count <= backward_count:
            cleaned = [*cleaned[: start_index + 1], point, *cleaned[end_index + 1 :]]
        else:
            cleaned = [*cleaned[start_index + 1 : end_index + 1], point]
    return tuple(cleaned)


def _first_path_intersection(
    points: list[tuple[float, float, float]],
) -> tuple[int, int, tuple[float, float, float]] | None:
    for start_index, end_index in combinations(range(len(points)), 2):
        if abs(start_index - end_index) <= 1 or {start_index, end_index} == {0, len(points) - 1}:
            continue
        intersection = _segment_intersection(
            points[start_index],
            points[(start_index + 1) % len(points)],
            points[end_index],
            points[(end_index + 1) % len(points)],
        )
        if intersection is not None:
            return start_index, end_index, intersection
    return None


def _segment_intersection(
    first_start: tuple[float, float, float],
    first_end: tuple[float, float, float],
    second_start: tuple[float, float, float],
    second_end: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    x1, _, z1 = first_start
    x2, y, z2 = first_end
    x3, _, z3 = second_start
    x4, _, z4 = second_end
    denominator = (x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4)
    if abs(denominator) < 1e-9:
        return None

    first_t = ((x1 - x3) * (z3 - z4) - (z1 - z3) * (x3 - x4)) / denominator
    second_t = ((x1 - x3) * (z1 - z2) - (z1 - z3) * (x1 - x2)) / denominator
    if not (1e-6 < first_t < 1 - 1e-6 and 1e-6 < second_t < 1 - 1e-6):
        return None
    return (x1 + first_t * (x2 - x1), y, z1 + first_t * (z2 - z1))


def _polygon_area(points: tuple[tuple[float, float, float], ...]) -> float:
    area = 0.0
    for point, next_point in zip(points, (*points[1:], points[0]), strict=True):
        area += point[0] * next_point[2] - next_point[0] * point[2]
    return area / 2


def _oriented_contour(
    points: tuple[tuple[float, float, float], ...],
    *,
    clockwise: bool,
) -> tuple[tuple[float, float, float], ...]:
    is_clockwise = _polygon_area(points) < 0
    return points if is_clockwise is clockwise else tuple(reversed(points))


def _append_vertical_loop(
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    uvs: list[tuple[float, float]],
    normals: list[tuple[float, float, float]],
    bottom_points: tuple[tuple[float, float, float], ...],
    top_points: tuple[tuple[float, float, float], ...],
    *,
    flip: bool,
    uv_scale: float,
) -> None:
    start = len(vertices)
    path_lengths = _path_lengths(bottom_points)
    path_normals = _path_normals(bottom_points, flip=flip)
    wall_height = max(0.001, abs(top_points[0][1] - bottom_points[0][1]))
    for index, (bottom, top) in enumerate(zip(bottom_points, top_points, strict=True)):
        vertices.extend((bottom, top))
        u = path_lengths[index] / uv_scale
        uvs.extend(((u, 0.0), (u, wall_height / uv_scale)))
        normals.extend((path_normals[index], path_normals[index]))

    for index in range(len(bottom_points)):
        next_index = (index + 1) % len(bottom_points)
        bottom = start + index * 2
        top = bottom + 1
        next_bottom = start + next_index * 2
        next_top = next_bottom + 1
        if flip:
            triangles.append((bottom, top, next_bottom))
            triangles.append((top, next_top, next_bottom))
        else:
            triangles.append((bottom, next_bottom, top))
            triangles.append((top, next_bottom, next_top))


def _append_vertical_band_segment(
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    uvs: list[tuple[float, float]],
    normals: list[tuple[float, float, float]],
    *,
    bottom_start: tuple[float, float, float],
    top_start: tuple[float, float, float],
    bottom_end: tuple[float, float, float],
    top_end: tuple[float, float, float],
    start_u: float,
    end_u: float,
    v: float,
    start_normal: tuple[float, float, float],
    end_normal: tuple[float, float, float],
    flip: bool,
) -> None:
    start = len(vertices)
    vertices.extend((bottom_start, top_start, bottom_end, top_end))
    uvs.extend(((start_u, 0.0), (start_u, v), (end_u, 0.0), (end_u, v)))
    normals.extend((start_normal, start_normal, end_normal, end_normal))
    if flip:
        triangles.append((start, start + 1, start + 2))
        triangles.append((start + 1, start + 3, start + 2))
    else:
        triangles.append((start, start + 2, start + 1))
        triangles.append((start + 1, start + 2, start + 3))


def _path_lengths(points: tuple[tuple[float, float, float], ...]) -> tuple[float, ...]:
    lengths: list[float] = [0.0]
    for point, next_point in pairwise(points):
        lengths.append(lengths[-1] + hypot(next_point[0] - point[0], next_point[2] - point[2]))
    return tuple(lengths)


def _path_normals(
    points: tuple[tuple[float, float, float], ...],
    *,
    flip: bool,
) -> tuple[tuple[float, float, float], ...]:
    normals: list[tuple[float, float, float]] = []
    for index, _ in enumerate(points):
        previous_point = points[index - 1]
        next_point = points[(index + 1) % len(points)]
        tangent_x = next_point[0] - previous_point[0]
        tangent_z = next_point[2] - previous_point[2]
        tangent_length = hypot(tangent_x, tangent_z)
        if tangent_length == 0:
            normals.append((0.0, 0.0, 1.0))
            continue
        normal_x = -tangent_z / tangent_length
        normal_z = tangent_x / tangent_length
        if flip:
            normal_x = -normal_x
            normal_z = -normal_z
        normals.append((normal_x, 0.0, normal_z))
    return tuple(normals)
