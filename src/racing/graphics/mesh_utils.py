"""Reusable mesh helpers for procedural racing visuals."""

from __future__ import annotations

from math import hypot
from typing import Any


def mesh_from_quads(ursina: Any, quads: tuple[tuple[tuple[float, float, float], ...], ...]) -> Any:
    """Build an Ursina mesh from a list of four-corner faces."""
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    normals: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    for quad in quads:
        if len(quad) < 3:
            continue
        normal = _face_normal(quad[0], quad[1], quad[2])
        start = len(vertices)
        vertices.extend(quad)
        normals.extend(normal for _ in quad)
        uvs.extend(_quad_uvs(quad))
        for index in range(1, len(quad) - 1):
            triangles.append((start, start + index, start + index + 1))
    return ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True)


def _quad_uvs(points: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float], ...]:
    min_x = min(point[0] for point in points)
    min_z = min(point[2] for point in points)
    span_x = max(0.001, max(point[0] for point in points) - min_x)
    span_z = max(0.001, max(point[2] for point in points) - min_z)
    return tuple(((point[0] - min_x) / span_x, (point[2] - min_z) / span_z) for point in points)


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
