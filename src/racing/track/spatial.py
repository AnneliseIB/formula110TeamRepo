"""Small spatial helpers shared by rendering, cameras, and race helpers."""

from __future__ import annotations

from math import cos, radians, sin
from typing import Any

from racing.track.world import TrackPoint


def track_forward_vector(heading_degrees: float) -> tuple[float, float]:
    """Return the X/Z unit vector that points along a track heading.

    Args:
        heading_degrees: Heading angle in degrees where zero points along +Z.

    Returns:
        A two-dimensional world-space vector in the X/Z plane.
    """
    heading = radians(heading_degrees)
    return sin(heading), cos(heading)


def track_left_vector(heading_degrees: float) -> tuple[float, float]:
    """Return the X/Z unit vector to the left of a heading.

    Args:
        heading_degrees: Heading angle in degrees where zero points along +Z.

    Returns:
        A two-dimensional world-space vector perpendicular to the heading.
    """
    forward_x, forward_z = track_forward_vector(heading_degrees)
    return -forward_z, forward_x


def node_position(node_path: Any) -> tuple[float, float, float]:
    """Read a Panda3D/Ursina node position as plain floats.

    Args:
        node_path: Object exposing Panda3D's ``getPos`` method.

    Returns:
        The node's ``(x, y, z)`` position tuple.
    """
    position = node_path.getPos()
    return float(position[0]), float(position[1]), float(position[2])


def track_point_from_node(node_path: Any) -> TrackPoint:
    """Project a 3D node position into a 2D track point.

    Args:
        node_path: Object exposing Panda3D's ``getPos`` method.

    Returns:
        A track point using the node's X/Z coordinates.
    """
    x, _, z = node_position(node_path)
    return TrackPoint(x=x, z=z)
