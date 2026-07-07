"""Lighting rigs for the racing scenes and car showcase."""

from __future__ import annotations

from typing import Any


def add_lighting(ursina: Any) -> None:
    """Add the standard night-race lighting rig."""
    sun = ursina.DirectionalLight(rotation=(35, -52, 18), shadows=True, color=(0.62, 0.66, 0.82, 1))
    light_node = getattr(sun, "_light", None)
    if light_node is not None and hasattr(light_node, "setShadowCaster"):
        light_node.setShadowCaster(True, 4096, 4096)
    ursina.DirectionalLight(rotation=(-18, 128, -8), shadows=False, color=(0.18, 0.23, 0.34, 1))
    ursina.AmbientLight(color=(0.24, 0.255, 0.285, 1))


def add_showcase_lighting(ursina: Any) -> None:
    """Add the bright studio lighting rig for the car showcase."""
    ursina.AmbientLight(color=(0.58, 0.58, 0.56, 1))
    key = ursina.DirectionalLight(rotation=(34, -46, 18), shadows=True, color=(0.98, 0.98, 0.96, 1))
    key_node = getattr(key, "_light", None)
    if key_node is not None and hasattr(key_node, "setShadowCaster"):
        key_node.setShadowCaster(True, 4096, 4096)
    fill = ursina.PointLight(position=(-2.8, 2.6, -2.1), color=(0.48, 0.58, 0.74, 1))
    fill._light.setAttenuation((1.0, 0.08, 0.030))
    rim = ursina.PointLight(position=(2.6, 1.8, 2.8), color=(0.78, 0.62, 0.42, 1))
    rim._light.setAttenuation((1.0, 0.10, 0.035))
