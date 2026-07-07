"""Panda3D process configuration helpers used before opening windows."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

DEFAULT_ANTIALIAS_MULTISAMPLES = 4


def quiet_panda_image_logs() -> None:
    """Reduce noisy image-loader logs that are not useful in this project."""
    panda3d_core = import_module("panda3d.core")
    load_prc_file_data = cast(Callable[[str, str], None], panda3d_core.loadPrcFileData)
    load_prc_file_data("", "notify-level-pnmimage error")


def configure_panda_antialiasing(multisamples: int = DEFAULT_ANTIALIAS_MULTISAMPLES) -> None:
    """Request a multisample framebuffer before opening a Panda3D window."""
    if multisamples < 1:
        raise ValueError("multisamples must be positive")
    core = cast(Any, import_module("panda3d.core"))
    core.loadPrcFileData("", "framebuffer-multisample 1")
    core.loadPrcFileData("", f"multisamples {multisamples}")


def enable_render_antialiasing(render: Any) -> None:
    """Enable Panda3D's automatic antialiasing attribute on a render root."""
    core = cast(Any, import_module("panda3d.core"))
    render.setAntialias(core.AntialiasAttrib.MAuto)


def configure_headless_panda() -> None:
    """Configure Panda3D for headless simulation."""
    configure_panda_y_up()
    core = cast(Any, import_module("panda3d.core"))
    core.loadPrcFileData("", "window-type none")
    core.loadPrcFileData("", "audio-library-name null")


def configure_panda_y_up() -> None:
    """Configure Panda3D to use the project Y-up coordinate system."""
    core = cast(Any, import_module("panda3d.core"))
    core.loadPrcFileData("", "coordinate-system y-up")


def patch_ursina_window_coordinate_system() -> Callable[[], None]:
    """Patch Ursina window setup so it preserves Panda3D Y-up coordinates."""
    window_module = cast(Any, import_module("ursina.window"))
    original_load_prc_file_data = cast(Callable[[str, str], None], window_module.loadPrcFileData)

    def load_prc_file_data(name: str, data: str) -> None:
        """Forward PRC settings while replacing Ursina's coordinate-system override."""
        if data.strip() == "coordinate-system y-up-left":
            original_load_prc_file_data(name, "coordinate-system y-up")
            return
        original_load_prc_file_data(name, data)

    window_module.loadPrcFileData = load_prc_file_data

    def restore() -> None:
        """Restore Ursina's original PRC-setting function."""
        window_module.loadPrcFileData = original_load_prc_file_data

    return restore
