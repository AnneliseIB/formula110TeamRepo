"""Ursina/Panda3D gamepad polling helpers for manual driving."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from racing.controls.keyboard import (
    GAMEPAD_ACCELERATOR_AXIS,
    GAMEPAD_REVERSE_AXIS,
    GAMEPAD_STEERING_AXIS,
)


@dataclass(frozen=True, slots=True)
class GamepadAxisSnapshot:
    """Current raw axis values for one connected gamepad."""

    source: str
    name: str
    left_stick_x: float
    left_stick_y: float
    right_trigger: float
    left_trigger: float


_native_discovery_started = False


def sync_gamepad_axes(held_keys: Any) -> tuple[GamepadAxisSnapshot, ...]:
    """Refresh Ursina-style held key axis values from currently connected gamepads."""
    core = _panda_core()
    base = _showbase()
    snapshots: list[GamepadAxisSnapshot] = []
    primary_axes_synced = False

    if core is not None and base is not None and hasattr(base, "devices"):
        devices = tuple(base.devices.getDevices(core.InputDevice.DeviceClass.gamepad))
        _remember_ursina_gamepads(devices)
        for index, device in enumerate(devices):
            key_prefix = "gamepad" if index == 0 else f"gamepad_{index}"
            left_stick_x_value = _panda_axis_value(device, core.InputDevice.Axis.left_x)
            left_stick_y_value = _panda_axis_value(device, core.InputDevice.Axis.left_y)
            right_trigger_value = _panda_axis_value(device, core.InputDevice.Axis.right_trigger)
            left_trigger_value = _panda_axis_value(device, core.InputDevice.Axis.left_trigger)
            left_stick_x = 0.0 if left_stick_x_value is None else left_stick_x_value
            left_stick_y = 0.0 if left_stick_y_value is None else left_stick_y_value
            right_trigger = 0.0 if right_trigger_value is None else right_trigger_value
            left_trigger = 0.0 if left_trigger_value is None else left_trigger_value
            held_keys[f"{key_prefix} left stick x"] = left_stick_x
            held_keys[f"{key_prefix} left stick y"] = left_stick_y
            held_keys[f"{key_prefix} right trigger"] = right_trigger
            held_keys[f"{key_prefix} left trigger"] = left_trigger
            has_standard_axes = any(
                value is not None
                for value in (left_stick_x_value, left_stick_y_value, right_trigger_value, left_trigger_value)
            )
            if has_standard_axes and not primary_axes_synced:
                _sync_primary_axes(
                    held_keys,
                    left_stick_x=left_stick_x,
                    right_trigger=right_trigger,
                    left_trigger=left_trigger,
                )
                primary_axes_synced = True
            snapshots.append(
                GamepadAxisSnapshot(
                    source="panda3d",
                    name=str(getattr(device, "name", f"gamepad {index}")),
                    left_stick_x=left_stick_x,
                    left_stick_y=left_stick_y,
                    right_trigger=right_trigger,
                    left_trigger=left_trigger,
                )
            )

    native_snapshots = _native_gamepad_snapshots()
    if len(native_snapshots) > 0:
        first_native = native_snapshots[0]
        _sync_primary_axes(
            held_keys,
            left_stick_x=first_native.left_stick_x,
            right_trigger=first_native.right_trigger,
            left_trigger=first_native.left_trigger,
        )
        primary_axes_synced = True
        snapshots.extend(native_snapshots)

    if not primary_axes_synced:
        _clear_primary_axes(held_keys)
    return tuple(snapshots)


def _sync_primary_axes(
    held_keys: Any,
    *,
    left_stick_x: float,
    right_trigger: float,
    left_trigger: float,
) -> None:
    held_keys[GAMEPAD_STEERING_AXIS] = left_stick_x
    held_keys[GAMEPAD_ACCELERATOR_AXIS] = right_trigger
    held_keys[GAMEPAD_REVERSE_AXIS] = left_trigger


def _clear_primary_axes(held_keys: Any) -> None:
    held_keys[GAMEPAD_STEERING_AXIS] = 0.0
    held_keys[GAMEPAD_ACCELERATOR_AXIS] = 0.0
    held_keys[GAMEPAD_REVERSE_AXIS] = 0.0


def _panda_core() -> Any | None:
    try:
        return import_module("panda3d.core")
    except ImportError:
        return None


def _showbase() -> Any | None:
    try:
        showbase_global = import_module("direct.showbase.ShowBaseGlobal")
    except ImportError:
        return None
    return getattr(showbase_global, "base", None)


def _remember_ursina_gamepads(devices: tuple[Any, ...]) -> None:
    try:
        input_handler: Any = import_module("ursina.input_handler")
    except ImportError:
        return
    input_handler.gamepads = devices
    if len(devices) > 0:
        input_handler.gamepad = devices[0]


def _panda_axis_value(device: Any, axis: Any) -> float | None:
    try:
        axis_state = device.findAxis(axis)
    except Exception:
        return None
    return float(axis_state.value)


def _native_gamepad_snapshots() -> tuple[GamepadAxisSnapshot, ...]:
    controller_class = _native_controller_class()
    if controller_class is None:
        return ()
    _start_native_discovery(controller_class)
    try:
        controllers = tuple(controller_class.controllers())
    except Exception:
        return ()

    snapshots: list[GamepadAxisSnapshot] = []
    for index, controller in enumerate(controllers):
        gamepad = _call(controller, "extendedGamepad")
        if gamepad is None:
            continue
        left_thumbstick = _call(gamepad, "leftThumbstick")
        snapshots.append(
            GamepadAxisSnapshot(
                source="macos-gamecontroller",
                name=_native_controller_name(controller, fallback=f"controller {index}"),
                left_stick_x=_native_axis_value(_call(left_thumbstick, "xAxis")),
                left_stick_y=_native_axis_value(_call(left_thumbstick, "yAxis")),
                right_trigger=_native_axis_value(_call(gamepad, "rightTrigger")),
                left_trigger=_native_axis_value(_call(gamepad, "leftTrigger")),
            )
        )
    return tuple(snapshots)


def _native_controller_class() -> Any | None:
    try:
        foundation: Any = import_module("Foundation")
        objc: Any = import_module("objc")
    except ImportError:
        return None
    try:
        bundle = foundation.NSBundle.bundleWithPath_("/System/Library/Frameworks/GameController.framework")
        if not bool(bundle.isLoaded()) and not bool(bundle.load()):
            return None
        return objc.lookUpClass("GCController")
    except Exception:
        return None


def _start_native_discovery(controller_class: Any) -> None:
    global _native_discovery_started
    if _native_discovery_started:
        return
    try:
        controller_class.startWirelessControllerDiscoveryWithCompletionHandler_(None)
    except Exception:
        return
    _native_discovery_started = True


def _native_controller_name(controller: Any, *, fallback: str) -> str:
    for accessor in ("vendorName", "productCategory", "detailedProductCategory"):
        value = _call(controller, accessor)
        if value is not None and str(value) != "":
            return str(value)
    return fallback


def _native_axis_value(axis_or_button: Any | None) -> float:
    value = _call(axis_or_button, "value")
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _call(obj: Any | None, name: str) -> Any | None:
    if obj is None:
        return None
    member = getattr(obj, name, None)
    if member is None:
        return None
    if callable(member):
        try:
            return member()
        except Exception:
            return None
    return member
