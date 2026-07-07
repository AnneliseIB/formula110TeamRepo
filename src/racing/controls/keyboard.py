"""Keyboard and gamepad command mapping for manual driving."""

from __future__ import annotations

from typing import Any

from racing.student.api import RobotCommand, clamp_command

GAMEPAD_STEERING_AXIS = "gamepad left stick x"
GAMEPAD_ACCELERATOR_AXIS = "gamepad right trigger"
GAMEPAD_REVERSE_AXIS = "gamepad left trigger"
GAMEPAD_STICK_DEADZONE = 0.10
GAMEPAD_TRIGGER_DEADZONE = 0.02


def keyboard_command(held_keys: Any) -> RobotCommand:
    """Map held arrow keys to a normalized robot command."""
    throttle_forward = _key_down(held_keys, "up arrow")
    throttle_reverse = _key_down(held_keys, "down arrow")
    brake = throttle_forward and throttle_reverse
    steer_right = _key_down(held_keys, "right arrow")
    steer_left = _key_down(held_keys, "left arrow")
    throttle = 0 if brake else throttle_forward - throttle_reverse
    return RobotCommand(throttle=float(throttle), steer=float(steer_left - steer_right), brake=float(brake))


def gamepad_command(held_keys: Any) -> RobotCommand:
    """Map the primary gamepad sticks/triggers to a normalized robot command."""
    steering_axis = _deadzone(_axis_value(held_keys, GAMEPAD_STEERING_AXIS), GAMEPAD_STICK_DEADZONE)
    accelerator = _trigger_amount(_axis_value(held_keys, GAMEPAD_ACCELERATOR_AXIS))
    reverse = _trigger_amount(_axis_value(held_keys, GAMEPAD_REVERSE_AXIS))
    brake = max(accelerator, reverse) if accelerator > 0.0 and reverse > 0.0 else 0.0
    throttle = 0.0 if brake > 0.0 else accelerator - reverse
    return RobotCommand(
        throttle=throttle,
        steer=-steering_axis,
        brake=brake,
    )


def manual_drive_command(held_keys: Any) -> RobotCommand:
    """Combine keyboard and gamepad input into one normalized robot command."""
    keyboard = keyboard_command(held_keys)
    gamepad = gamepad_command(held_keys)
    return clamp_command(
        RobotCommand(
            throttle=keyboard.throttle + gamepad.throttle,
            steer=keyboard.steer + gamepad.steer,
            brake=max(keyboard.brake, gamepad.brake),
        )
    )


def _key_down(held_keys: Any, key: str) -> int:
    return 1 if bool(held_keys[key]) else 0


def _axis_value(held_keys: Any, key: str) -> float:
    try:
        return float(held_keys[key])
    except (TypeError, ValueError, KeyError):
        return 0.0


def _deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    return value


def _trigger_amount(value: float) -> float:
    return _deadzone(max(0.0, min(1.0, value)), GAMEPAD_TRIGGER_DEADZONE)
