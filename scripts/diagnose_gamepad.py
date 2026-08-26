"""Open a small Ursina window and print racing gamepad input values."""

from __future__ import annotations

from importlib import import_module
from time import monotonic
from typing import Any, cast

from racing.controls.gamepad import GamepadAxisSnapshot, sync_gamepad_axes
from racing.controls.keyboard import manual_drive_command


def main() -> None:
    ursina = cast(Any, import_module("ursina"))
    app = ursina.Ursina(
        title="Racing Gamepad Diagnostic",
        borderless=False,
        fullscreen=False,
        development_mode=True,
        size=(820, 420),
    )
    ursina.window.color = ursina.color.black
    status = ursina.Text(
        text="Move the left stick and triggers.",
        position=(-0.48, 0.42),
        origin=(-0.5, 0.5),
        scale=0.85,
        color=ursina.color.white,
    )
    last_printed_at = 0.0

    def update() -> None:
        nonlocal last_printed_at
        snapshots = sync_gamepad_axes(ursina.held_keys)
        command = manual_drive_command(ursina.held_keys)
        lines = _diagnostic_lines(snapshots=snapshots, command=command)
        status.text = "\n".join(lines)
        now = monotonic()
        if now - last_printed_at >= 0.5:
            print("\n".join(lines), flush=True)
            last_printed_at = now

    ursina.Entity(name="gamepad_diagnostic_loop", update=update, eternal=True)
    app.run()


def _diagnostic_lines(*, snapshots: tuple[GamepadAxisSnapshot, ...], command: Any) -> tuple[str, ...]:
    if len(snapshots) == 0:
        return (
            "No Panda3D gamepad devices detected.",
            "",
            "Check that the controller is connected before launching this script.",
            "If macOS shows it but this stays empty, Panda/Ursina is not receiving the device.",
        )
    lines = [
        f"Detected gamepads: {len(snapshots)}",
        "",
    ]
    for index, snapshot in enumerate(snapshots):
        lines.extend(
            (
                f"{index}: [{snapshot.source}] {snapshot.name}",
                f"  left stick x:  {snapshot.left_stick_x: .3f}",
                f"  left stick y:  {snapshot.left_stick_y: .3f}",
                f"  right trigger: {snapshot.right_trigger: .3f}",
                f"  left trigger/reverse: {snapshot.left_trigger: .3f}",
                "",
            )
        )
    lines.extend(
        (
            "Racing command:",
            f"  throttle: {command.throttle: .3f}",
            f"  steer:    {command.steer: .3f}",
        )
    )
    return tuple(lines)


if __name__ == "__main__":
    main()
