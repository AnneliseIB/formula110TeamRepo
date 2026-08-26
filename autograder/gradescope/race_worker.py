#!/usr/bin/env python3
"""Trusted physics worker for one controller validation or race trial."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import pickle
import select
import shutil
import signal
import struct
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path
from types import TracebackType
from typing import Any, cast

RESULT_PREFIX = "FORMULA110_RESULT="
CONTROL_WORKER_PATH = Path(os.environ.get("FORMULA110_CONTROL_WORKER", "/opt/formula110-autograder/control_worker.py"))
MAX_RESPONSE_BYTES = 4096
COMMAND_TIMEOUT_SECONDS = 0.5
CONTROLLER_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
CONTROLLER_MEMORY_POLL_SECONDS = 0.02
CPU_ONLY_ENVIRONMENT = {
    "FORMULA110_DEVICE": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "JAX_PLATFORM_NAME": "cpu",
    "ONEAPI_DEVICE_SELECTOR": "*:cpu",
    "SYCL_DEVICE_FILTER": "cpu",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--module-file", type=Path, required=True)
    parser.add_argument("--function", default="control")
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


class ControllerClient:
    """Request commands from a persistent, unprivileged student process."""

    def __init__(self, *, submission: Path, module_file: Path, function_name: str) -> None:
        self.submission = submission
        self.module_file = module_file
        self.function_name = function_name
        self.process: subprocess.Popen[bytes] | None = None
        self.response_fd: int | None = None

    def __enter__(self) -> ControllerClient:
        response_read_fd, response_write_fd = os.pipe()
        worker_arguments = [
            str(CONTROL_WORKER_PATH),
            "--submission",
            str(self.submission),
            "--module-file",
            str(self.module_file),
            "--function",
            self.function_name,
            "--response-fd",
            str(response_write_fd),
        ]
        process_environment: dict[str, str] | None = None
        if os.environ.get("FORMULA110_LOCAL_CONTROL") == "1":
            command = [sys.executable, *worker_arguments]
            process_environment = {**os.environ, **CPU_ONLY_ENVIRONMENT}
        else:
            runuser = shutil.which("runuser") or "/usr/sbin/runuser"
            command = [
                runuser,
                "-u",
                "student",
                "--",
                "/usr/bin/env",
                "-i",
                "HOME=/tmp",
                "PATH=/autograder/venv/bin:/usr/bin:/bin",
                "PYTHONPATH=/opt/formula110-runtime",
                *(f"{name}={value}" for name, value in CPU_ONLY_ENVIRONMENT.items()),
                "/autograder/venv/bin/python",
                *worker_arguments,
            ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                pass_fds=(response_write_fd,),
                close_fds=True,
                start_new_session=True,
                env=process_environment,
            )
        finally:
            os.close(response_write_fd)
        self.response_fd = response_read_fd
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def command(self, sensors: object) -> Any:
        from racing import RobotCommand

        if self.process is None or self.process.stdin is None or self.response_fd is None:
            raise RuntimeError("controller process is not running")
        payload = pickle.dumps(sensors, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            self.process.stdin.write(struct.pack("!I", len(payload)) + payload)
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError(self._unexpected_exit_message()) from error
        response_size = struct.unpack("!I", self._read_exact(4))[0]
        if response_size > MAX_RESPONSE_BYTES:
            raise RuntimeError("controller response exceeded size limit")
        try:
            response = json.loads(self._read_exact(response_size))
        except json.JSONDecodeError as error:
            raise RuntimeError("controller returned an invalid response") from error
        if not isinstance(response, dict) or response.get("ok") is not True:
            message = (
                response.get("error", "controller call failed") if isinstance(response, dict) else "invalid response"
            )
            raise RuntimeError(str(message)[:1000])
        values = tuple(float(response[key]) for key in ("throttle", "steer"))
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("control must return finite command values")
        return RobotCommand(throttle=values[0], steer=values[1])

    def close(self) -> None:
        if self.process is not None:
            process_group_id = self.process.pid
            if self.process.stdin is not None:
                self.process.stdin.close()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=1.0)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process_group_id, signal.SIGKILL)
            if self.process.poll() is None:
                self.process.wait(timeout=1.0)
            self.process = None
        if self.response_fd is not None:
            os.close(self.response_fd)
            self.response_fd = None

    def _read_exact(self, byte_count: int) -> bytes:
        if self.response_fd is None:
            raise RuntimeError("controller response pipe is closed")
        chunks: list[bytes] = []
        remaining = byte_count
        deadline = time.monotonic() + COMMAND_TIMEOUT_SECONDS
        while remaining:
            self._check_memory_limit()
            timeout = deadline - time.monotonic()
            if timeout <= 0.0:
                raise TimeoutError("control call exceeded 0.5 seconds")
            readable, _, _ = select.select(
                [self.response_fd],
                [],
                [],
                min(timeout, CONTROLLER_MEMORY_POLL_SECONDS),
            )
            if not readable:
                continue
            chunk = os.read(self.response_fd, remaining)
            if not chunk:
                raise RuntimeError(self._unexpected_exit_message())
            chunks.append(chunk)
            remaining -= len(chunk)
        self._check_memory_limit()
        return b"".join(chunks)

    def _check_memory_limit(self) -> None:
        if self.process is None:
            return
        resident_memory_bytes = _linux_process_tree_resident_memory_bytes(self.process.pid)
        if resident_memory_bytes is None or resident_memory_bytes <= CONTROLLER_MEMORY_LIMIT_BYTES:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGKILL)
        limit_mib = CONTROLLER_MEMORY_LIMIT_BYTES // (1024 * 1024)
        used_mib = resident_memory_bytes / (1024 * 1024)
        raise MemoryError(f"controller exceeded {limit_mib} MiB memory limit ({used_mib:.1f} MiB resident)")

    def _unexpected_exit_message(self) -> str:
        if self.process is None:
            return "controller process exited unexpectedly"
        return_code = self.process.poll()
        if return_code is None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                return_code = self.process.wait(timeout=0.1)
        diagnostic = ""
        if return_code is not None and self.process.stderr is not None:
            diagnostic = self.process.stderr.read(2000).decode("utf-8", errors="replace").strip()
        message = "controller process exited unexpectedly"
        if return_code is not None:
            message += f" (exit {return_code})"
        if diagnostic:
            message += f": {diagnostic}"
        return message


def controller_client(args: argparse.Namespace) -> ControllerClient:
    return ControllerClient(
        submission=args.submission.resolve(),
        module_file=args.module_file.resolve(),
        function_name=str(args.function),
    )


def _linux_resident_memory_bytes(process_id: int) -> int | None:
    """Read one process's resident set size on the Linux grading host."""
    try:
        status = Path(f"/proc/{process_id}/status").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    for line in status.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            return int(fields[1]) * 1024
    return None


def _linux_process_tree_resident_memory_bytes(process_id: int) -> int | None:
    """Sum resident memory for a controller launcher and all descendants."""
    pending = [process_id]
    visited: set[int] = set()
    total_bytes = 0
    measured_process = False
    while pending:
        current_process_id = pending.pop()
        if current_process_id in visited:
            continue
        visited.add(current_process_id)
        resident_bytes = _linux_resident_memory_bytes(current_process_id)
        if resident_bytes is not None:
            total_bytes += resident_bytes
            measured_process = True
        try:
            children_text = Path(
                f"/proc/{current_process_id}/task/{current_process_id}/children"
            ).read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            continue
        pending.extend(int(child_process_id) for child_process_id in children_text.split())
    return total_bytes if measured_process else None


def validate_controller(args: argparse.Namespace) -> dict[str, object]:
    from racing import RobotCommand, RobotSensors

    with controller_client(args) as client:
        command = client.command(RobotSensors())
    if not isinstance(command, RobotCommand):
        raise TypeError(f"{args.function} must return racing.RobotCommand")
    return {"ok": True, "callable": True}


def run_trial(args: argparse.Namespace) -> dict[str, object]:
    from racing.graphics.panda_config import configure_headless_panda
    from racing.graphics.track_rendering import add_racing_scene_collisions
    from racing.physics import (
        FORMULA_VEHICLE_PHYSICS_CONFIG,
        PhysicsScene,
        apply_robot_vehicle_command,
        apply_wall_impact_damage,
        create_physics_world,
        create_robot_vehicle,
    )
    from racing.race.progress import default_track_progress_model, project_track_position
    from racing.race.runtime import (
        RaceCarRuntime,
        lap_progress_tracker_for_spawn_pose,
        race_contact_states,
        race_spawn_poses,
        robot_is_eliminated,
        robot_score_damage,
        robot_track_point,
        update_race_runtime_after_step,
    )
    from racing.race.sensors import build_robot_sensors

    seconds = float(args.seconds)
    seed = int(args.seed)
    if seconds <= 0.0:
        raise ValueError("trial duration must be positive")
    fixed_delta_seconds = 1.0 / 60.0
    configure_headless_panda()
    showbase = cast(Any, import_module("direct.showbase.ShowBase"))
    base = showbase.ShowBase(windowType="none")
    root: Any | None = None
    try:
        model = default_track_progress_model()
        physics_world = create_physics_world()
        physics_scene = PhysicsScene(world=physics_world, vehicles=[])
        root = base.render.attachNewNode(f"gradescope-{seed}")
        add_racing_scene_collisions(physics_world=physics_world, render=root)
        spawn_pose = race_spawn_poses(
            1,
            model=model,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            random_seed=seed,
            race_index=1,
        )[0]
        robot = create_robot_vehicle(
            world=physics_world,
            render=root,
            name=f"gradescope-{seed}-car",
            position=spawn_pose.position,
            heading_degrees=spawn_pose.heading_degrees,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        )
        physics_scene.vehicles.append(robot)
        runtime = RaceCarRuntime(
            robot=robot,
            tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=spawn_pose),
        )
        elapsed_seconds = 0.0
        lap_crossing_times: list[float] = []
        previous_lap_count = 0
        with controller_client(args) as client:
            while elapsed_seconds < seconds:
                if not robot_is_eliminated(runtime.robot):
                    sensors, runtime.sensor_state = build_robot_sensors(
                        physics_world=physics_world,
                        robot=runtime.robot,
                        track_model=model,
                        time_s=elapsed_seconds,
                        dt_s=fixed_delta_seconds,
                        previous_state=runtime.sensor_state,
                    )
                    apply_robot_vehicle_command(robot=runtime.robot, command=client.command(sensors))

                physics_scene.step(fixed_delta_seconds)
                next_elapsed_seconds = min(seconds, elapsed_seconds + fixed_delta_seconds)
                contact_state = race_contact_states(physics_world=physics_world, runtimes=(runtime,))[0]
                apply_wall_impact_damage(
                    physics_world=physics_world,
                    robots=(runtime.robot,),
                    fixed_time_step=physics_scene.fixed_time_step,
                )
                projection = project_track_position(model, robot_track_point(runtime.robot))
                update_race_runtime_after_step(
                    runtime=runtime,
                    projection=projection,
                    contact_state=contact_state,
                    elapsed_seconds=next_elapsed_seconds,
                    delta_seconds=fixed_delta_seconds,
                )
                while previous_lap_count < runtime.tracker.lap_count:
                    lap_crossing_times.append(next_elapsed_seconds)
                    previous_lap_count += 1
                elapsed_seconds = next_elapsed_seconds

        lap_durations = [
            crossing - (lap_crossing_times[index - 1] if index else 0.0)
            for index, crossing in enumerate(lap_crossing_times)
        ]
        damage = robot_score_damage(runtime.robot)
        return {
            "ok": True,
            "seed": seed,
            "elapsed_seconds": elapsed_seconds,
            "raw_distance_m": runtime.tracker.best_distance_m,
            "partial_laps": runtime.tracker.best_distance_m / model.total_length_m,
            "lap_count": runtime.tracker.lap_count,
            "damage": damage,
            "survived": not robot_is_eliminated(runtime.robot) and damage < 1.0,
            "wall_contact_seconds": runtime.tracker.wall_contact_seconds,
            "max_speed_mps": runtime.max_speed_mps,
            "first_lap_time_seconds": lap_crossing_times[0] if lap_crossing_times else None,
            "best_lap_time_seconds": min(lap_durations) if lap_durations else None,
        }
    finally:
        if root is not None:
            root.removeNode()
        base.destroy()


def main() -> None:
    args = parse_args()
    try:
        result = validate_controller(args) if args.validate_only else run_trial(args)
    except BaseException as error:
        result = {"ok": False, "seed": args.seed, "error": f"{type(error).__name__}: {error}"[:1000]}
    print(RESULT_PREFIX + json.dumps(result, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
