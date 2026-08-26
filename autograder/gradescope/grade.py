#!/usr/bin/env python3
"""Trusted Formula 110 Gradescope grading driver."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

AUTOGRADER_ROOT = Path("/autograder")
CONFIG_PATH = Path("/opt/formula110-autograder/config.json")
WORKER_PATH = Path("/opt/formula110-autograder/race_worker.py")
RUNTIME_PATH = Path("/opt/formula110-runtime")
VENV_PATH = AUTOGRADER_ROOT / "venv"
SUBMISSION_PATH = AUTOGRADER_ROOT / "submission"
RESULTS_PATH = AUTOGRADER_ROOT / "results" / "results.json"
RESULT_PREFIX = "FORMULA110_RESULT="
MAX_DIAGNOSTIC_CHARS = 6000


def read_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_results(results: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULTS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(RESULTS_PATH)


def blank_results(message: str) -> dict[str, Any]:
    return {
        "output": message,
        "output_format": "text",
        "test_output_format": "text",
        "test_name_format": "text",
        "stdout_visibility": "hidden",
        "tests": [],
        "leaderboard": [],
    }


def locate_module(module_name: str) -> tuple[Path | None, str]:
    relative = Path(*module_name.split(".")).with_suffix(".py")
    direct_candidates = [SUBMISSION_PATH / relative, SUBMISSION_PATH / "src" / relative]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate.resolve(), f"found {candidate.relative_to(SUBMISSION_PATH)}"

    matches = sorted(
        path.resolve() for path in SUBMISSION_PATH.rglob(relative.name) if path.is_file() and not path.is_symlink()
    )
    suffix_parts = relative.parts
    exact_suffix_matches = [path for path in matches if path.parts[-len(suffix_parts) :] == suffix_parts]
    if len(exact_suffix_matches) == 1:
        return exact_suffix_matches[0], f"found {exact_suffix_matches[0].relative_to(SUBMISSION_PATH)}"
    if len(matches) == 1:
        return matches[0], f"found flattened upload {matches[0].relative_to(SUBMISSION_PATH)}"
    if len(matches) > 1:
        names = ", ".join(str(path.relative_to(SUBMISSION_PATH)) for path in matches[:8])
        return None, f"multiple possible files named {relative.name}: {names}"
    return None, f"expected {relative} (or src/{relative}) in the submission"


def run_command(command: list[str], *, timeout_seconds: float) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=SUBMISSION_PATH,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
            text=True,
            errors="replace",
            env={
                "HOME": "/tmp/formula110-grader",
                "PATH": f"{VENV_PATH / 'bin'}:/usr/bin:/bin",
                "PYTHONPATH": str(RUNTIME_PATH),
            },
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout_seconds:.0f} seconds"
    output = completed.stdout[-MAX_DIAGNOSTIC_CHARS:].strip()
    if not output:
        output = "no diagnostics"
    return completed.returncode == 0, output


def static_checks(module_file: Path | None, check_id: str) -> dict[str, tuple[bool, str]]:
    if module_file is None:
        message = "cannot run because this module file is missing"
        return {"pyright": (False, message), "ruff_lint": (False, message), "ruff_format": (False, message)}
    files = [str(module_file)]

    pyright_config = {
        "pythonVersion": "3.11",
        "pythonPlatform": "Linux",
        "typeCheckingMode": "strict",
        "reportMissingTypeStubs": False,
        "executionEnvironments": [
            {
                "root": str(SUBMISSION_PATH),
                "extraPaths": [str(RUNTIME_PATH), str(SUBMISSION_PATH), str(SUBMISSION_PATH / "src")],
            }
        ],
    }
    config_path = RESULTS_PATH.parent / f"pyrightconfig-{check_id}.json"
    config_path.write_text(json.dumps(pyright_config), encoding="utf-8")
    pyright = run_command(
        [
            str(VENV_PATH / "bin" / "pyright"),
            "--pythonpath",
            str(VENV_PATH / "bin" / "python"),
            "--project",
            str(config_path),
            *files,
        ],
        timeout_seconds=30.0,
    )
    ruff_lint = run_command(
        [str(VENV_PATH / "bin" / "ruff"), "check", "--isolated", *files],
        timeout_seconds=15.0,
    )
    ruff_format = run_command(
        [str(VENV_PATH / "bin" / "ruff"), "format", "--check", "--isolated", *files],
        timeout_seconds=15.0,
    )
    return {"pyright": pyright, "ruff_lint": ruff_lint, "ruff_format": ruff_format}


def worker_command(arguments: list[str]) -> list[str]:
    prlimit = shutil.which("prlimit") or "/usr/bin/prlimit"
    return [
        prlimit,
        "--cpu=25",
        "--as=3221225472",
        "--fsize=1048576",
        "--nproc=128",
        "--",
        "/usr/bin/env",
        "-i",
        "HOME=/tmp",
        "PATH=/autograder/venv/bin:/usr/bin:/bin",
        "PYTHONPATH=/opt/formula110-runtime",
        str(VENV_PATH / "bin" / "python"),
        str(WORKER_PATH),
        *arguments,
    ]


def run_worker(arguments: list[str], *, timeout_seconds: float) -> dict[str, Any]:
    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=RESULTS_PATH.parent, prefix="worker-", delete=False) as output_file:
            output_path = Path(output_file.name)
            try:
                completed = subprocess.run(
                    worker_command(arguments),
                    cwd=SUBMISSION_PATH,
                    stdin=subprocess.DEVNULL,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {"ok": False, "error": f"timed out after {timeout_seconds:.0f} seconds"}

        output = output_path.read_bytes()[-1_048_576:].decode("utf-8", errors="replace")
        payload_line = next((line for line in reversed(output.splitlines()) if line.startswith(RESULT_PREFIX)), None)
        if payload_line is None:
            diagnostic = output[-2000:].strip()
            return {
                "ok": False,
                "error": f"worker exited {completed.returncode} without a result",
                "diagnostic": diagnostic,
            }
        try:
            payload = json.loads(payload_line[len(RESULT_PREFIX) :])
        except json.JSONDecodeError as error:
            return {"ok": False, "error": f"worker returned invalid JSON: {error}"}
        if not isinstance(payload, dict):
            return {"ok": False, "error": "worker result was not an object"}
        return payload
    finally:
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def validate_control(module_file: Path | None, function_name: str) -> dict[str, Any]:
    if module_file is None:
        return {"ok": False, "error": "module file is missing"}
    return run_worker(
        [
            "--submission",
            str(SUBMISSION_PATH),
            "--module-file",
            str(module_file),
            "--function",
            function_name,
            "--validate-only",
        ],
        timeout_seconds=10.0,
    )


def run_trials(
    module_file: Path | None,
    function_name: str,
    seeds: list[int],
    duration_seconds: float,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    if module_file is None:
        return [{"ok": False, "seed": seed, "error": "module file is missing"} for seed in seeds]
    return [
        run_worker(
            [
                "--submission",
                str(SUBMISSION_PATH),
                "--module-file",
                str(module_file),
                "--function",
                function_name,
                "--seed",
                str(seed),
                "--seconds",
                str(duration_seconds),
            ],
            timeout_seconds=timeout_seconds,
        )
        for seed in seeds
    ]


def test_case(name: str, passed: bool, points: float, output: str, number: str) -> dict[str, Any]:
    return {
        "name": name,
        "number": number,
        "score": points if passed else 0.0,
        "max_score": points,
        "status": "passed" if passed else "failed",
        "output": output[-MAX_DIAGNOSTIC_CHARS:],
        "visibility": "visible",
    }


def validation_output(module_name: str, result: dict[str, Any]) -> str:
    if result.get("ok") is True:
        return f"{module_name}.control loaded, accepted RobotSensors, and returned RobotCommand."
    return f"{module_name}: {result.get('error', 'validation failed')}"


def trial_summary(module_name: str, trials: list[dict[str, Any]]) -> str:
    lines = [module_name]
    for trial in trials:
        seed = trial.get("seed", "?")
        if trial.get("ok") is not True:
            lines.append(f"seed {seed}: ERROR — {trial.get('error', 'trial failed')}")
            continue
        lines.append(
            f"seed {seed}: {float(trial['partial_laps']):.3f} laps, "
            f"{float(trial['raw_distance_m']):.1f} m, "
            f"top {float(trial['max_speed_mps']):.2f} m/s, "
            f"damage {float(trial['damage']) * 100.0:.1f}%, "
            f"wall contact {float(trial['wall_contact_seconds']):.3f} s"
        )
    return "\n".join(lines)


def all_trials(trials: list[dict[str, Any]], predicate: Any) -> bool:
    return bool(trials) and all(trial.get("ok") is True and predicate(trial) for trial in trials)


def mean_metric(trials: list[dict[str, Any]], key: str) -> float:
    return sum(float(trial[key]) for trial in trials) / len(trials)


def leaderboard_for(trials: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    qualified = all_trials(trials, lambda trial: bool(trial["survived"]) and float(trial["damage"]) < 1.0)
    if not qualified:
        return [], "Leaderboard: DQ — the improved controller did not finish both 30-second seeded runs."

    leaderboard: list[dict[str, Any]] = [
        {"name": "Laps (partial)", "value": round(mean_metric(trials, "partial_laps"), 4)},
        {"name": "Top speed (m/s)", "value": round(mean_metric(trials, "max_speed_mps"), 3)},
    ]
    first_laps = [trial.get("first_lap_time_seconds") for trial in trials]
    best_laps = [trial.get("best_lap_time_seconds") for trial in trials]
    leaderboard.append(
        {
            "name": "First lap (s)",
            "value": round(sum(float(value) for value in first_laps) / len(first_laps), 3)
            if all(value is not None for value in first_laps)
            else "No lap",
            "order": "asc",
        }
    )
    leaderboard.append(
        {
            "name": "Best lap (s)",
            "value": round(sum(float(value) for value in best_laps) / len(best_laps), 3)
            if all(value is not None for value in best_laps)
            else "No lap",
            "order": "asc",
        }
    )
    return leaderboard, "Leaderboard: qualified; metrics average seeds 110 and 2026."


def grade() -> dict[str, Any]:
    started = time.monotonic()
    config = read_config()
    modules = config["modules"]
    first_name = str(modules["minimum_viable"])
    second_name = str(modules["improved"])
    function_name = str(config["control_function"])
    seeds = [int(seed) for seed in config["seeds"]]
    duration = float(config["duration_seconds"])
    trial_timeout = float(config["trial_timeout_seconds"])
    points = config["rubric"]

    first_file, first_location = locate_module(first_name)
    second_file, second_location = locate_module(second_name)
    first_checks = static_checks(first_file, "minimum")
    second_checks = static_checks(second_file, "improved")
    first_validation = validate_control(first_file, function_name)
    second_validation = validate_control(second_file, function_name)
    first_trials = (
        run_trials(first_file, function_name, seeds, duration, trial_timeout)
        if first_validation.get("ok") is True
        else [{"ok": False, "seed": seed, "error": "control validation failed"} for seed in seeds]
    )
    second_trials = (
        run_trials(second_file, function_name, seeds, duration, trial_timeout)
        if second_validation.get("ok") is True
        else [{"ok": False, "seed": seed, "error": "control validation failed"} for seed in seeds]
    )

    first_summary = trial_summary(first_name, first_trials)
    second_summary = trial_summary(second_name, second_trials)
    improvement_lines: list[str] = []
    improved_each_seed = True
    for seed, first, second in zip(seeds, first_trials, second_trials, strict=True):
        if first.get("ok") is not True or second.get("ok") is not True:
            improved_each_seed = False
            improvement_lines.append(f"seed {seed}: comparison unavailable")
            continue
        first_distance = float(first["raw_distance_m"])
        second_distance = float(second["raw_distance_m"])
        difference = second_distance - first_distance
        if difference <= 1e-6:
            improved_each_seed = False
        improvement_lines.append(
            f"seed {seed}: improved {second_distance:.1f} m vs minimum {first_distance:.1f} m ({difference:+.1f} m)"
        )

    tests = [
        test_case(
            "Minimum module: strict type checking",
            first_checks["pyright"][0],
            float(points["minimum_pyright_strict"]),
            first_checks["pyright"][1],
            "1.1",
        ),
        test_case(
            "Improved module: strict type checking",
            second_checks["pyright"][0],
            float(points["improved_pyright_strict"]),
            second_checks["pyright"][1],
            "1.2",
        ),
        test_case(
            "Minimum module: Ruff default lint rules",
            first_checks["ruff_lint"][0],
            float(points["minimum_ruff_lint"]),
            first_checks["ruff_lint"][1],
            "1.3",
        ),
        test_case(
            "Improved module: Ruff default lint rules",
            second_checks["ruff_lint"][0],
            float(points["improved_ruff_lint"]),
            second_checks["ruff_lint"][1],
            "1.4",
        ),
        test_case(
            "Minimum module: Ruff formatting",
            first_checks["ruff_format"][0],
            float(points["minimum_ruff_format"]),
            first_checks["ruff_format"][1],
            "1.5",
        ),
        test_case(
            "Improved module: Ruff formatting",
            second_checks["ruff_format"][0],
            float(points["improved_ruff_format"]),
            second_checks["ruff_format"][1],
            "1.6",
        ),
        test_case(
            "Minimum module defines a valid control function",
            first_validation.get("ok") is True,
            float(points["minimum_control"]),
            validation_output(first_name, first_validation),
            "2.1",
        ),
        test_case(
            "Improved module defines a valid control function",
            second_validation.get("ok") is True,
            float(points["improved_control"]),
            validation_output(second_name, second_validation),
            "2.2",
        ),
        test_case(
            "Minimum module completes a lap on both seeds",
            all_trials(first_trials, lambda trial: int(trial["lap_count"]) >= 1),
            float(points["minimum_lap"]),
            first_summary,
            "3.1",
        ),
        test_case(
            "Minimum module finishes 30 seconds with no damage on both seeds",
            all_trials(
                first_trials,
                lambda trial: bool(trial["survived"]) and float(trial["damage"]) <= 1e-9,
            ),
            float(points["minimum_no_damage"]),
            first_summary,
            "3.2",
        ),
        test_case(
            "Minimum module avoids all wall contact on both seeds",
            all_trials(first_trials, lambda trial: float(trial["wall_contact_seconds"]) <= 1e-9),
            float(points["minimum_no_walls"]),
            first_summary,
            "3.3",
        ),
        test_case(
            "Improved module survives 30 seconds on both seeds",
            all_trials(second_trials, lambda trial: bool(trial["survived"])),
            float(points["improved_survival"]),
            second_summary,
            "4.1",
        ),
        test_case(
            "Improved module goes farther on both seeds",
            improved_each_seed,
            float(points["improved_distance"]),
            "\n".join(improvement_lines),
            "4.2",
        ),
    ]
    leaderboard, leaderboard_message = leaderboard_for(second_trials)
    return {
        "execution_time": round(time.monotonic() - started, 3),
        "output": (
            f"Required modules:\n- {first_name}: {first_location}\n- {second_name}: {second_location}\n\n"
            f"{leaderboard_message}"
        ),
        "output_format": "text",
        "test_output_format": "text",
        "test_name_format": "text",
        "stdout_visibility": "hidden",
        "tests": tests,
        "leaderboard": leaderboard,
        "extra_data": {"minimum_trials": first_trials, "improved_trials": second_trials},
    }


def main() -> None:
    results = blank_results("The autograder started but did not finish.")
    write_results(results)
    try:
        results = grade()
    except BaseException as error:
        results = blank_results(f"Autograder infrastructure error: {type(error).__name__}: {error}")
    write_results(results)


if __name__ == "__main__":
    main()
