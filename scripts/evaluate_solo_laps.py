"""Measure laps, top speed, and lap times with the unmodified solo grading worker.

Each seed runs in a fresh process, including an isolated controller process.
This measures local physics performance; the live grader may use other seeds
or a newer runtime. Run from the repository root with the project Python.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "FORMULA110_RESULT="


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=Path("src/controllers/hybrid_control.py"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 110, 271, 997, 2027, 2026])
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--target-laps", type=float, default=4.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0 or args.target_laps <= 0:
        parser.error("seconds and target-laps must be positive")
    controller = args.controller.resolve()
    if not controller.is_file():
        parser.error(f"controller file does not exist: {controller}")

    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "FORMULA110_LOCAL_CONTROL": "1",
        "FORMULA110_CONTROL_WORKER": str(ROOT / "autograder/gradescope/control_worker.py"),
    }
    trials = []
    print(f"Solo trials: {args.seconds:g}s; target: {args.target_laps:g} laps", flush=True)
    print(f"{'seed':>6} {'laps':>9} {'top m/s':>9} {'first lap s':>11} {'best lap s':>11} {'damage %':>10} {'wall s':>8}")
    for seed in args.seeds:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "autograder/gradescope/race_worker.py"),
                "--submission",
                str(ROOT),
                "--module-file",
                str(controller),
                "--seed",
                str(seed),
                "--seconds",
                str(args.seconds),
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        payload = next(
            (line[len(RESULT_PREFIX) :] for line in completed.stdout.splitlines() if line.startswith(RESULT_PREFIX)),
            None,
        )
        if payload is None:
            raise RuntimeError(f"seed {seed}: worker returned no result: {completed.stderr[-2000:]}")
        trial = json.loads(payload)
        if trial.get("ok") is not True:
            raise RuntimeError(f"seed {seed}: {trial.get('error', 'trial failed')}")
        trials.append(trial)
        best_lap = trial["best_lap_time_seconds"]
        best_label = "--" if best_lap is None else f"{best_lap:.3f}"
        first_lap = trial["first_lap_time_seconds"]
        first_label = "--" if first_lap is None else f"{first_lap:.3f}"
        print(
            f"{seed:6d} {trial['partial_laps']:9.4f} {trial['max_speed_mps']:9.3f} "
            f"{first_label:>11} {best_label:>11} "
            f"{trial['damage'] * 100:10.3f} {trial['wall_contact_seconds']:8.3f}",
            flush=True,
        )
    mean_laps = sum(trial["partial_laps"] for trial in trials) / len(trials)
    minimum_laps = min(trial["partial_laps"] for trial in trials)
    report = {
        "controller": str(controller),
        "seconds": args.seconds,
        "target_laps": args.target_laps,
        "mean_partial_laps": mean_laps,
        "minimum_partial_laps": minimum_laps,
        "target_met_all_seeds": minimum_laps >= args.target_laps,
        "trials": trials,
    }
    print(f"Mean: {mean_laps:.4f} laps; minimum: {minimum_laps:.4f}; target: {args.target_laps:g}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
