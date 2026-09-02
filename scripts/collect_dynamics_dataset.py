"""Collect (state, action, next_state) transitions for training a learned dynamics model.

Drives with a smoothed random-exploration policy across several seeds so the
logged data covers a wide range of throttle/steer combinations, then writes
each transition as one flat-numeric JSON line, tagged with both a `run_id`
(one continuous drive) and the `seed` it came from. Both `mpc_baseline.py`'s
planner and this script use the same `LocalState` representation
(`controllers.mpc_lib.local_state_from_sensors`), so the dataset directly
matches what the planner will predict over.

`DEFAULT_SEEDS` deliberately does *not* overlap with
`racing.race.rules.HEAD_TO_HEAD_DEFAULT_SEED_SUITE`: that suite is reserved
for head-to-head evaluation of the resulting controller, so training the
dynamics model on different seeds keeps that evaluation an honest test of
generalization to track situations the model never trained on (see
`scripts/train_dynamics_model.py`'s seed-based held-out split for the same
reasoning applied within the training data itself).

Usage:
    uv run python scripts/collect_dynamics_dataset.py --seeds 1 2 3 --races-per-seed 2
"""

from __future__ import annotations

import argparse
import json
from itertools import count
from pathlib import Path

import numpy as np

from controllers.mpc_lib import LocalState, local_state_from_sensors
from racing import RobotCommand, RobotSensors, run_headless_head_to_head

DEFAULT_OUTPUT_PATH = Path("artifacts/dynamics_dataset.jsonl")
# Deliberately disjoint from `HEAD_TO_HEAD_DEFAULT_SEED_SUITE` -- see module docstring.
DEFAULT_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
DEFAULT_RACES_PER_SEED = 2
DEFAULT_ROUND_SECONDS = 40.0
# 8 ticks (~0.13s) never let speed actually build past a few m/s before resampling, so the
# dataset had almost no *dense* coverage of sustained higher speeds -- exactly the gap that
# forced `learned_dynamics_mpc` to cap itself at 6 m/s despite the vehicle safely handling much
# higher speeds under simple reactive control (see the 2026-09-01 lab notebook entry). Holding
# each action for longer lets speed genuinely build and cruise within a single hold.
ACTION_PERSISTENCE_STEPS = 45
STEER_LIMIT = 0.9  # stay off the extremes so the car doesn't just spin in place
# Biased toward forward driving rather than uniform(-1, 1): braking/reverse behavior is already
# densely covered from short holds in earlier data, and what's missing is sustained forward
# cruising and cornering at real speed, so most of the new exploration budget should go there.
THROTTLE_MIN = -0.2
THROTTLE_MAX = 1.0

# A "disturbance burst" mode was tried here on 2026-09-01: periodically inject a short, sharp
# high-throttle/hard-steer action to deliberately create high-speed, large-heading-error
# training examples (rare in plain random exploration, since a well-behaving policy rarely
# reaches that combination on its own). It measurably improved the model's fit and race behavior
# in exactly that regime -- but retraining on the resulting dataset was a net regression overall
# (5-seed total distance 780.75m -> 430.32m), evidently costing this small, fixed-capacity model
# accuracy elsewhere to accommodate the added disturbance examples. Reverted rather than shipped;
# see the 2026-09-01 lab notebook entry for the full comparison.


class RandomExplorationController:
    """Holds a random action for a few ticks at a time and logs every transition it sees.

    Logs into the shared `rows` list passed at construction, so every copy
    spawned by `copy_for_car` (one per race car) still writes into the same
    dataset. Each copy gets its own `run_id` (drawn from the shared
    `run_id_source` counter) so rows from the same continuous drive can be
    grouped back together later -- needed to check the trained model's
    predictions several steps ahead, not just one step ahead.
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        rows: list[dict[str, float]],
        run_id_source: count[int],
    ) -> None:
        self._rng = rng
        self._rows = rows
        self._run_id_source = run_id_source
        self._run_id = next(run_id_source)
        self._steps_remaining = 0
        self._current = RobotCommand()
        self._previous_state: LocalState | None = None
        self._previous_action: RobotCommand | None = None

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        state = local_state_from_sensors(sensors)
        if self._previous_state is not None and self._previous_action is not None:
            self._rows.append(
                _transition_row(
                    self._previous_state,
                    self._previous_action,
                    state,
                    run_id=self._run_id,
                    dt_s=sensors.dt_s,
                )
            )

        if self._steps_remaining <= 0:
            throttle = float(self._rng.uniform(THROTTLE_MIN, THROTTLE_MAX))
            steer = float(self._rng.uniform(-STEER_LIMIT, STEER_LIMIT))
            self._current = RobotCommand(throttle=throttle, steer=steer)
            self._steps_remaining = ACTION_PERSISTENCE_STEPS
        self._steps_remaining -= 1

        self._previous_state = state
        self._previous_action = self._current
        return self._current

    def copy_for_car(self) -> RandomExplorationController:
        return RandomExplorationController(
            rng=np.random.default_rng(self._rng.integers(0, 2**31 - 1)),
            rows=self._rows,
            run_id_source=self._run_id_source,
        )


def _transition_row(
    state: LocalState, action: RobotCommand, next_state: LocalState, *, run_id: int, dt_s: float
) -> dict[str, float]:
    return {
        "run_id": run_id,
        "dt_s": dt_s,
        "speed_mps": state.speed_mps,
        "heading_error_rad": state.heading_error_rad,
        "center_offset_m": state.center_offset_m,
        "lookahead_1_m": state.lookahead_offsets_m[0],
        "lookahead_2_m": state.lookahead_offsets_m[1],
        "lookahead_3_m": state.lookahead_offsets_m[2],
        "throttle": action.throttle,
        "steer": action.steer,
        "next_speed_mps": next_state.speed_mps,
        "next_heading_error_rad": next_state.heading_error_rad,
        "next_center_offset_m": next_state.center_offset_m,
        "next_lookahead_1_m": next_state.lookahead_offsets_m[0],
        "next_lookahead_2_m": next_state.lookahead_offsets_m[1],
        "next_lookahead_3_m": next_state.lookahead_offsets_m[2],
    }


def collect_dataset(
    *,
    seeds: tuple[int, ...],
    races_per_seed: int,
    round_seconds: float,
    random_seed: int = 110,
) -> list[dict[str, float]]:
    """Run headless races across `seeds` and return the logged transition rows."""
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, float]] = []
    run_id_source = count()

    for seed in seeds:
        challenger = RandomExplorationController(
            rng=np.random.default_rng(rng.integers(0, 2**31 - 1)), rows=rows, run_id_source=run_id_source
        )
        incumbent = RandomExplorationController(
            rng=np.random.default_rng(rng.integers(0, 2**31 - 1)), rows=rows, run_id_source=run_id_source
        )
        rows_before = len(rows)
        run_headless_head_to_head(
            challenger_controller=challenger,
            incumbent_controller=incumbent,
            race_count=races_per_seed,
            round_seconds=round_seconds,
            random_seed=seed,
        )
        for row in rows[rows_before:]:
            row["seed"] = seed

    return rows


def write_jsonl(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--races-per-seed", type=int, default=DEFAULT_RACES_PER_SEED)
    parser.add_argument("--round-seconds", type=float, default=DEFAULT_ROUND_SECONDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--random-seed", type=int, default=110, help="seed for the exploration policy's own RNG")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    rows = collect_dataset(
        seeds=tuple(args.seeds),
        races_per_seed=args.races_per_seed,
        round_seconds=args.round_seconds,
        random_seed=args.random_seed,
    )
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} transitions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
