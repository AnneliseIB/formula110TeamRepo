"""Compare two student controllers across the standard head-to-head seed suite.

Runs one head-to-head race per seed in `HEAD_TO_HEAD_DEFAULT_SEED_SUITE`
(the same suite used for competitive evaluation elsewhere in the project) and
reports each side's scored distance, top speed, and safety stats (wall
contact time, damage, off-track time), per seed and aggregated. Used to
compare `mpc_baseline` (pathway 5) against `learned_dynamics_mpc` (pathway 7)
under identical conditions -- same planner and cost function, only the
dynamics model differs -- so any difference in these numbers isolates the
effect of the dynamics model.

Usage:
    uv run python scripts/evaluate_seed_suite.py \\
        --challenger controllers.learned_dynamics_mpc --incumbent controllers.mpc_baseline
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from racing import load_student_controller, run_headless_head_to_head
from racing.race.head_to_head import HeadToHeadTeamRaceStats
from racing.race.rules import HEAD_TO_HEAD_DEFAULT_SEED_SUITE

DEFAULT_ROUND_SECONDS = 20.0


@dataclass(frozen=True, slots=True)
class SeedSummary:
    seed: int
    distance_m: float
    max_speed_mps: float
    wall_contact_s: float
    damage: float
    off_track_s: float


def _summarize(seed: int, stats: HeadToHeadTeamRaceStats) -> SeedSummary:
    return SeedSummary(
        seed=seed,
        distance_m=stats.team_sum_distance_m,
        max_speed_mps=stats.max_speed_mps,
        wall_contact_s=stats.total_wall_contact_seconds,
        damage=stats.average_damage,
        off_track_s=stats.total_off_track_seconds,
    )


def evaluate(
    *,
    challenger_module: str,
    incumbent_module: str,
    seeds: tuple[int, ...],
    round_seconds: float,
) -> tuple[list[SeedSummary], list[SeedSummary]]:
    """Run one race per seed; return (challenger_summaries, incumbent_summaries)."""
    challenger_summaries: list[SeedSummary] = []
    incumbent_summaries: list[SeedSummary] = []

    for seed in seeds:
        challenger = load_student_controller(challenger_module)
        incumbent = load_student_controller(incumbent_module)
        result = run_headless_head_to_head(
            challenger_controller=challenger,
            incumbent_controller=incumbent,
            challenger_name=challenger_module,
            incumbent_name=incumbent_module,
            race_count=1,
            round_seconds=round_seconds,
            random_seed=seed,
        )
        race = result.races[0]
        challenger_summaries.append(_summarize(seed, race.challenger))
        incumbent_summaries.append(_summarize(seed, race.incumbent))

    return challenger_summaries, incumbent_summaries


def _print_table(name: str, summaries: list[SeedSummary]) -> None:
    print(f"\n{name}")
    print(f"{'seed':>6} {'distance_m':>11} {'max_speed':>10} {'wall_s':>7} {'damage':>7} {'off_track_s':>12}")
    for row in summaries:
        print(
            f"{row.seed:>6} {row.distance_m:>11.2f} {row.max_speed_mps:>10.2f} "
            f"{row.wall_contact_s:>7.2f} {row.damage:>7.3f} {row.off_track_s:>12.2f}"
        )
    total_distance = sum(row.distance_m for row in summaries)
    total_wall = sum(row.wall_contact_s for row in summaries)
    total_off_track = sum(row.off_track_s for row in summaries)
    mean_damage = sum(row.damage for row in summaries) / len(summaries)
    print(
        f"{'sum/avg':>6} {total_distance:>11.2f} {'--':>10} {total_wall:>7.2f} "
        f"{mean_damage:>7.3f} {total_off_track:>12.2f}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenger", required=True, help="module path, e.g. controllers.learned_dynamics_mpc")
    parser.add_argument("--incumbent", required=True, help="module path, e.g. controllers.mpc_baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(HEAD_TO_HEAD_DEFAULT_SEED_SUITE))
    parser.add_argument("--round-seconds", type=float, default=DEFAULT_ROUND_SECONDS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    challenger_summaries, incumbent_summaries = evaluate(
        challenger_module=args.challenger,
        incumbent_module=args.incumbent,
        seeds=tuple(args.seeds),
        round_seconds=args.round_seconds,
    )
    _print_table(args.challenger, challenger_summaries)
    _print_table(args.incumbent, incumbent_summaries)


if __name__ == "__main__":
    main()
