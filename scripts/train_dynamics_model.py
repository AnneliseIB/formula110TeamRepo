"""Train the pathway 7 learned dynamics model on a collected transition dataset.

Reports one-step MSE (how well the model predicts a single tick ahead) *and*
multi-step rollout MSE (chaining the model's own predictions forward for a
planning horizon, exactly like `plan_action_sequence` does), since the two
can diverge sharply -- a model can look accurate one step ahead while its
errors compound into something useless a dozen steps out, which is what
actually matters once this feeds the planner.

Held-out data is whole *seeds*, not random rows or arbitrary `run_id`s --
holding out a `run_id` alone doesn't guarantee an unseen track region, since
the same seed can be raced multiple times. Held-out seeds also need to stay
contiguous per run for multi-step evaluation, which `run_id` still provides.

Usage:
    uv run python scripts/train_dynamics_model.py --dataset artifacts/dynamics_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from controllers.dynamics_model import DynamicsModel, create_dynamics_model

DEFAULT_DATASET_PATH = Path("artifacts/dynamics_dataset.jsonl")
DEFAULT_MODEL_PATH = Path("artifacts/dynamics_model.npz")
DEFAULT_HELD_OUT_SEEDS = 3
DEFAULT_HIDDEN_DIM = 32
DEFAULT_EPOCHS = 2000
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MOMENTUM = 0.9
DEFAULT_ROLLOUT_HORIZON = 12
DT_S_RELATIVE_TOLERANCE = 0.05

STATE_COLUMNS = (
    "speed_mps",
    "heading_error_rad",
    "center_offset_m",
    "lookahead_1_m",
    "lookahead_2_m",
    "lookahead_3_m",
)
# `mpc_lib.rollout_cost` only ever reads speed/heading_error/center_offset directly;
# the lookahead offsets matter only indirectly, as inputs the model itself uses to
# predict later steps. Upweight the three the planner actually costs.
DEFAULT_LOSS_WEIGHTS = (3.0, 3.0, 3.0, 1.0, 1.0, 1.0)
ACTION_COLUMNS = ("throttle", "steer")
NEXT_STATE_COLUMNS = tuple(f"next_{name}" for name in STATE_COLUMNS)


def load_rows(dataset_path: Path) -> list[dict[str, float]]:
    with dataset_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rows_to_arrays(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.array([[row[name] for name in STATE_COLUMNS] for row in rows], dtype=np.float64)
    actions = np.array([[row[name] for name in ACTION_COLUMNS] for row in rows], dtype=np.float64)
    next_states = np.array([[row[name] for name in NEXT_STATE_COLUMNS] for row in rows], dtype=np.float64)
    return states, actions, next_states


def resolve_training_dt_s(rows: list[dict[str, float]]) -> float:
    """Return the dataset's average tick length, warning if it isn't roughly fixed.

    The model predicts a delta over one fixed step size and has no way to
    account for a varying one, so a wide spread here would mean the whole
    approach needs rethinking, not just a warning -- but the game currently
    always runs headless races at a fixed physics timestep, so this is a
    sanity check, not an expected failure.
    """
    dt_values = np.array([row["dt_s"] for row in rows], dtype=np.float64)
    mean_dt_s = float(np.mean(dt_values))
    relative_spread = float(np.std(dt_values) / mean_dt_s) if mean_dt_s > 0 else float("inf")
    if relative_spread > DT_S_RELATIVE_TOLERANCE:
        print(
            f"WARNING: dataset dt_s varies by {relative_spread:.1%} relative to its mean "
            f"({mean_dt_s:.5f}s) -- the model's fixed-step-size assumption may not hold."
        )
    return mean_dt_s


def split_by_seed(
    rows: list[dict[str, float]], *, held_out_seeds: int
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Hold out whole seeds, not random rows or arbitrary `run_id`s.

    A held-out `run_id` alone doesn't guarantee an unseen track region --
    `run_id` just numbers individual controller copies, and a track section
    visited under one seed can recur under another. Holding out entire
    *seeds* is what actually tests whether the model generalizes to track
    situations it never trained on, rather than just interpolating within
    situations it already saw under a different run of the same seed.
    """
    seeds = sorted({int(row["seed"]) for row in rows})
    if held_out_seeds >= len(seeds):
        raise ValueError(f"held_out_seeds ({held_out_seeds}) must be less than the number of seeds ({len(seeds)})")
    held_out = set(seeds[-held_out_seeds:]) if held_out_seeds > 0 else set()
    print(f"Held-out seeds: {sorted(held_out)}")
    train_rows = [row for row in rows if int(row["seed"]) not in held_out]
    held_out_rows = [row for row in rows if int(row["seed"]) in held_out]
    return train_rows, held_out_rows


def one_step_mse(model: DynamicsModel, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> float:
    predicted = model.predict_fn(states, actions, model.training_dt_s)
    return float(np.mean((predicted - next_states) ** 2))


def multistep_rollout_mse(model: DynamicsModel, held_out_rows: list[dict[str, float]], *, horizon: int) -> float:
    """Chain the model's own predictions forward `horizon` steps within each held-out run.

    For every contiguous window of `horizon + 1` rows sharing a `run_id`,
    predict forward from the window's first state using the *actual* logged
    actions (not replanned ones -- this isolates dynamics-model error from
    planning behavior), and compare the final predicted state to the actual
    final logged state.
    """
    run_ids = sorted({int(row["run_id"]) for row in held_out_rows})
    squared_errors: list[np.ndarray] = []

    for run_id in run_ids:
        run_rows = [row for row in held_out_rows if int(row["run_id"]) == run_id]
        if len(run_rows) <= horizon:
            continue
        states, actions, _ = rows_to_arrays(run_rows)
        for start in range(len(run_rows) - horizon):
            current = states[start : start + 1]
            for step in range(horizon):
                current = model.predict_fn(current, actions[start + step : start + step + 1], model.training_dt_s)
            actual_final = states[start + horizon]
            squared_errors.append((current[0] - actual_final) ** 2)

    if not squared_errors:
        raise ValueError(
            f"no held-out run has more than {horizon} rows -- collect more data or lower --rollout-horizon"
        )
    return float(np.mean(np.stack(squared_errors)))


def train(
    *,
    dataset_path: Path,
    model_path: Path,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    momentum: float,
    loss_weights: tuple[float, ...],
    held_out_seeds: int,
    rollout_horizon: int,
    random_seed: int,
) -> None:
    rows = load_rows(dataset_path)
    training_dt_s = resolve_training_dt_s(rows)
    train_rows, held_out_rows = split_by_seed(rows, held_out_seeds=held_out_seeds)
    print(f"Loaded {len(rows)} transitions: {len(train_rows)} train, {len(held_out_rows)} held-out.")

    train_states, train_actions, train_next_states = rows_to_arrays(train_rows)
    rng = np.random.default_rng(random_seed)
    model = create_dynamics_model(
        states=train_states,
        actions=train_actions,
        next_states=train_next_states,
        hidden_dim=hidden_dim,
        training_dt_s=training_dt_s,
        rng=rng,
    )

    velocity = {name: np.zeros_like(getattr(model, name)) for name in ("w1", "b1", "w2", "b2")}
    loss_weights_array = np.array(loss_weights, dtype=np.float64)
    for epoch in range(epochs):
        loss = model.train_step(
            train_states,
            train_actions,
            train_next_states,
            learning_rate=learning_rate,
            momentum=momentum,
            velocity=velocity,
            loss_weights=loss_weights_array,
        )
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:5d}  train MSE (standardized) {loss:.6f}")

    held_out_states, held_out_actions, held_out_next_states = rows_to_arrays(held_out_rows)
    one_step_error = one_step_mse(model, held_out_states, held_out_actions, held_out_next_states)
    multistep_error = multistep_rollout_mse(model, held_out_rows, horizon=rollout_horizon)
    print(f"Held-out one-step MSE (raw state units): {one_step_error:.6f}")
    print(f"Held-out {rollout_horizon}-step rollout MSE (raw state units): {multistep_error:.6f}")

    model.save(model_path)
    print(f"Saved model to {model_path.resolve()}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM)
    parser.add_argument(
        "--loss-weights",
        type=float,
        nargs=6,
        default=list(DEFAULT_LOSS_WEIGHTS),
        metavar=tuple(name.upper() for name in STATE_COLUMNS),
        help="per-state-dimension training loss weight, in STATE_COLUMNS order",
    )
    parser.add_argument("--held-out-seeds", type=int, default=DEFAULT_HELD_OUT_SEEDS)
    parser.add_argument("--rollout-horizon", type=int, default=DEFAULT_ROLLOUT_HORIZON)
    parser.add_argument("--random-seed", type=int, default=110)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    train(
        dataset_path=args.dataset,
        model_path=args.output_model,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        loss_weights=tuple(args.loss_weights),
        held_out_seeds=args.held_out_seeds,
        rollout_horizon=args.rollout_horizon,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
