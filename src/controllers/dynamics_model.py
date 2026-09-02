"""Pathway 7 (Learned Dynamics) model: predicts `LocalState` deltas from (state, action).

A single-hidden-layer MLP trained with plain gradient descent (manual
backprop, no autodiff/torch -- the network is small enough that a hand-rolled
implementation is enough to learn something meaningful). Trained on logged
`(state, action, next_state)` transitions from
`scripts/collect_dynamics_dataset.py` to predict the delta between `state`
and `next_state` over one fixed simulator tick (`training_dt_s`), so it can
in principle pick up on how the track's curvature actually evolves as the
car drives -- the specific thing `mpc_baseline.kinematic_predict` gets wrong
on curves (see the lab notebook).

The model is trained on and predicts over a fixed step size; it has no way
to extrapolate to a different one, so any planner using it (see
`learned_dynamics_mpc.py`) must roll out with `dt_s == training_dt_s`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from controllers.mpc_lib import ACTION_DIM, STATE_DIM

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

INPUT_DIM = STATE_DIM + ACTION_DIM
OUTPUT_DIM = STATE_DIM
NORMALIZATION_STD_FLOOR = 1e-6


@dataclass(slots=True)
class DynamicsModel:
    """1-hidden-layer MLP predicting a fixed-`training_dt_s` state delta.

    Inputs and target deltas are standardized (zero mean, unit variance)
    using stats fit once on the training set: the state/action dimensions
    span very different physical scales (radians, meters, [-1, 1] controls),
    and plain gradient descent struggles without it.
    """

    w1: FloatArray  # (INPUT_DIM, hidden_dim)
    b1: FloatArray  # (hidden_dim,)
    w2: FloatArray  # (hidden_dim, OUTPUT_DIM)
    b2: FloatArray  # (OUTPUT_DIM,)
    input_mean: FloatArray  # (INPUT_DIM,)
    input_std: FloatArray  # (INPUT_DIM,)
    output_mean: FloatArray  # (OUTPUT_DIM,)
    output_std: FloatArray  # (OUTPUT_DIM,)
    training_dt_s: float

    @property
    def hidden_dim(self) -> int:
        return self.w1.shape[1]

    def predict_delta(self, states: FloatArray, actions: FloatArray) -> FloatArray:
        """Predict the (unstandardized) state delta for each (state, action) pair."""
        _, _, raw_output = self._forward(states, actions)
        return raw_output * self.output_std + self.output_mean

    def predict_fn(self, states: FloatArray, actions: FloatArray, dt_s: float) -> FloatArray:
        """`mpc_lib.PredictFn`-compatible next-state prediction.

        `dt_s` is accepted only to satisfy the shared planner interface and
        is otherwise ignored -- the model always predicts the delta over its
        own fixed `training_dt_s`. Callers must roll out with
        `dt_s == training_dt_s` (see `learned_dynamics_mpc.py`).
        """
        return states + self.predict_delta(states, actions)

    def train_step(
        self,
        states: FloatArray,
        actions: FloatArray,
        next_states: FloatArray,
        *,
        learning_rate: float,
        momentum: float = 0.0,
        velocity: dict[str, FloatArray] | None = None,
        loss_weights: FloatArray | None = None,
    ) -> float:
        """One full-batch gradient-descent step; returns the (possibly weighted) mean-squared error.

        Plain gradient descent converges very slowly on this network (see
        `scripts/train_dynamics_model.py`'s tuning notes) -- pass `momentum`
        (e.g. 0.9) and a `velocity` dict (start with zero arrays matching
        `w1`/`b1`/`w2`/`b2`; this method updates it in place) to use
        classical momentum instead. Momentum is plain arithmetic on the
        already hand-computed gradients below, not autodiff, so it doesn't
        pull in any additional dependency.

        `loss_weights` (shape `(OUTPUT_DIM,)`) lets some state dimensions
        matter more than others when training -- e.g. `rollout_cost` only
        ever reads `speed_mps`/`heading_error_rad`/`center_offset_m`
        directly, so upweighting those over the lookahead offsets spends the
        network's limited capacity on the dimensions the planner actually
        costs. Defaults to equal weighting.
        """
        normalized_input, hidden, raw_output = self._forward(states, actions)
        target_delta = next_states - states
        target_standardized = (target_delta - self.output_mean) / self.output_std

        weights = np.ones(OUTPUT_DIM, dtype=np.float64) if loss_weights is None else loss_weights
        error = raw_output - target_standardized
        loss = float(np.mean(weights[None, :] * error**2))

        sample_count = states.shape[0]
        d_raw_output = (2.0 / (sample_count * OUTPUT_DIM)) * weights[None, :] * error  # (N, OUTPUT_DIM)
        d_w2 = hidden.T @ d_raw_output
        d_b2 = np.sum(d_raw_output, axis=0)

        d_hidden = d_raw_output @ self.w2.T
        d_pre_activation = d_hidden * (hidden > 0.0)  # ReLU derivative
        d_w1 = normalized_input.T @ d_pre_activation
        d_b1 = np.sum(d_pre_activation, axis=0)

        if velocity is None:
            self.w2 -= learning_rate * d_w2
            self.b2 -= learning_rate * d_b2
            self.w1 -= learning_rate * d_w1
            self.b1 -= learning_rate * d_b1
        else:
            for name, grad in (("w2", d_w2), ("b2", d_b2), ("w1", d_w1), ("b1", d_b1)):
                velocity[name] = momentum * velocity[name] - learning_rate * grad
                setattr(self, name, getattr(self, name) + velocity[name])
        return loss

    def _forward(self, states: FloatArray, actions: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        inputs = np.concatenate([states, actions], axis=-1)
        normalized_input = (inputs - self.input_mean) / self.input_std
        hidden = np.maximum(normalized_input @ self.w1 + self.b1, 0.0)
        raw_output = hidden @ self.w2 + self.b2
        return normalized_input, hidden, raw_output

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            input_mean=self.input_mean,
            input_std=self.input_std,
            output_mean=self.output_mean,
            output_std=self.output_std,
            training_dt_s=np.array(self.training_dt_s, dtype=np.float64),
        )

    @staticmethod
    def load(path: Path) -> DynamicsModel:
        with np.load(path) as data:
            return DynamicsModel(
                w1=data["w1"],
                b1=data["b1"],
                w2=data["w2"],
                b2=data["b2"],
                input_mean=data["input_mean"],
                input_std=data["input_std"],
                output_mean=data["output_mean"],
                output_std=data["output_std"],
                training_dt_s=float(data["training_dt_s"]),
            )


def create_dynamics_model(
    *,
    states: FloatArray,
    actions: FloatArray,
    next_states: FloatArray,
    hidden_dim: int,
    training_dt_s: float,
    rng: np.random.Generator,
) -> DynamicsModel:
    """Initialize a fresh model with weights scaled for ReLU and stats fit on the given data.

    Normalization stats are computed once from `states`/`actions`/`next_states`
    (the training split) and then frozen into the model, so later calls to
    `train_step` or `predict_fn` reuse exactly these stats rather than
    recomputing them from whatever batch happens to be passed in.
    """
    inputs = np.concatenate([states, actions], axis=-1)
    input_mean = np.mean(inputs, axis=0)
    input_std = np.maximum(np.std(inputs, axis=0), NORMALIZATION_STD_FLOOR)

    target_delta = next_states - states
    output_mean = np.mean(target_delta, axis=0)
    output_std = np.maximum(np.std(target_delta, axis=0), NORMALIZATION_STD_FLOOR)

    scale1 = np.sqrt(2.0 / INPUT_DIM)
    scale2 = np.sqrt(2.0 / hidden_dim)
    return DynamicsModel(
        w1=rng.normal(0.0, scale1, size=(INPUT_DIM, hidden_dim)),
        b1=np.zeros(hidden_dim, dtype=np.float64),
        w2=rng.normal(0.0, scale2, size=(hidden_dim, OUTPUT_DIM)),
        b2=np.zeros(OUTPUT_DIM, dtype=np.float64),
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
        training_dt_s=training_dt_s,
    )


__all__ = ["INPUT_DIM", "OUTPUT_DIM", "DynamicsModel", "create_dynamics_model"]
