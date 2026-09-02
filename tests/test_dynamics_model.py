from __future__ import annotations

import numpy as np

from controllers.dynamics_model import INPUT_DIM, OUTPUT_DIM, DynamicsModel, create_dynamics_model
from controllers.mpc_lib import ACTION_DIM, STATE_DIM


def _toy_dataset(*, samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """States/actions/next_states where next_state = state + [throttle, steer, 0, 0, 0, 0].

    A trivial, easily learnable relationship: only used to check the model
    can actually fit *something*, not to validate real dynamics.
    """
    states = rng.uniform(-1.0, 1.0, size=(samples, STATE_DIM))
    actions = rng.uniform(-1.0, 1.0, size=(samples, ACTION_DIM))
    delta = np.zeros((samples, STATE_DIM), dtype=np.float64)
    delta[:, 0] = actions[:, 0]
    delta[:, 1] = actions[:, 1]
    next_states = states + delta
    return states, actions, next_states


def test_create_dynamics_model_shapes() -> None:
    rng = np.random.default_rng(0)
    states, actions, next_states = _toy_dataset(samples=32, rng=rng)

    model = create_dynamics_model(
        states=states, actions=actions, next_states=next_states, hidden_dim=16, training_dt_s=1 / 60, rng=rng
    )

    assert model.w1.shape == (INPUT_DIM, 16)
    assert model.w2.shape == (16, OUTPUT_DIM)
    assert model.input_mean.shape == (INPUT_DIM,)
    assert model.output_mean.shape == (OUTPUT_DIM,)
    assert model.training_dt_s == 1 / 60


def test_predict_delta_output_shape() -> None:
    rng = np.random.default_rng(1)
    states, actions, next_states = _toy_dataset(samples=8, rng=rng)
    model = create_dynamics_model(
        states=states, actions=actions, next_states=next_states, hidden_dim=8, training_dt_s=1 / 60, rng=rng
    )

    delta = model.predict_delta(states, actions)

    assert delta.shape == (8, STATE_DIM)


def test_predict_fn_adds_delta_to_state() -> None:
    rng = np.random.default_rng(2)
    states, actions, next_states = _toy_dataset(samples=8, rng=rng)
    model = create_dynamics_model(
        states=states, actions=actions, next_states=next_states, hidden_dim=8, training_dt_s=1 / 60, rng=rng
    )

    predicted = model.predict_fn(states, actions, 1 / 60)

    np.testing.assert_allclose(predicted, states + model.predict_delta(states, actions))


def test_train_step_reduces_loss_on_toy_data() -> None:
    rng = np.random.default_rng(3)
    states, actions, next_states = _toy_dataset(samples=256, rng=rng)
    model = create_dynamics_model(
        states=states, actions=actions, next_states=next_states, hidden_dim=16, training_dt_s=1 / 60, rng=rng
    )

    first_loss = model.train_step(states, actions, next_states, learning_rate=0.1)
    for _ in range(200):
        loss = model.train_step(states, actions, next_states, learning_rate=0.1)

    assert loss < first_loss


def test_train_step_with_momentum_reduces_loss_faster_than_plain_gd() -> None:
    rng = np.random.default_rng(5)
    states, actions, next_states = _toy_dataset(samples=256, rng=rng)

    plain_model = create_dynamics_model(
        states=states,
        actions=actions,
        next_states=next_states,
        hidden_dim=16,
        training_dt_s=1 / 60,
        rng=np.random.default_rng(9),
    )
    momentum_model = create_dynamics_model(
        states=states,
        actions=actions,
        next_states=next_states,
        hidden_dim=16,
        training_dt_s=1 / 60,
        rng=np.random.default_rng(9),
    )
    velocity = {name: np.zeros_like(getattr(momentum_model, name)) for name in ("w1", "b1", "w2", "b2")}

    plain_loss = momentum_loss = 0.0
    for _ in range(50):
        plain_loss = plain_model.train_step(states, actions, next_states, learning_rate=0.1)
        momentum_loss = momentum_model.train_step(
            states, actions, next_states, learning_rate=0.1, momentum=0.9, velocity=velocity
        )

    assert momentum_loss < plain_loss


def test_loss_weights_prioritize_the_upweighted_dimension() -> None:
    rng = np.random.default_rng(6)
    states, actions, next_states = _toy_dataset(samples=256, rng=rng)

    uniform_model = create_dynamics_model(
        states=states,
        actions=actions,
        next_states=next_states,
        hidden_dim=16,
        training_dt_s=1 / 60,
        rng=np.random.default_rng(11),
    )
    weighted_model = create_dynamics_model(
        states=states,
        actions=actions,
        next_states=next_states,
        hidden_dim=16,
        training_dt_s=1 / 60,
        rng=np.random.default_rng(11),
    )
    loss_weights = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    for _ in range(100):
        uniform_model.train_step(states, actions, next_states, learning_rate=0.1)
        weighted_model.train_step(states, actions, next_states, learning_rate=0.1, loss_weights=loss_weights)

    target_delta = next_states - states
    uniform_dim0_error = np.mean((uniform_model.predict_delta(states, actions)[:, 0] - target_delta[:, 0]) ** 2)
    weighted_dim0_error = np.mean((weighted_model.predict_delta(states, actions)[:, 0] - target_delta[:, 0]) ** 2)

    assert weighted_dim0_error < uniform_dim0_error


def test_save_and_load_round_trips_predictions(tmp_path) -> None:
    rng = np.random.default_rng(4)
    states, actions, next_states = _toy_dataset(samples=16, rng=rng)
    model = create_dynamics_model(
        states=states, actions=actions, next_states=next_states, hidden_dim=8, training_dt_s=1 / 45, rng=rng
    )
    for _ in range(20):
        model.train_step(states, actions, next_states, learning_rate=0.1)

    path = tmp_path / "model.npz"
    model.save(path)
    loaded = DynamicsModel.load(path)

    np.testing.assert_allclose(loaded.predict_delta(states, actions), model.predict_delta(states, actions))
    assert loaded.training_dt_s == 1 / 45
