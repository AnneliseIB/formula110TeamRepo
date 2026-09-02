# EXPERIMENTAL APPROACHES

## Planning and Model-Predictive Control

Model-predictive control is a methodology that looks into the future, compares several possible
future action sequences, and chooses the best one.

**Hypothesis:** Because model-predictive control evaluates future steering and throttle
sequences instead of only reacting to the current state, it may allow the car to anticipate
curves and choose actions that improve speed while remaining on the track.

**Minimum Experiment:**
- Look at the current state/sensors
- Generate possible future steering/throttle sequences
- Predict what will happen for each sequence
- Score each predicted future
- Choose the best sequence
- Only execute the first action
- Get new sensor readings and repeat the whole process

**Evaluation:** Evaluate the controller using a fixed random seed for reproducibility and compare
its performance with an existing baseline controller. Performance can be measured using lap
time, track completion, scored distance, maximum speed, off-track time, and wall contact.
Benchmarking against learned-dynamics MPC and the `smol_brain` reactive controller.

## Learned Dynamics and Model-Based Learning

Learned Dynamics or Model-Based Learning involves training a model to predict how the car's
state will change after taking a given action. The controller learns these dynamics from
simulator data and can then use the learned model to predict future states.

**Hypothesis:** A model trained on data from a random-exploration driving controller may learn
the relationship between the car's current state, steering and throttle actions, and resulting
next state well enough to accurately predict future car behavior and support improved control.

**Minimum Experiment:** Collect simulator data containing the current state, steering and
throttle action, and resulting next state, then train a small model to predict the next state
from the current state and action.

**Evaluation:** Consistent seed-based performance metrics compared against the manual MPC and
`smol_brain`.

---

## LAB NOTEBOOK ENTRIES

### Planning and Model-Predictive Control

**Date and time:** Monday, Aug 31

**Participants and contributions:** Sophia Moloo, experimental design.

**Question or objective:** Can a simple MPC controller reach competitive performance?

**What we investigated or changed:** Built a basic MPC controller that tested different future
steering and throttle actions. We adjusted the scoring system to prioritize speed and added plan
warm-starting so the car could reuse part of its previous plan instead of starting from scratch
every time.

Sources: `README.md`, `SENSORS.md`, `api.py`, `rules.py`
AI-agent assistance: Claude Code helped with implementation, running simulations, and analyzing
results.
Commits or code: `mpc_baseline.py`, `mpc_lib.py`, `test_mpc_lib.py`

**What we observed:** The car was stable but too slow at first. Reusing the previous plan made
the car less hesitant, but it also caused the car to drift to one side. We found that the basic
prediction system was not good enough at understanding upcoming curves.

**Decision and rationale:** We decided not to keep manually tuning the MPC controller because
the main problem seemed to be the quality of its future predictions, not just the scoring rules.

**Next steps:** Try a learned dynamics model that may make better predictions about what the car
will do next.

---

### Learned Dynamics / Model-Based Learning

**Date and time:** Tuesday, Sept 1

**Participants and contributions:** Sophia Moloo, testing the learned model, comparing it with
`smol_brain`, and improving performance.

**Question or objective:** Can a model trained from simulator data outperform the `smol_brain`
baseline?

**What we investigated or changed:**

- The first training data did not include enough high-speed driving, so the car stayed around
  6 m/s.
- We collected more data at higher speeds, around 15–18 m/s.
- Given the exact same situation twice, the planner sometimes picked opposite steering
  directions, because it added random noise to every future step separately and the noise drowned
  out the real signal. We fixed this by using one consistent random nudge for the whole plan
  instead, so the same situation reliably produces the same decision.
- We improved recovery behavior so the car could build speed again after hitting the wall.
- We tested pre-curve braking and extra high-speed training data, but removed those changes
  because they made overall performance worse.

**Evidence:**
- Sources or documentation: MPC research and documentation from the first experiment, plus
  collected state/action/next-state driving data.
- AI-agent assistance: Claude Code supported modeling and analyzing results.
- Commits or code: `collect_dynamics_dataset.py`, `train_dynamics_model.py`,
  `learned_dynamics_mpc.py` (all 148 tests verified).
- Experiment output:

  | Stage | 5-seed distance | `smol_brain`'s distance |
  |---|---:|---:|
  | First comparison | 340 m | ~950 m |
  | Better data + speed cap | 690 m | ~907 m |
  | Pre-curve braking (reverted) | 38.6 m | — |
  | Fixed recovery behavior | 719-780 m | — |
  | Extra high-speed training data (reverted) | 430 m | — |
  | Final | ~888 m | ~951 m (93%) |

  Race victory in a live head-to-head: 434.3 m vs 393.6 m.

**What we observed:** The model performed much better when the training data included
situations similar to what it experienced during the race. We also learned that a change can
help in one situation but still make overall performance worse, so every change needs to be
tested across the full race, not just the case it was meant to fix.

**Decision and rationale:** We continued with the learned-dynamics approach because it produced
much better performance and eventually beat the benchmark.

**Next steps:** Fix the remaining crash on the outlier seed and make sure any change improves
performance across all seeds, not just one.
