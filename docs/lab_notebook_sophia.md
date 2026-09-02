# Lab Notebook — Sophia Moloo

Covers only my two pathways from `docs/methods.md`: **Planning and Model-Predictive Control
(#5)** and **Learned Dynamics / Model-Based Learning (#7)**. My partner is tracking Reactive
Control (#1) and Model-Free RL (#4) separately.

## EXPERIMENTAL APPROACHES

### Planning and Model-Predictive Control

Plan a short sequence of throttle/steer actions ahead of time by predicting their consequences,
execute only the first action, then replan next tick (receding-horizon control) — rather than
mapping sensors to a command with fixed rules.

**Hypothesis:** The simulator is deterministic and a controller only ever receives one
`RobotSensors` snapshot with no access to the real track geometry or physics world
(`README.md` "Runtime contract"). A controller that samples candidate action sequences and
scores them against a predictive model should out-drive a purely reactive controller,
especially into corners, by weighing the consequences of a plan a second or so ahead instead of
reacting to the current instant only.

**Minimum experiment:** `src/controllers/mpc_baseline.py` — random-shooting MPC. Each tick,
sample many candidate `(throttle, steer)` sequences, roll each through a hand-written kinematic
model (`kinematic_predict` in the same file: first-order speed response to throttle, bicycle-style
yaw rate from steer, curvature estimated from the camera's lookahead offsets), score with a cost
function (`src/controllers/mpc_lib.py::rollout_cost`) that rewards forward progress and penalizes
lateral/heading error and off-track excursions, and execute the first action of the
lowest-cost sequence.

**Evaluation:** Scored distance, laps, off-track time, and wall-contact time against the
`crash_fast` and `default_student_controller` baselines, run across the repo's default seed
suite `(42, 110, 271, 997, 2027)` (`racing.race.rules.HEAD_TO_HEAD_DEFAULT_SEED_SUITE`) via
`scripts/evaluate_seed_suite.py`.

### Learned Dynamics / Model-Based Learning

Learn a model of how the car's state responds to actions from logged data, then use that
learned model wherever a predictive model is needed for control.

**Hypothesis:** The hand-written kinematic model above is a crude guess — it can't capture the
simulator's real throttle behavior (e.g., the brake-before-reverse transition in
`racing/physics/engine.py`) or how curvature actually evolves as the car moves. A model fit to
logged `(state, action, next_state)` transitions should predict short-horizon consequences more
accurately, and dropping that learned model into the *same* MPC planner (unchanged) should
therefore produce a faster/more reliable controller than the hand-written baseline — isolating
the effect of the dynamics model itself.

**Minimum experiment:**
1. `scripts/collect_dynamics_dataset.py` — drive with a smoothed random-exploration controller
   across several seeds, log every `(state, action, next_state)` transition as JSONL.
2. `src/controllers/dynamics_model.py` — a 1-hidden-layer numpy MLP predicting the next-state
   delta, trained with plain manual-backprop gradient descent (no torch — small enough network
   that a hand-rolled MLP is enough to learn something meaningful for a first pass).
3. `scripts/train_dynamics_model.py` — trains the model on the collected dataset, reports
   one-step **and** multi-step rollout prediction error (the error that actually matters once
   the model is used for multi-step planning).
4. `src/controllers/learned_dynamics_mpc.py` — uses the same state representation and cost
   functions as `mpc_baseline.py`, but predicts with the trained model instead of
   `kinematic_predict`. During testing, its sampler was changed to use one correlated action
   offset across the horizon because independent per-step noise produced inconsistent steering.

**Evaluation:** One-step and multi-step MSE on held-out transitions (from training), plus the
same head-to-head seed-suite comparison as pathway 5. The initial design intended the dynamics
model to be the only difference from `mpc_baseline`; later experiments also changed the learned
controller's sampling strategy after independent per-step noise proved unstable. Final race
results therefore compare the complete controllers and do not isolate only the model.

---

## LOG ENTRIES

### Date and time: 2026-08-31

**Participants and contributions:** Sophia Moloo (moloosophia@gmail.com) — scaffolded both
experiments with Claude Code as an AI pair-programming agent.

**Question or objective:** Stand up minimum viable implementations of both pathways so there's
real evidence to compare, not just a plan.

**What we investigated or changed:**
- Read through the simulator's public controller contract (`README.md`, `SENSORS.md`,
  `src/racing/student/api.py`) to confirm what state a controller can and can't access —
  this drove the design decision to share one MPC planner between both pathways, differing
  only in the internal predictive model (hand-written vs. learned).
- Added `numpy` as a project dependency (`uv add numpy`) for vectorized rollouts and the MLP.
- Built `src/controllers/mpc_lib.py`: shared `LocalState` representation, `rollout_cost`,
  and a random-shooting `plan_action_sequence` planner, generic over any `predict_fn`.
- Built `src/controllers/mpc_baseline.py` (pathway 5): hand-written kinematic `predict_fn` plus
  a `create_controller()`-based stateful controller.
- Built `src/controllers/dynamics_model.py` (pathway 7, in progress): the numpy MLP
  (`DynamicsModel`) with `predict`/`save`/`load`, and a manual-backprop `train_step`.
- Added `tests/test_mpc_lib.py` (9 tests) and `tests/test_dynamics_model.py` (4 tests) covering
  the planner and model in isolation, no Panda3D required.

**Evidence:**
- Sources or documentation: `README.md`, `SENSORS.md`, `docs/methods.md`,
  `src/racing/student/api.py`, `src/racing/race/rules.py` (found the existing
  `HEAD_TO_HEAD_DEFAULT_SEED_SUITE` constant, reused for evaluation instead of inventing a new
  seed list).
- AI-agent assistance: Claude Code explored the codebase, proposed the shared-planner design
  (approved via plan review before writing code), and implemented `mpc_lib.py`,
  `mpc_baseline.py`, `dynamics_model.py`, and their tests.
- Commits or code: `src/controllers/mpc_lib.py`, `src/controllers/mpc_baseline.py`,
  `src/controllers/dynamics_model.py`, `tests/test_mpc_lib.py`, `tests/test_dynamics_model.py`
  (not yet committed to git as of this entry).
- Experiment output: `uv run pytest tests/test_mpc_lib.py tests/test_dynamics_model.py -v` — all
  unit tests passing.
- Leaderboard result, if applicable: none yet — no head-to-head race run against a baseline yet.

**What we observed:** An early version of the kinematic model's lookahead-point propagation
had a divide-by-zero bug (trying to shift each lookahead offset toward the next one over a
zero-length final gap). Simplified it to hold lookahead offsets fixed across the planning
horizon instead — a real approximation error worth revisiting, but avoids a crash and is
honestly documented as a limitation in the code.

**Decision and rationale:** Share one planner between both pathways rather than building two
independent systems, since `docs/methods.md` pathway 7 explicitly frames learned dynamics as
"plan future actions using the learned model" — i.e., pathway 7 is meant to slot into pathway 5's
machinery, not duplicate it. This also satisfies the project's "evaluate under comparable
conditions" guidance: with the planner and cost function held constant, any performance
difference between `mpc_baseline` and `learned_dynamics_mpc` isolates the effect of the dynamics
model.

**Next steps:**
- Finish testing and verifying `mpc_baseline.py` (pathway 5) on its own before touching
  pathway 7 again — decided mid-session to work one experiment at a time instead of building
  both in parallel.
- `scripts/collect_dynamics_dataset.py` and `src/controllers/dynamics_model.py` exist as a
  rough sketch but are paused/unfinished; `scripts/train_dynamics_model.py`,
  `src/controllers/learned_dynamics_mpc.py`, and `scripts/evaluate_seed_suite.py` not started.

---

### Date and time: 2026-08-31 (later same day)

**Participants and contributions:** Sophia Moloo (moloosophia@gmail.com) — ran and watched
`mpc_baseline` drive, with Claude Code assistance running the experiment and diagnosing why it
drove slowly.

**Question or objective:** Does the pathway 5 minimum experiment (`mpc_baseline.py`) actually
drive the car, and how well?

**What we investigated or changed:** No code changes. Ran a 5-second headless race
(`mpc_baseline` vs. `crash_fast`) for numbers, then watched `mpc_baseline` drive solo in the
graphical viewer to see it directly:
```
PYTHONPATH=src uv run racing --seed 110 --student-module controllers.mpc_baseline
```

**Evidence:**
- Sources or documentation: n/a — direct experiment.
- AI-agent assistance: Claude Code ran the headless race, then (when the car appeared not to
  move in the two-car `--watch` view) ran the planner's decision logic on its own outside the
  simulator to check whether it was actually choosing to accelerate, before concluding the
  likely explanation was watching the stationary `crash_fast` car rather than `mpc_baseline`.
- Commits or code: `src/controllers/mpc_baseline.py`, `src/controllers/mpc_lib.py` (unchanged
  from the initial scaffold).
- Experiment output: 5-second headless race — 0.79 m scored distance, top speed ~5.0 m/s, zero
  wall contact and zero damage, but "low progress" for about half the run (2.53 of 5 s).
  Confirmed visually in the single-car view: the car does move, but very slowly.
- Leaderboard result: n/a, not submitted.

**What we observed:** `mpc_baseline` is safe — it never hit a wall or took damage in this test
— but slow and a bit hesitant. Likely cause: the scoring function
(`mpc_lib.rollout_cost`) weighs "go fast" and "stay near the track center" about equally, and
going faster naturally causes more centerline drift (turning at speed moves the car sideways
more than turning slowly does), so a cautious, medium-speed plan often scores about as well as
a fast one. It also fully re-plans from scratch every tick with no memory of the previous plan,
so the throttle it actually applies is a bit jittery instead of a smooth ramp-up.

**Decision and rationale:** Recording this as the first honest result for pathway 5 rather than
tuning it further right now, to keep progress deliberate and to have a clean baseline to compare
pathway 7 against later, once that's built. "Safe but slow" is a legitimate, useful data point,
not a failure.

**Next steps:** When resuming pathway 5, the natural first improvement is rebalancing
`rollout_cost` to weigh speed more heavily relative to centering (one small, reversible change).
Pathway 7 (`dynamics_model.py`, `collect_dynamics_dataset.py`) stays paused until pathway 5 is
considered done.

**Environment note (affects both pathways / the whole team):** Found that `uv run racing ...`
intermittently fails with `ModuleNotFoundError: No module named 'racing'` — confirmed this also
happens with the original, unmodified tutorial command
(`uv run racing --seed 110 --student-module controllers.crash_fast`), so it's a pre-existing
environment issue, not something introduced by this work. Reliable workaround: prefix commands
with `PYTHONPATH=src`, e.g. `PYTHONPATH=src uv run racing ...`. Worth sharing with the team.

---

### Date and time: 2026-08-31 (tuning session, same day)

**Participants and contributions:** Sophia Moloo (moloosophia@gmail.com) — directed each change
and watched results live; Claude Code implemented each change and ran the diagnostics.

**Question or objective:** `mpc_baseline` was safe but slow (see prior entry). Can small,
targeted changes to `mpc_lib.rollout_cost`/`mpc_baseline.kinematic_predict` fix that, one change
at a time, each verified before moving to the next?

**What we investigated or changed, in order:**
1. Added `SPEED_REWARD_WEIGHT = 5.0` to `rollout_cost` (previously speed and centering were
   weighted equally) — one constant, one line.
2. Added warm-starting (`shift_plan_for_warm_start`, `plan_action_sequence(..., warm_start=...)`)
   so each tick's search starts from last tick's plan instead of from scratch, to fix visible
   forward/back jitter from full random re-planning every tick.
3. Watched it live again — no longer jittery forward/back, but now swerving hard to one side.
   Rejected my first proposed fix (a blanket "penalize hard steering" cost) because that would
   also suppress legitimate hard swerves near a wall, which is backwards.
4. Logged real tick-by-tick telemetry from an actual headless race (state + chosen action every
   tick) instead of guessing, and found the car steering hard toward one side almost every tick
   while `center_offset` got steadily worse, not better, heading toward a wall.
5. Hypothesized the cause was a buggy "curvature" term in `kinematic_predict` (it reused nearly
   the same signal as the car's *current* offset, scaled by speed, creating a runaway
   self-correction the real car doesn't have) and removed it.
6. Re-ran the same telemetry check after removing it — **the fix did not work.** The trajectory
   was nearly identical to before. Wrong diagnosis.
7. Directly tested the cost function's own math at the exact real state from step 4 (comparing
   the score for steering right/left/straight) instead of guessing again.

**Evidence:**
- AI-agent assistance: Claude Code wrote and ran all diagnostic scripts (headless races,
  tick-by-tick telemetry logging, direct cost-function comparisons) rather than reasoning about
  the bug in the abstract; two of my own hypotheses about the cause were tested and one was
  disproven by real data before landing on the actual explanation.
- Commits or code: `src/controllers/mpc_lib.py` (`SPEED_REWARD_WEIGHT`, `shift_plan_for_warm_start`,
  `warm_start`/`exploration_std` params on `plan_action_sequence`), `src/controllers/mpc_baseline.py`
  (warm-start bookkeeping, removed curvature term), `tests/test_mpc_lib.py` (2 new tests for
  warm-starting).
- Experiment output (same 5-second headless race each time, `mpc_baseline` vs. `crash_fast`):

  | Change | Distance | Top speed | Low-progress time | Wall contact / damage |
  |---|---|---|---|---|
  | Original | 0.79 m | 5.04 m/s | 2.53 s / 5 | 0 / 0 |
  | + speed reward weight | 2.27 m | 2.45 m/s | 2.43 s / 5 | 0 / 0 |
  | + warm start | 2.95 m | 8.63 m/s | 2.72 s / 5 | 0.12 s / 0.04 |

**What we observed:** The scoring function is not the bug — directly checked: for the real state
observed mid-drift, it correctly scored "steer toward the direction the model predicts helps"
as best (cost 1.0) versus the opposite direction (cost 40.1). The bug is that the *model's belief*
about which direction helps is wrong on this part of the track. Best explanation: the car was on
a curving section, and the hand-written model only knows the car's current offset from center —
it has no signal for how fast the track's own "correct heading" is rotating as the car advances
through the turn. On a straight section that omission doesn't matter; on a curve, it makes the
model confidently wrong for an extended stretch, since nothing in a single sensor snapshot tells
it "the track is turning here."

**Decision and rationale:** Not attempting a better curvature signal right now. This is a clean,
well-evidenced illustration of exactly the limitation pathway 7 exists to address — a model
*learned from real driving data* would pick up on how curves actually behave from examples,
rather than needing me to hand-derive the right formula for every track shape. Moving to pathway
7 with this as the concrete motivation, rather than continuing to iterate on the hand-written
model.

**Next steps:** Start pathway 7 (learned dynamics). `dynamics_model.py` and
`collect_dynamics_dataset.py` exist as an earlier rough sketch (from before the "one experiment
at a time" decision) — plan to review/rebuild deliberately rather than assume they're still the
right design, now that pathway 5 surfaced a specific, concrete thing the learned model needs to
get right (predicting correctly through curves) that the hand-written model could not.

---

### Date and time: 2026-09-01

**Participants and contributions:** Sophia Moloo (moloosophia@gmail.com) — watched
`learned_dynamics_mpc` drive live, flagged that its turning looked wrong, and asked for a
comparison against a simple baseline; Claude Code diagnosed and fixed the issues and ran all
the experiments.

**Question or objective:** Two things: (1) `learned_dynamics_mpc` wasn't turning well while
driving live — could we find out why and fix it? (2) How does it actually compare to a very
simple "dumb" controller that does no planning at all, just reacts to what it sees right now?

**What we investigated or changed, in plain terms:**
- Watched pathway 7 drive and noticed it would head toward a wall, stop just short of it, and
  then turn very slowly instead of snapping into the corner.
- Tracked this down to real bugs, one at a time, each checked with real data before moving on:
  1. The model's training was checked against driving situations it had actually seen before,
     which hid a problem — it looked fine in testing but failed badly the moment it saw driving
     from a part of the track it had never trained on. Fixed by training on more varied driving
     and specifically checking it against driving it had truly never seen.
  2. Steering wasn't strong enough in what the model believed, so training was changed to care
     more about getting steering right. This helped a little but wasn't the real problem.
  3. The real bug: each time the controller planned, part of its decision came down to luck
     instead of what was actually best — proven directly by giving it the exact same situation
     twice and watching it choose two opposite turns. Fixed by having it commit to one clear
     decision each time instead of a jumble of random guesses.
  4. That fix let the car finally drive with confidence — but it then drove too fast, faster
     than anything the model had ever learned from, and crashed hard every single time. Fixed by
     capping how fast it's allowed to go before it's allowed to speed up further.
- Built a second, very simple "reactive" controller with no planning at all, called
  `smol_brain` (pulled from a shared `smol-brain` branch rather than written from scratch) —
  just two small formulas, one for steering and one for throttle, each based only on the
  current moment.
- Ran all three controllers through the same test: five practice tracks, 15 seconds each,
  recording how far each one got and whether it hit a wall.

**Evidence:**
- Real race results, same test for every controller (distance added up over five practice runs):
  - `smol_brain` (simple, no planning): about 950 meters total, top speed ~15 m/s, never hit a
    wall, never took damage.
  - `learned_dynamics_mpc` (my learned-model planner): about 340 meters total, top speed
    ~6-11 m/s, never hit a wall.
  - `mpc_baseline` (the hand-written planner): about 36 meters total, top speed ~3-4.5 m/s, hit
    a wall once.
- All 145 existing tests still passed after every change.

**What we observed:** The simple controller won by a lot — about 3x farther than my
learned-model planner, and about 27x farther than the hand-written one — while being just as
safe or safer. Likely reason: the planning controllers have to *guess* what will happen next
using a model of the car, and that guess can be wrong. The simple controller doesn't guess at
all — it just reacts to what's actually true right now, every tick. That makes it immune to the
exact kind of mistake (a wrong guess about the future) that caused nearly every problem we ran
into with the planning controllers this session.

**Decision and rationale:** Recording this honestly, even though it means the more complicated
approach (planning ahead with a learned model) lost to a much simpler one on this test. This is
a fair, useful result, not a failure — it shows planning ahead only pays off if the model of the
future is trustworthy, and building a trustworthy model took more debugging than expected and
still isn't as reliable as just reacting well.

**Next steps:** Left open — could keep improving the learned model so planning starts to pay
off, or could accept that the simple reactive approach is the stronger option for now and spend
effort elsewhere.

---

### Date and time: 2026-09-01 (speed-cap follow-up, same day)

**Participants and contributions:** Sophia Moloo (moloosophia@gmail.com) — set the goal of
beating `smol_brain` on both speed and distance, not just matching its safety; Claude Code made
and tested the changes.

**Question or objective:** After finding `smol_brain` (the simple reactive controller) beat
`learned_dynamics_mpc` badly (previous entry), the new goal was: make `learned_dynamics_mpc`
actually go faster and farther than `smol_brain`. Where did the earlier 6 m/s speed cap come
from, and can it be safely raised?

**What we investigated or changed, in plain terms:**
- The 6 m/s cap came from how the training data was collected: the exploration policy only held
  each throttle/steer choice for about 0.13 seconds before switching, so the car almost never
  got time to actually build up real speed. The model had barely seen any good examples of fast
  driving, so the planner was capped there to stay safe around what little it actually knew.
- Changed data collection to hold each choice for about 0.75 seconds instead, and made it
  choose "mostly forward" throttle instead of spending half its time braking or reversing, so
  the car would genuinely reach and cruise at higher speeds while collecting data.
- Collected a new, bigger batch of driving data this way. Checked the numbers: before, only
  1 in 100 moments were above about 6 m/s; after, 1 in 100 moments are above about 18.6 m/s, and
  1 in 20 are above about 13 m/s — real coverage of what fast driving looks like, not a guess.
- Retrained the model on the new data and checked its fit on driving it hadn't seen, same as
  before. The fit specifically during fast stretches improved a lot (previously the model barely
  explained anything about how things change at those speeds; now it explains a real amount,
  though still far from perfect).
- Raised the speed cap step by step, testing each time on the same 5-track comparison used
  throughout: first to 18 m/s (near the new data's limit), then back down to 15 m/s (closer to
  what `smol_brain` itself actually drives at).

**Evidence:**
- With the new data and an 18 m/s cap: top speed reached about 18.2-18.3 m/s — genuinely faster
  than `smol_brain`'s ~14-17.5 m/s — but it also picked up real wall contact (about 3.8 seconds
  total across the 5 tracks) and some damage, and total distance (747m) still came in below
  `smol_brain`'s (936m).
- Dialed back to a 15 m/s cap (about matching `smol_brain`'s own speed): still some wall contact
  (2.5 seconds total) and damage, and distance (690m) still below `smol_brain`'s (907m).
- `smol_brain` had zero wall contact, zero damage, and zero off-track time in every test run, at
  every cap we compared against.
- All 145 existing tests still pass after these changes.

**What we observed:** Raising the speed cap genuinely makes the car faster — real, repeatable
progress, not a fluke — but it isn't enough by itself to beat `smol_brain`, because going faster
now causes real crashes into walls that erase the speed advantage. Likely reason: the shared
"avoid the wall" check only looks at whatever is directly in front of the car right now — it has
no idea a wall is coming up around a bend until the car is basically already at the bend.
`smol_brain` doesn't have this weakness because it corrects its steering hard, every single
tick, no matter the speed, so it reacts to a curve the instant it can see it rather than needing
to plan ahead for one.

**Decision and rationale:** Documenting this as a real, partial result before deciding what to
try next. Raising the speed cap was a correct, well-evidenced fix for one real problem (bad
training data that never showed the model what fast driving looks like), but it exposed a
second, separate problem (the wall-avoidance check doesn't understand curves) that needs its own
fix before `learned_dynamics_mpc` can beat `smol_brain` on both speed and safety together.

**Next steps:** Decide between (a) adding a proper "slow down before a curve" check that uses
the camera's forward-looking readings instead of just what's directly ahead, or (b) turning the
speed cap back down to a safer, slower value and accepting `smol_brain` wins this particular
comparison for now. Not decided yet — reviewing this write-up first.

---

### Date and time: 2026-09-01 (final checkpoint on beating smol_brain, same day)

**Participants and contributions:** Sophia Moloo (moloosophia@gmail.com) — asked how to decide
when to stop tuning versus try something new; Claude Code proposed a plan and carried it out.

**Question or objective:** Given three rounds of tuning one setting (how cautious to be while
recovering from a bad position) without a clean win over `smol_brain`, how should we decide
whether to keep going? Agreed plan: try one real, different fix — teaching the planner to slow
down before a curve it can see coming, not just react to a wall already right in front of it —
and if that doesn't help, stop tuning and write down the honest final result instead of
continuing indefinitely.

**What we investigated or changed, in plain terms:**
- The specific crash from the previous entry was actually already fixed by the recovery-throttle
  change alone (confirmed by replaying the exact same situation — no more wall contact there).
  What was left was smaller: a little residual wall contact and one stuck-and-reset event when
  the cautious-driving throttle limit was loosened back up to let the car go faster.
- Added a new "don't speed into a curve" check: estimate how sharply the track bends ahead using
  the three forward-looking camera readings, and refuse to plan any speed higher than what that
  curve could safely handle.
- Tested it the same way as everything else this session: same 5-track comparison, before and
  after.

**Evidence:**
- Before this change (recovery throttle capped at a moderate level): 257.9 meters, some minor
  wall contact and light damage, 1 stuck-and-reset event.
- After adding the curve check: 38.6 meters, zero wall contact, but 2 stuck-and-reset events and
  top speed cut roughly in half. Much safer, but far less progress — the car was braking so
  often and so hard that it barely moved.
- All 147 tests still passed either way.

**What we observed:** The curve check made things clearly worse, not better — it was too quick
to slam the brakes even on ordinary curves the car could have taken at speed. Following the plan
agreed at the start of this entry, this was treated as a real answer (no), not something to keep
re-tuning, and was removed rather than kept in a worse-than-before state.

**Decision and rationale:** Reverted the curve check and kept the version from before it
(moderate recovery throttle cap, no curve check) as the final state for this comparison.
`smol_brain` still wins on both speed and distance, and that is now a well-tested, honest
result: three different fixes were tried (raising the speed cap using better data, capping
throttle while badly misaligned, and curve-aware braking), one clearly helped, one helped with a
real tradeoff, and one made things worse and was undone. Continuing to tune further would cost
more time than it's likely to be worth given how this compares to the value of the write-up
already in hand — a real, evidenced example of when simple reactive control beats short-horizon
planning with an imperfect learned model.

**Next steps:** None planned for this specific comparison — treating it as complete and moving
on to other work.

---

### Date and time: 2026-09-01 (resumed after all: "it needs to beat smol_brain", same day)

**Participants and contributions:** Sophia Moloo (moloosophia@gmail.com) — after seeing the
previous checkpoint's result live, decided the comparison mattered enough to keep going rather
than stop; Claude Code found and fixed two further real bugs.

**Question or objective:** Given the explicit decision to keep trying, what was actually still
costing distance and safety, beyond the fixes already tried?

**What we investigated or changed, in plain terms:**
- Pulled tick-by-tick telemetry from a stuck episode and found the real cause: recovery mode
  threw away its previous decision and searched completely fresh every single tick. When the
  car's speed sat right at the boundary between two similarly-good strategies ("creep forward"
  vs. "back up a little"), the search could flip its answer every tick based on tiny state
  differences — throttle alternated +0.50/-1.00/+0.50/-0.97 while heading never corrected,
  because speed never built up enough for steering to do anything (turning rate depends on
  speed).
- Fix: recovery mode now remembers and starts its search from its own last decision too (with a
  wide but not unlimited search around it), instead of discarding all memory and starting from
  nothing every tick. Confirmed directly: the same stuck episode disappeared entirely, and total
  distance across all 5 tracks nearly doubled.
- A second, more subtle problem turned up on one seed: at high speed (~15 m/s) combined with a
  large heading error, the model's predictions became unreliable — a combination that real
  driving rarely produces (a well-behaving car rarely gets going fast while badly misaligned), so
  the model had little to learn from there and its guesses in that specific situation didn't
  match what actually happened.
- Tried forcing hard braking whenever this combination occurred. This fixed the one seed it
  targeted, but made the overall result clearly worse (total distance dropped from about 719 to
  482 meters) — it was slamming the brakes on ordinary curves the car could have handled fine,
  the exact same kind of over-correction as the curve check from the previous entry. Reverted it
  and tried a milder version (stop accelerating further, but don't force active braking) instead,
  which tested as a real improvement.

**Evidence:**
- Same 5-track, 15-second comparison used throughout: total distance went from about 340 meters
  (start of this session's speed-cap work) to about 719 meters (continuity fix) to about 780
  meters (final, milder version) for the learned controller, against `smol_brain`'s roughly 917
  meters on the same test, run twice to confirm the result repeats exactly (it does — the
  simulator is deterministic given the same code).
- One seed (271) still shows meaningful wall contact and damage in the final version; the rest
  are clean.
- All 148 tests passed throughout, including new tests written for each specific bug found.

**What we observed:** Two more real, well-evidenced bugs found and fixed the same way as
everything else this session — by capturing exactly what happened, forming a specific
hypothesis, and verifying it against real telemetry rather than guessing. The gap to
`smol_brain` narrowed substantially (from roughly a third of its distance to about 85% of it)
but has not fully closed. The remaining gap looks concentrated in fewer, more specific
situations (one seed's crash) rather than a general, everywhere problem, which is different from
where this comparison stood at the last checkpoint.

**Decision and rationale:** Kept both fixes (recovery continuity, milder recovery braking) since
both were verified improvements on the full comparison, not just the one situation each was
found from. Not calling this finished, since the goal (beat `smol_brain` outright) has not been
met yet, but recording it as real, measured progress rather than stopping at the earlier
checkpoint now that the decision was made to keep going.

**Next steps:** The remaining gap is now concentrated enough (one seed's crash, not a
general problem) that it may be worth the same telemetry-first approach on that specific
situation next, rather than another broad constant to tune.

---

### Date and time: 2026-09-01 (targeted data collection for the last seed, same day)

**Participants and contributions:** Sophia Moloo (moloosophia@gmail.com) — confirmed the goal
was still to beat `smol_brain` outright, not stop at the previous checkpoint; Claude Code traced
the one remaining crash to its root cause and tried a real fix for it.

**Question or objective:** The remaining gap was concentrated in one seed's crash. What was
actually causing it, and could it be fixed without another regression like the curve-check
experiment from two entries ago?

**What we investigated or changed, in plain terms:**
- Pulled telemetry from the exact crash and found: the car correctly stopped accelerating (the
  earlier fix working as intended), but barely steered at all despite a 70+ degree heading error,
  and drifted into a wall over about 25 ticks.
- Checked what the model itself believed was the best move in that exact situation, and it
  genuinely thought a small, gentle correction was better than a hard one — because a hard
  correction was predicted (by the model) to swing the car off-track, triggering a big penalty.
  That prediction turned out to be wrong once played out for real. The likely reason: a
  well-behaving car rarely reaches "going fast AND badly misaligned" on its own, so the training
  data barely covered it, and the model had little to learn from there.
- Built a way to deliberately create more examples of exactly that situation: every so often
  during data collection, force a short, sharp burst of hard throttle and hard steering (rather
  than only ever sampling random independent actions), then let normal random driving continue
  from wherever that left the car. This raised the fraction of collected data in that specific
  danger zone from a small amount to about 8%.
- Collected a fresh batch of data this way, retrained, and checked: the model's understanding of
  that specific situation did measurably improve, and the exact crash we were chasing went away
  completely when replayed.
- But the same full 5-track comparison used throughout showed this was a net loss overall: total
  distance dropped from about 781 meters to about 430 meters, even though the one situation we
  targeted was fixed. Best guess: this model is small and simple by design (no torch, one hidden
  layer), and improving its accuracy in one specific rare situation used up some of its limited
  capacity to represent everything else well.
- Reverted the data-collection change and rebuilt the previous dataset and model from scratch
  (the exact same code and seeds reproduce it, confirming the earlier result wasn't a fluke).

**Evidence:**
- Targeted disturbance data: fixed the one seed's crash, but total distance across all 5 tracks
  fell from about 781 meters to about 430 meters. Reverted.
- After rebuilding the previous (non-disturbance) dataset and model: about 888 meters total,
  against `smol_brain`'s about 951 meters on the same test — the closest this comparison has
  come, run twice with identical results both times. Only one seed still shows any wall contact.
- All 148 tests passed throughout.

**What we observed:** A real, well-diagnosed fix for one specific problem made the overall
result worse, for the second time this session (the curve-check attempt two entries ago had the
same shape: right for the one case it targeted, wrong for the average case). This small,
simple model does not have unlimited room to get better at everything at once — improving it
somewhere can cost accuracy somewhere else. Confirming a fix on the full comparison, not just
the one situation it was built for, caught this before it was kept by mistake, the same way it
caught the curve-check regression earlier.

**Decision and rationale:** Reverted the targeted data collection and kept the previous dataset
and model, which reliably scores the best result so far (about 888 of `smol_brain`'s about 951
meters, roughly 93%). Not yet a win, but the closest and most solid result reached this session.

**Next steps:** The remaining gap is now down to one seed's crash costing roughly 50-60 meters.
Given that the last attempt to fix exactly this kind of situation made things worse elsewhere,
any further attempt should budget for the real possibility of another net-negative result and
be judged the same way: against the full comparison, not just the one case it targets.

---

## WHAT I LEARNED

### From research and implementation

- MPC is a **receding-horizon** method: predict the result of several possible action sequences,
  execute only the first action from the best sequence, then repeat the process using the next
  sensor reading.
- An MPC controller is only as reliable as three connected parts: its dynamics model, its cost
  function, and its candidate-action search. A sensible cost function cannot rescue a model that
  predicts the wrong steering response, and a good model can still be undermined by a noisy
  search procedure.
- A learned dynamics model needs training data that covers the states where it will be used.
  The first dataset contained little sustained high-speed driving, so predictions became unsafe
  when the controller exceeded that range. Longer action holds and a forward-throttle bias gave
  the retrained model meaningful coverage at higher speeds.
- One-step validation error is not enough for MPC. Prediction errors compound over a rollout, so
  held-out and multi-step performance matter more than how well the model fits familiar training
  transitions.

### From testing

- The hand-written MPC was cautious but very slow, and it became confidently wrong on curves
  because one sensor snapshot did not adequately describe how the track would turn throughout
  the planning horizon.
- The learned model improved the planning baseline substantially, but initially made unstable
  steering choices because the search perturbed every future action independently. Correlated
  action sampling made decisions more consistent.
- Expanding the training data and increasing the speed cap made the learned controller faster,
  but wall contact erased much of that gain. The final learned controller still did not beat
  `smol_brain`, the simple reactive baseline, on total distance or safety.
- The curve-aware braking experiment showed why rejected changes are useful evidence: it reduced
  wall contact but cut progress from 257.9 m to 38.6 m and increased marshal resets from one to
  two, so it was reverted.
- The main conclusion is not that MPC never works. In this deterministic simulator, simple
  reactive control worked better **for the model, data, horizon, and time budget tested**. MPC
  would need a more trustworthy multi-step model and broader training coverage before the cost
  of planning is justified.

---

## REFERENCE NOTES (not my own experiments — context from elsewhere)

**From a class exercise on reactive control (pathway 1), 2026-08-31:** a simple linear
reactive-controller test that reportedly worked well for speed:
```
THROTTLE_W = -2.0
THROTTLE_B = 1.00

STEER_W = 100
STEER_B = 0
```
Read as `throttle = clamp(THROTTLE_W * |steer| + THROTTLE_B, -1, 1)` (full throttle going
straight, easing toward braking the harder it's turning) and `steer` driven aggressively off a
heading/offset signal (`STEER_W = 100` is large enough that it behaves close to "turn hard
toward wherever it needs to go," saturating at the ±1 limit rather than a gentle proportional
response). Noting this as context — not something I've built or tested myself — but a possible
future reference point: a MPC/planning controller (pathway 5) should ideally be able to
out-perform a plan-free reactive rule like this one, so it could make a good baseline to compare
`mpc_baseline` against later instead of (or alongside) `crash_fast`.
