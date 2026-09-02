Your goal is to develop a controller that can drive successfully from any starting point on the track.

These pathways are starting points rather than an exhaustive menu. You may choose one, combine multiple pathways, or propose another well-motivated approach.

1. Reactive Control and Parameter Optimization
Construct explicit rules that connect sensor readings to throttle and steering.

Possible strategies:

Steer using weighted differences between left- and right-facing sensors.

Adjust throttle based on steering intensity or the visible space ahead.

Tune controller parameters using random search, Optuna, or another optimizer.

Questions to investigate:

What can the controller infer from its local sensors?

How should throttle and steering interact?

Do the rules generalize to unfamiliar tracks?

2. Imitation Learning
Train a model to reproduce the actions of a human or automated expert.

Possible strategies:

Train from demonstrations recorded from a human driver.

Train from a privileged controller that can access track geometry.

Use DAgger to collect expert corrections for states the model encounters.

Questions to investigate:

What makes a useful expert and demonstration dataset?

Does the dataset include varied tracks, mistakes, and recoveries?

How does the model behave in unfamiliar situations?

3. Evolutionary Computation and Neuroevolution
Evolve controller parameters or neural networks according to their racing performance.

Possible strategies:

Use CMA-ES to optimize the weights of a fixed, small neural network.

Use a genetic algorithm to evolve rule or controller parameters.

Use NEAT to evolve both network structure and weights.

Questions to investigate:

What information should an individual encode?

How should fitness balance speed, progress, and reliability?

How will evaluation discourage specialization to particular tracks?

4. Model-Free Reinforcement Learning
Train an agent through repeated interaction with the racing environment and a numerical reward.

Possible strategies:

Use PPO to learn from batches of recent driving experience.

Use SAC to learn continuous controls from a replay buffer.

Use TD3 as another replay-based continuous-control approach.

Questions to investigate:

What observations does the agent need?

What behavior does the reward function actually encourage?

Is the learned behavior reliable across tracks and training runs?

5. Planning and Model-Predictive Control
Search over possible future actions, execute the first action from the best sequence, and then plan again.

Possible strategies:

Evaluate randomly sampled action sequences.

Use the Cross-Entropy Method to improve candidate sequences.

Use beam search to retain several promising futures.

Questions to investigate:

How will the controller predict the consequences of actions?

How far ahead should it plan?

Can it plan quickly enough for interactive control?

6. Hybrid Approaches
Combine methods whose strengths address one another’s limitations.

Possible strategies:

Begin with imitation learning and fine-tune with reinforcement learning.

Add a learned correction to a reactive controller.

Use a planning controller as the expert for a faster learned policy.

Questions to investigate:

What responsibility does each component have?

Does the combination outperform its individual components?

What does an ablation study reveal about each component’s contribution?

7. Learned Dynamics or Model-Based Learning
Train a model to predict how the car and its observations change in response to actions.

Possible strategies:

Plan future actions using the learned model.

Generate imagined driving experience for policy training.

Learn a compressed world model that represents observations and dynamics.

Questions to investigate:

How accurate are the model’s multi-step predictions?

Can the controller exploit errors in the learned model?

Does the model generalize to unfamiliar track geometry?