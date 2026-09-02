This project has two overarching goals:

Practice evidence-based AI engineering. Your team should investigate alternatives, collaborate thoughtfully with an AI coding agent, make decisions based on evidence, and document how your thinking changes.
Build a fast, competitive car. Your final controller should perform well from any random starting-point seed on the given track and compete successfully on the course leaderboard.
A strong project must address both goals. A high leaderboard score does not replace a well-documented engineering process, and a thoughtful process does not remove the goal of producing a competitive car.

Process Overview
Stage	Product	Purpose
Throughout	Laboratory notebook	Preserve evidence of investigation, development, and decision-making
Exploration	Two experimental approaches and rough plans	Compare plausible alternatives before committing
Refinement	One selected approach and improvement plan	Focus effort on producing a competitive controller
Conclusion	Brief team presentation	Communicate the approach, results, and learning
1. Maintain a Laboratory Notebook
Your team must maintain a chronological laboratory notebook throughout the project. Update it during or immediately after substantive work sessions rather than reconstructing it at the end.

Each entry should record:

Who and when: Who participated, when the work occurred, and what each person contributed.

Question or goal: What the team intended to investigate, implement, debug, or evaluate.

Evidence: Sources consulted, AI-agent assistance, code changes, experiment configurations, logs, graphs, videos, commits, or leaderboard results.

Observations: What happened, including unexpected behavior and failed attempts.

Decisions: What the team decided and what evidence supported the decision.

Next steps: What the team plans to try next and why.

A useful entry template is:

Date and time:
Participants and contributions:

Question or objective:

What we investigated or changed:

Evidence:
- Sources or documentation:
- AI-agent assistance:
- Commits or code:
- Experiment output:
- Leaderboard result, if applicable:

What we observed:

Decision and rationale:

Next steps:
You do not need to preserve every conversation with an AI agent. Record the important assistance it provided, how you verified its suggestions, and any significant recommendation you rejected or changed.

Failed experiments are valuable evidence when the failure is documented and analyzed.

2. Explore Two Experimental Approaches
Before committing to one direction, investigate and conduct an initial experiment with two plausible control or training approaches.

The approaches should differ in more than a single parameter. They might come from different pathway families, or they might use meaningfully different learning or optimization mechanisms.

For each approach, produce a rough plan containing:

Hypothesis: Why might this approach produce a fast and reliable controller?

Minimum experiment: What is the smallest implementation that could provide useful evidence?

Evaluation: Which starting-point seeds and metrics will you use to judge it?

Implement enough of a minimum viable attempt at each approach to learn something meaningful. The two experiments do not need to be equally polished, but both should produce evidence that can inform your decision.

Evaluate them under comparable conditions whenever possible. Useful evidence may include:

Track-completion rate.

Progress before leaving the track.

Lap time.

Training or optimization time.

Consistency across starting-point seeds and repeated runs.

Development effort and remaining technical risk.

Early leaderboard performance.

If you have previously completed an ML course, we encourage you to investigate at least one approach with which you do not already have significant experience. The goal is to extend your understanding rather than simply repeat a familiar technique.

3. Select and Refine One Approach
After the initial experiments, select one primary control or training approach as a team for continued development.

Document the decision in the laboratory notebook, including:

The evidence supporting the selected approach.

The important strengths or limitations of both approaches.

Why the selected approach offers the best opportunity for improvement.

Any useful ideas from the other approach that you plan to retain.

Then establish a refinement plan. Identify the next experiments most likely to improve performance, such as:

Improving observations, training data, rewards, or fitness.

Adjusting the model, controller, or optimization process.

Increasing robustness across starting-point seeds.

Reducing unnecessary steering, braking, or hesitation.

Improving speed through curves or acceleration on straightaways.

Change a limited number of variables in each experiment so that you can determine what caused the result. Preserve previous checkpoints and compare each revision against both a baseline and the team’s best prior controller.

Use the leaderboard as performance evidence, but do not optimize for a single lucky run or favorable starting point. A strong controller should be fast and consistently successful across random starting-point seeds. Record meaningful leaderboard submissions and the changes that preceded them in your laboratory notebook.

4. Present Your Results
Conclude the project with a brief team presentation explaining:

Your approaches: The two alternatives you explored and the primary approach you ultimately selected.
Your development strategy: How you worked with one another and with an AI coding agent to plan, implement, test, and refine the controller.
Your evidence: Experimental results, representative runs, and final leaderboard performance.
Your learning: What the project taught you about machine learning, optimization, robustness, or intelligent control.
Your presentation should include at least one concrete result, such as a comparison table, learning curve, performance history, or representative race. Discuss failures or surprises that materially changed your approach.

All team members should be able to explain how the controller works, how it was evaluated, and why the team made its major decisions.

What Constitutes a Strong Project?
A strong project demonstrates:

Sustained, evidence-backed investigation.

Meaningful comparison before committing to an approach.

Deliberate iteration based on experimental results.

Responsible and verifiable use of an AI coding agent.

Understanding of the chosen control or training method.

Reliable performance across random starting-point seeds.

Competitive speed and leaderboard performance.

The objective is not merely to make the car move or to invoke an off-the-shelf training package. The objective is to use a thoughtful engineering process to produce the fastest and most capable controller your team can build.