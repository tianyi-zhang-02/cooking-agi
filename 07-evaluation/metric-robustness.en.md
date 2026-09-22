# Evaluation: is the metric robust?

[中文](metric-robustness.md) · **English**

A score going up does not mean the system got better. Two things mislead most easily: the average, and the evaluator itself.

## Why you must look at slices

One average recall, average win rate, or average judge score may contain, all at once:

- a large gain for mainstream users;
- a clear regression for new users;
- gains on simple tasks;
- failures on long tasks and multi-turn tasks;
- relevance going up while diversity goes down.

The average tells us what happened overall; only slices tell us **to whom, and under what conditions, it happened**.

## Evaluation needs to be evaluated too

An evaluator also drifts after it goes live. The model changes, the tasks change, the users change, and the old rubric may no longer cover the real failures.

So keep asking:

- has the evaluator's agreement with human judgment changed?
- does it favor longer, more confident, or particular styles of response?
- can it detect new failure modes?
- do metric gains correlate with real outcomes?
