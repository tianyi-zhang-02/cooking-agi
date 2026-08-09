# Evaluation: Why Should We Believe the System Improved?

[中文](README.md) · **English**

## In one sentence

Evaluation is not producing one score. It is collecting enough evidence to determine whether a change improves the intended behavior without creating a more important regression elsewhere.

## A simple example

A travel agent may produce a fluent itinerary while booking the wrong date. A style judge can score it highly, while a deterministic tool and state check correctly marks failure. No single evaluator covers every requirement.

## Start with the task contract

Define the user goal, success condition, invariants, tradeoffs, and slices before choosing metrics.

## An evaluation stack

1. **Deterministic checks** for schemas, state, tools, permissions, and rules.
2. **References or executors** for code, math, evidence, and task completion.
3. **LLM judges** for relevance, helpfulness, coherence, style, and open-ended quality.
4. **Human review** for rubric design, edge cases, and new failure discovery.
5. **Online and longitudinal outcomes** for real task completion and sustained experience.

Treat an LLM judge as a scalable semantic sensor, not final truth. Ask atomic questions, use references, swap pairwise order, allow ties, calibrate against human examples, and record evaluator versions.

For a fuller breakdown of demonstrations, references, judging modes, rubrics, and probabilistic aggregation, see [LLM-as-a-Judge](llm-as-a-judge.en.md).

Always inspect slices. Aggregate improvement can hide new-user regressions, long-task failures, or a relevance–diversity tradeoff.

Evaluators also drift and must be evaluated: track human agreement, style bias, new failure coverage, and correlation with real outcomes.
