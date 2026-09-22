# Evaluation: Why Should We Believe the System Improved?

[中文](README.md) · **English**

## Evaluation begins by defining what “good” means

Evaluation is not producing a score for the model. It is collecting enough evidence to judge whether a specific change really improved the target behavior without causing a more serious problem somewhere else.

## Why one average score is not enough

A travel agent produces a fluent response with a beautiful itinerary, but books the wrong date.

Scored only by an LLM judge that asks "is the writing clear," it may get a high score; check the tool calls and the final order, and it is an unambiguous failure.

This shows that different questions need different evidence:

- whether the format is correct can be checked with rules;
- whether the dates agree can be checked deterministically;
- whether the hotel can really be booked can be verified by calling a tool;
- whether the itinerary fits the user's preferences may need semantic judgment;
- whether the user is satisfied in the long run requires looking at later behavior.

No single evaluator can answer all of these alone.

## First write down what “good” means

Before evaluating, define the task contract:

```text
Goal        What does the user actually want to accomplish?
Success     What result counts as done?
Invariant   Which conditions must never be broken?
Tradeoff    How are speed, cost, quality, and risk balanced?
Slice       Which users, tasks, and environments must be observed separately?
```

If these questions have no answers, the final score means nothing, however precise it is.

## How this block reads

- [The evaluation stack](evaluation-stack.en.md): from deterministic checks to online outcomes, which question each layer is good at.
- [Is the metric robust?](metric-robustness.en.md): what an average hides, and how an evaluator drifts on its own.
- [LLM-as-a-Judge](llm-as-a-judge/README.en.md): how to handle the criterion, the scoring mode, and the biases when a model does the scoring.

## How it connects to other topics

- [Agent Observability](../06-systems/agent-observability.en.md) provides process evidence, instead of looking only at the final answer.
- [Human-in-the-Loop](../06-systems/human-in-the-loop.en.md) handles high-risk judgments and evaluator calibration.
- [Data and feedback](../01-data-and-feedback/README.en.md) decides whether the eval set represents real tasks.
- [Model Experience](../08-model-experience/README.en.md) connects offline metrics to how the experience feels over the long run.

## Engineering reading

- [Confident AI Blog](https://www.confident-ai.com/blog): continuously updated writing on LLM evaluation, agent evaluation, observability, Human-in-the-Loop, RAG metrics, and experimentation workflows. Good for learning engineering implementation and common design patterns; specific methods still need to be validated against the primary papers and your own data.
