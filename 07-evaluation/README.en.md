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

## An evaluation stack, from cheap to expensive

### 1. Deterministic checks

Good for schemas, formats, state changes, tool arguments, permissions, and business rules. Whatever can be decided in code should not be handed to a probabilistic model first.

### 2. Reference or executor

Code can run tests, a math problem can be checked against the answer, retrieval results can be checked against the evidence, and an agent's task completion can be verified. Signals of this kind are usually closer to ground truth than language-based scoring.

### 3. LLM-as-a-Judge

Good for relevance, helpfulness, coherence, style, and open-ended quality. It scales to large numbers of samples, but it is affected by the prompt, the model version, position, and writing style.

### 4. Human review

Good for defining rubrics, checking edge cases, and discovering new failures the automatic evaluators never anticipated. Humans are inconsistent too, so they need calibration and recorded rationales.

### 5. Online and long-term outcomes

In the end you still have to look at real task completion, user corrections, repeat use, and long-term behavior. An improvement in offline metrics does not necessarily turn into a better experience.

## How to use an LLM judge

Think of an LLM judge as a **semantic sensor**, not as the final truth on the judge's bench.

More robust practice includes:

- judge only one clear question at a time;
- provide a reference whenever you can;
- swap the A/B order when comparing two options;
- allow `tie` and `both bad`; don't force a winner;
- use human-labeled samples to check on which slices the judge fails;
- store the prompt, model version, temperature, and input evidence;
- don't ask the judge to make up a 1–10 score that looks precise but has no basis.

Writing the rubric as a DAG helps split the judgment apart, but the DAG itself does not automatically make the metric correct. What matters most is still whether each node corresponds to a real, verifiable standard.

Few-shot, reference-based, pairwise, and probability weighting are often discussed as if they were one thing, but they are different dimensions. For the fuller breakdown, see:

→ [LLM-as-a-Judge: how do criterion, reference, and scoring actually combine?](llm-as-a-judge.en.md)

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

## How it connects to other topics

- [Agent Observability](../06-systems/agent-observability.en.md) provides process evidence, instead of looking only at the final answer.
- [Human-in-the-Loop](../06-systems/human-in-the-loop.en.md) handles high-risk judgments and evaluator calibration.
- [Data and feedback](../01-data-and-feedback/README.en.md) decides whether the eval set represents real tasks.
- [Model Experience](../08-model-experience/README.en.md) connects offline metrics to how the experience feels over the long run.

## Engineering reading

- [Confident AI Blog](https://www.confident-ai.com/blog): continuously updated writing on LLM evaluation, agent evaluation, observability, Human-in-the-Loop, RAG metrics, and experimentation workflows. Good for learning engineering implementation and common design patterns; specific methods still need to be validated against the primary papers and your own data.
