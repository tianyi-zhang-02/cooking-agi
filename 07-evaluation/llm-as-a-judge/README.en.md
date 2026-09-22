# LLM-as-a-Judge: how exactly should a model do the scoring?

[中文](README.md) · **English**

## Quick learning: a judge is a measurement system, not one prompt

<details class="interview" markdown="1">
<summary>Criteria, scoring, calibration, and validation</summary>

**Quick memory**: define an observable criterion first, then choose pairwise, ordinal, or binary scoring; finally calibrate the judge with expert labels, perturbation tests, and distribution slices.

**Interview answer**

> LLM-as-a-Judge requires explicit input evidence, a single criterion, behavioral anchors, an output schema, and an aggregation rule. Pairwise scoring is usually steadier than absolute scores but still suffers from position, verbosity, self-preference, and reference-leakage biases. Before deployment it must be cross-validated against expert audits, and you report disagreement and confidence, not only a mean.

<details markdown="1">
<summary><b>Deep dive</b>: why can a weighted mean hide failure?</summary>

The same mean can come from "everyone gives a 3" or from "half give a 1, half give a 5," and the two carry completely different decision risks. Preserve the score distribution, entropy or disagreement, criterion slices, and the judge version; high-stakes cases should also be escalated to a human instead of collapsing the uncertainty into one scalar.

</details>
</details>

## A judge is an evaluation system, not a prompt

LLM-as-a-Judge is not one fixed prompt. It is a set of design choices that combine freely: **whether to give demonstrations, whether to give a reference answer, how candidates are compared, which criterion is judged, and how the scores are finally aggregated.**

Many of these concepts look alike because a single prompt often uses several of the methods at once.

## Why few-shot and reference-based are easy to confuse

They are two orthogonal dimensions:

| Dimension | Few-shot | Reference-based |
| --- | --- | --- |
| What it asks | Should the model first see a few scoring demonstrations? | Does the current answer have a reference answer to compare against? |
| Main function | Teaches the task format, how to use the rubric, and the output style | Gives the judge a task-level anchor |
| Core content | Several input → score examples | The current sample's `Expected Answer` |
| Opposite | Zero-shot | Reference-free |

So all four combinations exist; a prompt can use either, both, or neither:

| | Reference-free | Reference-based |
| --- | --- | --- |
| **Zero-shot** | Give only the criterion and let the judge score on its own | Give the criterion and the current reference answer, with no demonstrations |
| **Few-shot** | Give a few scoring demonstrations, but neither the demonstrations nor the current sample has a reference answer | Both the demonstrations and the current sample carry reference answers |

One sentence to remember:

> **Few-shot controls whether there are demonstrations; reference-based controls whether the current answer has a gold answer to compare against.**

## First separate a few fields that are easy to mix up

```text
Task / Input       What is the original task?
Candidate          Which response or trajectory is being evaluated?
Criterion          Which single quality dimension is judged this time?
Rubric             What observable behavior does each score stand for?
Reference          An ideal answer, key facts, or a set of acceptable answers
Demonstrations     Scoring examples that teach the judge how to use the rubric
Evidence           Retrieved content, tool results, environment state, or other verification information
Verdict            A score, category, preference, or pass/fail
Rationale          The specific evidence that supports the verdict
```

The difference between a reference and a demonstration matters most: a reference is **the answer anchor for the current task**; a demonstration is **a teaching example of how to carry out the evaluation task**.

## How this group reads

- [How to write a criterion](criteria.en.md): the five properties of a good criterion, and the criteria different tasks commonly use.
- [Choosing a scoring mode](scoring.en.md): what binary, ordinal, pairwise, listwise, QAG, and DAG scoring are each good for, and how I would choose.
- [Probability-weighted scores](probability-scores.en.md): G-Eval-style expected scores, repeated sampling, and why storing only the mean is not enough.
- [Biases and a robust workflow](bias-and-workflow.en.md): the judge's common biases, and a more robust minimal workflow.

## Recommended reading

- [G-Eval](https://arxiv.org/abs/2303.16634): CoT evaluation steps, form filling, and probability-weighted scoring.
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685): pointwise and pairwise evaluation, and common biases.
- [Prometheus](https://arxiv.org/abs/2310.08491): training a dedicated evaluator with fine-grained rubrics and reference answers.
- [Judging the Judges](https://arxiv.org/abs/2406.07791): a systematic analysis of pairwise position bias.

### Engineering practice

- [Confident AI Blog](https://www.confident-ai.com/blog): a fairly complete set of practical articles on LLM-as-a-Judge, G-Eval, Arena-style comparison, RAG evaluation, agent evaluation, and observability. A useful engineering companion for implementation ideas, but no substitute for the primary papers or for calibration experiments on your own task.

## How this connects to other chapters

- [Evaluation overview](../README.en.md)
- [Agent Observability](../../06-systems/agent-observability.en.md)
- [Human-in-the-Loop](../../06-systems/human-in-the-loop.en.md)
- [Data and feedback](../../01-data-and-feedback/README.en.md)
