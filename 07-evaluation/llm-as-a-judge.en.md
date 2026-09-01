# LLM-as-a-Judge: Criteria, References, and Scoring

[中文](llm-as-a-judge.md) · **English**

LLM judging combines several orthogonal choices:

1. **Demonstrations:** zero-shot or few-shot
2. **Reference:** reference-free or reference-based
3. **Judging mode:** pointwise, pairwise, or listwise
4. **Criterion and rubric:** what is judged and how score levels are anchored
5. **Aggregation:** hard label, ordinal score, decomposed ratio, token-probability expectation, or repeated sampling

Few-shot controls whether the judge sees examples of the evaluation task. Reference-based scoring controls whether the current candidate is compared against an expected answer. A prompt can use either, both, or neither.

## Quick learning: a judge is a measurement system, not one prompt

<details class="interview" markdown="1">
<summary>Criteria, scoring, calibration, and validation</summary>

**Quick memory**: define an observable criterion, choose pairwise, ordinal, or binary scoring, and calibrate the judge against expert labels, perturbations, and distribution slices.

**Interview answer**

> LLM-as-a-Judge requires explicit evidence, one criterion at a time, behavioral anchors, an output schema, and an aggregation rule. Pairwise scoring is often steadier than absolute scores but still has position, verbosity, self-preference, and reference-leakage biases. Validate against expert audits and report disagreement, not only a mean.

<details markdown="1">
<summary><b>Deep dive</b>: why can a weighted mean hide failure?</summary>

The same mean can come from unanimous scores of three or a polarized mix of ones and fives, which imply different decision risks. Preserve the score distribution, entropy or disagreement, criterion slices, and judge version; high-stakes ambiguous cases should escalate instead of collapsing uncertainty into one scalar.

</details>
</details>

## Criteria

Prefer atomic, observable, scoped criteria with clear boundaries and behavioral score anchors. Common dimensions include correctness, faithfulness, relevance, completeness, instruction following, tool correctness, state integrity, policy compliance, efficiency, recovery quality, memory appropriateness, breadth, and user control.

## Scoring modes

- **Binary / categorical:** best for one semantic condition.
- **Anchored ordinal:** 1–5 severity levels with observable definitions.
- **Pairwise:** A/B/tie/both-bad; swap order to measure position bias.
- **Listwise:** rank multiple candidates, with stronger order and context effects.
- **QAG:** decompose a target into yes/no propositions and compute a derived ratio.
- **DAG:** gate open-ended scoring behind hard requirements.
- **Probability-weighted rating:** compute `E[s] = Σ s·p(s)` over rating-token probabilities.

If rating token probabilities are unavailable, repeated sampling can estimate an empirical score distribution. Sampling 20 judgments and taking a weighted sum is therefore an approximation to the distribution, not a new criterion.

Store more than the mean. Two judges can both average 3 while one always returns 3 and another splits evenly between 1 and 5. Preserve probabilities, variance, agreement, and abstention rate.

## Practical minimum

Use deterministic checks first; judge one criterion at a time; write anchored rubrics; provide references when valid alternatives are not excluded; calibrate on human examples; swap pairwise order; preserve evidence and evaluator versions; inspect slices; and periodically remeasure agreement with humans.

## Reading

- [G-Eval](https://arxiv.org/abs/2303.16634)
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)
- [Prometheus](https://arxiv.org/abs/2310.08491)
- [Judging the Judges](https://arxiv.org/abs/2406.07791)
- [Confident AI Blog](https://www.confident-ai.com/blog) — implementation-oriented articles on LLM judging, RAG and agent evaluation, observability, and human review; useful as an engineering companion to primary papers.
