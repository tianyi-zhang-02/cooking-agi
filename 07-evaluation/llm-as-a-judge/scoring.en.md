# LLM-as-a-Judge: choosing a scoring mode

[中文](scoring.md) · **English**

The same criterion can be scored in different ways.

## 1. Binary / categorical

Output `pass / fail / unknown`, or one of a finite set of categories.

Good for well-defined conditions, such as "does it cite a fact that is not in the context?" It is usually easier to calibrate than a vague continuous score.

## 2. Anchored ordinal score

Output ordered levels such as 1–5, each with a clear behavioral anchor.

It fits cases that need to express severity, but don't assume the gap between 4 and 3 necessarily equals the gap between 3 and 2.

## 3. Pairwise comparison

Give two candidates, A and B, and let the judge choose `A / B / tie / both bad`.

Good for A/B comparison of models, prompts, or policies. It is usually more natural than absolute scoring but is prone to position bias, so swap the order and judge again.

## 4. Listwise ranking

Rank several candidates at once. It saves calls, but as the candidates multiply, context and order effects become more complex. It usually needs randomized order or grouped comparison.

## 5. QAG / decomposable ratio

First split a complex target into several answerable yes/no questions, then compute the score from the fraction that pass.

For completeness, for example, first extract five required points, then judge how many of them the response covers:

```text
completeness = required points covered / total required points
```

Where this score comes from is much clearer than directly asking the model to "rate completeness 0.73."

## 6. DAG / hierarchical gating

Check the inviolable conditions first, then evaluate open-ended quality. For example:

```text
Tool arguments wrong? → fail immediately
Fact not supported by the evidence? → score capped at 2
Otherwise → go on to judge helpfulness and quality of expression
```

The value of a DAG is that it expresses the evaluation logic; it does not automatically make LLM judgment deterministic.

## How I would choose a scoring mode

| Scenario | Better starting point |
| --- | --- |
| There is an explicit rule or an executor | Deterministic checks; an LLM is not the first choice |
| Judging whether one semantic condition is violated | Binary + rationale + evidence |
| Comparing two models or prompts | Pairwise, swap the order, allow ties |
| Judging how severe a quality problem is | A 1–5 ordinal rubric with behavioral anchors |
| A complex target made of several sub-conditions | QAG or DAG decomposition |
| Need a finer ranking and the API has logprobs | Rating-token probability expectation |
| No logprobs, but stability needs estimating | Repeated sampling, reporting the distribution and the variance as well |
| A high-quality gold answer exists | Reference-based |
| There is no single correct answer | Reference-free + a clear rubric + human calibration |
