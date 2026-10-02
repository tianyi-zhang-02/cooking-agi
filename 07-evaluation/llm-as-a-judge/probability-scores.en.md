# Probability scores: what does 4.1 actually tell us?

[中文](probability-scores.md) · **English**

Integer ratings often bunch together. Could we ask a judge 20 times and calculate a weighted score? Yes, but first distinguish the probabilities you actually have.

## Same mean, different disagreement

<div data-judge-lab="distribution"><p>This interaction needs JavaScript. Twenty scores of 3 and ten each of 1 and 5 both average 3, but have variances of 0 and 4.</p></div>

The controls edit **fictional score counts**. Switch between “All 3” and “Split 1 / 5”: the mean stays fixed while disagreement changes. Setting every count to zero means no data, not a score of zero.

## Method A: probabilities of rating labels

With probabilities for all allowed labels at a common rating position:

$$\bar{s}=\sum_{k=1}^{5}k\,p(k).$$

For \(p(3)=0.1,\ p(4)=0.7,\ p(5)=0.2\), the result is 4.1. [G-Eval](https://arxiv.org/abs/2303.16634) uses rating-token probabilities to refine discrete scores.

Having an API that returns logprobs is not sufficient by itself:

- Returned top-k tokens may omit rating labels. **Absent does not mean probability zero.**
- “4,” “ 4,” and multi-token labels may differ. Fix formatting, inspect tokenization, and identify the rating position.
- Renormalizing only a returned subset creates a **conditional distribution**. Report the probability mass retained.
- Incomplete top-k probabilities do not give the full expectation. Change the output protocol, use an interface exposing the needed probabilities, or use repeated sampling.

The probability of generating the token 4 is not a 70% probability that the answer is correct. **Token probabilities and calibration of judgment correctness are different.**

## Method B: separate calls, repeated sampling

If 20 calls return four 3s, twelve 4s, and four 5s:

$$\hat p(k)=\frac{n_k}{N},\qquad
\bar{s}=\sum_k k\frac{n_k}{N}=\frac{1}{N}\sum_{i=1}^{N}s_i=4.$$

This is the ordinary mean, rewritten using frequencies. It doesn't invent a separate weight for each of the 20 calls.

- One prompt requesting “20 scores” does not produce 20 independent calls.
- Keep the model, rubric, context, and sampling setup fixed. Independent sampling is a modeling assumption, not protection from systematic bias.
- Identical temperature-zero outputs do not establish a useful sampling distribution. Temperature isn't a quality-calibration dial.
- Two parsing failures mean 18 valid results from 20 attempts plus 2 errors. Don't report only the mean of the 18.

More draws can reveal sampling variation. They won't repair a biased rubric, and 20 is not a universally appropriate sample size.

## Keep more than a mean

| Record | What it tells you |
| --- | --- |
| Counts / probability distribution | Where scores concentrate; whether they are bimodal |
| Mean and variance | Location and dispersion under the chosen numeric encoding |
| Valid / attempted | How often a usable judgment is returned |
| Separate unknown and error counts | Missing evidence versus evaluator failure |
| Criterion and model versions | Whether two batches are comparable |

A value of 4.137 is not necessarily more meaningful than 4.100. A 1–5 scale is ordinal; treating its steps as equally spaced is a modeling choice whose usefulness needs validation against people and decisions.

## Two kinds of weighting

**Distribution weighting** calculates an expectation within one criterion. **Metric weighting** combines dimensions such as grounding, relevance, and style. The former cannot determine the latter's weights or justify offsetting false facts with better prose.

<details markdown="1">
<summary>If two groups average 3, should we prefer lower variance?</summary>

Not on variance alone. A consistently wrong judge can have zero variance. Check human judgments and the criterion before interpreting disagreement as sampling noise, task ambiguity, or incomplete evidence.

</details>

Continue: [Bias tests](bias-and-workflow.en.md) · [Calibration and thresholds](calibration.en.md)
