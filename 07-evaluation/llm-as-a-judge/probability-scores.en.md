# Probability scores: what does 4.1 actually tell us?

[中文](probability-scores.md) · **English**

Integer ratings often bunch together. Could we ask a judge 20 times and calculate a weighted score? Yes, but first distinguish the probabilities you actually have.

## Same mean, different disagreement

<div data-judge-lab="distribution"><p>This interaction needs JavaScript. Twenty scores of 3 and ten each of 1 and 5 both average 3, but have variances of 0 and 4.</p></div>

The controls edit **fictional score counts**. Switch between “All 3” and “Split 1 / 5”: the mean stays fixed while disagreement changes. Setting every count to zero means no data, not a score of zero.

## Method A: probabilities of rating labels

First define $p(k)$ as a distribution summing to one over rating labels 1–5. With the complete label probabilities at a common rating position:

$$\bar{s}=\sum_{k=1}^{5}k\,p(k).$$

For \(p(3)=0.1,\ p(4)=0.7,\ p(5)=0.2\), the result is 4.1. [G-Eval](https://arxiv.org/abs/2303.16634) uses rating-token probabilities to refine discrete scores.

Raw vocabulary probabilities need not sum to one over the rating labels. Suppose all five labels are available: ratings 3, 4, and 5 have probabilities 0.08, 0.56, and 0.16, ratings 1 and 2 have zero, and other tokens hold the remaining 0.20. The raw weighted sum is 3.28. Dividing by the rating mass of 0.80 gives 4.1, the mean conditional on a valid 1–5 label. Retain that mass rather than presenting the conditional score as an unconditional distribution.

```python
raw_rating_probabilities = {1: 0.0, 2: 0.0, 3: 0.08, 4: 0.56, 5: 0.16}
rating_mass = sum(raw_rating_probabilities.values())
weighted_sum = sum(score * probability for score, probability in raw_rating_probabilities.items())
conditional_mean = weighted_sum / rating_mass
assert abs(rating_mass - 0.80) < 1e-12
assert abs(weighted_sum - 3.28) < 1e-12
assert abs(conditional_mean - 4.1) < 1e-12
```

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

This is also part of G-Eval's implementation history. [Section 3.1](https://arxiv.org/html/2303.16634v3#S3.SS1) says its GPT-4 interface did not expose token probabilities at the time, so the authors estimated the rating distribution from 20 samples. That describes the paper's setup, not a restriction on every current interface. Twenty independent completions need not require twenty HTTP requests if the interface supports batch generation.

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
