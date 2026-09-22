# LLM-as-a-Judge: probability-weighted scores

[中文](probability-scores.md) · **English**

## What a probability-weighted score actually is

Suppose the judge can only output 1–5. Taking one discrete score produces many ties, and you cannot see the model hesitating between 3 and 4.

If you can get the probabilities of the rating tokens, you can compute the expectation `E[s] = Σ s·p(s)`:

```text
score = 1·p(1) + 2·p(2) + 3·p(3) + 4·p(4) + 5·p(5)
```

For example:

```text
p(3)=0.10, p(4)=0.70, p(5)=0.20
score = 3×0.10 + 4×0.70 + 5×0.20 = 4.10
```

That is the core idea of probability-weighted scoring in G-Eval: you end up with a finer, continuous score rather than only integers.

## What "generate 20 scores" means

Two implementations need to be told apart here.

### Method A: read token probabilities directly

If the model API exposes log probabilities for output tokens, you get `p(1)...p(5)` directly and then compute the weighted expectation.

### Method B: approximate the distribution by repeated sampling

If the full probabilities are not available, have the judge sample independently several times. Say 20 runs give:

```text
score 3:  4 times
score 4: 12 times
score 5:  4 times
```

The empirical probabilities are `0.2 / 0.6 / 0.2`, and the final expected score is 4.0.

So "generate 20 scores and take a weighted sum" is not a new criterion; it is **one way to estimate the score distribution**.

## Storing only the weighted mean is not enough

Both of the distributions below have a mean of 3:

```text
A: 100% give a 3
B: 50% give a 1, 50% give a 5
```

A means the judge consistently finds the answer mediocre; B means the judge is extremely uncertain, or the rubric admits two conflicting readings.

So it is best to store all of these together:

- the weighted mean;
- the variance or standard deviation;
- the probability of each score;
- the agreement rate across repeated samples;
- the `unknown / abstain` rate.

If the distribution is clearly bimodal, inspect the criterion and the input evidence first, rather than letting the mean paper over the disagreement.
