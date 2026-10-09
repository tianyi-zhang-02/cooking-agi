# Three KL estimators: similar values, different gradients

[中文](kl-estimators.md) · **English**

> Last reviewed: 2026-10 · Prerequisite: [Reference and Critic](reference-and-critic.en.md); derivations use expectations and logarithms

A training log reports negative `kl`, while another implementation uses an always-nonnegative expression. Is one wrong? Before reading the variable name, ask **which distributions are compared, which distribution produced the samples, and whether the value is logged or differentiated.**

For reading training logs, start with sections 1–3. For implementing a loss, continue to sampling and gradients. There is no need to memorize every derivation on a first pass.

## 1. Fix the direction first {#direction}

At a fixed prompt and prefix, let $q$ be the current next-token distribution and $p$ the reference. We estimate:

$$
D_{\rm KL}(q\|p)=\mathbb E_{x\sim q}\left[\log\frac{q(x)}{p(x)}\right].
$$

Initially assume strictly positive probabilities on the same finite vocabulary, and define $r(x)=p(x)/q(x)$. This is **reference divided by current policy**, not PPO's current/old ratio. The names $k_1,k_2,k_3$ are common conventions, not mandatory library names.

| Estimator | Can one sample be negative? | Expectation under $x\sim q$ |
| --- | --- | --- |
| $k_1=-\log r$ | Yes | Exactly KL |
| $k_2=\tfrac12(\log r)^2$ | No | Generally biased; a local approximation for nearby distributions |
| $k_3=r-1-\log r$ | No | Exactly KL under the support assumptions above |

A sample is not the whole distribution. Negative $k_1$ does not contradict nonnegative KL; sampling variation can also make a batch mean negative. Nonnegative $k_3$ does not make it the lowest-variance estimator for every problem.

## 2. Calculate two tokens by hand {#two-tokens}

Let $q=[0.8,0.2]$ and $p=[0.5,0.5]$. A two-token vocabulary lets us enumerate the entire distribution; it is not a claim about actual model vocabularies.

<div class="worked-table" markdown="1">

| Token | $q$ | $k_1$ | $k_2$ | $k_3$ |
| --- | --- | --- | --- | --- |
| A | 0.8 | 0.4700 | 0.1105 | 0.0950 |
| B | 0.2 | −0.9163 | 0.4198 | 0.5837 |

</div>

Weighting by sampling probabilities 0.8 and 0.2 gives about 0.1927 for both $k_1$ and $k_3$, but about 0.1723 for $k_2$. Giving the rows equal weight would change the sampling distribution.

```python
import math

def kl_sample_terms(log_current, log_reference):
    if not all(math.isfinite(value) and value <= 0
               for value in (log_current, log_reference)):
        raise ValueError("Expected finite log-probabilities no greater than zero")
    log_ratio = log_reference - log_current
    return (-log_ratio, 0.5 * log_ratio ** 2,
            math.expm1(log_ratio) - log_ratio)

current = [0.8, 0.2]
reference = [0.5, 0.5]
terms = [kl_sample_terms(math.log(policy), math.log(anchor))
         for policy, anchor in zip(current, reference)]
means = [math.fsum(probability * row[column]
                   for probability, row in zip(current, terms))
         for column in range(3)]
assert math.isclose(means[0], means[2], abs_tol=1e-12)
print([round(value, 4) for value in means])
```

This is a scalar check, not a trainer. `expm1(z)` is more stable than `exp(z)-1` near zero; it cannot prevent overflow for extreme ratios. Record nonfinite values and inspect support and precision. Silently clipping changes the estimator and its unbiasedness claim.

## 3. Why adding $r-1$ preserves the mean {#control-variate}

With the same support:

$$
\mathbb E_q[r-1]
=\sum_xq(x)\left(\frac{p(x)}{q(x)}-1\right)
=\sum_xp(x)-\sum_xq(x)=0.
$$

Thus $k_3=k_1+(r-1)$ adds a zero-mean term: a control variate can change sampling variation without changing the expectation. The inequality $\log r\le r-1$ makes $k_3$ nonnegative sample by sample.

$k_2$ has a different justification. For $z=\log r$, $k_3=e^z-1-z=\tfrac12z^2+O(z^3)$. When ratios are near one in relevant regions, $k_2$ keeps this local quadratic term. It is not exact for arbitrary policy shifts.

**Support matters.** If $q$ keeps only A while $p$ assigns half its mass to each of A and B, $\mathbb E_q[r]=0.5$, not 1. Top-k/top-p sampling changes the original softmax distribution. Model log-probabilities are not automatically actual sampling log-probabilities.

## 4. Old samples change the expectation {#old-samples}

Suppose an old policy $\mu$ generates the samples, then an update changes the current policy to $q$. Averaging recomputed $k_3$ estimates $\mathbb E_\mu[k_3(q,p)]$, not automatically $D_{\rm KL}(q\|p)$.

At a fixed prefix, if $\mu$ covers $q$, importance weighting gives:

$$
D_{\rm KL}(q\|p)=\mathbb E_{x\sim\mu}\left[\frac{q(x)}{\mu(x)}k_3(x)\right].
$$

For the same example, an old policy sampling both tokens equally gives an unweighted expectation of about 0.3394. Weights $q/\mu=[1.6,0.4]$ restore about 0.1927. Correction may increase variance; clipping changes the estimator again.

This corrects only the **action distribution at a given prefix**. If prefixes also come from an old policy, a current-token ratio does not correct the entire trajectory's state distribution.

## 5. An unbiased value need not give an unbiased gradient {#gradients}

Monitoring KL generally needs no backpropagation. Using it in a loss requires another question: does the sampling distribution depend on parameters, and what is detached?

For $q=q_\theta$ and fixed $p$:

$$
\nabla_\theta\mathbb E_{x\sim q_\theta}[k_3(x,\theta)]
=\mathbb E_q\left[\nabla_\theta k_3+k_3\nabla_\theta\log q_\theta(x)\right].
$$

Holding sampled tokens fixed and differentiating only $k_3$ omits the second term. A small counterexample: take $q=[\sigma(\theta),1-\sigma(\theta)]$, $\theta=\log4$, and $p=[0.5,0.5]$. The true KL derivative is about **0.2218**; differentiating $k_3$ under fixed sampling weights gives an expected gradient of **0.3000**.

This does not make every use of $k_3$ incorrect. An algorithm may deliberately define an old-data surrogate, or retain the required derivatives through importance weights. Inspect the full objective, not just an estimator's unbiasedness label. [DeepSeekMath's GRPO equations](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS1) include this nonnegative KL term; the distinction here is between a statistic and an update rule.

## 6. Checks before using it in training {#implementation}

| Check | Concrete question |
| --- | --- |
| Direction and sampling | Which model is current, reference, or old? Do behavior probabilities include temperature and truncation? |
| Valid positions | Answer tokens only? Include EOS? Exclude padding before computing statistics? |
| Units | Average over tokens or sum each response? How does length change the curve? |
| Purpose | Monitoring, reward shaping, or a differentiable regularizer? Where is detach applied? |
| Numerics | Does equality give zero? Have small shifts, extreme ratios, and missing support been tested? |

The chain rule decomposes full sequence KL into conditional KLs summed over current-policy prefixes. Arbitrary token averages or averages under old prefixes do not have the same definition.

Start by exactly enumerating this two-token example, then inspect sampled estimates over a large vocabulary. That is more informative than guessing what one logged `kl` value says about training.
