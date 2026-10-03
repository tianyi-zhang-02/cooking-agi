# Common distributions: derive more than their moments

[中文](distribution-toolkit.md) · **English** · [Review map](../README.en.md)

> Reading time: ~10 min · Prerequisites: expectation, variance, integration · Last reviewed: 2026-10

Choose a distribution from the mechanism. Counting successes in fixed trials, waiting for success, and counting arrivals during fixed time are different questions.

## 1 · Fix the parameter conventions

| Distribution | Parameters and support | Mean | Variance |
| --- | --- | --- | --- |
| Bernoulli | p; 0, 1 | p | p(1−p) |
| Binomial | n independent Bernoulli(p) trials | np | np(1−p) |
| Geometric | p; trial of first success, 1, 2… | 1/p | (1−p)/p² |
| Negative Binomial | Total trials to r successes | r/p | r(1−p)/p² |
| Poisson | λ>0; 0, 1… | λ | λ |
| Exponential | **Rate** λ>0 | 1/λ | 1/λ² |
| Gamma | Shape α>0, **rate** λ>0 | α/λ | α/λ² |
| Normal | μ, σ²>0 | μ | σ² |
| Beta | α,β>0; 0<x<1 | α/(α+β) | αβ/[(α+β)²(α+β+1)] |

Gamma's scale parameter is θ=1/λ. A Geometric variable counting failures is one less than our convention. Many apparent disagreements are parameter mismatches.

## 2 · Geometric and Negative Binomial: break waiting into stages

With independent success probability p, the first success occurs on trial k with probability $(1-p)^{k-1}p$.

For the rth success on trial n, the last trial must succeed and the previous n−1 trials contain r−1 successes:

$$
P(T_r=n)=\binom{n-1}{r-1}p^r(1-p)^{n-r},\qquad n\ge r.
$$

Successive waiting times are independent Geometric variables, giving additive means and variances. Varying success probabilities require a different model.

## 3 · Poisson: many opportunities, individually unlikely

Let $X_n\sim\operatorname{Binomial}(n,\lambda/n)$. For fixed k:

$$
P(X_n=k)=\frac{n(n-1)\cdots(n-k+1)}{k!}
\left(\frac\lambda n\right)^k
\left(1-\frac\lambda n\right)^{n-k}
\longrightarrow e^{-\lambda}\frac{\lambda^k}{k!}.
$$

This is a fixed-λ limit as n→∞, not a claim that all sparse data are Poisson. Clustering and changing rates can produce variance much larger than the mean.

<details markdown="1">
<summary>Derive E[X] and Var(X) without expanding a square</summary>

Shift the summation index once to obtain $E[X]=\lambda$, and twice to obtain $E[X(X-1)]=\lambda^2$. Since $X^2=X(X-1)+X$, the variance is λ.

</details>

## 4 · Exponential waiting and memorylessness

If $P(T>t)=e^{-\lambda t}$ for t≥0, the density is $\lambda e^{-\lambda t}$, and:

$$
P(T>s+t\mid T>s)=\frac{e^{-\lambda(s+t)}}{e^{-\lambda s}}=e^{-\lambda t}.
$$

Elapsed waiting does not change the remaining-wait distribution. This is a model property, not a universal fact about waiting. Regularly scheduled buses behave differently.

Tail integration gives $E[T]=\int_0^\infty e^{-\lambda t}dt=1/\lambda$. Similarly, $E[T^2]=\int_0^\infty2t e^{-\lambda t}dt=2/\lambda^2$, yielding variance $1/\lambda^2$.

The sum of r independent equal-rate exponentials is Gamma(r,λ), also called Erlang. In general:

$$
f(x)=\frac{\lambda^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\lambda x},\quad x>0.
$$

Substitute u=λx and use $\Gamma(\alpha+1)=\alpha\Gamma(\alpha)$ to derive moments. The Gamma recurrence follows from integration by parts.

## 5 · MGFs explain convenient sums

The moment-generating function is $M_X(t)=E[e^{tX}]$. Finiteness in a neighborhood of zero permits the standard differentiation identities for moments. Independence gives $M_{X+Y}=M_XM_Y$.

| Distribution | MGF | Domain |
| --- | --- | --- |
| Bernoulli(p) | $1-p+pe^t$ | All real t |
| Poisson(λ) | $\exp[\lambda(e^t-1)]$ | All real t |
| Gamma(α,λ) | $(\lambda/(\lambda-t))^\alpha$ | t<λ |
| Normal(μ,σ²) | $\exp(\mu t+\sigma^2t^2/2)$ | All real t |

Independent Poisson rates add; independent equal-rate Gamma shapes add; independent Normal means and variances add. Completing the square derives the Normal MGF.

**An MGF need not exist near zero.** A Lognormal variable has all positive integer moments but an infinite MGF for t>0. A finite mean is not permission to manipulate an MGF freely.

## 6 · Normal transformations and Beta updates

For standard Normal Z, $\mu+\sigma Z$ is Normal(μ,σ²). If $Y=e^X$ with Normal X, then Y is Lognormal and $E[Y]=e^{\mu+\sigma^2/2}$, not $e^{E[X]}$.

The Beta density is proportional to $p^{\alpha-1}(1-p)^{\beta-1}$. Multiplying by a Bernoulli likelihood with s successes and f failures gives Beta(α+s,β+f). Continue with [Bayesian updating](../methods/statistics.en.md).

<details markdown="1">
<summary>Exercise: independent X~Exp(2), Y~Exp(3). What is min(X,Y)?</summary>

$P(\min(X,Y)>t)=P(X>t)P(Y>t)=e^{-5t}$, so it is Exp(5), with mean 1/5. Adding rates relies on independence; identical waiting times do not satisfy the argument.

</details>

Continue: [Joint distributions and order statistics](joint-and-order.en.md). Reference: [MIT 6.041 lecture notes](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/pages/lecture-notes/).
