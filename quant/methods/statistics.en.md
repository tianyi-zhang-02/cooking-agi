# Statistical inference: what lies between a sample and a claim

[中文](statistics.md) · **English** · [Review map](../README.en.md)

> Reading time: ~11 min · Prerequisites: conditioning, variance, CLT, least squares · Last reviewed: 2026-10

Probability starts with a model and predicts data. Statistics starts with finite data and asks what a model or parameter might be. Ignoring how the data were collected can matter more than an arithmetic error.

## 1 · Estimators fluctuate too

For iid observations with finite mean μ and variance σ², $\bar X$ is unbiased with variance σ²/n. Mean squared error decomposes as:

$$
E[(\hat\theta-\theta)^2]=\operatorname{Var}(\hat\theta)+[E(\hat\theta)-\theta]^2.
$$

Add and subtract $E\hat\theta$, expand, and the cross term vanishes. Unbiasedness alone does not imply minimum MSE.

Why divide sample variance by n−1? The identity $\sum(X_i-\bar X)^2=\sum(X_i-\mu)^2-n(\bar X-\mu)^2$ has expectation $(n-1)\sigma^2$. Dividing by n−1 is unbiased for n>1.

## 2 · MLE: write the likelihood first

For n iid Bernoulli observations with s successes, $L(p)=p^s(1-p)^{n-s}$. For 0<s<n:

$$
\ell'(p)=\frac{s}{p}-\frac{n-s}{1-p}=0
\quad\Rightarrow\quad\hat p=\frac sn.
$$

The negative second derivative gives a unique interior maximum. At s=0 or n, the maximum is on the boundary.

For Normal observations, the mean MLE is the sample mean. With unknown variance, its MLE uses denominator n, not n−1. **Maximum likelihood and unbiasedness are different objectives.**

Fisher information measures local likelihood sensitivity. Bernoulli information per observation is $1/[p(1-p)]$ for p∈(0,1). Under regularity conditions, Cramér–Rao bounds the variance of unbiased estimators. Parameter-dependent support can break those conditions. We state the general result here rather than prove it.

## 3 · Bayes: make assumptions beyond the data explicit

A Beta(α,β) prior becomes Beta(α+s,β+n−s), with posterior mean $(\alpha+s)/(\alpha+\beta+n)$.

With Beta(2,2) and 7 successes in 10 trials, the posterior is Beta(9,5), mean 9/14; MLE is 0.7. Small-sample disagreement does not mean an arithmetic error.

<details markdown="1">
<summary>Basic example: what does a positive test imply?</summary>

With prevalence 1%, sensitivity 90%, and specificity 95%, the posterior disease probability is $0.9\cdot0.01/[0.9\cdot0.01+0.05\cdot0.99]=2/13$, about 15.4%. Among 10,000 people, expect 90 true positives and 495 false positives. This is a probability exercise, not medical guidance.

</details>

The observation mechanism also matters. In standard Monty Hall, the informed host must reveal an empty door and always offer a switch; switching wins with probability 2/3. A host opening a random door that happens to be empty defines a different conditional model.

## 4 · Confidence intervals describe a procedure

For iid Normal observations with unknown variance and n>1:

$$
\frac{\bar X-\mu}{S/\sqrt n}\sim t_{n-1}.
$$

This gives an exact t interval. For non-Normal data, large-sample approximations require care with tails and dependence. Confidence refers to repeated-sampling coverage; a Bayesian credible interval describes posterior probability under a model.

S measures dispersion of individual observations. S/√n is the standard error of the mean. They answer different questions.

## 5 · Significance is not usefulness

A p-value is the probability, **under the null and sampling model**, of a statistic at least as extreme as observed. It is not the probability that the null is true.

| Concept | Question to ask |
| --- | --- |
| Type I error | Is the probability of rejecting a true null controlled? |
| Type II error / power | Against which alternative and effect size are misses measured? |
| Effect size | How large is the difference, and does it matter? |
| Multiple testing | Were many attempts made but only the best reported? |

For m tests, Bonferroni uses level α/m per test. A union bound controls the probability of any false rejection without requiring independence. It can be conservative. Repeated peeking with a data-dependent stopping rule can also change error rates.

## 6 · Regression: algebra versus statistical guarantees

In $y=X\beta+\varepsilon$, full column rank and $E[\varepsilon\mid X]=0$ give conditional unbiasedness of OLS. If additionally $\operatorname{Var}(\varepsilon\mid X)=\sigma^2I$:

$$
\operatorname{Var}(\hat\beta\mid X)=\sigma^2(X^\top X)^{-1}.
$$

Substitute $\hat\beta=\beta+(X^\top X)^{-1}X^\top\varepsilon$. Normality is unnecessary for unbiasedness, but exact small-sample t inference generally needs additional distributional assumptions. Heteroskedasticity, correlated errors, and omitted variables are different problems.

## 7 · Research extension: sampling is part of the model

Use chronological train/test splits for time series and prevent future-information leakage. Autocorrelation can make iid standard errors optimistic. Ordinary observation-wise bootstrap does not preserve temporal dependence; appropriate block resampling may be needed.

For AR(1), $X_t=\phi X_{t-1}+\varepsilon_t$, independent zero-mean noise of variance σ² and |φ|<1 give a stationary solution with variance $\sigma^2/(1-\phi^2)$ and lag-k correlation φᵏ. Verify through the recurrence and independence. Do not use these stationary formulas when |φ|≥1.

<details markdown="1">
<summary>Explain aloud: why might doubling the sample leave bias unchanged?</summary>

If selection always omits the same population or labels favor the same behavior, more data can estimate the wrong target more precisely. A smaller standard error does not imply smaller selection bias. Specify the population, sampling mechanism, estimand, and validation data.

</details>

References: [MIT 18.05 inference materials](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/pages/classes-reading-and-in-class-materials/), [MIT regression notes](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote6/).
