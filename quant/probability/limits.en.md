# LLN and CLT: stable averages and the shape of error

[中文](limits.md) · **English**

> Reading time: ~8 min · Prerequisites: [Variance](expectation-proofs.en.md), [Chebyshev](inequalities.en.md) · Last reviewed: 2026-10

“More samples make it accurate” and “more samples make it normal” both skip important details. Which quantity changes, and under what assumptions? Separate those questions first.

## 1 · Weak LLN: large deviations of the average become unlikely

**Our version.** Let $X_1,X_2,\ldots$ be independent and identically distributed (iid), with mean $\mu$ and variance $\sigma^2<\infty$. Write $\bar X_n=(X_1+\cdots+X_n)/n$. For every fixed $\varepsilon>0$,

$$
P(|\bar X_n-\mu|\ge\varepsilon)\longrightarrow0.
$$

This is convergence in probability.

**The complete proof takes two steps.** Linearity and independence give $\mathbb E[\bar X_n]=\mu$ and $\operatorname{Var}(\bar X_n)=\sigma^2/n$. Apply Chebyshev:

$$
P(|\bar X_n-\mu|\ge\varepsilon)\le\frac{\sigma^2}{n\varepsilon^2}\longrightarrow0.
$$

Finite variance is an assumption of this short proof, not a requirement of every LLN version. Nor does the theorem promise that every additional sample reduces the realized error.

## 2 · A sample-size calculation

For independent Bernoulli($p$) observations, $\sigma^2=p(1-p)\le1/4$. To keep the probability of absolute error at least $0.05$ below $0.05$, Chebyshev gives the sufficient condition

$$
\frac1{4n(0.05)^2}\le0.05
\quad\Longrightarrow\quad n\ge2000.
$$

This is a conservative guarantee, not a minimum sample size or permission to collect 2000 observations in any manner. Dependence requires redoing the variance calculation.

<details markdown="1">
<summary>Counterexample: duplicate one observation many times</summary>

Let every $X_i=Z$, where $Z$ is one fair 0/1 outcome. Each variable has mean $1/2$ and variance $1/4$, but $\bar X_n=Z$ never becomes more stable.

For $\varepsilon=0.4$, $P(|\bar X_n-1/2|\ge0.4)=1$. Identically distributed does not mean independent, and more rows do not necessarily contain more independent information.

</details>

## 3 · CLT: magnify the shrinking fluctuations

**Classical iid version.** Assume finite mean $\mu$ and variance $0<\sigma^2<\infty$. The central limit theorem states

$$
Z_n=\frac{\sum_{i=1}^nX_i-n\mu}{\sigma\sqrt n}
=\frac{\sqrt n(\bar X_n-\mu)}{\sigma}
\xrightarrow{d}\mathcal N(0,1).
$$

Distributional convergence here means $P(Z_n\le z)\to\Phi(z)$. The original $X_i$ do not become normal, and a finite sample does not necessarily give an exactly normal sum.

| Question | Quantity | Result |
| --- | --- | --- |
| Does the average approach the mean? | $\bar X_n$ | Approaches $\mu$ in probability |
| What is the scale of the error? | $\bar X_n-\mu$ | Standard deviation $\sigma/\sqrt n$ |
| What shape does magnified error have? | $\sqrt n(\bar X_n-\mu)/\sigma$ | Distribution approaches standard normal |

Halving the standard error typically takes 4 times as many independent observations, not twice as many. The CLT gives no universal “30 observations are enough” rule.

## 4 · Why a normal distribution appears

<details markdown="1">
<summary>Advanced: characteristic-function proof sketch, with prerequisites stated</summary>

Set $W_i=(X_i-\mu)/\sigma$. Its characteristic function is $\varphi_W(t)=\mathbb E[e^{itW}]$, which always exists. Zero mean and unit variance give the local expansion

$$
\varphi_W(t)=1-\frac{t^2}{2}+o(t^2).
$$

Finite second moment is needed to control the remainder; treating an unbounded random variable as a bounded constant is not a justification. Independence makes characteristic functions multiply:

$$
\varphi_{Z_n}(t)=\left[\varphi_W\left(\frac{t}{\sqrt n}\right)\right]^n
=\left[1-\frac{t^2}{2n}+o(1/n)\right]^n
\longrightarrow e^{-t^2/2}.
$$

The limit is the standard normal characteristic function. Lévy's continuity theorem then turns this into convergence in distribution. Two analysis results—the justified second-order expansion and Lévy's theorem—are invoked here rather than proved in full.

</details>

## 5 · Convergence does not justify every interchange

**Convergence in probability does not automatically imply convergence of expectations.** Let $U$ be uniform on $(0,1)$ and $X_n=n\mathbf1_{\{U\le1/n\}}$. For fixed $\varepsilon>0$ and $n>\varepsilon$,

$$
P(|X_n|>\varepsilon)=\frac1n\to0,\qquad \mathbb E[X_n]=1.
$$

Increasingly rare large values keep the mean from shrinking. Being close to 0 most of the time does not justify moving a limit inside expectation. Conditions such as domination or uniform integrability address this problem.

<details markdown="1">
<summary>How is the strong law different?</summary>

The classical iid, integrable strong law says $\bar X_n\to\mu$ almost surely: outside a probability-zero set, the entire sample path converges. This is stronger than convergence in probability.

This page proves only the finite-variance weak law. A one-time Chebyshev bound is not presented as a proof of the strong law.

</details>

## Next

Advanced proof reference: [MIT 18.175: Characteristic functions and CLT](https://ocw.mit.edu/courses/18-175-theory-of-probability-spring-2014/resources/mit18_175s14_lecture15/).

[Continuous probability and calculus](continuous-calculus.en.md). References: [MIT 6.041: Weak LLN](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/resources/mit6_041f10_l19/), [Central limit theorem](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/resources/mit6_041f10_l20/). The characteristic-function sketch is optional; first make sure you can distinguish the two theorems.
