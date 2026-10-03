# Probability inequalities: you may not need an exact answer

[中文](inequalities.md) · **English**

> Reading time: ~9 min · Prerequisite: [Expectation and variance](expectation-proofs.en.md) · Last reviewed: 2026-10

What can we say with a mean or variance but no complete distribution? That is where inequalities help. They follow from nonnegativity, squares, and convexity rather than adding new probability definitions.

## 1 · Markov: large values cannot occur too often

**Assumptions.** $X\ge0$, $\mathbb E[X]<\infty$, and $a>0$.

$$
P(X\ge a)\le\frac{\mathbb E[X]}a.
$$

The proof is a pointwise comparison:

$$
X\ge a\mathbf1_{\{X\ge a\}}
\quad\Longrightarrow\quad
\mathbb E[X]\ge aP(X\ge a).
$$

If nonnegative request latency averages 100 ms, the probability of at least 500 ms is at most $0.2$. This does not identify its actual value.

<details markdown="1">
<summary>Can we drop nonnegativity? When is the bound exact?</summary>

No. If $X$ takes $-1,1$ with equal probability, its mean is 0 but $P(X\ge1)=1/2$, which is not at most 0.

If $X$ only takes $0,a$, then $\mathbb E[X]/a=P(X=a)$ and equality holds. Nonnegativity and a mean alone cannot always give a smaller bound.

</details>

## 2 · Chebyshev: make deviation nonnegative

**Assumptions.** $\mathbb E[X]=\mu$, $\operatorname{Var}(X)=\sigma^2<\infty$, and $t>0$.

$$
P(|X-\mu|\ge t)\le\frac{\sigma^2}{t^2}.
$$

Proof: apply Markov to $Z=(X-\mu)^2$ at threshold $t^2$. For $\sigma>0$, the probability of a deviation of at least 3 standard deviations is at most $1/9$.

No normality is assumed, so this is not the normal distribution's “3σ rule.” If the computed bound exceeds 1, use 1 instead: the available information is too weak for a useful bound.

## 3 · Cauchy–Schwarz: a square cannot be negative

**Assumptions.** $\mathbb E[X^2],\mathbb E[Y^2]<\infty$.

$$
(\mathbb E[XY])^2\le\mathbb E[X^2]\mathbb E[Y^2].
$$

<details markdown="1">
<summary>Proof: construct a square depending on t</summary>

Let $q(t)=\mathbb E[(X-tY)^2]\ge0$. If $\mathbb E[Y^2]>0$, minimize the quadratic with $t=\mathbb E[XY]/\mathbb E[Y^2]$:

$$
0\le \mathbb E[X^2]-\frac{(\mathbb E[XY])^2}{\mathbb E[Y^2]}.
$$

Rearrange. If $\mathbb E[Y^2]=0$, then $Y=0$ almost surely and both sides vanish. The product $XY$ is integrable because $2|XY|\le X^2+Y^2$.

</details>

Applying this to centered variables gives $|\operatorname{Cov}(X,Y)|\le\sigma_X\sigma_Y$. When both standard deviations are nonzero, correlation lies in $[-1,1]$. That range is a consequence, not an arbitrary convention.

## 4 · Jensen: averages and curved functions

**The version proved here.** $X$ takes values in an interval with $\mathbb E|X|<\infty$; $\phi$ is convex, differentiable at $\mu=\mathbb E X$, and $\mathbb E|\phi(X)|<\infty$.

$$
\phi(\mathbb E X)\le\mathbb E[\phi(X)].
$$

A convex function lies above its tangent:

$$
\phi(x)\ge\phi(\mu)+\phi'(\mu)(x-\mu).
$$

Take expectations. The last term vanishes because $\mathbb E[X-\mu]=0$. Supporting lines extend the proof to general convex functions; differentiating is not a proof of the nondifferentiable case.

For $\phi(x)=x^2$, we obtain $(\mathbb E X)^2\le\mathbb E[X^2]$. For positive variables, the concavity of $\log$ reverses the inequality: when the relevant expectations are finite, $\mathbb E[\log X]\le\log\mathbb E[X]$.

<details markdown="1">
<summary>Exercise: why is the average reciprocal at least the reciprocal average?</summary>

For $X>0$, assume $\mathbb E X$ and $\mathbb E[1/X]$ are finite. The function $\phi(x)=1/x$ is convex on the positive axis since $\phi''(x)=2/x^3>0$. Hence $\mathbb E[1/X]\ge1/\mathbb E[X]$.

For equally likely values 1 and 3, the two sides are $2/3$ and $1/2$. Averaging before taking a reciprocal changes the result.

</details>

## 5 · The Chernoff approach: exponentiate, then apply Markov

If $\lambda>0$ and the moment-generating function (MGF) $M_X(\lambda)=\mathbb E[e^{\lambda X}]$ is finite,

$$
P(X\ge a)\le e^{-\lambda a}M_X(\lambda).
$$

Monotonicity gives $\{X\ge a\}=\{e^{\lambda X}\ge e^{\lambda a}\}$. Every admissible $\lambda$ provides a bound; choose the best. This is a **method**, not a guarantee of a tight bound for every distribution.

For a sum $S$ of $n$ independent Bernoulli($p$) variables,

$$
M_S(\lambda)=(1-p+pe^\lambda)^n,\qquad
P(S\ge a)\le\inf_{\lambda>0}e^{-\lambda a}(1-p+pe^\lambda)^n.
$$

The product uses independence. Some heavy-tailed distributions have infinite MGF for every positive $\lambda$, preventing a useful upper-tail bound through this argument.

## Choosing a tool

| Information available | First tool to try |
| --- | --- |
| Nonnegativity and a mean | Markov |
| Mean and variance | Chebyshev |
| A product of two variables | Cauchy–Schwarz |
| A function around an expectation, or vice versa | Jensen; check convexity or concavity |
| Independent sum with a usable MGF | Exponential Markov / Chernoff |

Continue with [LLN and CLT](limits.en.md). References: [MIT 6.041: Chebyshev and weak LLN](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/resources/mit6_041f10_l19/), [MIT 18.440: Jensen](https://ocw.mit.edu/courses/18-440-probability-and-random-variables-spring-2014/resources/mit18_440s14_lecture32/).
