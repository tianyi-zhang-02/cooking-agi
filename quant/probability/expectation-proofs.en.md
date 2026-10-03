# Expectation and variance: where independence matters

[中文](expectation-proofs.md) · **English**

> Reading time: ~8 min · Prerequisite: [Random variables and distributions](distributions.en.md) · Last reviewed: 2026-10

“Expectations add” is right. Following it with “variances add” may not be. Separating the derivations reveals exactly where independence enters.

## 1 · Linearity does not factor the joint distribution

**Theorem.** If $\mathbb E|X|,\mathbb E|Y|<\infty$, then for constants $a,b$,

$$
\mathbb E[aX+bY]=a\mathbb E[X]+b\mathbb E[Y].
$$

For discrete variables with joint PMF $p(x,y)$,

$$
\begin{aligned}
\mathbb E[aX+bY]
&=\sum_{x,y}(ax+by)p(x,y)\\
&=a\sum_x x\sum_y p(x,y)+b\sum_y y\sum_xp(x,y)\\
&=a\mathbb E[X]+b\mathbb E[Y].
\end{aligned}
$$

We used linearity of sums and marginalization, **not** $p(x,y)=p_X(x)p_Y(y)$. Absolute integrability justifies rearrangement. For variables with a joint density, replace the sums by integrals. The general result comes from linearity of integration.

Nonnegative variables also admit an extended version allowing $+\infty$. That does not make $\infty-\infty$ a valid calculation.

## 2 · Expectation of a function without its new distribution

For a discrete variable, if $\mathbb E|g(X)|<\infty$,

$$
\mathbb E[g(X)]=\sum_xg(x)P(X=x).
$$

Often called LOTUS, this follows by grouping outcomes mapping to the same $z=g(x)$:

$$
\sum_z zP(g(X)=z)=\sum_z z\sum_{x:g(x)=z}P(X=x)=\sum_xg(x)P(X=x).
$$

If $X$ is uniform on $\{-1,0,1\}$, $\mathbb E[X]=0$ but $\mathbb E[X^2]=2/3$. **Applying a nonlinear function to an average generally changes the answer.**

## 3 · Indicators and the tail-sum formula

For a nonnegative integer-valued $N$, outcome by outcome,

$$
N=\sum_{k=1}^{\infty}\mathbf1_{\{N\ge k\}}.
$$

When $N=3$, the right side is $1+1+1+0+\cdots$. Therefore,

$$
\mathbb E[N]=\sum_{k=1}^{\infty}P(N\ge k).
$$

Finite linearity alone does not justify an infinite sum. Nonnegative summands allow monotone convergence, or Tonelli's theorem, to interchange expectation and summation. Infinity is permitted. The analogous formula for nonnegative continuous variables is $\mathbb E[X]=\int_0^\infty P(X>t)\,dt$.

**Example: waiting for the first success.** Independent repeated trials have success probability $p\in(0,1]$. Let $T$ include the successful trial. Since $P(T\ge k)=(1-p)^{k-1}$,

$$
\mathbb E[T]=\sum_{k=1}^{\infty}(1-p)^{k-1}=\frac1p.
$$

<details markdown="1">
<summary>Derive the same answer by conditioning on the first trial</summary>

Write $m=\mathbb E[T]$. Spend one trial; after failure, the remaining wait has the original distribution. Thus $m=1+(1-p)m$, giving $m=1/p$. The tail-sum argument already established finiteness, so the rearrangement is justified.

A recurrence for an expectation does not, by itself, prove that the expectation is finite.

</details>

## 4 · Variance: keep the cross term

Assume finite second moments. Expand the definition:

$$
\operatorname{Var}(X)=\mathbb E[(X-\mathbb E X)^2]=\mathbb E[X^2]-(\mathbb E X)^2.
$$

Expand the square for a sum in the same way:

$$
\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y).
$$

Here $\operatorname{Cov}(X,Y)=\mathbb E[XY]-\mathbb E[X]\mathbb E[Y]$. Independence implies $\mathbb E[XY]=\mathbb E[X]\mathbb E[Y]$, making covariance zero. Zero covariance does not imply independence.

| Case | Variance of the sum | Reason |
| --- | --- | --- |
| $Y=X$ | $4\operatorname{Var}(X)$ | Perfectly aligned copies |
| $Y=-X$ | $0$ | Exact cancellation |
| Independent $X,Y$ | $\operatorname{Var}(X)+\operatorname{Var}(Y)$ | Zero cross term |

<details markdown="1">
<summary>Find variables that are uncorrelated but dependent</summary>

Let $X$ be uniform on $\{-1,0,1\}$ and $Y=X^2$. Then $\mathbb E[X]=\mathbb E[X^3]=0$, so $\operatorname{Cov}(X,Y)=0$. But knowing $X=0$ forces $Y=0$, whereas $P(Y=0)=1/3$. They are not independent.

</details>

## 5 · Derive a familiar result again

The sum $S$ of independent Bernoulli($p$) variables $I_1,\ldots,I_n$ is Binomial($n,p$). Since $I_i^2=I_i$, each variance is $p-p^2$. Hence

$$
\mathbb E[S]=np,\qquad \operatorname{Var}(S)=np(1-p).
$$

The mean calculation does not require independence; the variance calculation uses it. If every $I_i$ copies the same coin result, the mean remains $np$ but the variance becomes $n^2p(1-p)$.

## Next

Continue with [Conditional expectation: average within groups first](conditioning-proofs.en.md). For comparison, see [MIT 6.041 lectures 5, 7, and 11](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/pages/lecture-notes/).
