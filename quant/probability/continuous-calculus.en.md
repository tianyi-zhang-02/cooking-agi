# Continuous probability and calculus: draw the region first

[中文](continuous-calculus.md) · **English**

> Reading time: ~8 min · Prerequisites: [PDFs and CDFs](distributions.en.md), basic calculus · Last reviewed: 2026-10

Writing an integral immediately often makes a continuous problem harder. Start with the support and the event region. This chapter practices area, transformations, and logarithms that simplify powers.

## 1 · When area represents probability

If $X,Y$ are independent and uniform on $[0,T]$, their joint density is $1/T^2$. Thus for a region $D$,

$$
P((X,Y)\in D)=\iint_D\frac1{T^2}\,dx\,dy
=\frac{\operatorname{Area}(D)}{T^2}.
$$

**Joint uniformity matters; two dimensions alone do not justify area ratios.** Arrivals clustered at certain times, or coordinated between people, may have nonuniform density or no two-dimensional density at all.

## 2 · Meeting times: subtract the two corners

Two people arrive independently and uniformly within $[0,T]$, each waiting $w$, where $0\le w\le T$. They meet exactly when $|X-Y|\le w$.

The missed-meeting region consists of two right triangles, each with legs $T-w$. Therefore,

$$
P(\text{meet})=1-\frac{2\cdot\frac12(T-w)^2}{T^2}
=1-\left(1-\frac wT\right)^2.
$$

For $T=60,w=15$, this is $7/16$. Check the endpoints: zero waiting gives probability 0; waiting the entire interval gives 1.

<details markdown="1">
<summary>What if one waits 10 minutes and the other waits 20?</summary>

A arrives at $X$ and waits $a$; B arrives at $Y$ and waits $b$, with $0\le a,b\le T$. They meet when $-b\le Y-X\le a$.

The two missed-meeting triangles have legs $T-a$ and $T-b$:

$$
P(\text{meet})=1-\frac{(T-a)^2+(T-b)^2}{2T^2}.
$$

For $T=60,a=10,b=20$, the result is $31/72$. Replacing the waits with their average changes the squared terms and the answer.

</details>

## 3 · Transformations: start with the CDF

Let $X\sim\operatorname{Uniform}(0,1)$ and $Y=X^2$. First identify the support $0\le Y\le1$. For $0\le y\le1$,

$$
F_Y(y)=P(X^2\le y)=P(X\le\sqrt y)=\sqrt y.
$$

Outside the interval, the CDF is 0 or 1. Differentiating inside gives

$$
f_Y(y)=\frac1{2\sqrt y},\qquad 0<y<1.
$$

The density is large near 0 but integrates to 1. It yields $\mathbb E[Y]=\int_0^1y/(2\sqrt y)\,dy=1/3$, agreeing with the direct calculation of $\mathbb E[X^2]$.

<details markdown="1">
<summary>What changes if X is Uniform(-1,1)?</summary>

Now $X^2\le y$ means $-\sqrt y\le X\le\sqrt y$; keeping only the positive root would be wrong. Its probability is still $(2\sqrt y)/2=\sqrt y$, so this particular change produces the same distribution of $Y$.

A nonmonotone transformation generally requires every inverse branch. The CDF method makes missing branches easier to notice.

</details>

## 4 · Compare powers using logarithms and monotonicity

Compare $e^\pi$ with $\pi^e$. Since $\log$ is strictly increasing, compare $\pi$ with $e\log\pi$, or equivalently $\log e/e$ with $\log\pi/\pi$.

For $g(x)=\log x/x$, $x>0$,

$$
g'(x)=\frac{1-\log x}{x^2}.
$$

It increases on $(0,e)$ and decreases on $(e,\infty)$. Since $\pi>e$, $g(e)>g(\pi)$, giving $e^\pi>\pi^e$.

The same derivative locates the maximum of $x^{1/x}$. Its logarithm is $g(x)$, so the maximum occurs at $x=e$ and equals $e^{1/e}$. **Check derivative signs across the domain; a stationary point alone is not enough.**

## 5 · When an approximation is justified

We often use $\log(1+u)\approx u$. Instead of only saying “$u$ is small,” control the error. Taylor's theorem gives some $\xi$ between 0 and $u$ such that

$$
\log(1+u)=u-\frac{u^2}{2(1+\xi)^2}.
$$

For $|u|\le1/2$, we have $|\log(1+u)-u|\le2u^2$. The neglected error is second order.

<details markdown="1">
<summary>Use the bound to show that (1+c/n)^n tends to exp(c)</summary>

Fix real $c$. For sufficiently large $n$, $|c/n|\le1/2$ and the base is positive. Thus

$$
\left|n\log(1+c/n)-c\right|\le\frac{2c^2}{n}\to0.
$$

Continuity of the exponential gives $(1+c/n)^n\to e^c$. Control the error before passing the limit through a continuous function.

</details>

## Next

[Markov chains and waiting times](markov-chains.en.md). Compare continuous distributions and transformations with [MIT 6.041 lectures 8–10](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/pages/lecture-notes/). The calculus examples above use derivative signs, Taylor remainders, and continuity.
