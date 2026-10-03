# 概率不等式：先别急着算精确答案

**中文** · [English](inequalities.en.md)

> 阅读时间：约 9 分钟 · 前置：[期望与方差](expectation-proofs.md) · 最近审阅：2026-10

有时不知道完整分布，只知道均值或方差，还能说什么？这几条不等式的价值就在这里。它们不是新的概率定义，而是从非负性、平方和凸性推出来的。

## 1 · Markov：大值不能太常出现

**条件。** $X\ge0$，$\mathbb E[X]<\infty$，$a>0$。

$$
P(X\ge a)\le\frac{\mathbb E[X]}a.
$$

证明只要一个逐点比较：

$$
X\ge a\mathbf1_{\{X\ge a\}}
\quad\Longrightarrow\quad
\mathbb E[X]\ge aP(X\ge a).
$$

若请求耗时非负，平均为 100 ms，那么超过或等于 500 ms 的概率至多为 $0.2$。这并没有告诉我们它实际有多大。

<details markdown="1">
<summary>非负这个条件能删吗？什么时候上界刚好取到？</summary>

不能删。若 $X$ 等概率取 $-1,1$，均值为 0，但 $P(X\ge1)=1/2$，不可能小于等于 0。

若 $X$ 只取 $0,a$，则 $\mathbb E[X]/a=P(X=a)$，上界刚好取到。只知道非负与均值时，不能指望总能得到更小的上界。

</details>

## 2 · Chebyshev：把偏离均值变成非负量

**条件。** $\mathbb E[X]=\mu$，$\operatorname{Var}(X)=\sigma^2<\infty$，$t>0$。

$$
P(|X-\mu|\ge t)\le\frac{\sigma^2}{t^2}.
$$

证明：把 Markov 用在 $Z=(X-\mu)^2$ 上，阈值用 $t^2$。因此偏离至少 3 个标准差的概率不超过 $1/9$（$\sigma>0$ 时）。

它不假设正态，所以不能把它和正态分布的“3σ 规则”混为一谈。若算出来的上界大于 1，改用 1 即可；那说明已有信息不够强。

## 3 · Cauchy–Schwarz：平方不能是负数

**条件。** $\mathbb E[X^2],\mathbb E[Y^2]<\infty$。

$$
(\mathbb E[XY])^2\le\mathbb E[X^2]\mathbb E[Y^2].
$$

<details markdown="1">
<summary>展开证明：构造一个关于 t 的平方</summary>

令 $q(t)=\mathbb E[(X-tY)^2]\ge0$。若 $\mathbb E[Y^2]>0$，选择使这个二次式最小的 $t=\mathbb E[XY]/\mathbb E[Y^2]$：

$$
0\le \mathbb E[X^2]-\frac{(\mathbb E[XY])^2}{\mathbb E[Y^2]}.
$$

移项即可。若 $\mathbb E[Y^2]=0$，则 $Y=0$ 几乎处处，左右两边都是 0。这里 $XY$ 可积，因为 $2|XY|\le X^2+Y^2$。

</details>

应用到中心化后的 $X-\mathbb E X$ 与 $Y-\mathbb E Y$，可得 $|\operatorname{Cov}(X,Y)|\le\sigma_X\sigma_Y$。两边标准差非零时，相关系数就在 $[-1,1]$ 内。这个范围不是约定出来的。

## 4 · Jensen：平均与弯曲的函数

**这里证明的版本。** $X$ 取值于一个区间，$\mathbb E|X|<\infty$；$\phi$ 为凸函数，在 $\mu=\mathbb E X$ 处可微，且 $\mathbb E|\phi(X)|<\infty$。

$$
\phi(\mathbb E X)\le\mathbb E[\phi(X)].
$$

凸函数在切线上方：

$$
\phi(x)\ge\phi(\mu)+\phi'(\mu)(x-\mu).
$$

两边取期望，最后一项因 $\mathbb E[X-\mu]=0$ 消失。一般的凸函数可用支撑直线扩展；这里不把不可微情形假装成求导证明。

例如 $\phi(x)=x^2$ 得 $(\mathbb E X)^2\le\mathbb E[X^2]$。对正变量，用凹函数 $\log$ 则方向反过来：在相关期望有限时，$\mathbb E[\log X]\le\log\mathbb E[X]$。

<details markdown="1">
<summary>小练习：为什么平均倒数不小于均值的倒数？</summary>

对 $X>0$，假设 $\mathbb E X$ 与 $\mathbb E[1/X]$ 有限。函数 $\phi(x)=1/x$ 在正半轴凸，因为 $\phi''(x)=2/x^3>0$。所以 $\mathbb E[1/X]\ge1/\mathbb E[X]$。

等概率取 1 和 3 时，左边是 $2/3$，右边是 $1/2$。不要先平均再取倒数。

</details>

## 5 · Chernoff 的入口：指数化以后再用 Markov

若 $\lambda>0$ 且矩母函数（MGF）$M_X(\lambda)=\mathbb E[e^{\lambda X}]$ 有限：

$$
P(X\ge a)\le e^{-\lambda a}M_X(\lambda).
$$

因为指数单调，$\{X\ge a\}=\{e^{\lambda X}\ge e^{\lambda a}\}$。对每个可用的 $\lambda$ 都有上界，再选择最小的。这是一个**方法**，不是对所有分布都自动给出很紧的界。

例如 $S$ 是 $n$ 个独立 Bernoulli($p$) 的和：

$$
M_S(\lambda)=(1-p+pe^\lambda)^n,\qquad
P(S\ge a)\le\inf_{\lambda>0}e^{-\lambda a}(1-p+pe^\lambda)^n.
$$

这里的乘积用到了独立。某些重尾分布在所有正 $\lambda$ 下 MGF 都发散，就不能这样得到有效的上尾界。

## 选择哪个工具

| 已知的信息 | 优先试 |
| --- | --- |
| 非负 + 均值 | Markov |
| 均值 + 方差 | Chebyshev |
| 两个变量的乘积 | Cauchy–Schwarz |
| 函数包着期望，或反过来 | Jensen，先检查凸凹 |
| 独立求和 + 可用的 MGF | 指数 Markov / Chernoff |

继续：[大数定律与 CLT](limits.md)。参考：[MIT 6.041：Chebyshev 与弱大数定律](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/resources/mit6_041f10_l19/)、[MIT 18.440：Jensen](https://ocw.mit.edu/courses/18-440-probability-and-random-variables-spring-2014/resources/mit18_440s14_lecture32/)。
