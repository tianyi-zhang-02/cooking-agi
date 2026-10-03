# 常用分布：别只背均值和方差

**中文** · [English](distribution-toolkit.en.md) · [复习总览](../README.md)

> 阅读时间：约 10 分钟 · 前置：期望、方差、积分 · 最近审阅：2026-10

先问“随机机制是什么”，再选分布。固定次数里数成功、等到一次成功、固定时间里数到达，是 3 个不同的问题。

## 1 · 一张表，先统一参数

| 分布 | 参数与取值 | 均值 | 方差 |
| --- | --- | --- | --- |
| Bernoulli | p；0、1 | p | p(1−p) |
| Binomial | n 次独立 Bernoulli(p) | np | np(1−p) |
| Geometric | p；第几次首次成功，1、2… | 1/p | (1−p)/p² |
| Negative Binomial | r 次成功所需总次数 | r/p | r(1−p)/p² |
| Poisson | λ>0；0、1… | λ | λ |
| Exponential | **rate** λ>0 | 1/λ | 1/λ² |
| Gamma | shape α>0、**rate** λ>0 | α/λ | α/λ² |
| Normal | μ、σ²>0 | μ | σ² |
| Beta | α,β>0；0<x<1 | α/(α+β) | αβ/[(α+β)²(α+β+1)] |

Gamma 若改用 scale θ，θ=1/λ。Geometric 若数失败次数，则要减 1。很多“答案差一个常数”的问题只是约定不同。

## 2 · 几何与负二项：把等待拆成几段

独立试验成功率 p，首次成功在第 k 次的概率为 $(1-p)^{k-1}p$。前 k−1 次必须失败，第 k 次成功。

第 r 次成功发生在第 n 次，最后一次必须成功，前 n−1 次中选 r−1 次成功：

$$
P(T_r=n)=\binom{n-1}{r-1}p^r(1-p)^{n-r},\qquad n\ge r.
$$

相邻成功之间的等待是独立 Geometric，所以均值、方差都能相加。参数 p 每次改变时，就不能直接套这一版。

## 3 · Poisson：从小概率的许多机会来

设 $X_n\sim\operatorname{Binomial}(n,\lambda/n)$。对固定 k：

$$
P(X_n=k)=\frac{n(n-1)\cdots(n-k+1)}{k!}
\left(\frac\lambda n\right)^k
\left(1-\frac\lambda n\right)^{n-k}
\longrightarrow e^{-\lambda}\frac{\lambda^k}{k!}.
$$

这是固定 λ、n→∞ 的极限，不是“稀疏数据都服从 Poisson”。若事件成团出现或速率剧烈变化，方差可以远大于均值。

<details markdown="1">
<summary>不展开平方，推 E[X] 与 Var(X)</summary>

把求和索引移一位，得到 $E[X]=\lambda$；移两位得到 $E[X(X-1)]=\lambda^2$。再用 $X^2=X(X-1)+X$，方差为 λ。

</details>

## 4 · 指数分布与无记忆性

取 $P(T>t)=e^{-\lambda t}$，t≥0，则密度为 $\lambda e^{-\lambda t}$。因为：

$$
P(T>s+t\mid T>s)=\frac{e^{-\lambda(s+t)}}{e^{-\lambda s}}=e^{-\lambda t}.
$$

已经等了多久，不改变剩余等待的分布。这是模型性质，不是所有等待过程的常识。公交固定间隔到达就不同。

连续尾积分给 $E[T]=\int_0^\infty e^{-\lambda t}dt=1/\lambda$；同理 $E[T^2]=\int_0^\infty2t e^{-\lambda t}dt=2/\lambda^2$，方差便是 $1/\lambda^2$。

r 个同速率独立指数变量之和是 Gamma(r,λ)，也叫 Erlang。一般 Gamma 密度为：

$$
f(x)=\frac{\lambda^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\lambda x},\quad x>0.
$$

代换 u=λx，再用 $\Gamma(\alpha+1)=\alpha\Gamma(\alpha)$，即可推均值和二阶矩；Gamma 递推本身来自分部积分。

## 5 · MGF：和的分布为什么好算

矩母函数（moment-generating function）定义为 $M_X(t)=E[e^{tX}]$。若在 0 的某个邻域有限，可以在相应条件下求导得到各阶矩；独立变量满足 $M_{X+Y}=M_XM_Y$。

| 分布 | MGF | t 的范围 |
| --- | --- | --- |
| Bernoulli(p) | $1-p+pe^t$ | 所有实数 |
| Poisson(λ) | $\exp[\lambda(e^t-1)]$ | 所有实数 |
| Gamma(α,λ) | $(\lambda/(\lambda-t))^\alpha$ | t<λ |
| Normal(μ,σ²) | $\exp(\mu t+\sigma^2t^2/2)$ | 所有实数 |

因此独立 Poisson 的速率相加；独立同 rate Gamma 的 shape 相加；独立正态的均值、方差相加。正态 MGF 可通过指数里的配方推出来。

**MGF 不一定存在。** 对数正态的所有正整数矩有限，却在 t>0 没有有限 MGF。不能把“有均值”当成“可以随意用 MGF”。

## 6 · 正态变换与 Beta：接到统计

Z 为标准正态时，$\mu+\sigma Z$ 是 Normal(μ,σ²)。若 $Y=e^X$ 且 X 正态，Y 为 Lognormal，$E[Y]=e^{\mu+\sigma^2/2}$，不是 $e^{E[X]}$。

Beta 密度正比于 $p^{\alpha-1}(1-p)^{\beta-1}$。观测 Bernoulli 的 s 次成功、f 次失败后，乘上似然，指数相加，得到 Beta(α+s,β+f)。这是[贝叶斯更新](../methods/statistics.md)的入口。

<details markdown="1">
<summary>练习：独立 X~Exp(2)、Y~Exp(3)，min(X,Y) 是什么？</summary>

$P(\min(X,Y)>t)=P(X>t)P(Y>t)=e^{-5t}$，所以是 Exp(5)，均值 1/5。速率可以相加依赖独立；若两个等待完全相同就不成立。

</details>

继续：[联合分布与顺序统计量](joint-and-order.md)。参考：[MIT 6.041 课程讲义](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/pages/lecture-notes/)。
