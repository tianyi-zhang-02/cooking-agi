# 条件期望：先把一部分随机性固定住

**中文** · [English](conditioning-proofs.en.md)

> 阅读时间：约 8 分钟 · 前置：[条件概率](conditional.md)、[期望](expectation-proofs.md) · 最近审阅：2026-10

问题里有好几层随机性时，不一定要一次算完。先固定一层，里面算清楚，再按外层的概率加回来。全概率、全期望、全方差其实都在做这件事。

## 1 · 全概率与 Bayes：从不交分组开始

假设 $A_1,A_2,\ldots$ 构成有限或可数个事件的划分：两两不交、并起来为 $\Omega$。只对 $P(A_i)>0$ 的组写条件概率，零概率组不贡献交集概率。

$$
P(B)=\sum_iP(B\cap A_i)=\sum_iP(B\mid A_i)P(A_i).
$$

第一步来自可加性，第二步来自条件概率的定义。若 $P(B)>0$，再把同一个交集从另一边表达：

$$
P(A_j\mid B)=\frac{P(B\mid A_j)P(A_j)}{\sum_iP(B\mid A_i)P(A_i)}.
$$

Bayes 没有多加一条神秘规则，只是把“已知什么”换了方向。

## 2 · 条件变了，问题也变了

从 52 张标准牌中不放回抽 2 张，里面有 4 张 A。问“两张都是 A”，下面两种信息不能混用。

| 你知道什么 | 条件概率 | 分母在数什么 |
| --- | --- | --- |
| 第一张是 A | $3/51=1/17$ | 剩下的牌 |
| 至少一张是 A | $\binom42/(\binom{52}2-\binom{48}2)=1/33$ | 所有包含 A 的无序牌对 |

第二行分母为 $198$，分子为 $6$。不是哪一种算法更聪明，而是看到的信息不同。若有人按某个规则选一张展示，还要把展示规则写进模型。

## 3 · 全期望：组内平均，再按组加权

先看离散的 $Y$。令 $m(y)=\mathbb E[X\mid Y=y]$。$m(y)$ 是一个数，$m(Y)=\mathbb E[X\mid Y]$ 却仍是随机变量，因为现在还不知道落在哪一组。

**定理。** 若 $\mathbb E|X|<\infty$，

$$
\mathbb E[\mathbb E[X\mid Y]]=\mathbb E[X].
$$

<details markdown="1">
<summary>展开离散证明</summary>

对所有正概率的 $y$ 求和：

$$
\begin{aligned}
\sum_y\mathbb E[X\mid Y=y]P(Y=y)
&=\sum_y\sum_xxP(X=x\mid Y=y)P(Y=y)\\
&=\sum_{x,y}xP(X=x,Y=y)\\
&=\mathbb E[X].
\end{aligned}
$$

没有要求 $X,Y$ 独立。连续联合密度情形把求和换成积分；一般形式来自条件期望的定义。这里先不展开测度论构造。

</details>

**例子。** 先公平地选一种硬币，再抛 10 次。两种硬币的正面概率分别是 $0.2$、$0.8$。设正面次数为 $S$，则两组条件均值分别为 2、8，总均值为 $(2+8)/2=5$。

这里的 10 次抛掷是**给定硬币后独立**，不是无条件独立：知道前几次很多正面，会改变你对选中哪种硬币的判断。

## 4 · 全方差：波动从哪里来

**定理。** 若 $\mathbb E[X^2]<\infty$，

$$
\operatorname{Var}(X)=\mathbb E[\operatorname{Var}(X\mid Y)]+\operatorname{Var}(\mathbb E[X\mid Y]).
$$

左边是整体波动。右边两项分别是组内波动的平均，以及不同组均值之间的波动。

<details markdown="1">
<summary>展开证明：为什么交叉项消失？</summary>

令 $m(Y)=\mathbb E[X\mid Y]$，$\mu=\mathbb E[X]$。先拆偏差：

$$
X-\mu=(X-m(Y))+(m(Y)-\mu).
$$

展开平方，交叉项为零，因为

$$
\mathbb E[(X-m(Y))(m(Y)-\mu)]
=\mathbb E[(m(Y)-\mu)\mathbb E[X-m(Y)\mid Y]]=0.
$$

给定 $Y$ 后，$m(Y)-\mu$ 已经固定，而剩余误差的条件均值为 0。两项平方取期望，正好就是全方差的两项。

</details>

回到硬币例子：每组方差都是 $10\times0.2\times0.8=1.6$，组均值 2、8 围绕 5 的方差是 9。因此总方差为 **10.6**，不是 Binomial($10,0.5$) 的 2.5。混合分布不能只保留平均成功率。

## 5 · 容易多走的一步

全期望说“可以分组算平均”，不代表任意函数都能搬过期望。例如：

$$
\mathbb E[(\mathbb E[X\mid Y])^2]\ne(\mathbb E[X])^2
$$

通常不相等，差值恰好是 $\operatorname{Var}(\mathbb E[X\mid Y])$。也不能仅凭“给定 Y 后独立”就删掉条件。

<details markdown="1">
<summary>小练习：若 X 完全由 Y 决定，全方差还剩哪一项？</summary>

此时 $\mathbb E[X\mid Y]=X$，条件方差为 0。所有波动都在不同的 $Y$ 之间，所以 $\operatorname{Var}(X)=0+\operatorname{Var}(X)$。反过来，如果 $X$ 与 $Y$ 独立，组均值恒定，第二项为 0。

</details>

## 接下来

[不等式：算不准，也能控制范围](inequalities.md)。进一步对照：[MIT 6.041：条件期望与全方差](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/resources/mit6_041f10_l12/)。
