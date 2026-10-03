# 计数：先弄清楚，到底在数什么

**中文** · [English](counting.en.md) · [复习总览](../README.md)

> 阅读时间：约 9 分钟 · 前置：事件与等可能模型 · 最近审阅：2026-10

同样是“抽 3 个”，放不放回、看不看顺序，答案能差很多。计数题最容易错的往往不是公式，而是分子和分母用了两套规则。

## 1 · 先回答 4 个问题

对象是否可区分？位置是否可区分？能不能重复？顺序算不算区别？例如，从 8 本不同的书里选 3 本：

| 模型 | 数量 | 为什么 |
| --- | --- | --- |
| 有序、不重复 | $8\cdot7\cdot6$ | 每个位置的选择数依次减少 |
| 无序、不重复 | $\binom83=56$ | 每组被有序模型数了 $3!$ 次 |
| 有序、可重复 | $8^3$ | 每个位置都有 8 个选择 |
| 无序、可重复 | $\binom{10}3=120$ | 给 8 类分配 3 个名额 |

最后一行不能简单用 $8^3/3!$：AAA 只有一种排列，ABC 却有 6 种。**只有每个对象被重复计数的次数相同，才能统一除掉。**

## 2 · 隔板法到底在做什么

非负整数解 $x_1+\cdots+x_m=n$，把 n 个星星与 m−1 根隔板排成一行。第 i 段的星星数就是 $x_i$，这个对应可逆，因此：

$$
\#\{x_i\ge0:\sum_i x_i=n\}=\binom{n+m-1}{m-1}.
$$

若每个 $x_i\ge1$，先给每类 1 个，再分剩下的 n−m 个，得到 $\binom{n-1}{m-1}$，要求 $n\ge m$。

<details markdown="1">
<summary>小练习：10 个相同名额分给 3 组，每组至少 2 个</summary>

先给出 6 个，剩下 4 个随意分。答案是 $\binom{4+3-1}{3-1}=15$。上限约束不能这样一次平移解决；可以接着用容斥，减掉超限的情况。

</details>

## 3 · 不放回抽样：为什么不是 Binomial

N 个对象中 K 个属于目标类，均匀无放回抽 n 个，目标数 X 的分布是 Hypergeometric：

$$
P(X=k)=\frac{\binom Kk\binom{N-K}{n-k}}{\binom Nn}.
$$

可行范围是 $\max(0,n-N+K)\le k\le\min(n,K)$。分子分母都在数无序子集，所以规则一致。

设 $p=K/N$。用抽取位置的指示变量，$E[X]=np$。但抽到了目标会降低下一次再抽到的概率，因此有负协方差。

<details markdown="1">
<summary>推导有限总体修正项</summary>

不同位置 i、j 同为目标的概率是 $K(K-1)/(N(N-1))$，于是：

$$
\operatorname{Cov}(I_i,I_j)=-\frac{p(1-p)}{N-1},\qquad
\operatorname{Var}(X)=np(1-p)\frac{N-n}{N-1}.
$$

这里 $N>1$。n=N 时方差为 0，正好符合“全拿走后数量已确定”。样本占总体很小才适合考虑 Binomial 近似。

</details>

## 4 · 生日碰撞与占用：事件和个数别混

m 个人的生日独立均匀落在 d 天。m≤d 时，无碰撞概率是：

$$
P(\text{no collision})=\prod_{j=0}^{m-1}\left(1-\frac jd\right).
$$

m>d 则为 0。小的 j/d 下，用 $\log(1-u)\approx-u$ 得到无碰撞概率约为 $\exp[-m(m-1)/(2d)]$；当高阶项累计不可忽略时，近似会差。

相同模型也可理解为 m 个球进入 d 个箱子。用“某个箱子有没有球”的指示变量：

$$
E[\text{occupied boxes}]=d\left[1-\left(1-\frac1d\right)^m\right].
$$

这与碰撞的人对数 $\binom m2/d$ 是不同的量。一箱 3 个球，贡献 1 个占用箱，却贡献 3 对碰撞。

## 5 · 容斥：从固定点数到完全错排

均匀随机排列 n 个元素。$A_i$ 表示位置 i 保持不动。指定 k 个位置固定的排列数为 $(n-k)!$，所以：

$$
P(\text{no fixed points})=\sum_{k=0}^n\frac{(-1)^k}{k!}.
$$

<details markdown="1">
<summary>为什么每项恰好是 1/k!？</summary>

容斥第 k 层有 $\binom nk$ 种选法；每个交集有 $(n-k)!$ 个排列。除以总数 n!，就是 $1/k!$。n 越大趋近 $e^{-1}$，但有限 n 不能直接写等号。

</details>

## 6 · 收集齐全：不要假设各阶段等难

每次独立均匀拿到 n 种中的 1 种，已收集 k 种时，新品概率是 $(n-k)/n$。这段等待的期望为 $n/(n-k)$，逐段加起来：

$$
E[T]=n\sum_{j=1}^n\frac1j=nH_n.
$$

这里只用期望线性性；无需先证明各段独立。不均匀概率下，“已经有几种”通常不足以描述状态，还得知道缺哪几种。

回头检查：n=1 时只要 1 次；最后一种平均要 n 次，所以最后一小段可能很慢。相关思路见[期望证明](expectation-proofs.md)。

继续：[常用分布怎么推](distribution-toolkit.md)。背景读物：[MIT Mathematics for Computer Science 的计数章节](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/pages/readings/)。
