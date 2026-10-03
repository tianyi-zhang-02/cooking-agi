# 综合复习：记条件，比记公式更重要

**中文** · [English](review.en.md) · [复习总览](README.md)

> 阅读时间：约 10 分钟 · 适合复习时边写边看 · 最近审阅：2026-10

这一页不是新的知识清单。挑几个问题，用纸写或直接讲出来；卡住再回对应章节。先不追求速度，能解释为什么这样做更重要。

## 1 · 常用定理的条件速查

| 想用什么 | 至少先确认 | 经常混淆的地方 |
| --- | --- | --- |
| 等可能计数 | 有限结果是否真的等可能 | “随机”不等于均匀 |
| 期望线性性 | 有限绝对期望，或非负求和的适当版本 | 不需要独立 |
| 方差相加 | 二阶矩与协方差项 | 仅同分布不够 |
| Bayes | 观察事件概率正、观察机制明确 | 基准率和采样程序不能省 |
| MGF 相乘 | 独立、MGF 在所用区间有限 | 所有矩存在也不够 |
| Jensen | 凸性、变量与函数值的可积条件 | 方向由凸凹决定 |
| WLLN / CLT | 使用哪个版本的独立性和矩条件 | CLT 不承诺 n=30 就够 |
| 有界停止 | 鞅、停止时间、有界 | 最终会停不等于可以停 |
| Wald | iid 可积增量、停止时间、E[T]<∞ | T 不能偷看未来 |
| t 区间 | 精确版本用 iid 正态、非零方差 | 一般样本只是近似 |
| Newton 二次收敛 | 简单根附近、光滑、导数非零 | 全局不保证收敛 |
| 风险中性定价 | 模型与无套利、可容许复制等条件 | q 不是真实上涨概率 |

这些是本组所用版本，不是每条定理最弱的可能假设。详细条件回相应章节看。

## 2 · 先做 4 道“换条件”的题

<details markdown="1">
<summary>1. 两次 Bernoulli 的边缘成功率都是 1/2，至少一次成功一定是 3/4 吗？</summary>

不是。独立时 3/4；若两次完全相同，只有 1/2；若第二次永远是第一次的反面，则为 1。边缘分布不决定联合分布。回[条件与独立](probability/conditional.md)。

</details>

<details markdown="1">
<summary>2. N=20、K=5，不放回抽 n=4，目标数的期望与方差？</summary>

期望 1；方差 $4(1/4)(3/4)(16/19)=12/19$。若有放回且每次均匀，方差变为 3/4。不同模型下均值一样，不代表波动一样。回[计数](probability/counting.md)。

</details>

<details markdown="1">
<summary>3. 无限期等到公平游走首次 +1，能直接用停止定理算 E[S_T]=0 吗？</summary>

不能。T 虽然几乎必然有限，但期望无限；需要的极限控制失效。停止时 S_T 恒为 1，已经足够反驳不加条件的套用。回[鞅](processes/martingales.md)。

</details>

<details markdown="1">
<summary>4. 把样本按时间随机打散，为什么测试分数可能特别好？</summary>

相邻样本共享信息，未来特征或标签计算也可能泄漏进训练；随机切分改变了真正要评估的未来泛化问题。先固定可用信息的时刻，再做时间切分。不能仅凭“训练测试没有同一行”就断言无泄漏。回[统计](methods/statistics.md)。

</details>

## 3 · 4 道从头推的小题

<details markdown="1">
<summary>5. 独立指数等待 rate 为 a,b，谁先到？</summary>

$P(X<Y)=\int_0^\infty ae^{-at}e^{-bt}dt=a/(a+b)$；最早到达等待为 Exp(a+b)。a、b 要正，且独立。a 很大时 X 先到的概率应接近 1。回[分布工具](probability/distribution-toolkit.md)。

</details>

<details markdown="1">
<summary>6. 为什么协方差矩阵不能有负特征值？</summary>

对任意 v，$v^\top\Sigma v=\operatorname{Var}(v^\top X)\ge0$。若存在负特征值，取对应特征向量就矛盾。二阶矩需要有限。回[线代](methods/linear-algebra.md)。

</details>

<details markdown="1">
<summary>7. 一次有概率 p 成功，直到成功；为什么方差不等于均值？</summary>

Geometric(1,2,…) 的均值为 1/p、方差为 (1−p)/p²；Poisson 才具有均值等于方差的性质。可用首次递推推二阶矩，或对几何级数求导。p=1 时等待确定为 1，方差为 0。回[基础练习第 4 题](probability/practice.md)。

</details>

<details markdown="1">
<summary>8. 一步股票价格 100→120/80，R=1.05，call payoff 20/0，为什么不用真实上涨率？</summary>

半股与借款 800/21 在两个状态都复制 payoff。价格为 250/21，由复制而非预测决定。改变真实上涨率不会改变这套理想模型里的复制成本。回[无套利](finance/README.md)。

</details>

## 4 · 再挑 4 道，把知识串起来

<details markdown="1">
<summary>9. n 个 Uniform(0,1) 的最大值均值为什么是 n/(n+1)？</summary>

最大值≤x 需要全部≤x，CDF 是 xⁿ。密度 nxⁿ⁻¹，积分 $\int_0^1x\cdot nx^{n-1}dx=n/(n+1)$。n=1 时是 1/2，n 增大趋近 1。回[顺序统计](probability/joint-and-order.md)。

</details>

<details markdown="1">
<summary>10. Monte Carlo 结果波动大，你有哪些办法？</summary>

先检查目标、实现、尾部与方差是否有限，再增加样本或利用结构做 control variates、antithetic、分层或重要性采样。比较同样计算预算下的误差；不要只展示一次好看的随机种子。回[数值方法](methods/numerical-and-coding.md)。

</details>

<details markdown="1">
<summary>11. 为什么 d(W²) 比 2W dW 多一个 dt？</summary>

平方增量累积不消失：均分区间时平方和 L² 趋于时间长度。Taylor 的二阶项留下 dt。这个解释不能取代一般 Itô 定理证明，但能检查具体公式。回[Itô](processes/brownian-ito.md)。

</details>

<details markdown="1">
<summary>12. 掷骰子最多 3 次，为什么第一轮只收 5、6？</summary>

只剩 1 次价值 3.5；剩 2 次价值 17/4=4.25；第一次看到 x，就比较 x 与 4.25。因此只收 5、6，最优期望 14/3。回[动态规划](processes/dynamic-programming.md)。

</details>

## 5 · 用 1 分钟说清楚一道题

可以照这个顺序，不必背成固定话术：

“我先假设……；把目标写成……。直接算比较麻烦，所以先用……拆开。这里需要……这个条件。算出来是……；当参数取……时退化到……，和直觉一致。如果取消……，这一步就不能用了。”

复习时把“我记得答案”替换成“我知道哪一步让问题变简单”。已经能做到，就去做变式，而不是重复读相同答案。

本页练习是通用教学例子与自编变式，不对应任何未公开面试题。继续到[覆盖表](README.md)挑缺的部分。
