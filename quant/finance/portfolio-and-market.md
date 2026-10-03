# 组合与市场：收益之外，还在承担什么

**中文** · [English](portfolio-and-market.en.md) · [复习总览](../README.md)

> 阅读时间：约 8 分钟 · 基础模型与拓展 · 最近审阅：2026-10

本篇只作概念复习，不推荐任何资产或交易策略。先算清组合的风险，再分清理论价格与真实成交。

## 1 · 两个资产不等于两份独立风险

收益向量均值 μ、协方差 Σ，权重 w：

$$
E[R_p]=w^\top\mu,\qquad
\operatorname{Var}(R_p)=w^\top\Sigma w.
$$

两个同波动率 σ、等权资产的组合方差为 $\sigma^2(1+\rho)/2$。ρ=1 时没有这个意义上的分散收益；ρ=0 时方差减半。ρ=−1 且等波动率时风险可完全抵消，但现实里估计相关性与未来相关性未必相同。

## 2 · 最小方差权重：从约束优化推出来

允许任意实数权重、Σ 正定、总权重为 1，最小化 $w^\top\Sigma w$。Lagrange 一阶条件 2Σw−λ1=0，因此：

$$
w^*=\frac{\Sigma^{-1}\mathbf1}{\mathbf1^\top\Sigma^{-1}\mathbf1}.
$$

加入不准做空、交易成本、目标收益等约束，答案会变。Σ 估得不准或接近奇异时，公式可放大误差；实际求解不要显式算逆。

<details markdown="1">
<summary>两个独立资产，方差分别 1 与 4，权重是多少？</summary>

最小化 $w^2+4(1-w)^2$，导数为 10w−8，所以第一个资产权重 0.8、第二个 0.2，组合方差 0.8。等权不是自动最稳。

</details>

## 3 · CAPM 与 Sharpe：知道模型在承诺什么

Sharpe ratio 用超额收益均值除以其标准差，需要统一频率。只有在适当独立、稳定等条件下，年化才有熟悉的 √时间缩放；自相关下不能机械套。

CAPM 在其均衡与市场假设下给 $E[R_i]-r_f=\beta_i(E[R_m]-r_f)$，其中 $\beta_i=\operatorname{Cov}(R_i,R_m)/\operatorname{Var}(R_m)$。这是模型关系，不是任何资产未来回报的保证，也不是“历史回归显著就能赚钱”。

## 4 · 债券价格为什么怕利率上升

确定正现金流 Cₖ、连续复利统一收益率 y：

$$
P(y)=\sum_k C_ke^{-yt_k},\qquad
D=-\frac{P'(y)}{P(y)}=\frac{\sum_kt_kC_ke^{-yt_k}}{P(y)}.
$$

D 是此约定下的 duration，价格小变动约为 ΔP/P≈−DΔy。二阶项使用 $P''/P$ 的 convexity。收益率曲线非平行移动时，一个 duration 不能概括全部风险。

## 5 · VaR 与尾部：分位点不等于最坏损失

把 L 定义为损失，α-VaR 是其 α 分位数。它没有告诉你超过阈值之后会多糟。Expected Shortfall 用最坏 1−α 比例的平均损失；对连续分布可写 $E[L\mid L\ge\operatorname{VaR}_\alpha]$，有原子时要按尾部概率质量正确处理。

分布假设、样本稀少和 regime change 都会影响尾部估计。先说清损失正负号、持有期和置信水平。

## 6 · 市场机制：价差不是白捡的钱

bid 是别人愿买的价格，ask 是别人愿卖的价格。主动成交通常跨越 spread，限价单则面对能否成交、排队和逆向选择风险。

逆向选择（adverse selection）可以这样理解：你的卖单刚被吃掉，价格随后上涨；成交本身可能说明对手信息或需求与你不同。库存风险、费用、延迟与部分成交也会影响结果。具体交易所规则不同，本篇不把一种撮合机制写成通用规则。

## 7 · 一句话串起来

概率模型给分布，统计从有限数据估参数，优化选权重，定价用复制关系，市场机制决定能否以想象中的价格成交。每一步都有自己的假设，不能把前一步的漂亮公式当成后一步的保证。

参考：[MIT Portfolio Theory](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote14/)。回到[覆盖表](../README.md)检查下一块。
