# Brownian motion 与 Itô：为什么多了半个二阶导

**中文** · [English](brownian-ito.en.md) · [复习总览](../README.md)

> 阅读时间：约 10 分钟 · 进阶入门 · 最近审阅：2026-10

这篇的目标是能推几个核心例子、读懂定价里的 Itô 步骤，不是假装 10 分钟完成测度论和随机积分课程。Itô 公式在这里给出推导直觉；一般定理的完整证明见课程讲义。

## 1 · Brownian motion 的尺度

标准 Brownian motion Wₜ 从 0 出发，路径连续，独立平稳增量，$W_t-W_s\sim N(0,t-s)$，t>s。均值 0、方差 t，并且：

$$
\operatorname{Cov}(W_s,W_t)=\min(s,t).
$$

对 s≤t，写 Wₜ=Wₛ+(Wₜ−Wₛ)，后项与 Wₛ 独立，协方差就是 s。小时间步 Δt 的典型增量尺度是 √Δt，不是 Δt。

## 2 · 二次变差：普通微积分漏掉的量

把 [0,T] 均分成 n 段。每段增量独立正态，所以平方和的期望是 T、方差是 $2T^2/n$：

$$
Q_n=\sum_{j=1}^n(\Delta W_j)^2,\qquad Q_n\longrightarrow T\quad\text{in }L^2.
$$

这就解释了记号 $(dW)^2=dt$：它不是逐点的普通代数等式，而是平方增量累积的极限规则。平滑路径的平方增量和趋于 0，Brownian 路径则不同。

## 3 · Itô 公式：保留会累计下来的项

若 $dX_t=a_tdt+b_tdW_t$，系数适应于过去信息并满足使积分存在的条件，f∈C¹,²，则：

$$
df(t,X_t)=\left(f_t+a_tf_x+\frac12b_t^2f_{xx}\right)dt+b_tf_xdW_t.
$$

Taylor 的二阶项乘上 $(\Delta W)^2$，累积后留下 dt；dt·dW 与 (dt)² 消失。**这是解释路线，不是一般定理的完整证明。**

<details markdown="1">
<summary>例子：f(x)=x² 为什么不是普通链式法则？</summary>

取 X=W，得到 $d(W_t^2)=2W_tdW_t+dt$，所以：

$$
\int_0^TW_t\,dW_t=\frac12(W_T^2-T).
$$

期望为 0，与左端适应性积分的性质一致。若错误地照普通积分写成 W_T²/2，期望会变成 T/2。

</details>

## 4 · 积分的均值和方差也有条件

对可预测、平方可积的过程 H：

$$
E\left[\int_0^T H_t\,dW_t\right]=0,\qquad
E\left[\left(\int_0^T H_t\,dW_t\right)^2\right]=E\int_0^T H_t^2dt.
$$

后一个是 Itô isometry。它可先对分段常值、只使用左端已有信息的 H 证明：不同增量交叉项条件均值为 0，再通过 L² 极限延伸。

不能把 Hₜ=W_T 当作合法的无预见性策略：它偷看了终点。一般的 local martingale 也不能不查可积条件就声称期望不变。

## 5 · GBM：为什么 drift 少了 σ²/2

常系数几何 Brownian motion，S₀>0：

$$
dS_t=\mu S_tdt+\sigma S_tdW_t,\qquad
S_t=S_0\exp[(\mu-\sigma^2/2)t+\sigma W_t].
$$

对 logS 用 Itô 即可得到。借正态 MGF，$E[S_t]=S_0e^{\mu t}$；中位数为 $S_0e^{(\mu-\sigma^2/2)t}$。均值增长和典型路径的对数增长不是同一个量。

同样，$\exp(\theta W_t-\theta^2t/2)$ 是常 θ 下的指数鞅：给定过去，把独立正态增量的 MGF 算掉即可。

## 6 · 反射原理：最大值也能变成终点问题

对 a>0：

$$
P\left(\max_{0\le s\le T}W_s\ge a\right)=2P(W_T\ge a).
$$

直觉是首次碰到 a 后反射后续路径，把“碰过 a 但终点低于 a”的路径与“终点高于 a”的路径对应。严格论证使用强 Markov 性与对称性；这里给的是证明骨架。

带漂移、非连续路径或不同边界不能直接原样套。由上式可推出最大值的分布等于 |W_T|，均值为 $\sqrt{2T/\pi}$。

## 7 · 模拟时保留正确尺度

Euler–Maruyama 一步为 $X_{t+h}=X_t+a_th+b_t\sqrt hZ$，Z~N(0,1)。它是离散近似，有时间步误差；GBM 有精确的单步 lognormal 更新时，不必为了形式统一强用 Euler。

继续：[无套利定价](../finance/README.md)。参考：[MIT Itô calculus](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote18/)。
