# Black–Scholes：公式背后是对冲，不是猜涨跌

**中文** · [English](options-and-greeks.en.md) · [复习总览](../README.md)

> 阅读时间：约 10 分钟 · 前置：Itô、无套利复制 · 最近审阅：2026-10

本篇是数学模型复习。理解 PDE 与 Greeks，比只记住 d₁、d₂ 更有用。模型假设不等于真实市场保证。

## 1 · 先把理想条件写出来

无分红标的服从 GBM，常数 σ>0、r；可连续交易、无交易成本，可按同一利率借贷，可做空，采用标准自融资可容许策略。欧式合约只在到期支付。

现实的跳跃、离散对冲、波动率变化、流动性与融资约束，都可能使模型偏离实际。

## 2 · Delta hedge 怎样消去随机项

设价格 V(t,S)。Itô 给出 $dV=(V_t+\mu SV_S+\frac12\sigma^2S^2V_{SS})dt+\sigma SV_SdW$。

取自融资复制组合中股票持仓 Δ=V_S，使随机项一致。剩下现金账户为 V−ΔS，赚无风险利率。匹配 drift 得到：

$$
V_t+\frac12\sigma^2S^2V_{SS}+rSV_S-rV=0.
$$

μ 消失了：定价用复制消除局部风险，而不是因为我们知道真实上涨均值。这里不能直接把 $d(\Delta S)$ 写成 ΔdS 而忽略再平衡；**自融资条件**承担了现金变化的记账。

## 3 · 风险中性期望与公式

剩余期限 τ=T−t，call 终值 $(S_T-K)^+$。在风险中性测度下，S 的 drift 为 r：

$$
C=S\Phi(d_1)-Ke^{-r\tau}\Phi(d_2),\qquad
d_1=\frac{\log(S/K)+(r+\sigma^2/2)\tau}{\sigma\sqrt\tau},\quad
d_2=d_1-\sigma\sqrt\tau.
$$

<details markdown="1">
<summary>这两个 Φ 从哪来？</summary>

将 $e^{-r\tau}E[(S_T-K)^+]$ 拆成 $e^{-r\tau}E[S_T\mathbf1_{\{S_T>K\}}]-Ke^{-r\tau}P(S_T>K)$。logS_T 是正态，后项直接得到 Φ(d₂)。前项的正态密度乘上指数后配方，均值平移 σ√τ，得到 SΦ(d₁)。

所以 Φ(d₂) 是该模型下的风险中性价内概率，Δ=Φ(d₁) 却不是同一个概率，更不是现实成功率。

</details>

## 4 · Greeks：价格对什么变动敏感

下面是无分红欧式 call 的 Greeks；φ 为标准正态密度：

| 量 | 定义与公式 | 在问什么 |
| --- | --- | --- |
| Delta | $\partial C/\partial S=\Phi(d_1)$ | 标的微小变化时的一阶响应 |
| Gamma | $\partial^2C/\partial S^2=\varphi(d_1)/(S\sigma\sqrt\tau)$ | Delta 变化有多快 |
| Vega | $\partial C/\partial\sigma=S\varphi(d_1)\sqrt\tau$ | 波动率参数敏感度 |
| Theta | 对日历时间 t 求导 | 时间流逝的价格影响 |
| Rho | $\partial C/\partial r=K\tau e^{-r\tau}\Phi(d_2)$ | 利率敏感度 |

Vega 的 σ 按小数单位；若报“波动率上涨 1 个百分点”，要乘 0.01。Theta 按日还是按年、对 t 还是 τ，符号与单位必须明确。

## 5 · 一个可手算方向的例子

S=K=100、r=0、σ=0.2、τ=1，d₁=0.1、d₂=−0.1。call 约 7.97，Delta 约 0.54，Vega 约 39.70。

意味着局部 σ 从 0.20 增至 0.21，价格约增加 0.397，不是 39.70。它是局部近似，大幅变化要重新计算。

<details markdown="1">
<summary>怎么不靠记忆检查这组数？</summary>

r=0、平值时，call 与 put 相等；价格应介于 0 与 100。σ→0 时价格趋近确定性 payoff 的现值；τ→0 时趋近立即到期 payoff。Gamma、Vega 对这类 call 为正。

</details>

## 6 · Implied volatility 是反解，不是预言

给市场价格，求使模型价格一致的 σ。正 Vega 使标准条件下的价格对 σ 单调，适合 bracketed root finding；但先检查价格是否在可达界内。深度价内外、短期限时 Vega 很小，反解容易对报价噪声敏感。

更复杂的利率产品、随机波动率和奇异期权不在这里声称完整覆盖。先把无套利、PDE、期望和 Greeks 接通。

参考：[MIT Black–Scholes 讲义](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote19/)。下一篇：[组合、风险与市场机制](portfolio-and-market.md)。
