# Portfolios and markets: risk beyond the expected return

[中文](portfolio-and-market.md) · **English** · [Review map](../README.en.md)

> Reading time: ~8 min · Basic models and extensions · Last reviewed: 2026-10

This is conceptual review, not an asset or trading recommendation. Calculate portfolio risk, then distinguish theoretical prices from executable trades.

## 1 · Two assets need not mean independent risks

For return mean vector μ, covariance Σ, and weights w:

$$
E[R_p]=w^\top\mu,\qquad
\operatorname{Var}(R_p)=w^\top\Sigma w.
$$

Two equal-volatility σ assets with equal weights have variance $\sigma^2(1+\rho)/2$. At ρ=1 there is no variance diversification; at ρ=0 variance halves. At ρ=−1 with equal volatility it cancels entirely. Estimated correlations need not persist.

## 2 · Derive minimum-variance weights

Allow real weights, assume positive-definite Σ, and constrain their sum to one. Minimizing $w^\top\Sigma w$ gives first-order condition 2Σw−λ1=0:

$$
w^*=\frac{\Sigma^{-1}\mathbf1}{\mathbf1^\top\Sigma^{-1}\mathbf1}.
$$

No-short constraints, costs, or target returns change the answer. Noisy or nearly singular covariance estimates amplify errors. Solve linear systems rather than explicitly forming the inverse.

<details markdown="1">
<summary>Two independent assets with variances 1 and 4: what weights minimize variance?</summary>

Minimize $w^2+4(1-w)^2$. Its derivative is 10w−8, so weights are 0.8 and 0.2, with variance 0.8. Equal weighting is not automatically minimum variance.

</details>

## 3 · CAPM and Sharpe: what is the model claiming?

Sharpe divides mean excess return by its standard deviation, on a consistent time scale. Familiar √time annualization needs suitable independence and stability assumptions; autocorrelation changes it.

Under its equilibrium and market assumptions, CAPM gives $E[R_i]-r_f=\beta_i(E[R_m]-r_f)$, where $\beta_i=\operatorname{Cov}(R_i,R_m)/\operatorname{Var}(R_m)$. This is a model relationship, not a guarantee of future returns or profitable historical regressions.

## 4 · Why bond prices fall as yields rise

For positive deterministic cash flows Cₖ discounted at a common continuously compounded yield y:

$$
P(y)=\sum_k C_ke^{-yt_k},\qquad
D=-\frac{P'(y)}{P(y)}=\frac{\sum_kt_kC_ke^{-yt_k}}{P(y)}.
$$

D is duration under this convention. Locally ΔP/P≈−DΔy. The second-order term uses convexity $P''/P$. One duration does not summarize nonparallel yield-curve shifts.

## 5 · VaR is a quantile, not a worst-case loss

Define L as loss. α-VaR is its α quantile; it says little about severity beyond that threshold. Expected Shortfall averages the worst 1−α probability mass. For continuous distributions it can be written $E[L\mid L\ge\operatorname{VaR}_\alpha]$; atoms require correct partial tail-mass treatment.

Tail estimates depend on distributional assumptions, scarce observations, and regime changes. Specify sign conventions, horizon, and confidence level.

## 6 · A spread is not free money

Bid is a price at which others will buy; ask is a price at which they will sell. Aggressive execution typically crosses the spread. Limit orders face execution uncertainty, queues, and adverse selection.

For example, your sell order fills just before the price rises. Execution itself may indicate different information or demand on the other side. Inventory, fees, latency, and partial fills matter too. Venue rules vary; one matching mechanism is not universal.

## 7 · Connect the pieces

Probability describes distributions; statistics estimates parameters; optimization chooses weights; pricing uses replication; market mechanics determine execution. Each stage has assumptions that do not disappear because an earlier formula is elegant.

Reference: [MIT Portfolio Theory](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote14/). Return to the [coverage map](../README.en.md).
