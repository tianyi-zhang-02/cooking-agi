# Black–Scholes: hedging rather than guessing direction

[中文](options-and-greeks.md) · **English** · [Review map](../README.en.md)

> Reading time: ~10 min · Prerequisites: Itô and replication · Last reviewed: 2026-10

This is mathematical review. Understanding the PDE and sensitivities matters more than memorizing d₁ and d₂. Model assumptions are not market guarantees.

## 1 · State the idealized assumptions

A non-dividend stock follows GBM with constant σ>0 and r. Trading is continuous and frictionless; borrowing and lending use the same rate; shorting and standard admissible self-financing strategies are allowed. European payoffs occur at expiry.

Jumps, discrete hedging, changing volatility, liquidity, and financing constraints can invalidate the idealization.

## 2 · Delta hedging removes the random term

Itô gives $dV=(V_t+\mu SV_S+\frac12\sigma^2S^2V_{SS})dt+\sigma SV_SdW$.

Hold Δ=V_S shares in a self-financing replicating strategy. The cash balance V−ΔS earns the risk-free rate. Matching drift gives:

$$
V_t+\frac12\sigma^2S^2V_{SS}+rSV_S-rV=0.
$$

μ disappears because of replication, not because the real-world expected return is known. Do not naively replace $d(\Delta S)$ by ΔdS while ignoring rebalancing: the **self-financing condition** accounts for cash changes.

## 3 · Risk-neutral expectation and the formula

For remaining time τ=T−t and payoff $(S_T-K)^+$, the risk-neutral stock drift is r:

$$
C=S\Phi(d_1)-Ke^{-r\tau}\Phi(d_2),\qquad
d_1=\frac{\log(S/K)+(r+\sigma^2/2)\tau}{\sigma\sqrt\tau},\quad
d_2=d_1-\sigma\sqrt\tau.
$$

<details markdown="1">
<summary>Where do the two Φ terms come from?</summary>

Split $e^{-r\tau}E[(S_T-K)^+]$ into $e^{-r\tau}E[S_T\mathbf1_{\{S_T>K\}}]-Ke^{-r\tau}P(S_T>K)$. LogS_T is Normal, so the second term gives Φ(d₂). Completing the square after multiplying the Normal density by the exponential shifts its mean, producing SΦ(d₁).

Thus Φ(d₂) is the model's risk-neutral in-the-money probability. Delta Φ(d₁) is different, and neither is a real-world success forecast.

</details>

## 4 · Greeks: sensitivity to which input?

For a non-dividend European call, with φ the standard Normal density:

| Quantity | Definition and formula | Meaning |
| --- | --- | --- |
| Delta | $\partial C/\partial S=\Phi(d_1)$ | First-order stock sensitivity |
| Gamma | $\partial^2C/\partial S^2=\varphi(d_1)/(S\sigma\sqrt\tau)$ | Change in Delta |
| Vega | $\partial C/\partial\sigma=S\varphi(d_1)\sqrt\tau$ | Volatility-parameter sensitivity |
| Theta | Derivative with respect to calendar time t | Effect of passing time |
| Rho | $\partial C/\partial r=K\tau e^{-r\tau}\Phi(d_2)$ | Interest-rate sensitivity |

Vega uses σ in decimal units. A one-percentage-point volatility move multiplies it by 0.01. Specify daily versus annual Theta and differentiation with respect to t versus τ.

## 5 · A numerical sanity check

For S=K=100, r=0, σ=0.2, τ=1, d₁=0.1 and d₂=−0.1. The call is about 7.97, Delta about 0.54, and Vega about 39.70.

A local move from σ=0.20 to 0.21 therefore adds roughly 0.397, not 39.70. Large changes require repricing rather than a linear approximation.

<details markdown="1">
<summary>How can you check these numbers without memorizing them?</summary>

At zero rates and at-the-money, call and put prices agree; the price lies between zero and 100. As σ→0 it approaches the discounted deterministic payoff, and as τ→0 it approaches expiry payoff. Gamma and Vega are positive for this call.

</details>

## 6 · Implied volatility is an inversion, not a prediction

Given a market price, solve for σ matching the model. Positive Vega gives monotonicity under standard conditions, useful for bracketed root finding. Check attainable price bounds first. Small Vega near extreme strikes or short maturities makes inversion sensitive to quote noise.

Interest-rate derivatives, stochastic volatility, and exotic options are not claimed to be comprehensively covered here. First connect replication, PDEs, expectation, and sensitivities.

Reference: [MIT Black–Scholes notes](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote19/). Continue: [Portfolios, risk, and market mechanics](portfolio-and-market.en.md).
