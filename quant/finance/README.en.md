# No-arbitrage pricing: replicate cash flows first

[中文](README.md) · **English** · [Review map](../README.en.md)

> Reading time: ~9 min · Introductory financial mathematics · Last reviewed: 2026-10

These are idealized models, not investment recommendations. A pricing problem often begins with replication rather than forecasting.

## 1 · Put cash flows on the same time basis

At constant continuously compounded rate r, a certain payment K at T is worth $Ke^{-rT}$. With one-period gross risk-free return R, it is worth K/R. R=1+r and eʳ are different compounding conventions.

A zero-coupon bond pays face value only at maturity. A coupon bond sums discounted cash flows. A term structure requires maturity-specific discount factors, not an arbitrary average rate.

## 2 · Why identical cash flows imply identical prices

Assume frictionless markets, permitted trading and financing, performance of contracts, and no arbitrage. Two portfolios paying exactly the same in every future state must have the same current price; otherwise buy the cheaper and sell the dearer.

Equal expected cash flow alone is not enough. Different risks and distributions are not interchangeable.

## 3 · One-step replication without a real-world up probability

Let S₀=100, next-period prices 120 or 80, and gross risk-free return R=1.05. A strike-100 call pays 20 or zero. Replicate using Δ shares and cash B:

$$
120\Delta+1.05B=20,\qquad80\Delta+1.05B=0.
$$

Subtract to obtain Δ=1/2 and B=−800/21, meaning borrowing. Initial cost is $100/2-800/21=250/21\approx11.90$.

<details markdown="1">
<summary>Read the risk-neutral probability from replication</summary>

For up/down multipliers u>d>0 with d<R<u:

$$
q=\frac{R-d}{u-d},\qquad
V_0=\frac{qV_u+(1-q)V_d}{R}.
$$

q lies in (0,1) and makes the stock's expected gross return R. It follows from no-arbitrage replication, not a subjective forecast. Here q=5/8.

</details>

If R is outside the open interval, the standard tradable model has arbitrage or a degenerate boundary. Check before using q.

## 4 · Forwards and put–call parity

For a non-dividend-paying asset at constant r, with frictionless borrowing and shorting, the zero-value forward delivery price is $F_{0,T}=S_0e^{rT}$. Known dividends require subtracting their present value first; carrying benefits and costs matter.

European call and put options on the same underlying with strike K and expiry T satisfy:

$$
C_0-P_0=S_0-Ke^{-rT}.
$$

Compare terminal payoffs: $(S_T-K)^+-(K-S_T)^+=S_T-K$. Long call plus short put replicates a stock financed by borrowing the present value of K. Black–Scholes is unnecessary for this identity.

## 5 · Bounds provide quick checks

A non-dividend European call satisfies $\max(0,S_0-Ke^{-rT})\le C_0\le S_0$. A put is bounded above by $Ke^{-rT}$. These rely on the stated trading and financing assumptions.

Call price is nonincreasing and convex in strike K. Each fixed-S_T payoff has those properties, preserved by a positive discounted expectation. This underlies vertical-spread and butterfly no-arbitrage checks.

## 6 · American exercise changes the problem

American options permit early exercise. At each tree node compare immediate exercise value with discounted continuation value.

For a non-dividend stock, nonnegative rates, and the standard frictionless model, early exercise of an American call is generally not optimal: waiting preserves optionality and delays paying K. Do not apply this unchanged to puts, dividends, or negative rates.

<details markdown="1">
<summary>Exercise: r=0, S₀=K=50, European call price 6. What is the put price?</summary>

For matching maturity and no dividends under the assumptions, parity gives C−P=0, so P=6. If quotes disagree, first check maturity, exercise style, dividends, and timestamps rather than immediately declaring an arbitrage.

</details>

Continue: [Black–Scholes and Greeks](options-and-greeks.en.md). Reference: [MIT Black–Scholes and risk-neutral valuation](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote19/).
