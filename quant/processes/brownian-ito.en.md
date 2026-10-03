# Brownian motion and Itô: why half a second derivative appears

[中文](brownian-ito.md) · **English** · [Review map](../README.en.md)

> Reading time: ~10 min · Advanced introduction · Last reviewed: 2026-10

The aim is to derive core examples and understand Itô steps in pricing, not replace a stochastic-calculus course. We motivate Itô's formula; the full general proof is left to the cited lectures.

## 1 · The scale of Brownian motion

Standard Brownian motion Wₜ starts at zero, has continuous paths and independent stationary increments, with $W_t-W_s\sim N(0,t-s)$ for t>s. Its mean is zero, variance t, and:

$$
\operatorname{Cov}(W_s,W_t)=\min(s,t).
$$

For s≤t, write Wₜ=Wₛ+(Wₜ−Wₛ). The independent increment leaves covariance s. Typical increments scale as √Δt, not Δt.

## 2 · Quadratic variation: what ordinary calculus misses

Divide [0,T] into n equal intervals. Independent Normal increments give a sum of squares with mean T and variance $2T^2/n$:

$$
Q_n=\sum_{j=1}^n(\Delta W_j)^2,\qquad Q_n\longrightarrow T\quad\text{in }L^2.
$$

This motivates $(dW)^2=dt$. It is shorthand for accumulated squared-increment limits, not a pointwise algebraic identity. Smooth paths have vanishing sums of squared increments; Brownian paths do not.

## 3 · Itô retains terms that accumulate

For $dX_t=a_tdt+b_tdW_t$, with adapted coefficients satisfying the required integrability conditions, and f∈C¹,²:

$$
df(t,X_t)=\left(f_t+a_tf_x+\frac12b_t^2f_{xx}\right)dt+b_tf_xdW_t.
$$

The second-order Taylor term multiplied by $(\Delta W)^2$ survives accumulation. Mixed dt·dW and (dt)² terms vanish. **This motivates the result; it is not a full general proof.**

<details markdown="1">
<summary>Example: why does f(x)=x² differ from the ordinary chain rule?</summary>

With X=W, $d(W_t^2)=2W_tdW_t+dt$, hence:

$$
\int_0^TW_t\,dW_t=\frac12(W_T^2-T).
$$

Its expectation is zero, as expected for this adapted integral. The ordinary-calculus answer W_T²/2 would incorrectly have mean T/2.

</details>

## 4 · Integral means and variances have assumptions

For predictable square-integrable H:

$$
E\left[\int_0^T H_t\,dW_t\right]=0,\qquad
E\left[\left(\int_0^T H_t\,dW_t\right)^2\right]=E\int_0^T H_t^2dt.
$$

The second identity is Itô isometry. First prove it for step processes using information available at left endpoints: cross terms vanish by conditioning. Extend through an L² limit.

Hₜ=W_T anticipates future information and does not qualify as an ordinary nonanticipating integrand. A general local martingale also needs integrability checks before asserting constant expectation.

## 5 · GBM and the −σ²/2 correction

For constant coefficients and S₀>0:

$$
dS_t=\mu S_tdt+\sigma S_tdW_t,\qquad
S_t=S_0\exp[(\mu-\sigma^2/2)t+\sigma W_t].
$$

Apply Itô to logS. The Normal MGF gives $E[S_t]=S_0e^{\mu t}$, while the median is $S_0e^{(\mu-\sigma^2/2)t}$. Mean growth and typical log growth differ.

Likewise $\exp(\theta W_t-\theta^2t/2)$ is a martingale for constant θ: condition on the past and evaluate the independent Normal increment's MGF.

## 6 · Reflection: a maximum becomes an endpoint question

For a>0:

$$
P\left(\max_{0\le s\le T}W_s\ge a\right)=2P(W_T\ge a).
$$

After first hitting a, reflect the remaining path. This pairs paths hitting a but ending below it with paths ending above a. A rigorous proof uses the strong Markov property and symmetry; this is the proof outline.

Do not reuse the formula unchanged for drift, jumps, or different boundaries. It implies the maximum has the distribution of |W_T| and mean $\sqrt{2T/\pi}$.

## 7 · Simulate with the correct scale

Euler–Maruyama uses $X_{t+h}=X_t+a_th+b_t\sqrt hZ$, Z~N(0,1). It introduces time-discretization error. For GBM, an exact Lognormal step is available, so Euler is not obligatory.

Continue: [No-arbitrage pricing](../finance/README.en.md). Reference: [MIT Itô calculus](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote18/).
