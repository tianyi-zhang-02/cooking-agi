# Martingales and stopping: when may n be replaced by T?

[中文](martingales.md) · **English** · [Review map](../README.en.md)

> Reading time: ~11 min · Prerequisites: conditional expectation, random walks · Last reviewed: 2026-10

“Each step is fair, but I will stop when ahead.” This confuses a fixed time with a random stopping time. Martingales make the missing justification visible.

## 1 · A martingale preserves conditional means, not sample paths

Let $\mathcal F_n$ contain information available by time n. An integrable adapted process Mₙ is a martingale when:

$$
E[M_{n+1}\mid\mathcal F_n]=M_n.
$$

Replace equality by ≤ for a supermartingale and ≥ for a submartingale. These describe drift, not desirability.

For independent fair ±1 increments ξᵢ, $S_n=\sum_{i=1}^n\xi_i$ is a martingale. Expanding the square shows $S_n^2-n$ is another: the next squared position gains one in conditional expectation. With drift, center using $S_n-nE[\xi_1]$.

## 2 · Stopping times cannot look ahead

T is a stopping time if whether T≤n is known at time n. First hitting a boundary qualifies. The time of the maximum over an entire future path generally does not.

**Bounded optional stopping:** for a discrete integrable martingale and stopping time T≤K, $E[M_T]=E[M_0]$.

<details markdown="1">
<summary>Prove the bounded version with indicators</summary>

Use a finite sum:

$$
M_T=M_0+\sum_{j=1}^K(M_j-M_{j-1})\mathbf1_{\{T\ge j\}}.
$$

The event T≥j is known at j−1. Conditioning on $\mathcal F_{j-1}$ makes each summand's expectation zero. Finiteness avoids an unjustified interchange of limits and expectation. An unbounded T requires additional control.

</details>

## 3 · Gambler's ruin: two proofs of the same result

Start a fair walk at i and stop at 0 or N. In every N-step block, probability at least $2^{-N}$ of consecutive moves in one direction ensures absorption. Thus a geometric block bound gives Eτ<∞.

Apply the martingale to the stopped position at τ∧m, bounded in [0,N]. Bounded convergence yields $E[S_\tau]=i$, so:

$$
P(S_\tau=N)=\frac iN,\qquad E[\tau]=i(N-i).
$$

For the duration, apply $S_n^2-n$: $E[\tau\wedge m]=E[S_{\tau\wedge m}^2]-i^2$. Monotone convergence and bounded positions give the limit. Compare the [first-step recursion](../probability/markov-chains.en.md).

For upward probability p and q=1−p, with 0<p<1 and p≠q:

$$
h_i=\frac{1-(q/p)^i}{1-(q/p)^N},\qquad
E_i[\tau]=\frac{Nh_i-i}{p-q}.
$$

Find hitting probability from the recurrence, then duration from the centered walk. The limit p→1/2 must recover the fair result. Direct evaluation near that limit can suffer cancellation.

## 4 · A counterexample to unrestricted stopping

A fair integer walk starts at zero; T is its first hit of +1. It hits almost surely, but E[T]=∞. Thus S_T=1 although E[S_n]=0 at every fixed n.

Almost-sure termination alone does not imply $E[S_T]=E[S_0]$. The stopped variables lack the needed uniform-integrability control. We invoke these standard hitting-time facts here rather than provide their full proof.

## 5 · Wald's identity also has assumptions

Let Yᵢ be iid with $E|Y_1|<\infty$. Let T be a stopping time for those increments with E[T]<∞. Then:

$$
E\left[\sum_{i=1}^TY_i\right]=E[T]E[Y_1].
$$

<details markdown="1">
<summary>Where independence enters the proof</summary>

Write the random sum as $\sum_{i\ge1}Y_i\mathbf1_{\{T\ge i\}}$. The event T≥i depends only on the first i−1 increments and is independent of Yᵢ. Absolute summability follows from $E|Y_1|\sum_iP(T\ge i)=E|Y_1|E[T]<\infty$, justifying interchange. Factor each expectation.

</details>

The argument fails if T uses future information or lacks finite expectation. “Mean count times mean increment” is not unconditional permission.

## 6 · Choose the simpler tool

For a small state space with clear boundaries, first-step equations are often simplest. Linear or quadratic conserved structure can make a martingale shorter. Neither is intrinsically more advanced.

Exercise: a fair walk starting at 2 between 0 and 5 reaches the top with probability 2/5 and stops after 6 steps on average. Explain both with recurrences and stopped martingales.

Continue: [Finite-horizon decisions](dynamic-programming.en.md). Reference: [MIT random walks and martingales](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/resources/mit6_262s11_chap07/).
