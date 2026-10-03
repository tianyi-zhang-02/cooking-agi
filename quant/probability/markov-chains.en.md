# Markov chains: how much history must a waiting problem remember?

[中文](markov-chains.md) · **English**

> Reading time: ~10 min · Prerequisite: [Conditional expectation](conditioning-proofs.en.md) · Last reviewed: 2026-10

How long until repeated coin flips first contain HHT? A length-3 window has probability $1/8$, but that alone does not prove a mean waiting time of 8. Neighboring windows overlap; for HTH the answer is 10. The difference becomes clear once we choose the state.

## 1 · Markov does not mean independent steps

A finite-state, time-homogeneous Markov chain satisfies

$$
P(X_{t+1}=j\mid X_t=i,X_{t-1},\ldots,X_0)=P(X_{t+1}=j\mid X_t=i)=Q_{ij}.
$$

The conditioned history must have positive probability. **Given the current state, earlier history adds no predictive information.** The next state can still depend on the present one.

The transition matrix $Q$ has nonnegative entries and row sums 1. Conditioning on the intermediate state gives

$$
P(X_{t+2}=j\mid X_t=i)=\sum_kQ_{ik}Q_{kj}=(Q^2)_{ij}.
$$

This uses total probability and the Markov property, not unconditional independence of adjacent transitions. Induction gives the $n$-step matrix $Q^n$.

## 2 · Waiting for HHT: keep the useful suffix

Flips are independent, with $P(H)=p\in(0,1)$ and $q=1-p$. Record the prefix of HHT currently matched by the ending of the history:

| State | Information retained | Next H | Next T |
| --- | --- | --- | --- |
| 0 | No useful suffix | H | 0 |
| H | Suffix matches H | HH | 0 |
| HH | Suffix matches HH | HH | Done |
| Done | HHT has appeared | Stop | Stop |

After HH followed by H, the latest two symbols are still HH. **Do not reset to zero.** It is the same longest-useful-suffix idea used in string matching.

## 3 · First-step analysis: start each equation with 1

Let $E_0,E_H,E_{HH}$ be expected additional flips from the respective states. The completed state has remaining time 0. Spend one flip, then average over the next state:

$$
\begin{aligned}
E_0&=1+pE_H+qE_0,\\
E_H&=1+pE_{HH}+qE_0,\\
E_{HH}&=1+pE_{HH}.
\end{aligned}
$$

<details markdown="1">
<summary>Establish finiteness before solving</summary>

Split the sequence into nonoverlapping blocks of 3 flips. Each block equals HHT with probability $p^2q>0$, independently of other blocks. The number of blocks until such a match is geometric. The waiting time is at most 3 times that count, so its expectation is at most $3/(p^2q)$. The recurrence therefore uses finite expectations.

The third equation gives $E_{HH}=1/q$; the first gives $E_0=1/p+E_H$. Substitute into the second:

$$
pE_0=\frac1p+1+\frac pq=\frac1{pq},
\qquad E_0=\frac1{p^2q}.
$$

For a fair coin, $p=q=1/2$, so $E_0=8$. Here the result equals the reciprocal window probability, but the justification comes from the state equations.

</details>

## 4 · Why HTH does not take 8 flips

Use states $0,H,HT,\text{Done}$. H followed by H stays at H; HT followed by T returns to 0. For a fair coin,

$$
E_0=1+\tfrac12E_H+\tfrac12E_0,\quad
E_H=1+\tfrac12E_H+\tfrac12E_{HT},\quad
E_{HT}=1+\tfrac12E_0.
$$

Solving gives $E_0=10$. Both targets have window probability $1/8$, yet different first waiting times. **A pattern's overlap with itself changes how much progress survives a retry.** The nonoverlapping-block argument again gives a finite geometric upper bound.

<details markdown="1">
<summary>One more step: derive HTH for a biased coin</summary>

Keep the same states and replace $1/2$ with the appropriate $p,q$. We obtain $E_0=1/p+E_H$, $E_H=1/q+E_{HT}$, and $E_{HT}=1+qE_0$. Therefore,

$$
E_0=\frac1{p^2q}+\frac1p.
$$

For a fair coin this adds 2 flips compared with HHT. The [exact equation checker on the exercise page](practice.en.md) can verify several other values of $p$.

</details>

## 5 · Random walks: boundary conditions belong to the problem

Walk on integers $0,\ldots,N$, moving one step left or right with equal probability, stopping at 0 or N. Let $h_i$ be the probability of hitting N first from state $i$.

$$
h_0=0,\quad h_N=1,\qquad h_i=\tfrac12h_{i-1}+\tfrac12h_{i+1}.
$$

Rearranging gives $h_{i+1}-h_i=h_i-h_{i-1}$, so adjacent differences are constant. The boundaries imply $h_i=i/N$.

<details markdown="1">
<summary>What is the expected time to either boundary?</summary>

Call it $t_i$. The boundaries are $t_0=t_N=0$; interior states satisfy

$$
t_i=1+\tfrac12t_{i-1}+\tfrac12t_{i+1},
\qquad t_{i+1}-2t_i+t_{i-1}=-2.
$$

A constant second difference of $-2$ suggests $t_i=-i^2+ai+b$. The boundaries give $b=0,a=N$, so $t_i=i(N-i)$.

Finiteness also needs justification. From any nonabsorbing state, the next N steps all moving in one fixed direction has probability at least $2^{-N}$ and ensures absorption. Bounding tails in N-step blocks gives a geometric bound. For uniqueness, the difference of two solutions has zero boundaries and zero second difference, so it is identically zero.

</details>

## 6 · Stationarity does not always mean convergence

A stationary distribution is a probability row vector $\pi$ satisfying $\pi Q=\pi$. Starting from it leaves the distribution unchanged. This does not guarantee convergence to it from every starting state.

The smallest counterexample alternates deterministically between two states: $Q=\begin{pmatrix}0&1\\1&0\end{pmatrix}$. The vector $(1/2,1/2)$ is stationary, but a chain starting at state 0 keeps alternating.

For a finite chain, irreducibility guarantees a unique stationary distribution; adding aperiodicity gives the familiar convergence result from any initial distribution. That theorem is for further study; the counterexample here explains why its assumptions matter.

## Next

[Proof techniques: finding a useful first step](proof-toolbox.en.md). Further reading: [Harvard Stat 110: Markov Chains](https://stat110.hsites.harvard.edu/resource/markov-chains). It connects states, matrices, and long-run behavior; the pattern examples here focus on first-step analysis.
