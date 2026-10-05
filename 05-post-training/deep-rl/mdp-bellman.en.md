# MDPs and Bellman: breaking the future into one step

[中文](mdp-bellman.md) · **English**

> Reading time: ~6–9 min · Last reviewed: 2026-10

A robot can stop now for 2 points, or charge for zero immediate reward and receive 4 points on the following step. Which is better? It depends on how we count the future. This tiny problem already contains a central structure of RL.

## Specify what the environment provides

An MDP (Markov decision process) includes states, actions, transition probabilities, rewards, and a discount. Markov does not mean the past is useless. It means **given the current state and action, the distribution of the next state and reward needs no additional history**.

A single image may not reveal whether an object is approaching or receding. Several frames, memory, or a belief state—a distribution over hidden states—can provide the missing information. An observation is not automatically a full state. In a finite-horizon task, the same location may have different values with 10 steps left and with 1. Below, remaining time is included in the state; otherwise write $V_t^\pi(s)$ explicitly.

Define the discounted return from the current step:

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k},\qquad
V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s].
$$

$Q^\pi(s,a)$ fixes the first action and follows $\pi$ afterward. $V^\pi$ also averages over that first action, so $V^\pi(s)=\mathbb E_{a\sim\pi}[Q^\pi(s,a)]$.

Below, $r(s,a,s')$ denotes the expected immediate reward conditional on that transition; the reward itself may be random.

## Bellman decomposes the same objective

Separate the first reward: $G_t=r_t+\gamma G_{t+1}$. Apply the law of total expectation, conditioning on the next state:

$$
V^\pi(s)=\mathbb E_{a\sim\pi,\,s'\sim P}
\left[r(s,a,s')+\gamma V^\pi(s')\right].
$$

This is policy evaluation: assessing a fixed policy. For an optimal policy, replace the average over the existing action distribution with the best available action:

$$
V^*(s)=\max_a\mathbb E_{s'\sim P}[r+\gamma V^*(s')].
$$

Do not interchange $\max_a\mathbb E[\cdot]$ and $\mathbb E[\max_a\cdot]$. The latter effectively lets the decision-maker see the random future before choosing.

## Two value-iteration steps by hand

Initialize the values of both “start” and “charged” to zero. Each iteration uses the **previous iteration's** complete table:

| Iteration | Charged | Start, $\gamma=0.9$ |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 4 | $\max(2,0)=2$ |
| 2 | 4 | $\max(2,0.9\times4)=3.6$ |

Reward propagates from states near the end toward earlier states. With $\gamma=0.4$, charging is worth only 1.6, so taking 2 immediately is better. The algorithm did not become less intelligent; the objective changed.

<div class="drl-lab" data-drl-lab="bellman"><p>Adjust the discount and step through updates when the interactive loads. The table retains the complete static example.</p></div>

## Why the tabular update converges

For finite states and actions, bounded rewards, and $0\le\gamma<1$, the optimal Bellman operator is a contraction in the infinity norm:

$$
\|TV-TW\|_\infty\le\gamma\|V-W\|_\infty.
$$

Two facts do the work: the difference of maxima is bounded by the largest pointwise difference, and a probability-weighted average cannot enlarge the maximum error. After each update, the distance between two value tables is at most $\gamma$ times its previous value. Repeated updates converge to a unique fixed point.

**This is not a convergence proof for neural DQN.** Sampling error, function approximation, and parameter updates change the problem. Finite-horizon tasks with $\gamma=1$ can use backward induction; the strict-contraction argument above does not apply unchanged.

## Policy iteration: evaluate, then improve

Value iteration applies the optimal Bellman backup each round. Policy iteration first evaluates $V^\pi$, then chooses the action maximizing $Q^\pi(s,a)$ at each state, producing $\pi'$.

Why can this not worsen the exact solution? An average under the old policy cannot exceed the maximum:

$$
T_{\pi'}V^\pi(s)=\max_aQ^\pi(s,a)\ge V^\pi(s).
$$

By monotonicity of the Bellman operator, repeated application of $T_{\pi'}$ cannot reduce these values. Under the discounted assumptions above it converges to $V^{\pi'}$, so $V^{\pi'}\ge V^\pi$. This justifies exact policy improvement. Finite evaluation steps and neural approximation introduce errors; the guarantee does not transfer unconditionally.

## What happens when the system has a loop?

The charging example ends after two steps. Now fix a policy: state A gives reward 1, then stays in A with probability one half or moves to B with probability one half. B gives reward 2, then returns to A. With $\gamma=0.5$:

$$
V_A=1+0.5(0.5V_A+0.5V_B),\qquad V_B=2+0.5V_A.
$$

Substituting the second equation into the first gives $0.625V_A=1.5$, so $V_A=2.4,V_B=3.2$. This evaluates a fixed policy; there is no maximization over actions.

```text
           50% stay in A
                ↺
        A, reward 1 ──50%──→ B, reward 2
              ↑__________________|
                   100% to A
```

In matrix notation, $V=r+\gamma P_\pi V$, or $(I-\gamma P_\pi)V=r$. Solving this linear system and repeatedly applying the Bellman update address the same policy-evaluation problem. Explicitly storing and solving the matrix can be costly for many states, motivating sample-based updates.

## What does a small residual guarantee?

Suppose a value table satisfies $\|TV-V\|_\infty\le\varepsilon$. Contraction and the triangle inequality give:

$$
\|V-V^*\|_\infty\le\|V-TV\|_\infty+\|TV-TV^*\|_\infty
\le\varepsilon+\gamma\|V-V^*\|_\infty.
$$

Rearranging yields $\|V-V^*\|_\infty\le\varepsilon/(1-\gamma)$. A residual of 0.01 gives a bound of 0.1 at $\gamma=0.9$, but 1 at $\gamma=0.99$. Valuing the distant future more requires a tighter residual for the same value-error guarantee.

This needs the true Bellman residual over the entire state space. A small minibatch MSE does not establish that uniform bound: sampling noise and uncovered states matter.

## Change one assumption

<details markdown="1">
<summary>What if charging leads to 4 points with probability one half and zero otherwise?</summary>

At $\gamma=0.9$, charging is worth $0.9(0.5\times4+0.5\times0)=1.8$, so take 2 now. Compare expected returns, not the best possible outcome. Risk sensitivity or constraints require an explicitly different objective, not a silent change to this calculation.

</details>

Reference: [RL foundations and value definitions](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html). Next: [Learning value from samples when transitions are unknown](returns-and-td.en.md).
