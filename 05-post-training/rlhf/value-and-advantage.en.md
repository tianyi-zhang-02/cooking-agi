# RLHF: reward, value, and advantage

[中文](value-and-advantage.md) · **English**

> Reading time: ~4 min · Last reviewed: 2026-10

## Reward, return, and credit assignment

A **reward** $r_t$ is immediate feedback at one step. A **return** $G_t$ accumulates future rewards from that step onward:

$$G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.$$

In classic preference-based RLHF, the main reward often arrives only after a complete answer is scored. Every earlier token must then share responsibility for the terminal score. Was a poor result caused by the opening direction or a factual mistake halfway through? That is the **credit-assignment problem**. Implementations often add per-token KL penalties as denser shaping rewards, but those penalties are not human preference themselves.

## Value, Q, and advantage

The Critic does not judge whether a completed answer is good. It predicts the cumulative return expected from the current prefix:

$$V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s].$$

Conditioning additionally on the current action gives the action value:

$$Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].$$

Their difference is the **advantage**:

$$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).$$

It asks not whether the total score is high, but how much better this action was than the normal expectation at that state. The same return of $0.6$ is disappointing if the Critic predicted $0.8$ and encouraging if it predicted $0.2$. Subtracting this baseline leaves the expected policy gradient unchanged and, when the baseline is accurate, usually reduces its variance substantially.

## Work through a three-token answer {#worked-returns}

Put these definitions on one concrete answer. It has 3 tokens and $\gamma=1$. The Reward Model gives 1 only at the end, and each token also carries a KL shaping term of −0.1. The Critic values the 3 prefixes at `[0.5, 0.6, 0.8]`, and nothing follows once the answer ends. All numbers are teaching assumptions.

| Position $t$ | $r_t$ | $G_t$ (sum from here on) | $V(s_t)$ | $G_t-V(s_t)$ |
| --- | --- | --- | --- | --- |
| 0 | −0.1 | 0.7 | 0.5 | +0.2 |
| 1 | −0.1 | 0.8 | 0.6 | +0.2 |
| 2 | −0.1 + 1 = 0.9 | 0.9 | 0.8 | +0.1 |

$G_t$ is easiest to compute from the back: $G_2=0.9$, $G_1=-0.1+0.9=0.8$, $G_0=-0.1+0.8=0.7$. All three advantages are positive: at every prefix this answer did a little better than the Critic expected. The Critic already expected a lot at the last step (0.8), so only 0.1 is left over there. If the Critic's estimates are off, this column is off too; TD and GAE below deal with that trade-off.

## Bellman equations update value, not reward

The Bellman equation writes long-term value as a one-step recursion:

$$
V^\pi(s_t)
=\mathbb E_{a_t\sim\pi,\,s_{t+1}\sim P}
\left[r_t+\gamma V^\pi(s_{t+1})\right].
$$

It is not a rule for modifying rewards. The environment, Reward Model, or verifier normally supplies $r_t$; the Bellman relation uses that observed reward and the next state's value to update the current **value estimate**. Expanding the recursion gives

$$
V^\pi(s_t)=\mathbb E[r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots].
$$

When $0<\gamma<1$, distant rewards receive less weight. The discount also defines an effective planning horizon and helps keep returns finite in continuing tasks. Discounting distant outcomes is not part of the definition of RL, however: finite LLM episodes often use $\gamma=1$, so a terminal reward does not shrink merely because a token occurred earlier.

One observed transition gives a one-step TD target:

$$
y_t=r_t+\gamma V_\psi(s_{t+1}),
\qquad
\delta_t=y_t-V_\psi(s_t).
$$

$\delta_t$ is the **temporal-difference error**: the difference between the new target—one-step reward plus future value—and the old estimate. Monte Carlo uses the complete $G_t$, giving low bias but high variance. TD bootstraps from a value estimate, reducing variance while introducing approximation bias. PPO commonly uses GAE to interpolate between these behaviors.

Continue with the answer above: $\delta_0=-0.1+0.6-0.5=0$, $\delta_1=-0.1+0.8-0.6=0.1$, and $\delta_2=0.9+0-0.8=0.1$. They sum to 0.2, exactly $G_0-V(s_0)$: with $\gamma=1$ the intermediate values cancel term by term. Using $\delta_t$ alone as the advantage is the $\lambda=0$ end of GAE; adding up all the later $\delta$ terms is the $\lambda=1$ end, which is the Monte Carlo advantage in the table.

The central policy-gradient expression is therefore

$$\nabla_\theta J(\theta)\approx\mathbb E\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t\right].$$

When $\hat A_t>0$, increase the probability of the sampled action; when it is negative, decrease it. The sum runs only over sampled positions, but at one state the softmax probabilities share a normalizer: the gradient of the sampled log-probability with respect to the logits is $e_a-\pi$. For positive advantage, a direct update of these logits raises the sampled logit and lowers the others. That does not mean every other probability falls: they are normalized together.

For example, starting from `[0.2, 0.7, 0.1]`, sampling the first token and taking a logit-gradient step of 0.1 with advantage 1 gives approximately `[0.22374, 0.67402, 0.10224]`. The third probability rises too; only the sum of the unsampled probabilities must fall in this direct-update example. Shared parameters carry the effect to other states as well.

The Critic regresses $V_\psi(s_t)$ toward returns or bootstrapped targets. Its main contribution is **variance reduction**; it does not choose the Actor's next token.
