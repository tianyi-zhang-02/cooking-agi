# RLHF: reward, value, and advantage

[中文](value-and-advantage.md) · **English**

> Reading time: ~2 min · Level: core · Last reviewed: 2026-09

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

It asks not whether the total score is high, but how much better this action was than the normal expectation at that state. The same return of $0.6$ is disappointing if the Critic predicted $0.8$ and encouraging if it predicted $0.2$. Subtracting this baseline leaves the expected policy gradient unchanged while greatly reducing its variance.

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
y_t=r_t+\gamma V_\phi(s_{t+1}),
\qquad
\delta_t=y_t-V_\phi(s_t).
$$

$\delta_t$ is the **temporal-difference error**: the difference between the new target—one-step reward plus future value—and the old estimate. Monte Carlo uses the complete $G_t$, giving low bias but high variance. TD bootstraps from a value estimate, reducing variance while introducing approximation bias. PPO commonly uses GAE to interpolate between these behaviours.

The central policy-gradient expression is therefore

$$\nabla_\theta J(\theta)\approx\mathbb E\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t\right].$$

When $\hat A_t>0$, increase the probability of the sampled action; when it is negative, decrease it. The expression directly updates sampled tokens, with shared parameters carrying the effect to other states. The Critic regresses $V_\phi(s_t)$ toward returns or bootstrapped targets. Its main job is **variance reduction**, not choosing the Actor's next token.
