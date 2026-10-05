# Actor–Critic and GAE: making credit assignment concrete

[中文](actor-critic-gae.md) · **English**

> Reading time: ~10–12 min · Last reviewed: 2026-10

A high-scoring trajectory does not make every action in it good. Actor–Critic asks: **at that state, how much more expected return does this action offer than following the existing policy's action choice?** The Actor chooses actions; the Critic estimates value. Here we use a state-value Critic with a stochastic policy. TD3 and SAC later use Q Critics instead.

## What each network receives

| Component | Input | Output | Training signal |
| --- | --- | --- | --- |
| Actor $\pi_\theta$ | Current state | Action distribution | A detached advantage |
| Critic $V_\phi$ | Current state | Expected future return | A detached return target |
| Environment | Action | Next state and reward | Not trained by these losses |

The networks can share an encoder, but then their losses interact through shared parameters. A small value MSE does not establish a good policy; the target itself may be wrong.

<div class="drl-paths" aria-label="Separate Actor and Critic update paths">
<section class="drl-path"><h3>Actor: change future choices</h3><ol><li>State → action distribution</li><li>Sampled action → environment feedback</li><li>Fixed advantage × log-probability</li><li>Update through log-probability only</li></ol><small>Detach the scoring signal; do not backpropagate through environmental reward.</small></section>
<section class="drl-path"><h3>Critic: revise expectations</h3><ol><li>State → current value</li><li>Rewards + tail estimate → fixed target</li><li>Compare prediction with target</li><li>Update through the current value</li></ol><small>Detach the target; a shared encoder still couples the two updates.</small></section>
</div>

## From a TD residual to advantage

If the Critic were exactly $V^\pi$, the expected one-step TD residual conditional on $(s_t,a_t)$ would equal $A^\pi(s_t,a_t)$:

$$
\delta_t=r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t),\qquad
\mathbb E[\delta_t\mid s_t,a_t]=Q^\pi(s_t,a_t)-V^\pi(s_t).
$$

In practice $V_\phi$ is approximate. Generalized Advantage Estimation (GAE) combines future TD residuals with geometrically decreasing weights:

$$
\hat A_t^{\rm GAE}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}
=\delta_t+\gamma\lambda\hat A_{t+1}.
$$

At $\lambda=0$, only one step contributes. At $\lambda=1$, a complete terminated trajectory telescopes to $G_t-V(s_t)$. A truncated rollout retains its bootstrap tail and is not full MC. Intermediate values trade value-estimation error against sampling noise; no $\lambda$ is universally best.

## Why GAE has these weights

Sum $n$ discounted residuals. Intermediate value terms cancel, leaving:

$$
\hat A_t^{(n)}=\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}
=\sum_{l=0}^{n-1}\gamma^l r_{t+l}+\gamma^n V(s_{t+n})-V(s_t).
$$

This is an $n$-step return minus the starting value estimate. Rather than choose one $n$, GAE mixes lengths. Take $N$ remaining steps within the same episode, without crossing a reset. Retain a tail bootstrap if the segment ends before the task does:

$$
\hat A_t^{\rm GAE}
=(1-\lambda)\sum_{n=1}^{N-1}\lambda^{n-1}\hat A_t^{(n)}
+\lambda^{N-1}\hat A_t^{(N)}.
$$

The final term takes the remaining weight. Do not mechanically multiply it by another $(1-\lambda)$ in a finite trace. These weights sum to 1; collecting coefficients of each residual gives $(\gamma\lambda)^l$.

| Three remaining steps, λ=0.8 | Mixture weight | How far it looks |
| --- | --- | --- |
| 1-step advantage | 0.2 | 1 reward, then the Critic |
| 2-step advantage | 0.16 | 2 rewards, then the Critic |
| 3-step advantage | 0.64 | End of this segment, then termination or bootstrap |

The endpoints become intuitive: $\lambda=0$ selects one step; $\lambda=1$ selects the longest segment. Longer does not automatically mean more accurate: reward noise and Critic error both matter.

## Calculate a trace, then move the slider

For $\delta=[1,2,-1]$, $\gamma=0.9$, and $\lambda=0.8$:

$$
\hat A_0=1+0.72\times2+0.72^2\times(-1)=1.9216.
$$

<div class="drl-lab" data-drl-lab="gae"><p>Static example: A₀ is 1 at λ=0, 1.9216 at λ=0.8, and 1.99 at λ=1. The slider shows each residual's contribution.</p></div>

The residuals are fixed teaching data. Moving the slider does not retrain a Critic; it changes only the estimator.

## Keep the two masks separate

One mask asks **whether the next state has future value**. The other asks **whether later residuals belong to this trajectory**. True termination disables both. External truncation can bootstrap, but the GAE recurrence must not cross a reset into another episode.

| Boundary | Use next-state value? | Add the next residual? | Observation used |
| --- | --- | --- | --- |
| Normal continuation | Yes | Yes | Next state |
| True task termination | No | No | No bootstrap |
| External timeout followed by reset | Yes | No | Final observation before reset |
| Full buffer, unfinished environment | Yes | Stop in this segment | Next state at the buffer tail |

If running out of time defines task completion, that is true termination; remaining time should generally be part of a finite-horizon task's state. External truncation here means the task could continue but the collector stopped.

```python
def gae(rewards, values, next_values, terminated, truncated,
        gamma=0.99, trace_decay=0.95):
    lengths = {len(part) for part in
               (rewards, values, next_values, terminated, truncated)}
    if len(lengths) != 1:
        raise ValueError("trajectory arrays must have equal lengths")
    result = [0.0] * len(rewards)
    carry = 0.0
    for step in reversed(range(len(rewards))):
        bootstrap = 0.0 if terminated[step] else next_values[step]
        residual = rewards[step] + gamma * bootstrap - values[step]
        same_episode = not (terminated[step] or truncated[step])
        carry = residual + gamma * trace_decay * same_episode * carry
        result[step] = carry
    return result
```

Each next_values entry must use that transition's final observation, not a reset observation. A rollout buffer that ends before the task does can also bootstrap its final value while stopping recurrence at the buffer boundary.

## Separate the Actor and Critic updates

The Actor minimizes $-\log\pi_\theta(a_t\mid s_t)\,\mathrm{stopgrad}(\hat A_t)$. The Critic often fits $\mathrm{stopgrad}(\hat A_t+V_{\rm old}(s_t))$. Fix this target before optimization rather than letting it drift with the value being trained.

PPO adds a new/old probability ratio and clipping to the Actor objective. GAE is not PPO itself. The existing [four-case PPO clipping interactive](../rlhf/ppo-clipping.en.md) shows that additional mechanism.

Reference: [Original GAE paper](https://arxiv.org/abs/1506.02438). Next, see [how far a policy should update](trust-region.en.md). For the Q-based route, continue with [DQN](dqn.en.md).
