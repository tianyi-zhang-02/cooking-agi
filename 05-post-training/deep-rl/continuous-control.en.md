# From DDPG to TD3: choosing continuous actions

[中文](continuous-control.md) · **English**

> Reading time: ~8–10 min · Last reviewed: 2026-10

Discrete actions can be enumerated; steering angles cannot. DDPG learns an additional network $\mu_\theta(s)$ that proposes an action, then improves it along the Critic's slope. TD3 modifies this approach to reduce the damage from estimation errors.

## The Actor outputs coordinates, not a category

The Critic maps $(s,a)$ to scalar Q. The Actor maps $s$ to a bounded continuous action. Its objective encourages actions the Critic values:

$$
\nabla_\theta J\approx
\mathbb E_{s\sim\mathcal D}
[\nabla_aQ_\phi(s,a)|_{a=\mu_\theta(s)}\,\nabla_\theta\mu_\theta(s)].
$$

This is the practical replay-state update. Writing it as a gradient does not eliminate questions about off-policy state distributions. During the Actor step, Critic parameters are fixed, but **the derivative of Q with respect to the action must remain connected**.

For $Q(s,a)=-(a-0.7)^2$ and current action 0.2, the slope is $-2(0.2-0.7)=1$, pushing toward 0.7. If that peak is a Critic approximation error, the Actor will just as confidently climb a fictitious hill.

## One DDPG update

Sample replay data, generate next actions with a target Actor, and evaluate them with a target Critic. Fit the current Critic to this bootstrap target. Then update the Actor with $-Q_\phi(s,\mu_\theta(s))$ and slowly synchronize target parameters.

```text
replay (s, a, r, s')
    → target actor(s') → target critic(s', a') → detached target
    → critic(s, a) → regression update
    → actor(s) → critic(s, actor(s)) → actor update
```

Behavior collection adds noise to deterministic actions. This exploration noise is not SAC's policy entropy, nor the target smoothing below.

## Why TD3 combines 3 changes

| Change | Mechanism | Main issue addressed |
| --- | --- | --- |
| Twin Critics | Use the smaller target Q | Reduce exploitation of overestimated values |
| Delayed Actor updates | Update Critics more frequently than the Actor | Avoid chasing temporarily unreliable Q estimates |
| Target policy smoothing | Add clipped noise near the next action | Prevent sharp, fragile Q peaks from dominating targets |

The TD3 target is:

$$
\tilde a'=\mathrm{clip}(\mu_{\bar\theta}(s')+\mathrm{clip}(\epsilon,-c,c)),
\qquad
y=r+\gamma(1-d)\min_{i=1,2}Q_{\bar\phi_i}(s',\tilde a').
$$

Here $d$ denotes true termination, $\epsilon\sim\mathcal N(0,\sigma^2I)$ is smoothing noise, and $c$ bounds its magnitude. The outer clip then enforces the environment's action bounds. The Q-networks are not statistically independent, and their minimum can underestimate value. The Actor commonly updates through the first current Critic.

## Trace one update round

Make the update switches explicit. For illustration, update the Actor once every 2 Critic steps:

| Branch | Networks used | Gradient destination | Timing |
| --- | --- | --- | --- |
| Next-step target | Target Actor + two target Critics | No backpropagation through this branch | Every Critic step |
| Current-Q regression | Two current Critics | Each Critic's parameters | Every update |
| Current-action improvement | Current Actor + first current Critic | Through Q's action derivative into the Actor | Every 2 updates |
| Slow synchronization | Current parameters → target parameters | Not an optimizer gradient | Usually with delayed Actor updates |

A common convention is $\bar\phi\leftarrow(1-\tau)\bar\phi+\tau\phi$. Other code uses $\rho$ for the old-parameter weight, equal to $1-\tau$. A coefficient close to 1 therefore does not necessarily mean fast synchronization.

Work a target by hand: $r=1,\gamma=0.9$, and the same smoothed next action receives target-Q estimates 4 and 6. A nonterminal target is $y=1+0.9\times4=4.6$; a terminal target is 1. If current Q estimates are 3 and 5, squared errors against the frozen target are 2.56 and 0.16. No Actor update has happened yet.

| Three easily confused noises | Where it appears | Purpose |
| --- | --- | --- |
| Behavioral exploration noise | Actions executed in the environment | Collect different data |
| Target-smoothing noise | Next-action target branch during training | Prevent sharp local peaks from dominating bootstrap targets |
| Q estimation error | Learned value estimates | An error to manage, not deliberate exploration |

This approach is attractive for continuous control when interaction is expensive and historical transitions can be reused. It adds replay, several target networks, update schedules, and action-scale decisions. The update-to-data ratio counts optimizer updates per environment step. Increasing it can save interaction or overfit stale data.

<details markdown="1">
<summary>Can the whole Critic call run under no_grad during the Actor update?</summary>

No. The Actor needs Q's action derivative. You can temporarily freeze Critic parameters to avoid accumulating their gradients while retaining the graph from Q through the action to the Actor. The Critic bootstrap-target branch is the part that should use no_grad.

</details>

References: [DDPG](https://arxiv.org/abs/1509.02971) · [TD3](https://arxiv.org/abs/1802.09477) · [TD3 algorithm guide](https://spinningup.openai.com/en/latest/algorithms/td3.html). Next: [Why SAC includes randomness in the objective](soft-actor-critic.en.md).
