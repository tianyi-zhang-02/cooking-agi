# From DDPG to TD3: choosing continuous actions

[中文](continuous-control.md) · **English**

> Reading time: ~10–12 min · Last reviewed: 2026-10

Discrete actions can be enumerated; steering angles cannot. DDPG learns an additional network $\mu_\theta(s)$ that proposes an action, then improves it along the Critic's slope. TD3 modifies this approach to reduce the damage from estimation errors.

## The Actor outputs coordinates, not a category

The Critic maps $(s,a)$ to scalar Q. The Actor maps $s$ to a bounded continuous action. Its objective encourages actions the Critic values:

$$
\nabla_\theta J\approx
\mathbb E_{s\sim\mathcal D}
[\nabla_aQ_\phi(s,a)|_{a=\mu_\theta(s)}\,\nabla_\theta\mu_\theta(s)].
$$

This is the practical replay-state update. Writing it as a gradient does not eliminate questions about off-policy state distributions. During the Actor step, Critic parameters are fixed, but **the derivative of Q with respect to the action must remain connected**.

For $Q(s,a)=-(a-0.7)^2$ and current action 0.2, the slope is $-2(0.2-0.7)=1$, pushing toward 0.7. If that high value is a Critic approximation error, the Actor still moves toward it, exploiting the error.

## One DDPG update

Sample replay data and form $y=r+\gamma(1-d)Q_{\bar\phi}(s',\mu_{\bar\theta}(s'))$, where $d$ denotes true termination only. Fit the current Critic to this target. Then update the Actor with $-Q_\phi(s,\mu_\theta(s))$ and slowly synchronize target parameters.

```text
replay (s, a, r, s', d)
    → target actor(s') → target critic(s', a') → r + γ(1-d)Q' → detached target
    → critic(s, a) → regression update
    → actor(s) → critic(s, actor(s)) → actor update
```

Behavior collection adds noise to deterministic actions. This exploration noise is not SAC's policy entropy, nor the target smoothing below.

Replay stores the executed action, including exploration noise and boundary clipping. An external time-limit truncation, unlike task termination, still bootstraps from the final pre-reset $s'$, not the new reset state. This boundary applies to both DDPG and TD3.

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

## Follow the Actor gradient back to parameters

Keep $Q(a)=-(a-0.7)^2$ and let the Actor output $a=\tanh\theta$. At $a=0.2$:

$$
\frac{dQ}{d\theta}=\frac{dQ}{da}\frac{da}{d\theta}
=-2(a-0.7)\cdot(1-a^2)=1\times0.96=0.96.
$$

Gradient ascent increases $\theta$, moving toward 0.7. Near ±1, the factor $1-a^2$ becomes small and the action saturates. Bounds and scaling affect both environment meaning and optimization; treating controls with different units as identically scaled is not neutral.

Freezing Critic parameters does not mean freezing Q as a function of the action. You may disable gradient accumulation on Critic parameters while retaining the Q → action → Actor graph. Otherwise the Actor has no direction to follow.

## Can the minimum of two Q estimates be too conservative?

Yes. Give two estimates of the same true zero-valued action independent, equally likely ±1 errors. The four minima are $[-1,-1,-1,1]$, averaging −0.5. Suppressing optimism can introduce pessimism. Real Critics have correlated errors, but the example disproves the intuition that the smaller estimate must be more accurate.

For smoothing, consider a quadratic peak $Q(a)=-(a-a^*)^2$ and zero-mean noise with variance $\sigma^2$, temporarily ignoring boundary clipping:

$$
\mathbb E_\epsilon[Q(a+\epsilon)]=-(a-a^*)^2-\sigma^2.
$$

Smoothing evaluates a neighborhood, not just a point. For a peak $Q(a)=-k(a-a^*)^2$, smoothing subtracts $k\sigma^2$: the sharper the peak (larger $k$), the larger the penalty, so narrow spikes are less able to dominate the target. Neural-network peaks can have more complex shapes; noise truncation and action clipping change the distribution. Interpret noise relative to action range rather than copying a scale between environments.

## Can you take the max over continuous actions without training an actor?

DDPG's Actor is essentially an "approximate argmax function": in each state it finds a high-scoring action on the Critic's behalf. For the $\max_{a'}Q(s',a')$ in a Q-learning target, there are two other options that train no separate network.

**Maximize by sampling.** Whenever you need the max, sample $N$ actions, score each with the Critic, and keep the best; this is random shooting. CEM iterates a few more rounds: keep the best few, fit a new Gaussian, and sample again (a one-step hand calculation is in [planning and LQR](planning-and-control.en.md)).

[QT-Opt (Kalashnikov et al., 2018)](https://arxiv.org/abs/1806.10293) replaces the Actor with CEM for vision-based grasping: 64 samples per round, keep 6, two rounds. The same procedure computes training targets and chooses actions on real robots. The authors argue that this avoids actor–critic instability and say CEM is moderately robust to local optima in low-dimensional problems.

Counting candidates, one max requires Q values for $64\times2=128$ actions, and every sample in a batch needs such a search for its target. Candidates within a round can be evaluated in a batch; the next round depends on the previous round's elites. Thus 128 candidate evaluations do not mean 128 serial network calls, and latency also depends on batching and hardware. The best sampled value never exceeds the Q curve's own maximum, so targets run low with few samples. Relative to the true value, errors in Q itself are still amplified by the max.

**Make Q easy to maximize.** [NAF (Gu et al., 2016)](https://arxiv.org/abs/1603.00748) writes Q as a state value minus a term quadratic in the action:

$$
Q(s,a)=V(s)-\tfrac12\big(a-\mu(s)\big)^\top P(s)\big(a-\mu(s)\big),\qquad P(s)=L(s)L(s)^\top .
$$

$L(s)$ is a lower-triangular matrix output by the network, with exponentiated diagonal entries so that $P(s)$ is positive definite. Q then peaks at $a=\mu(s)$ with value $V(s)$, and the max in the target needs no search. What you give up is expressiveness: in each state, Q has a single peak in the action. The paper itself acknowledges that the quadratic form is more restrictive than a general network; the QT-Opt authors argue that such shapes are a poor fit for grasping, where reaching toward objects is valuable and reaching into the gaps between them is not, so Q has several peaks in the action.

Work it once by hand. Take a one-dimensional action $a\in[-1,1]$ and a made-up Q curve $Q(a)=\max\big(1-8(a-0.5)^2,\;0.8-2(a+0.5)^2\big)$ with two peaks: a narrow, high one on the right reaching the true maximum 1 at $a=0.5$, and a broad, low one on the left reaching 0.8 at $a=-0.5$.

- **8 random samples**: suppose one uniform draw happens to give −0.85, −0.6, −0.35, −0.1, 0.15, 0.3, 0.75, 0.95, with Q values 0.555, 0.78, 0.755, 0.48, 0.02, 0.68, 0.5, −0.62. The best is $a=-0.6$, on the low left peak; 0.3, the sample closest to the true peak, scores only 0.68 because the right peak is so narrow.
- **CEM**: use these 8 samples as round 1 and keep the best 2 (−0.6, −0.35); the new Gaussian has mean −0.475 and standard deviation 0.125. Beating 0.8 requires $1-8(a-0.5)^2>0.8$, or $a\in(0.5-\sqrt{0.2/8},\,0.5+\sqrt{0.2/8})$, with endpoints approximately 0.3419 and 0.6581; $a>0.34$ is only a rough necessary condition. The interval's lower end is more than 6.5 standard deviations above the new mean, so a small next batch from this Gaussian is extremely unlikely to hit it, but the probability is not zero. If later samples remain near the left peak, search may keep contracting toward −0.5; the first round does not guarantee that convergence. A variance floor, restarts, or a mixture with uniform samples changes subsequent search.
- **NAF-style quadratic Q**: suppose a function quadratic in the action is fit to this curve with equal weight over $[-1,1]$ (a teaching assumption; real training uses Bellman regression, and the data distribution decides which peak it leans toward). The fitted peak sits at $a\approx-0.23$, where the true Q is only about 0.65, lower than either peak.
- **Actor gradient ascent**: DDPG climbs along $\partial Q/\partial a$. Starting from $a=0$, the left piece is higher with slope $-2$, and the actor climbs to −0.5; starting from $a=0.2$, the right piece is higher with slope 4.8, and it climbs to 0.5.

| Method | Computation per max | Action found | True Q there |
| --- | --- | --- | --- |
| 8 random samples | 8 candidate Q evaluations, batchable | −0.6 | 0.78 |
| CEM, 8 per round, keep 2 | 8 candidate Q evaluations per round, batchable | About −0.5 (if search stays near the left peak) | About 0.8 |
| NAF-style quadratic Q | 1 forward pass, read off $\mu(s)$ | About −0.23 | About 0.65 |
| Learned Actor | 1 forward pass, plus Actor training | Depends on the start: −0.5 or 0.5 | 0.8 or 1 |
| True maximum | — | 0.5 | 1 |

The example illustrates mechanisms, not performance: none of the methods automatically finds the narrow peak. With only a few action dimensions, large-scale training, and expensive tuning, sampling or CEM removes a hard-to-tune network at the cost of many Q calls at inference. When value is close to unimodal in the action (say, "reach a position"), NAF-style methods are cheapest. With dozens of action dimensions and a high control rate, the number of samples needed grows quickly with dimension, and learning an Actor (DDPG, TD3) pays off, provided you accept that it also climbs only locally and chases the Critic's errors.

## Trace one update round

Make the update switches explicit. For illustration, update the Actor once every 2 Critic steps:

| Branch | Networks used | Gradient destination | Timing |
| --- | --- | --- | --- |
| Next-step target | Target Actor + two target Critics | No backpropagation through this branch | Every Critic step |
| Current-Q regression | Two current Critics | Each Critic's parameters | Every update |
| Current-action improvement | Current Actor + first current Critic | Through Q's action derivative into the Actor | Every 2 updates |
| Slow synchronization | Current parameters → target parameters | Not an optimizer gradient | Usually with delayed Actor updates |

Initialize each target network by copying its current network's parameters. A common synchronization convention is $\bar\phi\leftarrow(1-\tau)\bar\phi+\tau\phi$, with the same update for the target Actor. Other code uses $\rho$ for the old-parameter weight, equal to $1-\tau$. A coefficient close to 1 therefore does not necessarily mean fast synchronization.

Work a target by hand: $r=1,\gamma=0.9$, and the same smoothed next action receives target-Q estimates 4 and 6. A nonterminal target is $y=1+0.9\times4=4.6$; a terminal target is 1. If current Q estimates are 3 and 5, squared errors against the frozen target are 2.56 and 0.16. No Actor update has happened yet.

| Three easily confused noises | Where it appears | Purpose |
| --- | --- | --- |
| Behavioral exploration noise | Actions executed in the environment | Collect different data |
| Target-smoothing noise | Next-action target branch during training | Prevent sharp local peaks from dominating bootstrap targets |
| Q estimation error | Learned value estimates | An error to manage, not deliberate exploration |

This approach is attractive for continuous control when interaction is expensive and historical transitions can be reused. It adds replay, several target networks, update schedules, and action-scale decisions. Here, update-to-data ratio counts Critic minibatch update rounds per newly collected environment transition: updating both Q-functions on one batch counts as one round, without adding Actor optimizer steps. Report batch size separately. Increasing the ratio can save interaction or overfit stale data.

<details markdown="1">
<summary>Can the whole Critic call run under no_grad during the Actor update?</summary>

No. The Actor needs Q's action derivative. You can temporarily freeze Critic parameters to avoid accumulating their gradients while retaining the graph from Q through the action to the Actor. The Critic bootstrap-target branch is the part that should use no_grad.

</details>

References: [DDPG](https://arxiv.org/abs/1509.02971) · [TD3](https://arxiv.org/abs/1802.09477) · [TD3 algorithm guide](https://spinningup.openai.com/en/latest/algorithms/td3.html). Next: [Why SAC includes randomness in the objective](soft-actor-critic.en.md).
