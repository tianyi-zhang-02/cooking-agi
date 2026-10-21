# MC and TD: wait for the outcome or bootstrap?

[中文](returns-and-td.md) · **English**

> Reading time: ~6–9 min · Last reviewed: 2026-10

Bellman equations are convenient, but an environment rarely hands you a transition table. You have samples $(s,a,r,s')$. The first choice is simple: **wait for the full outcome, or update now using an estimate of what follows?**

## Monte Carlo uses the observed outcome

Suppose rewards are $[1,0,2]$, $\gamma=0.9$, and the final step terminates the episode:

$$
G_0=1+0.9\times0+0.9^2\times2=2.62.
$$

With current $V(s_0)=1$ and step size $\eta=0.2$, the update gives $1+0.2(2.62-1)=1.324$.

A complete on-policy episode provides an unbiased MC sample of the corresponding value, but it can be noisy and requires waiting. Unbiased does not mean accurate from one sample. Truncating without a tail estimate or mixing behavior policies changes the assumptions.

## TD combines one observation with an imperfect estimate

If the current next-state value is 1.5, the one-step TD target and error are:

$$
y_0=1+0.9\times1.5=2.35,\qquad
\delta_0=y_0-V(s_0)=1.35.
$$

The same step size gives 1.27. This is not an incorrect MC calculation; it is a different estimator. Bootstrapping constructs a new target from an existing estimate. It permits earlier updates but also propagates estimation error.

An n-step return lies between the two:

$$
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^nV(s_{t+n}).
$$

Stop the sum at true termination and remove the tail value. Larger $n$ usually relies more on observed trajectories and less on bootstrapping. The actual bias–variance change depends on the environment and value quality; it is not a monotonic performance guarantee.

## An episode boundary is not always termination

| Situation | Bootstrap? | Which next state? |
| --- | --- | --- |
| Success or failure ends the task | No | Terminal value is zero |
| An external collection limit stops a continuing task | Yes | The final observation before truncation |
| The environment automatically resets | Depends on the reason | Never substitute the next episode's initial observation |

<div class="drl-lab" data-drl-lab="boundary"><p>Static example: reward=1, next value=5, γ=0.9. The target is 1 at true termination and 5.5 at external truncation.</p></div>

If “finish within 20 steps” defines the task itself, reaching the limit may be genuine termination; remaining time should also be represented in the state. Not every time limit has the same semantics.

```python
def td_target(reward, next_value, gamma, terminated):
    if terminated or gamma == 0:
        return reward
    return reward + gamma * next_value

assert td_target(1.0, 5.0, 0.9, True) == 1.0
assert td_target(1.0, 5.0, 0.9, False) == 5.5
assert td_target(1.0, float("nan"), 0.9, True) == 1.0
```

Return the reward directly rather than multiplying an invalid successor estimate by zero: `0 * NaN` is still NaN. A zero discount also needs no successor estimate; callers should skip those unnecessary value evaluations.

## SARSA versus Q-learning: whose next action?

Both can update $Q(s,a)$ through TD. SARSA uses the next action $a'$ actually sampled by the behavior policy. Q-learning uses the maximum next-state Q, targeting a greedy policy:

$$
y_{\rm SARSA}=r+\gamma Q(s',a'),\qquad
y_{\rm Q}=r+\gamma\max_{a'}Q(s',a').
$$

These expressions assume a nonterminal transition. At true termination, both continuation terms are zero.

Let $Q(s',\cdot)=[2,5]$ and exploration select the first action. For $r=1,\gamma=0.9$, SARSA targets 2.8 and Q-learning targets 5.5. The distinction is not whether exploration exists; it is which continuation the target evaluates.

This is off-policy control: behavior may explore while the target uses a greedy continuation. One-step Q-learning conditions on $(s,a)$ and uses its actual transition in the same MDP, so it needs no importance ratio for the current action. Adequate state–action coverage still matters. This does not justify uncorrected multi-step off-policy returns.

## Compare three targets on the same trajectory

Keep rewards $[1,0,2]$, frozen estimates $V(s_1)=1.5,V(s_2)=1$, genuine termination after the third reward, and $\gamma=0.9$:

| Update | Target at the starting state | Observation versus estimate |
| --- | --- | --- |
| 1-step TD | $1+0.9\times1.5=2.35$ | One observed reward plus the next-state estimate |
| 2-step TD | $1+0.9\times0+0.9^2\times1=1.81$ | Two observed rewards plus a later estimate |
| Full MC | $1+0.9\times0+0.9^2\times2=2.62$ | Complete trajectory, no post-terminal value |

The two-step target happens to be farther from MC because $V(s_2)$ underestimates the remaining reward. A bias–variance tradeoff concerns repeated sampling, not a promise that more steps improve every trajectory. With an accurate Critic, additional observed steps may introduce more reward noise.

## Why is TD a semi-gradient update?

For linear approximation $V_\theta(s)=\theta^\top x(s)$, the update is:

$$
\theta\leftarrow\theta+\eta\,[r+\gamma V_\theta(s')-V_\theta(s)]x(s).
$$

The target $r+\gamma V_\theta(s')$ is treated as fixed. Gradients flow through the current-state estimate, not through $V_\theta(s')$. Differentiating the full squared residual introduces a next-state-feature term and produces a different update. Two implementations mentioning MSE need not optimize the same thing.

MC targets do not depend on current parameters and resemble ordinary supervised regression. TD targets move with the estimate. That helps propagate local information sooner, while making approximation stability more delicate. [Fitted Q](fitted-q.en.md) provides a concrete counterexample.

## Even tabular convergence has assumptions

For a finite-state MDP with bounded rewards and $0\le\gamma<1$, on-policy tabular TD(0) for a fixed policy converges under suitable sampling and learning-rate conditions, including infinitely many visits to each state being estimated. Standard stochastic-approximation conditions require each state’s step sizes to satisfy $\sum_t\eta_t=\infty$ and $\sum_t\eta_t^2<\infty$: keep enough opportunity to correct errors while gradually reducing sampling noise.

A constant step size may help track change, but it does not inherit an exact-convergence conclusion. Nonlinear approximation, off-policy data, and poor coverage require further care. Verify updates in a tiny tabular environment before adding the network.

## Check the implementation assumption

<details markdown="1">
<summary>Does replacing done with terminated fix every boundary bug?</summary>

Not necessarily. A vector environment may already have reset, and the reset timing depends on the mode. As of October 2026, Gymnasium 1.x vector environments default to next-step autoreset: a finished sub-environment resets on the following `step`, which returns the new episode's first observation. This reset is not a real transition, so drop or mask it before it enters the buffer. Only same-step mode resets on the ending step itself and puts the last observation separately in `info["final_obs"]`. See [Farama's explanation of vector autoreset modes (2025)](https://farama.org/Vector-Autoreset-Mode).

GAE also needs separate masks for bootstrapping and recurrence across episode boundaries. The [Actor–Critic chapter](actor-critic-gae.en.md) explains them separately. Inspect the environment API before choosing how to store arrays.

</details>

References: [Gymnasium v1.1.0 on termination and truncation](https://gymnasium.farama.org/v1.1.0/tutorials/gymnasium_basics/handling_time_limits/), [Watkins and Dayan (1992), §2–3: Q-learning sampling and convergence](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf). Next, see [how to update a policy directly](policy-gradients.en.md).
