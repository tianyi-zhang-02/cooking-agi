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
    return reward + gamma * (not terminated) * next_value

assert td_target(1.0, 5.0, 0.9, True) == 1.0
assert td_target(1.0, 5.0, 0.9, False) == 5.5
```

## SARSA versus Q-learning: whose next action?

Both can update $Q(s,a)$ through TD. SARSA uses the next action $a'$ actually sampled by the behavior policy. Q-learning uses the maximum next-state Q, targeting a greedy policy:

$$
y_{\rm SARSA}=r+\gamma Q(s',a'),\qquad
y_{\rm Q}=r+\gamma\max_{a'}Q(s',a').
$$

These expressions assume a nonterminal transition. At true termination, both continuation terms are zero.

Let $Q(s',\cdot)=[2,5]$ and exploration select the first action. For $r=1,\gamma=0.9$, SARSA targets 2.8 and Q-learning targets 5.5. The distinction is not whether exploration exists; it is which continuation the target evaluates.

## Check the implementation assumption

<details markdown="1">
<summary>Does replacing done with terminated fix every boundary bug?</summary>

Not necessarily. A vector environment may already have reset, so you must retain the final observation. GAE also needs separate masks for bootstrapping and recurrence across episode boundaries. The Actor–Critic chapter separates them. Inspect the environment API before choosing how to store arrays.

</details>

Reference: [Gymnasium on termination and truncation](https://farama.org/Gymnasium-Terminated-Truncated-Step-API). Next, see [how to update a policy directly](policy-gradients.en.md).
