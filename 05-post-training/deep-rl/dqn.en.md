# DQN: why learning Q with a network can be unstable

[中文](dqn.md) · **English**

> Reading time: ~7 min · Last reviewed: 2026-10

With actions such as “left, right, stop,” a network can output 3 Q-values and choose the largest. It resembles regression, except the labels are not fixed ground truth: **the target contains the network's own estimates.**

For a concrete case where regression loss improves while values drift, start with the [two-state fitted Q counterexample](fitted-q.en.md). This chapter explains how DQN mitigates related problems.

## Turn a transition into a loss

A replay buffer stores $(s,a,r,s',d)$. The online network $Q_\theta$ predicts the current action's value; a slower target network $Q_{\bar\theta}$ constructs its label:

$$
y=r+\gamma(1-d)\max_{a'}Q_{\bar\theta}(s',a'),\qquad
L(\theta)=\mathbb E[(Q_\theta(s,a)-\mathrm{stopgrad}(y))^2].
$$

Here $d$ means true termination, not every reset. Replay reduces strong sequential correlations and reuses interaction. A target network slows target drift. These mitigate problems; they do not guarantee convergence with neural approximation.

This is a semi-gradient update: differentiate the current Q, not the target. It is not a full gradient through every occurrence in a Bellman residual.

## Double DQN separates selection from evaluation

A maximum tends to select actions whose noisy estimates are too high. Double DQN selects with the online network and evaluates with the target network:

$$
a^*=\arg\max_a Q_\theta(s',a),\qquad
y_{\rm DDQN}=r+\gamma(1-d)Q_{\bar\theta}(s',a^*).
$$

Let online Q be $[5,4]$, target Q be $[2,6]$, and $r=1,\gamma=0.9$. DQN uses the target maximum of 6, giving 6.4. Double DQN selects the first action using online Q, then uses its target value of 2, giving 2.8.

<div class="drl-lab" data-drl-lab="double-q"><p>Static comparison: DQN target=6.4; Double DQN target=2.8. The interactive changes the online ranking.</p></div>

We do not know the true $Q^*$ here, so **a smaller target is not evidence of greater accuracy**. This example illustrates decoupling only. The networks remain correlated and can also underestimate.

## Inspect shapes and gradients in code

The code uses the same Double DQN target but replaces squared error with Smooth L1—equivalent here to Huber loss with threshold 1—to limit the influence of large TD errors on gradients. Action selection and evaluation are unchanged.

```python
import torch

def double_dqn_loss(online, target, states, actions, rewards,
                    next_states, terminated, gamma=0.99):
    prediction = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_actions = online(next_states).argmax(dim=1, keepdim=True)
        next_q = target(next_states).gather(1, next_actions).squeeze(1)
        expected = rewards + gamma * (~terminated).float() * next_q
    return torch.nn.functional.smooth_l1_loss(prediction, expected)
```

states has shape [batch, state_dim], actions is integer [batch], and terminated is boolean [batch]. Do not accidentally broadcast a [batch, 1] prediction against a [batch] target into [batch, batch]. With dropout or batch normalization, no_grad does not automatically select evaluation mode; explicitly define the target network's mode.

## Similar names, different changes

| Change | What it changes | What it does not change |
| --- | --- | --- |
| Double DQN | Decouples next-action selection and evaluation | Still Q-learning |
| Dueling network | Combines value and action advantage into Q | Not two target networks |
| Prioritized replay | Samples some transitions more frequently | Sampling shifts require attention to correction and bias |

Exploration often uses $\epsilon$-greedy. Random training actions collect information; document whether evaluation disables them. Otherwise low scores from exploration and low scores from the policy itself become hard to separate.

## Check these before tuning

First test that terminal targets equal reward. Then verify that targets have no gradients, replay actions align with their states, and target synchronization uses the intended step units. If loss falls while return does not improve, inspect these semantics before enlarging the network.

Reference: [Double DQN](https://arxiv.org/abs/1509.06461). Next: [What if continuous actions make enumeration impossible?](continuous-control.en.md)
