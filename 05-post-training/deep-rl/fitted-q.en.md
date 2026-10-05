# From tables to networks: why better regression can mean worse values

[中文](fitted-q.md) · **English**

> Reading time: ~7 min · Last reviewed: 2026-10

The [Bellman chapter](mdp-bellman.en.md) established convergence for exact tabular iteration under discounting and bounded rewards. Why does replacing the table with a neural network change the conclusion? **Every update must now fit back into a restricted function class.**

## Fitted Q-iteration makes labels, then fits them

Fix transitions $\mathcal D=\{(s_i,a_i,r_i,s'_i,d_i)\}$, where $d_i$ means true termination. At each iteration, construct labels with the previous function:

$$
y_i^{(k)}=r_i+\gamma(1-d_i)\max_{a'}Q_k(s'_i,a'),
\qquad
Q_{k+1}\in\arg\min_{Q\in\mathcal F}\sum_i\left(Q(s_i,a_i)-y_i^{(k)}\right)^2.
$$

This is fitted Q-iteration (FQI). Labels remain fixed within a regression round and change at the next round. DQN performs related approximate iterations using replay, gradient steps, and periodic target updates, rather than solving each regression exactly.

<div class="drl-flow drl-sequence" aria-label="Fitted Q-iteration">
<span>Old Q<br><small>Construct fixed labels</small></span><b>→</b><span>Regression<br><small>Stay in the function class</small></span><b>→</b><span>New Q<br><small>Build labels next round</small></span>
</div>

## The extra operation is a projection

Temporarily ignore finite-sample and optimization errors. Write regression as a projection $\Pi_\mu$, where $\mu$ is the fitting distribution:

$$
Q_{k+1}=\Pi_\mu\mathcal TQ_k.
$$

The Bellman operator $\mathcal T$ is a $\gamma$-contraction in the maximum norm $\|\cdot\|_\infty$. Least-squares orthogonal projection onto a linear function space is nonexpansive in the corresponding weighted $L_2(\mu)$ norm. **Those guarantees use different norms and cannot simply be multiplied.**

With a nonconvex neural function class, projection may also be nonunique and optimization may miss its optimum. Even the ideal linear-projection properties no longer apply automatically. This does not mean all function approximation diverges: specific function classes, distributions, and algorithms can still admit convergence guarantees.

## A two-state counterexample

Consider an original value-iteration example: both states transition deterministically to state 2, all rewards are zero, and $\gamma=0.9$. There is one action per state, so Q and V are equivalent and the true values are both zero.

Restrict the representation to $V_\theta=[\theta,2\theta]$ and give the states equal fixed regression weight. Bellman labels are $[1.8\theta_k,1.8\theta_k]$. Exact least squares gives:

$$
\theta_{k+1}
=\arg\min_u\left[(u-1.8\theta_k)^2+(2u-1.8\theta_k)^2\right]
=1.08\theta_k.
$$

Starting at $\theta_0=1$, multiplying by 1.08 each round moves away from zero. The true answer is in the function class, yet the iteration still fails.

| Iteration | $\theta$ | Predicted values |
| --- | --- | --- |
| 0 | 1 | [1, 2] |
| 1 | 1.08 | [1.08, 2.16] |
| 2 | 1.1664 | [1.1664, 2.3328] |

In the first regression, mean squared error against **the same old labels** falls from 0.34 to 0.324. The inner fit really improves. But labels change next round, so this is not progress toward the true value function.

The fixed uniform fitting distribution is not this Markov process's stationary distribution, which concentrates on state 2. This example does not show that all on-policy TD diverges. It refutes unconditional inheritance of tabular convergence for arbitrary sampling distributions and projections.

<div class="drl-lab" data-drl-lab="projection"><p>Step through fits and compare value curves with each round's MSE table. At γ=0.9, θ multiplies by 1.08; at γ=0.5, it multiplies by 0.6. Both fit fixed labels within each round.</p></div>

<details markdown="1">
<summary>What changes if we only change the discount?</summary>

The general recurrence is $\theta_{k+1}=1.2\gamma\theta_k$. At $\gamma=0.5$, the multiplier is 0.6 and this example converges. Lowering the discount also changes how much the task values the future, so it is not a free fix. The accompanying [Python checks](code/rl_checks.py) verify both cases.

</details>

## Gradient descent on which objective?

FQI may use gradient descent to fit each fixed set of labels. Q-learning's semi-gradient is also the gradient against its current fixed target. The outer targets change with Q, however: **neither is full gradient descent on one permanently fixed overall Bellman residual.**

Differentiating $\mathbb E[(Q-\mathcal TQ)^2]$ fully introduces estimation of products of conditional expectations under stochastic transitions. Reusing one next-state sample for both terms is generally insufficient. This double-sampling issue is different from Double DQN's separation of action selection and evaluation.

## Why can one next-state sample miss the full gradient?

Set aside the RL notation. Let $Y_\theta$ be a random target and $c_\theta$ the current prediction. The intended squared residual against the expected target is:

$$
L(\theta)=(c_\theta-\mathbb E[Y_\theta])^2.
$$

Squaring a single sampled residual instead gives:

$$
\mathbb E[(c_\theta-Y_\theta)^2]
=(c_\theta-\mathbb E[Y_\theta])^2+\operatorname{Var}(Y_\theta).
$$

If target variance depends on parameters, that extra term has a gradient. For $c_\theta=0$ and a target equally likely to be $+\theta$ or $-\theta$, the intended residual is always zero, while expected sample-squared error is $\theta^2$. The update also optimizes variance.

A full Bellman-residual gradient contains a product involving conditional expectations of the residual and target derivative. Reusing one transition for both generally introduces dependence. Independent next-state samples can address the product, but a real environment may not let you resample exactly the same state–action pair. Semi-gradients avoid part of this computation by changing the update, not by becoming unbiased full gradients.

## Which error decreased when the loss fell?

| Error | Question | Diagnostic |
| --- | --- | --- |
| Optimization | Did the network fit this round’s fixed targets? | Freeze a tiny batch and its targets |
| Approximation | Can the function class express the needed Q? | Compare representations where true values are known |
| Statistical and coverage | Do finite samples represent the regions the policy needs? | Inspect state coverage and independent rollouts |

A larger model mainly changes representation capacity, not missing data. More gradient steps mainly improve the current regression, not necessarily the outer iteration. The two-state example diverges even with exact inner fitting, separating these concerns.

## Each stabilization component addresses one problem

| Component | Main issue it addresses | What it does not guarantee |
| --- | --- | --- |
| Replay buffer | Temporal correlations and wasted samples | Adequate coverage or unbiased sampling |
| Target network | Labels moving immediately with current updates | Convergence of the outer iteration |
| Double Q | Selection bias from maximizing noisy estimates | More accurate Q everywhere or no underestimation |
| Independent rollout evaluation | Training loss diverging from actual behavior | Coverage of every failure with finite evaluation |

Reference: [A classic batch fitted Q-iteration method](https://jmlr.org/papers/v6/ernst05a.html). The two-state calculation above is an independently constructed example, not an experimental result from that paper. Continue with [DQN implementation and checks](dqn.en.md).
