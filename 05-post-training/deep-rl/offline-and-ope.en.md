# Offline RL and OPE: what can old data tell us?

[中文](offline-and-ope.md) · **English**

> Reading time: ~6–8 min · Last reviewed: 2026-10

Having a replay buffer does not make a method offline RL. Online off-policy learning can still collect new data. Offline RL has only fixed logs. If a model proposes an uncovered action, **no fresh feedback arrives to correct its imagination**.

## Check support before choosing an algorithm

Suppose logs at state A contain only a conservative action, never an aggressive one. A Q-network can assign the aggressive action a high value, and the Bellman maximum can propagate that value backward. This is not evidence of a better policy; the estimate is untested.

More data from the same distribution can reduce estimation variance within covered regions. But repeating the same behavior a million times does not reveal the outcome of an action that behavior never takes. This is a support problem: the target policy needs states or actions absent from the data. Also ask whether observations omit variables affecting both action choice and outcomes.

## OPE evaluates without deploying

Off-policy evaluation assesses a fixed policy; offline RL learns one. Basic trajectory importance sampling gives:

$$
\hat J_{\rm IS}=\frac1N\sum_{i=1}^Nw_iR_i,\qquad
w_i=\prod_t\frac{\pi(a_t^i\mid s_t^i)}{\mu(a_t^i\mid s_t^i)}.
$$

Here $R_i$ is the return of complete trajectory $i$. The target policy and logs must share the initial-state distribution and environment, behavior probabilities $\mu$ must be reliable, and target trajectories must have behavioral support. Self-normalization replaces $N$ with the sum of weights; it does not divide the preceding mean a second time:

$$
\hat J_{\rm SNIS}=\frac{\sum_iw_iR_i}{\sum_iw_i},\qquad \sum_iw_i>0.
$$

This is generally biased at finite sample size but may reduce variance. Clipping weights similarly trades bias for stability.

A useful weight diagnostic is:

$$
{\rm ESS}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
$$

With 3 trajectory weights $[1,1,8]$, ESS is approximately 1.52, not 3. It warns that a few trajectories may dominate. It does not establish unbiasedness or absence of hidden confounding.

## What CQL and IQL try to avoid

| Method | Central idea | Remaining dependence |
| --- | --- | --- |
| CQL | Penalize unsupported high Q estimates conservatively | Regularization, coverage, and approximation quality |
| IQL | Learn from dataset actions, then extract a policy with advantage-weighted regression | Useful behavior must exist in the dataset |
| Behavior cloning | Learn logged actions directly | Demonstration quality and similar deployment distribution |

IQL's expectile value regression weights positive and negative residuals differently. For $u=Q(s,a)-V(s)$, the weight is $|\tau-\mathbf1(u<0)|$; $\tau>0.5$ emphasizes better in-dataset actions. An expectile is not a quantile: it uses weighted squared error, not pinball loss.

Conservatism alone is not a safety guarantee. Excessive conservatism may prevent improvements, and no algorithm can guarantee knowledge absent from the data.

## A useful evaluation order

Start with time- or entity-separated train / validation / test splits to avoid trajectory leakage. Inspect behavioral support, weight tails, and effective sample size. Compare several OPE or model-based estimates and check ranking stability. Where permissible, follow with limited, controlled real validation.

Do not repeatedly select policies on the same logs and report the winning OPE score as independent evidence. Policy selection can overfit the estimator.

## Scope of this introduction

This chapter develops support, importance weighting, conservative learning, and in-dataset learning. It does not derive complete OPE confidence bounds or offline-RL convergence theorems. Read further: [CQL](https://arxiv.org/abs/2006.04779) · [IQL](https://arxiv.org/abs/2110.06169). For basic importance weighting, revisit [policy gradients](policy-gradients.en.md).
