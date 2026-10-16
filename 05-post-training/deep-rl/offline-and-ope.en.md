# Offline RL and OPE: what can old data tell us?

[中文](offline-and-ope.md) · **English**

> Reading time: ~6–8 min · Last reviewed: 2026-10

Having a replay buffer does not make a method offline RL. Online off-policy learning can still collect new data. Offline RL has only fixed logs. If a model proposes an uncovered action, **no fresh feedback arrives to correct its imagination**.

## Check support before choosing an algorithm

Suppose logs at some state $s$ contain only a conservative action, never an aggressive one. A Q-network can assign the aggressive action a high value, and the Bellman maximum can propagate that value backward. This is not evidence of a better policy; the estimate is untested.

More data from the same distribution can reduce estimation variance within covered regions. But repeating the same behavior a million times does not reveal the outcome of an action that behavior never takes. This is a support problem: the target policy needs states or actions outside the behavior distribution's support. An action missing from finite logs need not have zero probability under the behavior policy, but low probabilities and sparse samples can still make estimates unreliable. Also ask whether observations omit variables affecting both action choice and outcomes.

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

For finite, nonnegative weights with at least one positive weight, a useful diagnostic is:

$$
{\rm ESS}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
$$

With 3 trajectory weights $[1,1,8]$, ESS is approximately 1.52, not 3. It warns that a few trajectories may dominate. It does not establish unbiasedness or absence of hidden confounding.

Multiplying every weight by the same positive constant leaves ESS unchanged. Divide by the maximum before sums and squares to avoid overflow or squared-weight underflow. Starting from log-weights, subtract their largest finite value before exponentiation; rescaling cannot recover weights that have already overflowed to infinity or all underflowed to zero. If all weights are genuinely zero, ESS and SNIS are undefined, not valid zero-score estimates. This fixes numerical evaluation, not missing behavioral coverage.

## Calculate weights in a two-action bandit first

Left and right have fixed rewards 1 and 3. The logging policy chooses them with probabilities 90% and 10%; the target policy chooses each with 50%. Suppose 100 logged observations contain exactly 90 left and 10 right actions:

| Action | Count | Reward | $\pi/\mu$ | Total weighted reward |
| --- | --- | --- | --- | --- |
| Left | 90 | 1 | $0.5/0.9$ | 50 |
| Right | 10 | 3 | $0.5/0.1=5$ | 150 |

The raw mean is 1.2. IS gives $(50+150)/100=2$, the target policy’s true expectation. Right’s reward did not change; ten observations now represent its greater target-policy frequency. These are convenient expected counts; actual finite logs fluctuate.

If the logging policy never goes right, there is no right-action feedback. Adding a small constant to the denominator cannot invent those observations. In multistep settings the ratios multiply, and one rare trajectory may dominate the estimate. Inspect support and weight distributions first. How to multiply fewer ratios, and how a value model can cancel noise, is in [per-decision weights and doubly robust estimation](ope-estimators.en.md).

## What CQL and IQL try to avoid

| Method | Central idea | Remaining dependence |
| --- | --- | --- |
| CQL | Penalize unsupported high Q estimates conservatively | Regularization, coverage, and approximation quality |
| IQL | Learn from dataset actions, then extract a policy with advantage-weighted regression | Useful behavior must exist in the dataset |
| Behavior cloning | Learn logged actions directly | Demonstration quality and similar deployment distribution |

IQL's expectile value regression weights positive and negative residuals differently. For $u=Q(s,a)-V(s)$, the weight is $|\tau-\mathbf1(u<0)|$; $\tau>0.5$ emphasizes better in-dataset actions. An expectile is not a quantile: it uses weighted squared error, not pinball loss.

Conservatism alone is not a safety guarantee. Excessive conservatism may prevent improvements, and no algorithm can guarantee knowledge absent from the data.

## Write one step of conservative and in-data learning

A simplified discrete-action CQL penalty at a fixed state takes the form:

$$
\alpha\left[\log\sum_a\exp Q(s,a)-\mathbb E_{a\sim\mathcal D(\cdot\mid s)}Q(s,a)\right].
$$

Optimize it alongside Bellman fitting. The first term pushes down large Q values, while the second pushes data-action values back up relative to the others; it does not just subtract a constant from every Q. With only A observed, $Q(A)=1,Q(B)=5$ gives about 4.018 inside the brackets. Lowering unseen B to 0 gives about 0.313. This illustrates regularization pressure, not knowledge that B is actually bad.

IQL instead fits an expectile value from data-action Q values. For equally weighted values 0 and 4 with $\tau=0.8$, the optimum satisfies $0.8(4-V)=0.2V$, giving $V=3.2$. The mean is 2 and this distribution’s 0.8 quantile is 4: expectiles are different from quantiles.

Then regress $Q(s,a)$ toward $r+\gamma V(s')$ and extract a policy through exponentially advantage-weighted behavior cloning. Implementations still need fixed targets, termination handling, and control of extreme weights. Avoiding a maximization over unseen actions during training does not ensure a parameterized policy never leaves the data distribution at deployment.

## Can we estimate Q without trustworthy behavior probabilities?

Fitted Q Evaluation (FQE) fixes the policy $\pi$ being evaluated and repeatedly regresses toward:

$$
y=r+\gamma(1-d)\mathbb E_{a'\sim\pi(\cdot\mid s')}Q_{\rm old}(s',a').
$$

Here $d=1$ means true task termination, giving $y=r$. A data-collection time limit is not termination if the original task continues: bootstrap from the final successor $s'$ before the reset, not the new initial state after it.

Unlike FQI, it averages under a fixed policy rather than maximizing over actions. Estimate performance by averaging the fitted Q over initial states and policy actions. This avoids explicit trajectory weights but introduces approximation and extrapolation risks. Missing support does not disappear. Agreement between estimators sharing the same blind spot is not independent validation.

## A useful evaluation order

Start with time- or entity-separated train / validation / test splits to avoid trajectory leakage. Inspect behavioral support, weight tails, and effective sample size. Compare several OPE or model-based estimates and check ranking stability. Where permissible, follow with limited, controlled real validation.

Do not repeatedly select policies on the same logs and report the winning OPE score as independent evidence. Policy selection can overfit the estimator.

## Scope of this introduction

This chapter develops support, importance weighting, conservative learning, and in-dataset learning; per-decision importance sampling, doubly robust estimation, and LSTD follow in the [next note](ope-estimators.en.md). Neither note derives complete OPE confidence bounds or offline-RL convergence theorems. Read further: [CQL, §3.2, Eq. (4)](https://arxiv.org/abs/2006.04779) · [IQL, §§4.1–4.3](https://arxiv.org/abs/2110.06169) · [OPE and weight normalization, §§2–5](https://proceedings.mlr.press/v48/thomasa16.html) · [FQE, §3.2, Algorithm 3](https://proceedings.mlr.press/v97/le19a.html). For basic importance weighting, revisit [policy gradients](policy-gradients.en.md).
