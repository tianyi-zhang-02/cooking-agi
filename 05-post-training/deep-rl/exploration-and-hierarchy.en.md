# Exploration and hierarchy: randomness is not a strategy

[中文](exploration-and-hierarchy.md) · **English**

> Reading time: ~6–8 min · Last reviewed: 2026-10

A maze rewards only reaching the exit. More random movement may not help when success requires a coherent sequence. **Random actions and discovering useful behavior are different things.**

## Why epsilon and entropy may be insufficient

$\epsilon$-greedy occasionally picks a random action; an entropy bonus keeps action distributions spread out. If success requires 20 consecutive correct decisions, independent randomness may repeatedly interrupt a promising route. Exploration may need temporal consistency, not just more variation per step.

Possible approaches include state coverage, diverse policies, meaningful skills, and predictive uncertainty. The most unexplored region need not matter to the task, and real systems impose safety constraints.

## Why can twenty consecutive steps defeat randomness?

Consider a toy task that rewards twenty consecutive right moves; any left move resets progress. With independent equally likely left/right actions, a twenty-step attempt succeeds with probability $2^{-20}$, roughly one in a million.

If the agent already has “move right for twenty steps” and “move left for twenty steps” as skills, choosing each with probability one half gives a one-half success chance. But that skill library is extra structural knowledge, not a renamed version of the same random policy. Discovering useful skills and selecting them are central difficulties in hierarchical learning.

Inspect sustained progress, useful state coverage, and cost, not just action entropy. A toy maze without safety constraints is not a direct model of real-robot or online-product exploration.

## Intrinsic reward scores novelty

Random Network Distillation (RND) uses a fixed random network $f$ and a learned predictor $\hat f_\psi$. Error in predicting features of the current observation provides an intrinsic reward:

$$
r^{\rm int}(s)=\|\hat f_\psi(s)-f(s)\|^2,\qquad
r^{\rm total}=r^{\rm task}+\beta r^{\rm int}.
$$

An important distinction: prediction error for the *next frame* can attract an agent to a randomly flickering TV, whose future is inherently unpredictable. RND does not predict the next frame. Its target is deterministic given the current observation, avoiding that source of error from a stochastic prediction target.

Unfamiliar still does not mean useful. Features, finite data, and optimization error affect RND scores. Check observation normalization and the scale and decay of intrinsic rewards alongside actual task progress. An RND score is not automatically calibrated epistemic uncertainty.

## Shaping rewards without silently changing the goal

Arbitrarily rewarding movement toward a goal can invite reward-farming loops. Potential-based shaping uses:

$$
r'_t=r_t+\gamma\Phi(s_{t+1})-\Phi(s_t).
$$

The discounted sum telescopes, leaving the original return plus $-\Phi(s_0)+\gamma^T\Phi(s_T)$. For a fixed initial state and zero terminal potential, or a vanishing infinite-horizon discounted tail, policy comparisons are preserved. **Without these conditions, policy invariance is not guaranteed.**

## Calculate the shaping cancellation explicitly

A three-step trajectory has zero original rewards, $\gamma=0.9$, and potentials $[-3,-2,-1,0]$ at its four states. Added rewards are:

| Step | Added reward $\gamma\Phi(s')-\Phi(s)$ | Discounted contribution |
| --- | --- | --- |
| 0 | $0.9(-2)-(-3)=1.2$ | 1.2 |
| 1 | $0.9(-1)-(-2)=1.1$ | 0.99 |
| 2 | $0.9(0)-(-1)=1$ | 0.81 |

The sum is 3, exactly $-\Phi(s_0)$. Other trajectories from the same starting state with zero terminal potential receive the same constant addition, preserving their original ordering. Shaping redistributes feedback across steps rather than arbitrarily paying extra for a shorter route.

RND bonuses generally lack this telescoping structure and change the objective. An agent can prefer novelty after adding the bonus even though external rewards are unchanged. Preserve original task return as a separate evaluation measure.

## An option selects a chunk of behavior

An option usually specifies an initiation set, an internal policy, and a termination rule. A high-level controller selects “walk to the doorway,” and lower-level actions run for a variable duration $\tau$. Its target must account for time:

$$
y=\sum_{k=0}^{\tau-1}\gamma^kr_{t+k}
+\gamma^\tau V(s_{t+\tau}).
$$

After a 3-step option, the tail is discounted by $\gamma^3$, not $\gamma$. This is semi-Markov structure. Hierarchy can shorten high-level decision chains, but poorly chosen skills may restrict reachable policies.

## What to test first

Hold the external reward fixed and check whether exploration coverage and success improve together. Then isolate bonuses, skills, and budgets. Increased internal novelty is not task success.

This is an introduction, not a complete derivation of option-critic or Bayesian exploration. References: [RND](https://arxiv.org/abs/1810.12894) · [Potential-based shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) · [Options and semi-MDPs](https://www.sciencedirect.com/science/article/pii/S0004370299000521). Next: [Why policies can drift despite good demonstrations](imitation-and-rewards.en.md).
