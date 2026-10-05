# Exploration and hierarchy: randomness is not a strategy

[中文](exploration-and-hierarchy.md) · **English**

> Reading time: ~6–8 min · Last reviewed: 2026-10

A maze rewards only reaching the exit. More random movement may not help when success requires a coherent sequence. **Random actions and discovering useful behavior are different things.**

## Why epsilon and entropy may be insufficient

$\epsilon$-greedy occasionally picks a random action; an entropy bonus keeps action distributions spread out. If success requires 20 consecutive correct decisions, independent randomness may repeatedly interrupt a promising route. Exploration may need temporal consistency, not just more variation per step.

Possible approaches include state coverage, diverse policies, meaningful skills, and predictive uncertainty. The most unexplored region need not matter to the task, and real systems impose safety constraints.

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
