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

The discounted sum telescopes, leaving the original return plus $-\Phi(s_0)+\gamma^T\Phi(s_T)$. With zero terminal potential, every policy's return shifts by $-\Phi(s_0)$ from each starting state. Expected-return rankings are therefore preserved not only for a fixed initial state but also under the same policy-independent initial-state distribution. On an infinite horizon, $0\le\gamma<1$ and bounded $\Phi$ ensure the tail vanishes. **The boundary contribution must not depend on the policy; arbitrary nonzero terminal potentials or collection cutoffs cannot simply be ignored.**

## Calculate the shaping cancellation explicitly

A three-step trajectory has zero original rewards, $\gamma=0.9$, and potentials $[-3,-2,-1,0]$ at its four states. Added rewards are:

| Step | Added reward $\gamma\Phi(s')-\Phi(s)$ | Discounted contribution |
| --- | --- | --- |
| 0 | $0.9(-2)-(-3)=1.2$ | 1.2 |
| 1 | $0.9(-1)-(-2)=1.1$ | 0.99 |
| 2 | $0.9(0)-(-1)=1$ | 0.81 |

The sum is 3, exactly $-\Phi(s_0)$. Other trajectories from the same starting state with zero terminal potential receive the same constant addition, preserving their original ordering. Shaping redistributes feedback across steps rather than arbitrarily paying extra for a shorter route.

Now calculate the reward-farming loop mentioned above. A corridor has cells 0, 1, and 2; the agent starts at 0, reaching 2 earns 1 and ends the episode, and $\gamma=0.9$. Compare an ad hoc bonus, "+1 for each step toward the goal, no penalty for stepping back," with the potential version $\Phi=[-2,-1,0]$:

| Policy | Original return | +1 per step toward the goal | Potential version |
| --- | --- | --- | --- |
| Go straight 0→1→2 | 0.9 | $1+0.9\times(1+1)=2.8$ | $1.1+0.9\times2=2.9$ |
| Shuttle between 0 and 1, never finishing | 0 | $1/(1-0.81)\approx5.26$ | $(1.1-0.9\times0.8)/(1-0.81)=2$ |

The ad hoc bonus makes shuttling (5.26) worth more than finishing the task (2.8): stepping back costs nothing, so the same stretch of corridor pays out again and again. In the potential version, a step back costs $0.9\times(-2)-(-1)=-0.8$, both routes gain only $-\Phi(s_0)=2$, and the ranking matches the original 0.9 versus 0.

RND bonuses generally lack this telescoping structure and change the objective. An agent can prefer novelty after adding the bonus even though external rewards are unchanged. Preserve original task return as a separate evaluation measure.

## An option selects a chunk of behavior

An option usually specifies an initiation set, an internal policy, and a termination rule. A high-level controller selects “walk to the doorway,” and lower-level actions run for a variable duration $\tau$. Its target must account for time:

$$
y=\sum_{k=0}^{\tau-1}\gamma^kr_{t+k}
+\gamma^\tau(1-d)V(s_{t+\tau}).
$$

Here $d=1$ means true environment termination, not option termination. Completing a skill still bootstraps from the value $V$ of subsequent high-level choices; only environment termination removes that tail. $V$ evaluates the subsequent high-level policy; for optimal control, use the maximum Q over options available in that state.

After a 3-step option, the tail is discounted by $\gamma^3$, not $\gamma$. For rewards $[1,0,2]$, $\gamma=0.9$, and successor $V=5$, a completed skill with a continuing episode gives $y=2.62+0.9^3\times5=6.265$; true episode termination gives $y=2.62$. This is semi-Markov structure. Hierarchy can shorten high-level decision chains, but poorly chosen skills may restrict reachable policies.

## What to test first

Hold the external reward fixed and check whether exploration coverage and success improve together. Then isolate bonuses, skills, and budgets. Increased internal novelty is not task success.

This is an introduction, not a complete derivation of option-critic or Bayesian exploration. References: [RND, §§2.2–2.4](https://arxiv.org/abs/1810.12894) · [Potential-based shaping, §3, Theorem 1](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) · [Options and semi-MDPs, §§2, 3.2](https://www.sciencedirect.com/science/article/pii/S0004370299000521). Next: [Why policies can drift despite good demonstrations](imitation-and-rewards.en.md).
