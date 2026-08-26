# After PPO: every algorithm deletes one of its parts

[中文](after-ppo.md) · **English**

> Reading time: ~7 min · Type: chapter · Last reviewed: 2026-08

## In one sentence

The long list of algorithms that followed PPO looks like a pile of separate inventions. It isn't: each one **deletes a component of PPO and then deals with whatever surfaces afterwards.** Work out what each removed and what it cost, and you no longer have to memorize the table.

## What PPO has available to delete

Stage three holds four models in memory, two of them training:

| Model | Training? | Job |
| --- | --- | --- |
| Policy | yes | the model you're trying to produce |
| Critic | **yes** | estimates state value, the baseline for the advantage |
| Reward | frozen | scores responses |
| Reference | frozen | KL anchor, keeps the policy from drifting |

**The Critic is the expensive one** — full-size network, and it's training. So most of what follows is a story about it.

## The main line: delete the Critic

The Critic exists for exactly one reason: **to supply a baseline for the advantage**, to cut variance. If all you need is a baseline, it does not have to be a learned network.

| Algorithm | Where the baseline comes from | Cost |
| --- | --- | --- |
| **GRPO** | sample a group of responses per prompt; use the group mean (divided by its std) | multiple samples per prompt, so sampling cost goes up |
| **RLOO** | leave-one-out: one sample's baseline is the mean reward of the others | same sampling cost; and it treats the whole response as **one action**, giving up token-level credit assignment |
| **REINFORCE++** | batch-level statistics for advantage normalization | a coarser baseline, but it keeps PPO's clipping and KL stabilization |

What they share: **the baseline goes from learned to sampled.** That removes a full-size training model — and plants the seed of every problem below.

RLOO's baseline, with $k$ samples per prompt:

$$\hat A_i = r_i - \frac{1}{k-1}\sum_{j \neq i} r_j$$

Its argument is that RLHF starts from a trained SFT model rather than a randomly initialized network, so PPO's machinery for stabilizing unstable training (GAE, per-token value estimation) may not be necessary.

## What surfaces once the Critic is gone

This section is the point. **A sampled baseline fails differently than a learned one.**

**One: when a group's rewards don't differ, the baseline carries no information.**
If every response in a group is right, or every one is wrong, subtracting the group mean leaves all advantages at zero and the group **produces no gradient at all**. The samples were wasted. On tasks that are too easy or too hard, that share gets large.

**Two: normalization introduces bias.**
Dividing by the group's standard deviation looks like standardization but weights groups unequally; normalizing by sequence length pushes systematically toward one length. That is Dr. GRPO's argument: those terms bias the model toward **longer answers that aren't necessarily better**. Remove them and token efficiency improves visibly.

**Three: symmetric clipping can't lift low-probability tokens.**
PPO clips the probability ratio to $[1-\epsilon, 1+\epsilon]$. For a token whose current probability is tiny, the ceiling $1+\epsilon$ permits only a minuscule absolute increase — it has almost no path back. The long-run consequence is entropy collapse: the model gets more certain and less diverse.

**Four: sequence-level loss dilutes long answers.**
Average the loss per sample and a 1000-token response carries the same weight as a 50-token one, so each token in the long answer receives a thinner share of gradient. Reasoning tasks are exactly the ones that need long answers.

DAPO's four changes map onto these four problems: dynamic sampling filters out all-right/all-wrong groups, the clip bounds are decoupled (Clip-Higher), the loss becomes token-level, and overlong responses get soft shaping instead of a hard cut.

## Another line: delete the RL loop entirely

DPO goes further — **it removes the Reward model, the Critic, and the sampling loop together.**

The key derivation is that KL-constrained reward maximization has a closed-form optimum; invert it and the reward can be written in terms of the policy itself, so the preference loss can be differentiated with respect to the policy directly. Details in [the three stages of RLHF](rlhf-pipeline.en.md).

The cost is that **it can only consume offline preference pairs**. The policy keeps changing during training while the data still comes from the old one, and that distribution mismatch is DPO's main limitation — and why it falls behind the PPO family in settings that suit RL scaling.

Two follow-ups:

- **IPO** — DPO overfits preference data fast. IPO adds a regularizer so the model converges without tricks like early stopping.
- **KTO** — no pairs required, just a "good" or "bad" label per sample. An order of magnitude cheaper to collect, at the price of a coarser signal.

## And one more: replace the reward model with a program

Math has answers to check. Code has tests to run. For these, the reward **doesn't need to be learned** — write a checker.

What this deletes is the Reward Model. And **what can be gamed is a fitted reward, not a verified one** — a learned reward model offers a gradient to climb; a deterministic checker doesn't. So verifiable rewards mostly dissolve reward hacking as a side effect.

The limitation is obvious: it only applies where you can write the checker.

## The table

| | Deletes | Replaced by | Bad fit when |
| --- | --- | --- | --- |
| PPO | — | — | memory is tight, tuning budget is small |
| GRPO | Critic | group-relative reward | rewards within a group barely differ |
| RLOO | Critic + token-level credit | leave-one-out baseline | you need fine-grained credit assignment |
| REINFORCE++ | Critic | batch-statistic normalization | reward scales vary a lot across prompts |
| DPO | Critic + RM + sampling | closed-form direct optimization | you want online scaling |
| RLVR | RM | a verifier | you can't write a checker |

## How to choose

**Ask first whether the reward can be verified.** If you can write a checker, write it — that pays more than any algorithm swap.

**Then ask whether you can sample online.** No (static preference data only) → the DPO family, accepting the mismatch. Yes → the GRPO family.

**Only then ask which component to delete.** Memory-bound → drop the Critic. Long outputs and worried about entropy collapse → ship DAPO's fixes alongside.

Don't invert that order — **picking the algorithm first and retrofitting the data is the most common source of rework in these projects.**

## Down to a checklist

1. Is my reward learned or verified? If learned, am I watching for reward hacking?
2. With a group-relative baseline, what fraction of groups is all-right or all-wrong? That fraction was sampled for nothing.
3. What bias do my normalization terms introduce? Is response length climbing monotonically through training?
4. Is each token in a long answer getting a thinner gradient than one in a short answer?
5. If using DPO: which policy produced my preference data, and how far is it from the current one?

## Where to read next

- [The three stages of RLHF, and what came after](rlhf-pipeline.en.md): what each model does, and the DPO derivation
- [Data and feedback](../01-data-and-feedback/): the quality of preference labels themselves
- [Evaluation](../07-evaluation/): telling whether alignment actually helped

## Starting papers

- [PPO](https://arxiv.org/abs/1707.06347) — clipped objective and trust region
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO
- [DAPO](https://arxiv.org/abs/2503.14476) — four fixes for four GRPO failure modes
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) — Dr. GRPO, the length bias from normalization
- [Back to Basics](https://arxiv.org/abs/2402.14740) — RLOO
- [REINFORCE++](https://arxiv.org/abs/2501.03262) — drop the Critic, keep PPO's stabilizers
- [DPO](https://arxiv.org/abs/2305.18290) · [IPO](https://arxiv.org/abs/2310.12036) · [KTO](https://arxiv.org/abs/2402.01306)

## Further reading (Chinese)

- [大模型中的强化学习 — RL in large models](https://zhuanlan.zhihu.com/p/693582342) — by @大家好我是爱因 on Zhihu. This chapter is organized along a single axis, "what got deleted," and covers only the trunk. That piece is the encyclopedic view — MDP elements, the Bellman equation, MC/TD/GAE, then each algorithm in detail. Start there if you want the fuller map.
