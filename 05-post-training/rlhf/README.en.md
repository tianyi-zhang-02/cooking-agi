# RLHF: why the detour is necessary

[中文](README.md) · **English**

> Reading time: ~3 min · Level: core · Last reviewed: 2026-09

<div class="lesson-recipe">
  <div><span>The problem</span><strong>turning "people prefer this answer" into something optimisable</strong></div>
  <div><span>Prerequisites</span><strong>SFT · preference data · next-token prediction</strong></div>
  <div><span>Core mechanism</span><strong>learn a reward model, then optimise it with RL</strong></div>
  <div><span>Common mistakes</span><strong>thinking all four models train; thinking the KL term is optional</strong></div>
</div>

## Quick learning: Actor, RM, Critic, and Reference

<details class="interview" markdown="1">
<summary>Explain Base → SFT → Preference → RL → Evaluation in two minutes</summary>

**Quick memory**: the Actor generates, the RM scores complete answers, the Critic estimates prefix return, and the Reference limits policy drift. Evaluation checks whether reward still matches the product objective.

**Interview answer**

> SFT first creates a usable policy; preference data trains a Reward Model; PPO updates the Actor with reward-derived advantages while clipping and reference KL constrain change. The Critic is a variance-reduction baseline, not the Reward Model. Current-policy sampling and independent evaluation close the loop.

<details markdown="1">
<summary><b>Deep dive</b>: why are Reference KL and PPO clipping different constraints?</summary>

Clipping limits one optimizer update relative to the rollout policy. Reference KL limits long-term drift from a fixed SFT policy. The former is a local trust region and the latter a behavioral prior; deleting either is not automatically compensated by the other.

</details>
</details>

## Why the detour is necessary

The direct approach to "answer better" would be a loss function. But *better* has no closed form and no reference answer to compare against.

People can **compare**, though: shown two answers, they can say which is better. All of RLHF is turning that into an objective:

```text
people can compare → learn a model that predicts which they'd prefer
                   → use that model as a reward function for RL
```

The cost is a layer of indirection. You are no longer optimising human preference; you are optimising *a model's fit to* human preference. Every difficulty below comes from that.

## How this series reads

1. [Mapping ordinary RL onto a language model](rl-for-language-models.en.md): what state, action, reward, and trajectory are in a language model
2. [Reward, value, and advantage](value-and-advantage.en.md): return, value, advantage, and the Bellman equation, and why a baseline is subtracted
3. [On-policy and off-policy](on-off-policy.en.md): who generates the data, and why a Reward Model is not a Critic
4. [Three stages and four models](three-stages.en.md): SFT → reward model → RL, and which models train in stage three (step through it)
5. [Why both a Reference and a Critic](reference-and-critic.en.md): why the KL term is not optional, and why the Critic only reduces variance
6. [What happened next](after-rlhf.en.md): GRPO drops the Critic, DPO drops the RL loop, RLVR turns the reward into a program
7. [The four cases of PPO clipping](ppo-clipping.en.md): drag ρ, flip the sign of A, and watch when the encouragement stops
8. [Evaluation closes the loop, and review questions](evaluation-and-review.en.md): how to confirm it really improved; interview questions and a self-check
