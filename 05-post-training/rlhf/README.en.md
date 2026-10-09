# RLHF: learning from feedback

[中文](README.md) · **English**

> Reading time: ~2 min · Level: core · Last reviewed: 2026-10

Ask, “I'm learning Python for the first time. What could I try today?” One answer gives half a page of language history; another walks you through a few lines of code you can run immediately. Both might be factually correct, but the second is more useful for this request.

It is difficult to capture every such distinction in a reference answer. Comparing two responses is often easier: which addresses the question, which omits something important, and which is padding?

RLHF uses feedback like this to keep adjusting a model's behavior after pretraining and SFT.

<span id="why-the-detour-is-necessary"></span>

## How does feedback become a training signal?

The classic approach collects human comparisons and trains a Reward Model to predict those preferences. The language model then generates new answers, receives scores, and updates through RL.

```text
People compare answers → learn preferences → generate new answers → update from scores
```

The middle step deserves attention: training directly optimizes the reward model's score. If that model mistakes length for quality, answers may get longer rather than more useful. We therefore need a separate check of answer quality after training.

[InstructGPT](https://arxiv.org/abs/2203.02155) is a public example of this route. Later methods, including GRPO and DPO, change some of these steps; we will examine them separately.

<span id="quick-learning-actor-rm-critic-and-reference"></span>

## Which names will come up?

The model generating answers—the one we want to improve—is the **Actor**. The Reward Model scores outcomes. A **Critic** estimates the return to expect from continuing a prefix, and a **Reference** retains an anchor policy to help measure how much behavior has changed.

There is no need to memorize every role at once. Follow how one answer is generated and receives feedback, then return to why each component is there.

<span id="how-this-series-reads"></span>

## Where to go next

1. [Why language models can use RL](rl-for-language-models.en.md): how generating a token becomes an action
2. [Reward, value, and advantage](value-and-advantage.en.md): the result you got versus the result you expected
3. [Can we train on old answers?](on-off-policy.en.md): what changes when the model moves beyond its data
4. [Three stages and four models](three-stages.en.md): demonstrations, preferences, and policy updates
5. [Reference and Critic](reference-and-critic.en.md): estimating drift versus estimating return
6. [PPO clipping](ppo-clipping.en.md): why one good result should not keep earning more encouragement
7. [GRPO, DPO, and RLVR](after-rlhf.en.md): which part each changes
8. [Did training help?](evaluation-and-review.en.md): separating a score change from a useful improvement

On a first pass, follow that sequence. When implementing a loss or inspecting training logs, turn to [Three KL estimators](kl-estimators.en.md): why a sample estimate can be negative, and why a correct value does not necessarily give the gradient you intended.

For the underlying reinforcement-learning ideas, start with [Deep RL foundations](../deep-rl/README.en.md). If they are familiar, go straight to the question you want to explore.
