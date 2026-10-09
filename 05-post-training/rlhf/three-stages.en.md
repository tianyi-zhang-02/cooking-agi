# RLHF: three stages and four models

[中文](three-stages.md) · **English**

> Reading time: ~3 min · Level: core · Last reviewed: 2026-10

To teach a model to summarize a long email in three sentences, we could show it a good summary, ask it to compare two summaries, or let it write one and learn from feedback. These roughly correspond to the three stages of classic RLHF.

Step through the figure or read on. This page describes a common PPO-style pipeline; other post-training methods need not retain every stage.

<!-- widget:tx-rlhf -->

<span id="the-three-stages"></span>

## Demonstrations, comparisons, then new attempts

**SFT: learn from a demonstration.** Each example pairs an email with a suitable three-sentence summary. The model learns to produce that output from the input. A useful starting policy makes worthwhile samples easier to obtain later; if even occasional success is hard to sample, RL may struggle for signal too.

**Reward modeling: learn to compare.** Now pair the same email with two summaries. One preserves the deadline; the other reads smoothly but omits it. An annotator prefers the first. From comparisons like this, a model learns to predict preferences without requiring people to assign an absolute score to every sentence.

**RL: generate new attempts.** The Actor writes a summary, the reward model scores it, and the policy updates from that feedback. This adds an active attempt: training evaluates what the current model writes, rather than only imitating summaries already in the dataset.

<details markdown="1">
<summary>How can “the first is better” train a numerical score?</summary>

A common choice is a Bradley–Terry preference model. With preferred answer $y_w$ and alternative $y_l$, its training loss is:

$$
\mathcal L(\phi)=-\mathbb E_{(x,y_w,y_l)}\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big).
$$

It favors a higher score for the preferred answer. The loss directly uses the **score difference**: adding 10 to both scores leaves it unchanged. An individual score is therefore not a fixed unit of satisfaction; comparisons across prompts, domains, or versions require checking scale and calibration.

</details>

<span id="the-four-models-and-which-two-actually-train"></span>

## Who does what during the RL stage?

The Actor writes the summary. The Reward Model evaluates the completed result, the Critic estimates the return expected from continuing a prefix, and the Reference retains the anchor behavior.

| Model | Common starting point | Updated at this stage? | Role |
|---|---|---|---|
| Actor | SFT model | Yes | Generates answers; the model we want to improve |
| Critic | Pretrained backbone with a value head; may start from the reward model | Yes | Estimates state value to help construct advantages |
| Reward Model | Model trained in the preceding stage | Usually frozen | Scores completed answers |
| Reference | Copy of the SFT model | Usually frozen | Provides the KL anchor |

This setup primarily updates the Actor and Critic. Actor and Reference can start with identical weights; the Actor then trains while the Reference stays put.

These are roles, not a memory budget. Shared backbones, caching, sharding, and offloading change the resident weights required. See [InstructGPT](https://arxiv.org/abs/2203.02155) for a public pipeline, or continue with [Reference and Critic](reference-and-critic.en.md) to separate their roles.
