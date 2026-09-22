# RLHF: what happened next (GRPO, DPO, RLVR)

[中文](after-rlhf.md) · **English**

> Reading time: ~2 min · Level: core · Last reviewed: 2026-09

## What happened next

Four models, two of them training, one of which is a full-size Critic. That cost is what everything since has been cutting.

![how many models each method keeps resident](../assets/rlhf-model-count.svg)

**GRPO drops the Critic.** Sample a group of answers for the same prompt and use the group's own spread as the baseline:

$$\hat A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

If all you needed was a baseline for variance reduction, the group mean supplies one — no separate network required. That removes a full-size model *that was being trained*.

**DPO drops the RL loop entirely.** The key derivation: KL-constrained reward maximisation has a closed-form optimum, and inverting it expresses the reward in terms of the policy itself, so the preference loss can be taken **directly against the policy**:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

No reward model, no sampling, no Critic. Standard DPO usually uses **offline** preference pairs: the policy changes while the data distribution does not, so it cannot actively discover the current policy's new failures. Online DPO can resample and reduce this mismatch. The deeper distinction is whether data follows the current policy and whether feedback comes from preferences, a Reward Model, a verifier, or an environment.

**RLVR replaces the reward with a program.** Maths problems have answers to check; code has tests to run. Rewards like that need not be learned at all. The learned reward model disappears, and so does most of the room for reward hacking — **what gets gamed is a fitted reward, not a verified one.**
