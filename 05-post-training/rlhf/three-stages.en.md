# RLHF: three stages and four models

[中文](three-stages.md) · **English**

> Reading time: ~2 min · Level: core · Last reviewed: 2026-09

Step through the figure below: first what each of the three stages produces, then how the four models of stage three work together.

<!-- widget:tx-rlhf -->

## The three stages

**Stage 1, SFT.** Finetune the pretrained model on human demonstrations to get something that at least follows the instruction format. It becomes RL's initial policy, and RL cannot rescue a bad starting point.

**Stage 2, the reward model.** Collect **pairwise** rankings — two answers to the same prompt, labelled which is better — and train $r_\phi$ to score answers with the Bradley–Terry loss:

$$\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]$$

It learns **relative** order. Its zero point is unidentifiable: adding the same constant to every score changes no loss. Raw scores therefore are not literal units of “human satisfaction,” and comparisons across prompts or data distributions require calibration checks. The objective directly constrains the chosen–rejected gap for the same prompt.

**Stage 3, optimise with RL.** Four models are present at once, in quite different roles.

## The four models, and which two actually train

This is the part most explanations blur:

| Model | Comes from | Trains? | Role |
| --- | --- | --- | --- |
| **Actor** (policy) | copy of the SFT model | **yes** | the thing being optimised, and what ships |
| **Critic** (value) | often initialised from the reward model | **yes** | estimates $V_t$ to reduce gradient variance |
| **Reward** | stage 2's output | frozen | scores complete answers |
| **Reference** | copy of the SFT model | frozen | the KL anchor that keeps the Actor from drifting |

**Only the first two update weights.** Reward and Reference run forward passes and nothing else.

Actor and Reference start as two copies of the same weights. The Actor trains and drifts; the Reference stays put as the measuring stick.

These are four **conceptual roles**, not necessarily four independent full models that remain GPU-resident at all times. A Critic may be a value head on a shared backbone, while frozen models can be sharded or offloaded. Their training relationships stay the same.
