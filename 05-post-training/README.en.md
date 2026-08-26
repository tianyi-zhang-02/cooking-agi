# Post-Training

[中文](README.md) · **English**

## Working thesis

Post-training is fundamentally a problem of turning imperfect observations of human behavior into defensible learning objectives. The data-generating policy, feedback delay, selection effects, and evaluation design matter as much as the optimizer.

## Questions

- What can SFT learn that preference optimization or online RL cannot, and vice versa?
- How should objectives change when feedback is sparse, delayed, or confounded?
- How can training datasets preserve longitudinal and causal structure?
- Which behavioral gains survive distribution shift and repeated interaction?

## How to read this series

Post-training comes down to two things: **teaching by demonstration (SFT)** and **teaching by outcome (RL)**. The series is ordered by dependency — skip ahead and the later pieces won't land.

**I. Foundations: why more teaching is needed**

1. Why pretraining isn't enough (this page) — what learning problem SFT, preference learning, and RL each solve
2. [SFT: how far imitation goes, and where it stops](sft-and-its-ceiling.en.md) — why cross-entropy can't see the pivotal token, and how demonstrations that always contain an answer train hallucination in
3. *Where preferences come from* — to write. Training a reward model, Bradley-Terry, and the quality of preference labels themselves

**II. Teaching by outcome: the RL line**

4. [The three stages of RLHF, and what came after](rlhf-pipeline.en.md) — four models, which train and which are frozen
5. [After PPO: every algorithm deletes one of its parts](after-ppo.en.md) — one reading that covers GRPO / RLOO / REINFORCE++ / DAPO / DPO
6. *Verifiable rewards: when the reward doesn't need learning* — to write. Why it narrows reward hacking as a side effect, and where it stops applying

**III. Actually running it**

7. [Post-training infrastructure: sampling, numerics, context](post-training-infrastructure.en.md) — none of the three changes the objective; all of them decide what you can do to it. **The least discussed layer, and often where runs actually stall**
8. [How far one base model can go](same-base-different-posttraining.en.md) — a rare natural experiment: base pinned, post-training only, showing what those three are worth

**IV. What it costs**

9. *The alignment tax: what you lose by becoming agreeable* — to write. Diversity loss, entropy collapse, sycophancy, and the generalization/diversity tradeoff

Italics are unwritten. **This isn't roadmap theater — it's the todo list.** Drop the italics as each one lands.

## Starting papers

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
