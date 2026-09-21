# Post-Training: what does a model still have to learn after pretraining?

[中文](README.md) · **English**

> Reading time: ~12 min · Type: overview · Last reviewed: 2026-08

## Post-training changes how the model behaves

Pretraining teaches a model what the world usually contains. Post-training teaches it how to behave and act when it faces a particular kind of task.

## What each stage does, intuitively

- **Pretraining**: learn language, knowledge, and general capability from large-scale corpora.
- **Continued Pretraining**: keep adapting to the data distribution and terminology of one domain.
- **SFT**: learn instruction following and target behavior from high-quality demonstrations.
- **Preference Learning**: learn preferences from comparisons between responses.
- **RL**: adjust the whole action policy according to outcomes or verifiable rewards.

They are not interchangeable buttons; they solve different learning problems.

## From Base Model to Aligned Model: get the map right first

### 1. Three stages use three different kinds of supervision

A common high-level path is

$$
\boxed{
\text{Pre-Training}
\longrightarrow
\text{SFT}
\longrightarrow
\text{Preference Alignment (RLHF / DPO / RLVR)}
}
$$

| Stage | Main data | Training signal | Common result label |
| --- | --- | --- | --- |
| Pre-training | large-scale text, code, and related corpora | token targets constructed from the text itself | Base Model |
| SFT | instruction–response demonstrations | selected ideal responses | Instruction Model |
| Preference alignment | chosen/rejected pairs, rewards, or verifiers | which complete behavior is better | Aligned Model / Policy |

These labels describe functional stages; an organization need not save exactly three
separate checkpoints. The useful distinction is that pre-training learns language,
knowledge, and foundational capability from the data distribution; SFT teaches the
model to invoke that capability through demonstrations; preference alignment selects
which of several plausible behaviors better serves the objective.

**Pre-training precedes classic RLHF; it is not RLHF's first stage.** In the full model
lifecycle it is upstream, while the classic RLHF pipeline usually begins from an SFT
policy.

### 2. GPT and BERT use different pre-training objectives

GPT-style decoder-only models use causal language modeling:

$$
\mathcal L_{\text{causal}}
=-\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

They see only the left context and naturally support autoregressive continuation.
BERT-style encoders commonly use masked language modeling: selected tokens are hidden
and recovered using both left and right context. That objective more naturally supports
bidirectional representation, classification, and extraction than generation.

A pretrained Base Model can continue text, imitate textual patterns, and encode a great
deal of knowledge, yet still fail to treat a user prompt as an instruction requiring a
direct answer. It may continue the question, imitate a webpage, or ignore a constraint
such as “use three sentences,” because its original objective predicts text rather than
behaving as a chat assistant.

“Pre-training learns capability; SFT teaches the model how to use it” is a useful
approximation, not a law. SFT can also change knowledge and capability, although its
scale and objective usually emphasize behavioral shaping.

### 3. Post-training is a scope; LoRA is an update mechanism

Post-training broadly covers training performed after initial foundation pre-training
and is larger than RLHF:

$$
\text{Post-Training}\supset
\{\text{SFT, preference optimization, RL, domain/safety tuning, distillation, tool use}\}.
$$

Classic RLHF trains a Reward Model from preference pairs and then optimizes the policy
with an RL method such as PPO. DPO directly optimizes the policy from chosen/rejected
pairs, with neither a separate Reward Model nor an online RL loop. DPO is therefore not
strictly reinforcement learning, although both approaches belong to preference
alignment.

Keep two orthogonal axes separate:

- **what is trained (objective / data):** SFT, DPO, language modeling, distillation, RL;
- **how parameters are updated (parameterization):** full-parameter fine-tuning or
  LoRA / adapters.

The same SFT or DPO objective can use full updates or LoRA. Listing “LoRA” beside “SFT”
as if both were training stages confuses the objective with the update mechanism.

### 4. Continued pre-training happens later but retains a language-modeling objective

A common domain-adaptation path is

```text
General Base Model
  → continue next-token prediction on a large medical corpus
  → Medical Domain Base Model
  → Medical SFT / preference alignment
```

The middle step is Continued Pre-Training (CPT) or Domain-Adaptive Pre-Training. It
occurs after initial pre-training in time but still uses a language-modeling objective
and non-instruction domain text; SFT uses demonstrations to shape behavior.

The boundary between “pre-training” and “post-training” can therefore depend on whether
the speaker is classifying by **chronological stage** or by **training objective**. When
terminology is ambiguous, ask four concrete questions: what is the data, what is the
objective, which checkpoint is the starting point, and what kind of model should result?

English terminology also differs: **pre-training** is a process, while a
**pre-trained model** has completed that process. In casual usage, “pretrained LLM” may
even refer broadly to a general-purpose model that has already been aligned.

## What SFT does

SFT uses input–output demonstrations to teach the model to imitate a target behavior. It fits when you want to:

- learn a fixed task format;
- establish basic instruction following;
- distill an expert's process into the model;
- give the model a reasonably stable behavioral starting point first.

Its limit is that it can only imitate behavior that appears in the data. For long-tail cases the demonstrations never covered, the model may not know what to do.

## What preference learning does

When the best answer is hard to write down directly but people can compare A with B, you can learn from preferences.

Methods such as DPO turn preference pairs directly into a policy training objective. That is simpler than building a full reward pipeline, but it still rests on several assumptions: that the preference labels are stable, that the candidates differ enough, and that the training distribution is close to real use.

## What RL does

RL is the better fit for tasks that need multi-step action, have delayed outcomes, or require the policy to learn through exploration.

But RL does not automatically turn weak feedback into strong feedback. If the reward is sparse, ambiguous, or contaminated by policy bias, the model may only learn to exploit loopholes in the metric.

In recommendation, for example, one click depends at the same time on exposure position, the title, and how much time the user has. Use the click directly as the reward, and the model may learn to attract clicks more effectively rather than to deliver more long-term content value.

## Why data often becomes the bottleneck before the algorithm does

If the data lacks longitudinal structure, so that one user's behavior is cut into many isolated samples, the model can hardly learn how long-term intent changes.

If the training data cannot say which policy produced each sample, it is also hard to tell whether the model learned the user's preference or the old system's bias.

So you need an objective-aware data layer that makes explicit:

- what the current training objective is;
- which samples fit that objective;
- how they are sampled and combined;
- how data and model versions are tracked;
- whether field semantics agree between training and serving;
- whether different experiments are actually comparable.

**Working thesis.** Post-training is fundamentally a problem of turning imperfect observations of human behavior into defensible learning objectives. The data-generating policy, feedback delay, selection effects, and evaluation design matter as much as the optimizer.

**Questions I keep open:**

- What can SFT learn that preference optimization or online RL cannot, and vice versa?
- How should objectives change when feedback is sparse, delayed, or confounded?
- How can training datasets preserve longitudinal and causal structure?
- Which behavioral gains survive distribution shift and repeated interaction?

## How to choose a method

Start by asking:

| Question | More likely starting point |
| --- | --- |
| Do you have high-quality gold answers? | SFT |
| Is the answer hard to write, but easy for people to compare? | Preference learning / DPO |
| Does it need multi-step exploration, with a final outcome you can verify? | RL |
| Does the model lack domain knowledge and distribution coverage? | Continued pretraining |
| Is the feedback itself sparse, biased, or impossible to attribute? | Fix data and evaluation first, rather than switching algorithms first |

## How to evaluate

Training loss or one aggregate reward is not enough. Also check:

- whether the new behavior comes from the intended mechanism rather than from data leakage;
- whether different user and task slices improve consistently;
- whether the model gives up diversity, calibration, or safety in exchange for one metric;
- whether offline preference translates into real task outcomes;
- whether the policy, log-probabilities, and data semantics agree between training and inference.

## How to read this series

Model adaptation begins with two questions: **what supplies the learning signal**, and **which parameters may change**. Post-training then shapes behavior into a usable product through demonstrations, preferences, and outcomes. The series is ordered by dependency — skip ahead and the later pieces will not land.

**I. Foundations: why more teaching is needed**

1. Why pretraining isn't enough (this page) — what learning problem SFT, preference learning, and RL each solve
2. [Model adaptation: Full Fine-Tuning, LoRA, Prompt Tuning, and Distillation](model-adaptation.en.md) — separate the learning objective from parameterization, and explain how the system constrains classification output
3. [SFT: how far imitation goes, and where it stops](sft-and-its-ceiling.en.md) — why cross-entropy can't see the pivotal token, and how demonstrations that always contain an answer train hallucination in
4. [Where preferences come from: the reward model and what it learns](where-preferences-come-from.en.md) — Bradley-Terry learns order but not scale, and a reward model **expires** as the policy drifts

**II. Teaching by outcome: the RL line**

5. [The three stages of RLHF, and what happened next](rlhf-pipeline.en.md) — four models, which train and which are frozen
6. [After PPO: every algorithm deletes one of its parts](after-ppo.en.md) — one reading that covers GRPO / RLOO / REINFORCE++ / DAPO / DPO
7. [Verifiable rewards: when the reward doesn't need learning](verifiable-rewards.en.md) — it narrows reward hacking without eliminating it; verifiability is a spectrum, not a binary

**III. Actually running it**

8. [Post-training infrastructure: sampling, numerics, context](post-training-infrastructure.en.md) — none of the three changes the objective; all of them decide what you can do to it. **The least discussed layer, and often where results actually stall**
9. [How far one base model can go](same-base-different-posttraining.en.md) — a rare natural experiment: base pinned, only post-training changed, showing what the three things in note 8 are worth

**IV. What it costs**

10. [The alignment tax: what you lose by becoming agreeable](alignment-tax.en.md) — trading distribution width for out-of-distribution robustness. **pass@1 up with pass@k down means you compressed the distribution into a point**

The ten notes form one line: **choose supervision and parameterization → teach by demonstration → teach by outcome → make it actually run → what it costs.**

Read in order it's about an hour. If you're here for one specific problem, the "In one sentence" opener of each piece is the index.

## Continue reading

- [Data and feedback](../01-data-and-feedback/README.en.md)
- [Evaluation](../07-evaluation/README.en.md)
- [Model Experience](../08-model-experience/README.en.md)

## Starting papers

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
