# Frontier: first pin down what is meant

[中文](README.md) · **English**

> Reading time: ~3 min · Last reviewed: 2026-10

In RL, a “world model” might predict the next state. In a video system, the same term might describe a scene you can explore with a keyboard. “Self-improvement” is just as broad: training on model-generated data is different from using AI to help researchers write code.

This section explains these developing areas: how the methods work, what papers and public systems have demonstrated, and which questions remain open.

## Why a separate block

The foundations chapters mostly cover established methods. Here we focus on newer approaches and unresolved questions. For each result, we ask what was measured and under which conditions, separating the measurements from the authors' interpretations and predictions. Each note has a “Last reviewed” date, with reviews planned every six months.

A topic belongs here if it is already changing how real systems are designed and has public evidence readers can check. Popularity alone does not warrant a separate note.

Each series distinguishes findings supported by existing evidence, approaches that still need testing, and useful follow-up experiments. New systems and performance figures come with dates, sources, and experimental conditions. Company-reported numbers are labeled as such so you can check the original account.

| A claim you read | Ask first |
| --- | --- |
| "X is a world model" | What does it predict, in what space, conditioned on what, and what is it used for? |
| "AI is improving itself" | Which object changes, who decides the change, and what signal accepts it? |
| "We measured an N-fold speed-up" | Who measured it, in what unit, with how much human intervention, and did compute grow at the same time? |

## Six series

1. [World models: what the term means now](world-models/README.en.md). Start with Dreamer training policies in imagined trajectories and JEPA predicting representations, then explore interactive video and models used to evaluate robot policies. The series closes by asking whether these models actually understand physics.
2. [Recursive self-improvement: first ask what changes](self-improvement/README.en.md). Separate changing weights, changing code, speeding up research, and self-generated data; see why the verifier sets the ceiling, and how "AI speeding up AI" is measured today.
3. [Reasoning models and test-time compute: when thinking longer helps](test-time-compute/README.en.md). Sampling and voting, verifiers, how long chains of thought are trained, and how to compare at equal compute.
4. [Training agents with RL: from single answers to multi-step tasks](agentic-rl/README.en.md). Where environments and rewards come from, how they get gamed, and which tokens receive gradient in a multi-step episode.
5. [Mechanistic interpretability: opening the model to see how it computes](interpretability/README.en.md). Probes, superposition, sparse autoencoders and circuits: what each can and cannot show.
6. [Diffusion language models: denoising a whole block instead of left to right](diffusion-lm/README.en.md). What masked diffusion trains, how it relates to BERT's MLM, and why parallel decoding is not automatically faster.

## Worth reading first

The world-model series builds on [Model-based RL](../05-post-training/deep-rl/model-based.en.md) and [Planning & LQR](../05-post-training/deep-rl/planning-and-control.en.md); the new Deep RL note [Which model to learn](../05-post-training/deep-rl/model-types.en.md) is also a good way in. The self-improvement series uses the [Deep RL notation](../05-post-training/deep-rl/README.en.md) and basic ideas from [Evaluation](../07-evaluation/README.en.md). None of these is a hard prerequisite; come back to them when a term is unfamiliar.
