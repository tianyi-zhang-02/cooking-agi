# Frontier: first pin down what is meant

[中文](README.md) · **English**

> Reading time: ~3 min · Last reviewed: 2026-10

"World model" can mean a reinforcement-learning component that predicts the next step in a latent space, or a generated video you can walk into with your keyboard. "Self-improvement" can mean a model filtering its own answers and training on them, or a lab saying that AI now writes most of its code. **The same word often covers several different things; work out which one is meant before weighing the evidence.** This block collects directions of that kind, ones that are still moving.

## Why a separate block

Things change fast here: within six months there may be new systems and new claims, so every note carries a "Last reviewed" date and is rechecked every six months. Most other blocks on this site teach methods that have settled. This one is more about reading the news well: what was measured, under which conditions, and what is only a statement or a prediction.

A direction gets notes here only if it already changes how real systems are designed and there is public evidence you can check. A word that is merely popular does not get its own note.

Each series keeps three kinds of content apart: conclusions that have largely settled, parts that are still changing fast, and the signals worth watching next. Every fast-moving fact carries its date and says who measured it and under which conditions; numbers that companies report about themselves are marked as self-reported. When a piece of news goes stale, you still know which line to check a newer result against.

| A claim you read | Ask first |
| --- | --- |
| "X is a world model" | What does it predict, in what space, conditioned on what, and what is it used for? |
| "AI is improving itself" | Which object changes, who decides the change, and what signal accepts it? |
| "We measured an N-fold speed-up" | Who measured it, in what unit, with how much human intervention, and did compute grow at the same time? |

## Six series

1. [World models: what the term means now](world-models/README.en.md). From Dreamer training a policy inside imagination, to JEPA predicting representations, to video you can act in and models that evaluate robot policies, ending with whether they actually understand physics.
2. [Recursive self-improvement: first ask what changes](self-improvement/README.en.md). Separate changing weights, changing code, speeding up research, and self-generated data; see why the verifier sets the ceiling, and how "AI speeding up AI" is measured today.
3. [Reasoning models and test-time compute: when thinking longer helps](test-time-compute/README.en.md) (draft, pending review). Sampling and voting, verifiers, how long chains of thought are trained, and how to compare at equal compute.
4. [Training agents with RL: from single answers to multi-step tasks](agentic-rl/README.en.md) (draft, pending review). Where environments and rewards come from, how they get gamed, and which tokens receive gradient in a multi-step episode.
5. [Mechanistic interpretability: opening the model to see how it computes](interpretability/README.en.md) (draft, pending review). Probes, superposition, sparse autoencoders and circuits: what each can and cannot show.
6. [Diffusion language models: denoising a whole block instead of left to right](diffusion-lm/README.en.md) (draft, pending review). What masked diffusion trains, how it relates to BERT's MLM, and why parallel decoding is not automatically faster.

## Worth reading first

The world-model series builds on [Model-based RL](../05-post-training/deep-rl/model-based.en.md) and [Planning & LQR](../05-post-training/deep-rl/planning-and-control.en.md); the new Deep RL note [Which model to learn](../05-post-training/deep-rl/model-types.en.md) is also a good way in. The self-improvement series uses the [Deep RL notation](../05-post-training/deep-rl/README.en.md) and basic ideas from [Evaluation](../07-evaluation/README.en.md). None of these is a hard prerequisite; come back to them when a term is unfamiliar.
