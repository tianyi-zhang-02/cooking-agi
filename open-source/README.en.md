# Open source: getting an idea into someone else's codebase

[中文](README.md) · **English**

> Reading time: ~4 min · Type: Section index · Freshness: Evolving · Last reviewed: 2026-08

## Why contribute upstream

In your own repo, code only has to convince you. Upstream, it has to convince someone who does not know you, has no time, and assumes by default that you are wrong. Those require different skills.

Contributing upstream forces you to answer questions you can dodge in your own project:

- **Why is this change correct?** "I ran it and nothing broke" is not an answer. You need an argument someone can check without running anything.
- **What does it break that you don't know about?** In a codebase of tens of thousands of lines, the function you touched has five call sites and you have read three of them.
- **How do you make someone believe you?** The same fact framed as "I optimized performance" and as "94% of this tensor is computed and then discarded, and the discarded columns provably cannot influence the ones kept" gets very different responses.

For me this is the cheapest test of whether I actually understood something. It is easy to fool yourself into thinking you understood a paper. It is much harder to fool yourself into an accepted claim that a term in a real training framework is redundant.

## What this section covers

Not "I merged N PRs". Rather:

- What the library does inside a larger system, and where its hard parts are
- The **reasoning** behind each change — why I suspected something, how I proved it, which of my own guesses I killed
- What I got wrong. More than once I nearly published a conclusion that did not survive checking

## Projects so far

| Project | What it is | Notes |
| --- | --- | --- |
| [NVIDIA NeMo-RL](nemo-rl.en.md) | NVIDIA's LLM post-training framework (RLHF / GRPO / distillation) | Objective math, silently-ignored config, the trainer↔inference seam |

## One thread running through it

Looking back, almost nothing I submitted was "this is written wrong". It was one of two shapes:

**Code doing something it does not need to do.** Computing a normalizer that cancels out. Materializing a full vocabulary to read 64 of its columns. Projecting a matrix into 128k dimensions to keep 8k of them.

**Code not doing what it says it does.** A function that prints "masking out these positions", computes the mask, and discards it. A config key that is documented, accepted, and never read. A carefully written error message that can never fire.

Both shapes are **provable by reading**. No GPUs, no experiments, no paper required. That is how someone with a single consumer GPU can keep finding things in a framework built for clusters.
