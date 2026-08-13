# Open source

[中文](README.md) · **English**

> Reading time: ~3 min · Type: Section index · Freshness: Evolving · Last reviewed: 2026-08

In your own repo, code only has to convince you. Upstream, it has to convince someone who does not know you, has no time, and assumes by default that you are wrong. Those turn out to need fairly different skills.

Contributing upstream forces you to answer questions you can dodge in your own project. Why is this correct? "I ran it and nothing broke" does not count — you need an argument someone can check without running anything. What does it break that you don't know about? In a codebase of tens of thousands of lines, the function you touched has five call sites and you have read three. And then there is the question of how you get someone to believe you: the same fact framed as "I optimized performance" and as "94% of this tensor's columns are computed and then discarded, and the discarded ones provably cannot influence the ones kept" gets very different responses.

For me this is the cheapest test of whether I actually understood something. It is easy to fool yourself into thinking you understood a paper. It is much harder to fool yourself into an accepted claim that a term in a real training framework is redundant.

This section is not a list of merged PRs. What I want to write down is what the library does inside a larger system, where its hard parts are, and the reasoning behind each change — why I suspected something, how I proved it, which of my own guesses I ended up throwing away. Including the mistakes: more than once I nearly published a conclusion that did not survive checking.

## Projects so far

| Project | What it is | Notes |
| --- | --- | --- |
| [NVIDIA NeMo-RL](nemo-rl.en.md) | NVIDIA's LLM post-training framework (RLHF, GRPO, distillation) | Four categories, easiest first: config that silently does nothing, computation that cancels, functions that misstate themselves, the trainer↔inference seam |

## One thread running through it

Looking back, almost none of it was "this line is wrong". It all fell into two other shapes.

Code doing something it does not need to do: computing a normalizer that cancels out, materializing a full vocabulary to read 64 of its columns, projecting a matrix into 128k dimensions to keep 8k of them.

Code not doing what it says it does: a function that prints "masking out these positions", computes the mask, and discards it; a config key that is documented, accepted, and never read; a carefully written error message that can never fire.

Both shapes share one useful property — **they are provable by reading**. No GPUs, no experiments, no paper required. Which probably explains how someone with a single consumer GPU keeps finding things in a framework built for clusters.
