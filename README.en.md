# AGI Study Notes

[中文](README.md) · **English**

### 📖 Read it at **<https://tianyi-zhang-02.github.io/cooking-agi/>**

## Why I write these notes

This is the learning path I am building for myself. Understanding one idea at a time is usually manageable; the harder part is connecting the pieces: what the model learns from, how it remembers and retrieves information, when it should use a tool, and why a better metric should count as real progress.

I care about more than “how to train a bigger model.” I want to understand how data, representation, memory, search, feedback, training, and evaluation work together so that an AI system can **understand a need more accurately, find evidence, and revise itself when the feedback says it was wrong**.

I try to write every note in the same order: where the problem comes from, the smallest example that builds intuition, the mathematics and implementation, the engineering trade-offs, and finally the assumptions and open questions. The perspective is modern-first: I start with today's LLM systems and bring in history only when it helps explain a current design.

These notes reflect what I understand today, and I expect them to keep changing as I learn and build more.

## Where to start

- **To learn systematically**: start with [Foundations](00-foundations/README.en.md) and move through the sections; the figures in [the Transformer lab](00-foundations/transformer-lab.en.md) are interactive.
- **To understand why model families make different choices**: use the [model family deep dives](00-foundations/model-families/README.en.md) to compare Llama, Qwen, DeepSeek, and Gemma through the same questions instead of memorising parameter tables.
- **To prepare for interviews**: use [Interviews](interview/README.en.md) for a quick review; [Job search](career/README.en.md) records what I learned while looking for MLE / RE roles.
- **To look something up**: browse the [paper notes](papers/README.en.md); the [editorial policy](EDITORIAL.en.md) explains how the material is maintained.

## Write it with us

If something is wrong, unclear, or missing a better example, please point it out; fixing a typo counts too. This repository is the source for the site, and [CONTRIBUTING.md](CONTRIBUTING.md) explains how to take part.

Only foundational principles, derivations, public papers, reproducible experiments, and open-source project notes are published here; nothing from a specific company or a recruiting process goes in.
