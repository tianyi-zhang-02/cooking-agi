# Modern-First Editorial and Freshness Policy

[中文](EDITORIAL.md) · **English** · [Back to home](README.en.md)

> Reading time: ~5 minutes · Last reviewed: 2026-08

## Core principle

The repository begins with real problems in current AI systems rather than an algorithm chronology:

```text
user goals
→ data, memory, search, and tools
→ LLM / multimodal policy
→ training, serving, and runtime
→ evaluation, feedback, and continual improvement
```

An older work enters the main path only when it explains a mechanism, constraint, or failure that remains active today.

## What belongs in the main path

- current LLM and multimodal behavior and systems mechanisms;
- agent memory, search, retrieval, tools, and state;
- SFT, preference learning, RL, distillation, and model updating;
- evaluation, LLM judges, human feedback, and online outcomes;
- training infrastructure, inference, GPUs, data, and observability;
- personal AI, longitudinal interaction, control, privacy, and model experience;
- modern hands-on projects that verify these concepts.

## What does not receive standalone coverage

- catalog-style introductions to traditional classifiers, CNNs, RNNs, or SVMs;
- paper histories ordered only by year and disconnected from current decisions;
- deprecated framework API tutorials;
- performance claims without versions, hardware, and workloads;
- lists of terms without system relationships or verification methods;
- background added for completeness that changes neither understanding nor practice.

Exception: historical mechanisms remain when they are still active. Stable softmax, embedding retrieval, SIMD versus SIMT, quadratic attention intermediates, and exposure bias all directly affect modern systems.

## Five minutes by default

Each note normally has:

- one central question;
- roughly five minutes of reading;
- one core example or mental model;
- one quantity, experiment, or judgment the reader can verify;
- a clear next note or related module.

Large topics use an index plus short notes. A project overview remains short, while detailed milestones move into child notes or code directories.

## Bilingual reading contract

Chinese is the default reading surface, without hiding the canonical English
terminology:

- the Chinese concept card is the front, with technical terms linked to their
  standard English names through the glossary;
- every concept card has a complete, one-to-one English back—not merely a summary;
- readers switch language in place on the current card, without returning to the top
  or navigating away;
- formulas, shapes, examples, and caveats remain equivalent on both sides;
- the standalone English page remains available for continuous English reading, but
  it is not the primary bilingual-comparison interface.

When a concept changes, both sides are updated together. If an accurate counterpart
is not ready, do not publish a permanently drifting pair.

## Diagram standard

A diagram must explain a relationship that prose would make harder to scan. Keep one visual question per diagram, use short labels, and preserve a consistent grammar: rounded nodes for outcomes, cylinders for versioned state or evidence, diamonds for gates, solid arrows for the forward path, and dotted arrows for feedback or control. Group by phase only when it improves the reading order, and render-check every Mermaid block before publishing.

## Freshness

| Content | Review cadence |
| --- | --- |
| mathematics, memory, statistics, and systems foundations | annually |
| search, evaluation, and post-training methods | every 6 months |
| APIs, serving engines, distributed stacks, low precision, and hardware support | quarterly |
| explicit versions or product defaults | re-verify before publication |

Fast-moving pages carry `Last reviewed: YYYY-MM`. A current fact that cannot be verified should not be written as a permanent conclusion.

## Source order

1. official documentation and specifications;
2. original papers;
3. reproducible code, experiments, and benchmarks;
4. high-quality secondary explanations.

Facts, measured results, and inference must remain distinct. Benchmarks include model, data, hardware, version, configuration, and workload.

## Modern does not mean chasing novelty

Modern-first does not mean following every new term. Ask:

- Has it changed mainstream system design?
- Is the evidence credible?
- Does it solve a real bottleneck?
- Does it connect clearly to existing modules?
- Is it likely to help a decision six months from now?

If the only argument is that a topic is currently popular, it belongs in an experiment log before the main learning path.

AI Infrastructure follows a more specific [five-minute note and freshness policy](ai-infra/EDITORIAL.en.md).
