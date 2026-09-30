# Writing clear notes that are easy to revisit

[中文](EDITORIAL.md) · **English** · [Back to home](README.en.md)

> Reading time: ~5 minutes · Last reviewed: 2026-08

## Core principle

Start with the reader's question, then introduce the concepts needed to answer it. Pick a concrete part of this chain:

```text
user goals
→ data, memory, search, and tools
→ LLM / multimodal policy
→ training, serving, and runtime
→ evaluation, feedback, and continual improvement
```

Older methods belong when they explain later designs. We don't need a paper-by-year chronology, but we shouldn't skip foundations just to appear current.

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

## Answer one question at a time

Each note normally has:

- one central question;
- roughly five minutes of reading;
- one core example or mental model;
- one quantity, experiment, or judgment the reader can verify;
- a clear next note or related module.

Split large topics into a guide and shorter notes. The guide explains the order and why each next step matters. Keep details in their own notes rather than reintroducing the whole system on every page.

## Review cards are not summaries

Each existing content chapter has 2–8 bilingual cards in `site/review.json`, shared across its pages. Ask readers to explain, calculate, or decide—not just expand an acronym.

- `question`: a directly answerable question or a scenario with sufficient conditions.
- `answer`: reasoning and assumptions; decision criteria rather than a fake single correct answer for open questions.
- `pitfall`: a specific misconception, not a generic warning to be careful.
- `source`: the related Chinese Markdown path; the builder connects the English version.
- `id`: keep it stable when editing wording. Content changes automatically invalidate old mastery marks.

Prefer small synthetic examples. Real measurements require public sources and experimental conditions. Review correctness, bilingual equivalence, and useful source links. Run `python3 -m unittest discover -s site/tests` to check chapter coverage and references.

Chinese should sound like an explanation to a person. “Fix the candidate pool before comparing” is more useful than abstract process language. Keep standard English terms where helpful, without translating sentences word for word.

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
