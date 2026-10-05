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

An overview can be short; a lesson should not stop at an outline. Do not remove reasoning or assumptions to meet a five-minute reading target, or add repetitive definitions just to make a page longer.

Build each lesson around a question the reader should be able to work through afterward:

- Start with a concrete situation and name the prerequisites.
- Carry one small example through the mechanism. Explain the symbols, calculate a result, and change an assumption.
- Give readers something they can check: a calculation, runnable code, or an interactive experiment. A diagram is not a substitute for an explanation.
- Compare at least one reasonable alternative: what does each save, sacrifice, and fail to handle?
- Separate established results, teaching assumptions, and untested designs, with links to the relevant original sources.

Order the material by how understanding develops, not by labels for beginners and experts. Someone looking for intuition can read the example and diagram; someone exploring the mechanism can follow the derivation; someone building it can find code and experimental conditions. A few in-page links can locate these parts. Do not add rows of proficiency badges or hide essential reasoning inside collapsed panels.

Split long topics by question, not word count. Guides explain the reading order; lessons develop the subject. Reuse examples and terminology across adjacent notes instead of introducing the whole system again. Career reflections and section landing pages do not need to follow a technical-lesson template.

## Keep practice close to the explanation

Where practice helps, place it beside the relevant explanation: calculate a small example, change a line of code, or test whether a conclusion survives a different assumption. Not every note needs exercises, and pages do not get an automatic deck of cards, mastery scores, or check-in controls.

Explain the assumptions and reasoning in a worked answer. For open questions, offer decision criteria rather than a single supposedly correct answer. Point out a specific misconception instead of saying “watch the details.”

Prefer small synthetic examples. Real measurements require public sources and experimental conditions. Review correctness, bilingual equivalence, and useful links.

Write Chinese as a clear spoken explanation, not a word-for-word translation. Keep familiar names such as Transformer, SFT, and LLM. Use the glossary for English equivalents at first mention: at most two automatic annotations per paragraph, without duplicating existing parenthetical explanations. Keep headings, controls, and diagrams uncluttered. Prefer concrete inputs, outputs, and actions to abstract jargon or unnecessary language switching.

Chinese should sound like an explanation to a person. “Fix the candidate pool before comparing” is more useful than abstract process language. Keep standard English terms where helpful, without translating sentences word for word.

## Bilingual reading contract

Preferred reading language and familiar terminology are separate choices. Some readers use only Chinese, some learned the subject in English but enjoy Chinese explanations, and others read only English. Do not infer a preference from nationality, education, or browser language.

- **Chinese must stand on its own.** Explain what a concept does in Chinese before supplying its English name. Keep standard names such as Transformer and SFT, but explain essential acronyms at first use. Readers should not need to look up English just to continue.
- **Bilingual readers choose freely.** Switch directly between the Chinese and English versions of the same article. Show English terms at first mention on Chinese pages by default, at most two per paragraph; avoid sentences built from unexplained jargon.
- **English is not an appendix.** Preserve the derivations, examples, code, diagrams, interaction instructions, review questions, and caveats. Navigation, controls, and accessibility labels must also work in English. No essential material should require a trip to the Chinese page.
- **Compare concepts in place.** Both faces of a bilingual card must be complete, with equivalent formulas, tensor shapes, examples, and assumptions. Open on the page's language and allow switching without losing the reading position.
- **No forced redirects.** Respect the language of the URL the reader opened. Do not choose the page language from their identity or browser settings.

Review each note three ways: can someone understand the Chinese without relying on English? Can an English-trained reader recognize the concepts? Can someone complete the examples and exercises using only the English page? `site/paritycheck.py` checks structural parity, not translation quality; it cannot replace this review.

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
