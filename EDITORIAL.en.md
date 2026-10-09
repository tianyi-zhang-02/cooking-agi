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

Place examples where a reader is likely to get stuck, rather than inventing a story for every paragraph. For candidate softmax, ask what happens when a fourth equally scored candidate joins three others. Seeing the probability fall from $1/3$ to $1/4$ makes dependence on the comparison pool tangible before the formula. Change one important condition at a time and keep the numbers checkable. A constructed example explains a mechanism; it does not establish model performance.

Review with three questions: without the derivation, can readers explain the problem being solved? If they continue, can they see why the result holds and where it fails? If they want to build it, can they find an implementation and checks? Stopping early should still be useful, without making the rest of the article shallow.

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

A diagram should show something concrete: how a vector changes, where a request goes, or how two settings differ. Reuse the example in the text where possible and label arrows with what they carry. Boxes containing abstract nouns do not explain a mechanism. Use shapes and colors for meaningful roles, with a short legend when needed; do not force every topic into the same flowchart.

The default view should be informative without interaction. Controls support comparisons and exploration, not access to essential explanations. Label illustrative numbers and cite measured results. Check narrow screens, both languages, and the rendered output of every Mermaid block. Captions should explain what the figure does and does not show, rather than offer a slogan.

## Freshness

| Content | Review cadence |
| --- | --- |
| mathematics, memory, statistics, and systems foundations | annually |
| search, evaluation, and post-training methods | every 6 months |
| APIs, serving engines, distributed stacks, low precision, and hardware support | quarterly |
| explicit versions or product defaults | re-verify before publication |

Fast-moving pages carry `Last reviewed: YYYY-MM`. A current fact that cannot be verified should not be written as a permanent conclusion.

Name the version and verification date; changing the year on an old report is not an update. Explain new releases when they change architecture, interfaces, or evaluation. An announcement supports only what it discloses. State missing architecture and training details directly rather than inferring them from a family name.

Reference notes identify gaps; each topic still needs primary-source, example, and scope checks. Track reading, drafting, integration, tests, and release acceptance separately. Site-wide revisions follow `CONTENT_RELEASE_CHECKLIST.md` and are not pushed before acceptance.

## Source order

1. official documentation and specifications;
2. original papers;
3. reproducible code, experiments, and benchmarks;
4. high-quality secondary explanations.

Facts, measured results, and inference must remain distinct. Benchmarks include model, data, hardware, version, configuration, and workload.

For a specific method, read the relevant paper sections, experimental setup, and limitations—not only the abstract or reference notes. When background helps, explain the practices and bottlenecks at the time and what the authors actually compared. A paper's baselines do not describe the entire industry; later results are not evidence that was available then.

Place paper links beside the explanations they support, preferably with year, version, and relevant sections. Add official code or model cards when available. Mark only material actually read as checked. State missing public details rather than guessing, and keep these links within lessons instead of adding a separate Papers section.

## Modern does not mean chasing novelty

Modern-first does not mean following every new term. Ask:

- Has it changed mainstream system design?
- Is the evidence credible?
- Does it solve a real bottleneck?
- Does it connect clearly to existing modules?
- Is it likely to help a decision six months from now?

If the only argument is that a topic is currently popular, it belongs in an experiment log before the main learning path.
