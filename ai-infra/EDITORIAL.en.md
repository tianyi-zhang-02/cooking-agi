# Five-Minute Notes and Freshness Policy

[中文](EDITORIAL.md) · **English** · [Back to index](README.en.md)

> Reading time: ~5 minutes · Type: editorial policy · Last reviewed: 2026-08

## One page answers one question

A title should become a precise question, such as “Why is BF16 more suitable than FP16 for many training workloads?” rather than “Everything about low precision.” A page that simultaneously explains hardware, algorithms, APIs, benchmarks, and career paths should be split.

Suggested body targets:

- roughly 500–900 English words;
- roughly 1,000–2,200 Chinese characters;
- one central mental model;
- at most one or two necessary formulas or diagrams;
- one executable exercise or diagnostic question.

Code, tables, and citations should not be mechanically removed to meet a word count, but each must serve the same question.

## Standard structure

```text
Title
Reading time · Level · Freshness · Last reviewed
Why this matters
Core mental model
What to calculate or observe
Hands-on check
Common failure
Next note
```

A project can be larger, but its overview should still target five minutes. Detailed milestones become separate child notes.

## Classification principles

The main index follows working problems rather than paper chronology or vendors:

- foundations;
- training systems;
- inference systems;
- platforms and reliability;
- data, evaluation, and learning loops;
- hands-on projects.

Each note has one primary category. Other relationships appear under related notes so the same link is not repeated everywhere.

## Freshness labels

| Label | Meaning | Review cadence |
| --- | --- | --- |
| Stable foundation | long-lived architecture, memory, or numerical concepts | annually |
| Evolving | the design remains useful while tools and implementations change | every 6 months |
| Fast-moving | APIs, hardware formats, serving, and distributed implementations | quarterly |
| Version-specific | tied to an explicit release or device | re-verify before publishing |

Every page carries `Last reviewed: YYYY-MM`. This records when the content was rechecked, not when its sources were published.

## Triggers for an update

- An official API, default, or support matrix changes.
- New hardware changes mainstream dtypes, communication, or execution paths.
- Mainstream serving or training architecture changes materially.
- An old benchmark no longer represents real workloads.
- Links break or sources become deprecated.
- Hands-on evidence contradicts the written conclusion.

Update the mental model and decision rule first; do not merely append release notes.

## What stays out of the main path

- Catalogs of traditional models unrelated to current LLM systems.
- Long historical surveys that do not explain a current design choice.
- Performance numbers without measurement conditions.
- Vendor features without a version.
- Tutorials that show framework calls but not data movement or bottlenecks.
- Soon-to-be-obsolete APIs presented as permanent concepts.

Historical material remains only when it explains a current system, such as stable softmax, SIMD versus SIMT, or attention's I/O problem.

## Source rules

Prefer:

1. official hardware and software documentation;
2. original research papers;
3. reproducible code and benchmarks;
4. high-quality engineering explanations as supporting context.

Claims about current APIs, hardware support, or defaults must be rechecked against official sources. Clearly distinguish inference from measured evidence.

## Completion check

- Can the page's question be stated in one sentence?
- Can it be read in roughly five minutes?
- Does it include a quantity or exercise the reader can verify?
- Has history unrelated to a current decision been removed?
- Do fast-moving claims have a review date and reliable source?
- Does the next link feel natural rather than forcing a linear path?

Only notes satisfying these conditions enter the main index.
