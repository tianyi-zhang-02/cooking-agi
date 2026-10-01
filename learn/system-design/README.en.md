# System design: one problem, several defensible answers

[中文](README.md) · **English**

The goal isn't to memorize a diagram. Start with explicit constraints, build a workable version, then change a condition and see where the design breaks.

These are **original teaching exercises**, not company interview questions or production configurations.

## Choose an exercise

| Exercise | Main skill | Easy mistake |
| --- | --- | --- |
| [Recommendation feed](feed.en.md) | Retrieval, ranking, freshness, and fallbacks | Ranking cannot recover an item retrieval missed |
| [Knowledge-base answers with citations](rag.en.md) | Indexing, retrieval, permissions, and evidence | Seeing a document doesn't make an answer supported |
| [Editable long-term memory](memory.en.md) | Writes, conflicts, retrieval, and deletion | Superseded information can still be retrieved |

## How to practice

Spend five minutes writing goals, non-goals, and constraints. Draw the data flow and give each arrow an input and output. Then challenge the design with a timeout, stale information, a distribution shift, or a revoked permission.

More components don't necessarily mean a better design. Removing one while preserving the important properties often shows deeper understanding.

## Design exercises versus implementation studies

A design exercise lets you state assumptions and compare alternatives. [Industry Practice](../../practice/README.en.md) returns to public evidence: what does the code actually implement, and what remains unknown? The two connect, but a diagram you designed isn't evidence of someone else's production architecture.

The [system-design reading guide](../../interview/system-design.en.md) keeps the resource list. Return to the [study map](../README.en.md) for foundations.
