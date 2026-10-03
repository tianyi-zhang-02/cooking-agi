# Probability review: derive the formulas again

[中文](study-guide.md) · **English**

For linear algebra, statistics, processes, and finance, use the [full coverage map](../README.en.md). This page keeps the probability route focused rather than becoming one long topic list.

> Reading time: ~4 min · Foundations and proofs · Last reviewed: 2026-10

Some formulas feel familiar until I try to explain them from scratch. These notes are for closing that gap: why the first step works, and whether the argument survives a change in assumptions.

The route runs from probability axioms through expectation, conditioning, inequalities, limits, and Markov chains. The examples are general mathematics exercises, not a company-specific interview collection. Proofs stay short where possible; arguments requiring extra background are explicitly labeled as sketches.

## Pick a route

| What you need | Suggested route |
| --- | --- |
| Refresh the basics | [Axioms and random variables](README.en.md) → [Distributions](distributions.en.md) → [Conditioning](conditional.en.md) |
| Turn familiar formulas into proofs | [Event proofs](event-proofs.en.md) → [Expectation and variance](expectation-proofs.en.md) → [Conditional expectation](conditioning-proofs.en.md) |
| Revisit common theorems | [Inequalities](inequalities.en.md) → [LLN and CLT](limits.en.md) |
| Get unstuck on continuous variables or waiting times | [Continuous probability and calculus](continuous-calculus.en.md) → [Markov chains](markov-chains.en.md) |
| Practice finding a first step | [Proof techniques](proof-toolbox.en.md) → [Exercises and counterexamples](practice.en.md) |
| Expand the common problem types | [Counting and sampling](counting.en.md) → [Distributions](distribution-toolkit.en.md) → [Joint and order statistics](joint-and-order.en.md) |

One chapter at a time is enough. Before opening a proof, write down the assumptions and the goal. Look at the explanation when you can identify the step you cannot justify.

## How the chapters connect

<div class="lesson-recipe">
  <div><span>1 · Events</span><strong>Separate overlapping cases before adding probabilities</strong></div>
  <div><span>2 · Random variables</span><strong>Turn occurrences into numbers to study averages and variation</strong></div>
  <div><span>3 · Conditioning and bounds</span><strong>Fix some information; when an exact answer is hard, control its size</strong></div>
  <div><span>4 · Limits and processes</span><strong>What changes with repetition, and what must a state remember?</strong></div>
</div>

Two connections are particularly useful:

- **Indicators → Markov's inequality → Chebyshev → weak LLN.** An event probability can be controlled through the expectation of a nonnegative random variable.
- **Total probability → conditional expectation → first-step analysis.** Split on the next outcome, then express the remaining problem using the same states.

## Common theorems: check the assumptions first

| Theorem or tool | What to check | Starting point |
| --- | --- | --- |
| Union bound | Finitely or countably many events; no independence needed | [Remove overlaps](event-proofs.en.md) |
| Linearity of expectation | Here we assume absolute integrability; no independence needed | [Expand the joint distribution](expectation-proofs.en.md) |
| Tail-sum formula | Nonnegative integer-valued variable; infinity is allowed | [Stack indicators](expectation-proofs.en.md) |
| Tower property | Here we assume integrability | [Average within groups, then across groups](conditioning-proofs.en.md) |
| Total variance | Finite second moment | [Within-group and between-group variation](conditioning-proofs.en.md) |
| Markov / Chebyshev | Nonnegativity / finite variance; positive threshold | [Pointwise comparison, then expectation](inequalities.en.md) |
| Cauchy–Schwarz / Jensen | Second moments / convexity and integrability conditions | [Nonnegative squares / supporting lines](inequalities.en.md) |
| Weak LLN | Our proof uses iid variables with finite variance | [Variance of the sample mean shrinks](limits.en.md) |
| Central limit theorem | Our version uses iid variables with finite, nonzero variance | [Standardize, then study the distribution](limits.en.md) |
| First hitting time | A sufficient state; finite expectation or a separate justification | [First-step cases and boundary conditions](markov-chains.en.md) |

A sufficient condition is not necessarily a necessary one. Finite variance makes our LLN proof short, for example, but other versions need less.

## Four questions when a proof stalls

1. Am I manipulating an event, a random variable, or a fixed number?
2. Does this equality come from a definition or a theorem? Are its assumptions satisfied?
3. Am I double-counting, ignoring ties, or silently dropping a condition?
4. Can I build a counterexample without independence, nonnegativity, or finite variance?

Proofs and hints expand where they are needed, without streaks or mastery scores. The [exercise page](practice.en.md) also links a Python standard-library checker. Enumeration and exact fractions help catch mistakes; they do not replace general proofs.

## Original lecture notes

The [MIT 6.041 lecture index](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/pages/lecture-notes/) is useful for looking up topics. [Harvard Stat 110's Markov chain handout](https://stat110.hsites.harvard.edu/resource/markov-chains) continues into states, transition matrices, and long-run behavior.

Start here: [Event proofs: why the decomposition works](event-proofs.en.md).
