# Math and quant review: turn formulas into explanations

[中文](README.md) · **English**

> Reading time: ~6 min · Coverage map and reading routes · Last reviewed: 2026-10

These are notes I want to revisit myself. When a problem changes its wording, the goal is to know what to define, which property to try, and how to check the answer.

Start with the [probability route](probability/study-guide.en.md) if that is what you need. The sections below connect it to mathematics, stochastic processes, and finance.

## How this relates to the “Green Book”

The working reference is Xinfeng Zhou's *A Practical Guide to Quantitative Finance Interviews*. Its [bibliographic description](https://books.google.com/books?id=RosxmAYFFosC) lists logic puzzles, calculus, linear algebra, probability, stochastic processes, finance, and programming.

**This is an independent topic review, not a reproduction of the book's questions and solutions.** Major areas are represented, but the exact edition's contents and exercises have not been checked item by item. It is therefore not labeled “100% of the book covered.” Full derivations, proof sketches, and introductory statements are distinguished.

| Area | Where to review | Current depth |
| --- | --- | --- |
| Problem solving and explanation | [Proof toolbox](probability/proof-toolbox.en.md), [mixed review](review.en.md) | Modeling, first steps, checks, and explanations |
| Logic and brain teasers | [Invariants, pigeonholes, induction, double counting](methods/README.en.md) | Proofs and variations, not a riddle-answer bank |
| Calculus | [Continuous probability](probability/continuous-calculus.en.md), [calculus and optimization](methods/calculus-optimization.en.md) | Integration, transformations, MVT, Taylor, convexity; introductory KKT |
| Linear algebra | [Projection, least squares, PSD, SVD](methods/linear-algebra.en.md) | Key derivations; spectral/SVD existence theorems invoked, not fully proved |
| Probability foundations | [Axioms through limits](probability/study-guide.en.md) | Event, conditioning, expectation, inequality proofs; CLT proof outline |
| Counting and random variables | [Counting](probability/counting.en.md), [distributions](probability/distribution-toolkit.en.md), [joint and order statistics](probability/joint-and-order.en.md) | Derivations, examples, counterexamples |
| Stochastic processes | [Markov chains](probability/markov-chains.en.md), [Poisson](processes/README.en.md), [martingales](processes/martingales.en.md), [dynamic programming](processes/dynamic-programming.en.md) | Finite-state recurrences, bounded stopping and Wald proofs, finite-horizon decisions |
| Stochastic calculus | [Brownian motion and Itô](processes/brownian-ito.en.md) | Quadratic variation, Itô intuition and examples; not a rigorous course replacement |
| Finance | [Replication and trees](finance/README.en.md), [options](finance/options-and-greeks.en.md), [portfolios and risk](finance/portfolio-and-market.en.md) | Replication, BS derivation route, Greeks, basic risk concepts |
| Algorithms and numerics | [Numerics and streaming algorithms](methods/numerical-and-coding.en.md), [existing algorithm track](../interview/leetcode.en.md) | Methods, complexity, executable examples; C++ semantics remain incomplete |
| Extensions | [Statistical inference](methods/statistics.en.md) | MLE, Bayes, intervals, tests, regression, temporal dependence, validation |

## Choose a route by purpose

<div class="lesson-recipe">
  <div><span>1 · Rebuild foundations</span><strong>Events, counting, distributions, expectation: define what is random</strong></div>
  <div><span>2 · Recover the proofs</span><strong>Conditioning, inequalities, limits: justify each equality</strong></div>
  <div><span>3 · Understand processes and models</span><strong>Recurrences, stopping, optimization: reduce a large problem</strong></div>
  <div><span>4 · Check the conclusion</span><strong>Sampling error, numerical error, model assumptions: look beyond the final number</strong></div>
</div>

- **Probability first:** [event proofs](probability/event-proofs.en.md) → [counting](probability/counting.en.md) → [expectation](probability/expectation-proofs.en.md) → [conditioning](probability/conditioning-proofs.en.md) → [distributions](probability/distribution-toolkit.en.md).
- **Research-oriented mathematics:** [inequalities](probability/inequalities.en.md) → [LLN / CLT](probability/limits.en.md) → [linear algebra](methods/linear-algebra.en.md) → [inference](methods/statistics.en.md) → [numerics](methods/numerical-and-coding.en.md).
- **Waiting, decisions, and finance:** [Markov chains](probability/markov-chains.en.md) → [Poisson](processes/README.en.md) → [martingales](processes/martingales.en.md) → [dynamic programming](processes/dynamic-programming.en.md) → [Itô](processes/brownian-ito.en.md) → [replication](finance/README.en.md).

These are learning routes, not claims about a particular company's interviews. Deeper derivatives material depends on the role.

## Use each chapter actively

1. **State the model.** What is random? What is independent? What is the sample space?
2. **Derive the key step.** Find the equality that simplifies the problem.
3. **Change an assumption.** Replacement versus no replacement; fair versus biased; fixed time versus stopping.
4. **Check once.** Small instances, extreme parameters, units, bounds, or a second method.
5. **Use code to catch mistakes.** Numerical examples do not replace proofs; simulations need uncertainty estimates.

Hints and solutions expand in place. No streaks, mastery scores, or rankings.

## What remains worth expanding

- An edition-specific subsection map, especially programming languages and problem types; this needs the actual contents.
- Complete spectral and SVD existence proofs, general optional stopping, and weak convergence.
- Deeper mathematical statistics, time series, and numerical linear algebra; the current level is a foundation for further lectures.
- Specialized interest-rate models, exotic derivatives, and a complete C++ track are not marked finished.

Short on time? Use [mixed review: assumptions and variations](review.en.md). If the foundations feel shaky, return to the [probability route](probability/study-guide.en.md).
