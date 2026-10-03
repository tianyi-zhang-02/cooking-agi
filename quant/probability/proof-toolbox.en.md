# Proof techniques: finding a useful first step

[中文](proof-toolbox.md) · **English**

> Reading time: ~7 min · Prerequisites: events, expectation, and conditioning from earlier chapters · Last reviewed: 2026-10

A proof trick is often a change of representation rather than a flash of genius. Turn an awkward object into one you already know how to handle. These moves transfer without memorizing an entire solution.

## 1 · For “at least,” try the complement

For $n$ independent trials with success probability $p$, computing exactly 1 success, 2 successes, and so on is unnecessary. No successes gives one product:

$$
P(\text{at least one success})=1-(1-p)^n.
$$

**Check assumptions.** Taking a complement always works; factoring the failure probability needs independence. Marginal success probabilities of $p$ alone do not justify the answer.

## 2 · For “how many,” try indicators

Let $N$ be the number of fixed points in a uniformly random permutation of $n$ elements. Let $I_i$ indicate that element $i$ stays in place:

$$
N=\sum_{i=1}^nI_i,\qquad \mathbb E[N]=\sum_{i=1}^nP(I_i=1)=n\cdot\frac1n=1.
$$

This avoids finding the full distribution of $N$. Fixed-point indicators are dependent, but the calculation does not need independence.

**Transfer.** For collisions, repetitions, or adjacent qualifying pairs, first identify the objects being counted. Counting same-birthday pairs calls for one indicator per pair, not one per person.

## 3 · For symmetry, keep ties in the calculation

A flips $n+1$ fair coins and B flips $n$, with all flips independent. What is the probability that A gets more heads?

Set aside A's extra flip. The remaining counts $X,Y$ are iid. Write $r=P(X=Y)$. Exchanging the two people preserves the distribution, so $P(X>Y)=P(Y>X)=(1-r)/2$.

<details markdown="1">
<summary>Put the extra coin back</summary>

If A already leads, the extra flip cannot remove the lead. If A trails, it can at most produce a tie. From a tie, an extra head creates a lead. Therefore,

$$
P(\text{A leads})=\frac{1-r}{2}+\frac r2=\frac12.
$$

Keep the tie term rather than skipping it. Symmetry equates the two strict-lead probabilities; it does not make each $1/2$ when ties are possible.

</details>

If every coin has head probability $p$, the first $n$ flips remain exchangeable between people, but the extra flip is no longer fair. The answer becomes $1/2+(p-1/2)r$. **State exactly which part of the model is symmetric.**

## 4 · For stages, condition on the first one

Choose a coin and then flip it repeatedly: [condition on the selected coin](conditioning-proofs.en.md) to obtain independent Bernoulli trials. Wait for a pattern: [condition on the next step](markov-chains.en.md), after which the remaining problem starts from another state.

An expected-time recurrence has “**cost of this step + remaining expectation**.” A success-probability recurrence has no extra 1. Their shapes may look similar, but their units differ.

## 5 · For a bound, find a pointwise relation

Markov begins with $a\mathbf1_{\{X\ge a\}}\le X$; Cauchy–Schwarz begins with $(X-tY)^2\ge0$. Prove a relation for each outcome, then average, rather than guessing the final inequality.

This also suggests counterexamples. If $X$ can be negative, which outcomes break the first comparison? That explains Markov's nonnegativity assumption.

## 6 · For many terms, try induction or telescoping

Prove the finite union bound by induction: the $n=1$ case holds; assuming the $n$-event case, combine those events into one and apply the two-event bound with event $n+1$. State both the base case and the induction step.

Another common move is telescoping:

$$
\sum_{k=1}^n\frac1{k(k+1)}
=\sum_{k=1}^n\left(\frac1k-\frac1{k+1}\right)
=1-\frac1{n+1}.
$$

The decomposition creates cancellation. Turning a random-walk recurrence into adjacent differences serves a similar purpose: make the relation something we can sum.

## 7 · Find counterexamples in the smallest useful model

| Suspicious claim | First example to try |
| --- | --- |
| Uncorrelated means independent | $X\in\{-1,0,1\}$, $Y=X^2$ |
| More identically distributed samples always stabilize the mean | Copy the same $Z$ n times |
| Independent means disjoint | Positive-probability disjoint events on a die |
| Expectation passes through every function | $X=\pm1$, then square |
| Reciprocal window probability is the waiting time | Compare HHT and HTH |

A two- or three-point sample space often defeats an overstrong statement. Start there before constructing an elaborate distribution.

## Explaining a proof aloud

Give the route before the algebra: “Instead of counting whole permutations, I will count fixed positions using indicators and linearity.” Then define the variables, state the conditions, justify the key equalities, and check a small case or boundary.

Try the [Exercises and counterexamples](practice.en.md) next. Every argument links back through the [review guide](study-guide.en.md); there is no separate vocabulary of tricks to memorize.
