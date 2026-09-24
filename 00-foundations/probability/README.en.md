# Probability: starting from three axioms

[中文](README.md) · **English**

> Reading time: ~5 min · Level: beginner · Last reviewed: 2026-09

<div class="lesson-recipe">
  <div><span>The problem</span><strong>what a probability actually is, and which facts are definitions and which are derived</strong></div>
  <div><span>Prerequisites</span><strong>intersection, union and complement of sets</strong></div>
  <div><span>Core mechanism</span><strong>a sample space Ω, events, a probability function P, and three axioms</strong></div>
  <div><span>Common mistakes</span><strong>mixing up events and random variables, e.g. writing P(X)</strong></div>
</div>

## Three pieces

Take "does it rain tomorrow" as the example:

- **Sample space Ω**: the set of all possible outcomes. Here $\Omega = \{\text{rain}, \text{no rain}\}$.
- **Event**: a subset of Ω. $A = \{\text{rain}\}$, and its complement is $A^c = \{\text{no rain}\}$.
- **Probability P**: a function that assigns a number to each event, obeying the three axioms below.

Note that "it rains tomorrow" is an **event**, not a random variable. Getting that distinction clean now saves a lot of trouble later.

## Three axioms

Kolmogorov, in 1933, defined a probability as a function P satisfying:

1. $P(A) \ge 0$ for every event A;
2. $P(\Omega) = 1$;
3. if $A_1, A_2, \dots$ are pairwise disjoint, then $P(A_1 \cup A_2 \cup \cdots) = P(A_1) + P(A_2) + \cdots$.

**That is the whole definition.** Everything else in probability is either a theorem derived from these three lines or a further definition laid on top (conditional probability, independence, random variables).

The full object is a probability space $(\Omega, \mathcal{F}, P)$. $\mathcal{F}$ is the collection of events you are allowed to assign probability to: every subset in the discrete case; in the continuous case, intervals and everything built from them by countable unions and complements. You will almost never need to think about it in an interview.

## How you compute it is not how it is defined

- **Equally likely outcomes**: when Ω is finite and every outcome is equally likely, $P(A) = |A| / |\Omega|$. This is a consequence of axiom 3, not an axiom. Dice, cards and coins all live here.
- **The continuous case**: $P(a \le X \le b) = \int_a^b f(x)\,dx$. The density f is a device that makes P satisfy the axioms; it is not itself a probability.
- **Long-run frequency**: repeat the experiment n times and the fraction of times A occurs converges to $P(A)$. That is the law of large numbers, a theorem — and the intuition for why these are the right axioms.

## Six consequences you can derive at once

Each takes at most three lines. Try them yourself before reading:

1. $P(A^c) = 1 - P(A)$. A and $A^c$ are disjoint and their union is Ω, so $P(A) + P(A^c) = P(\Omega) = 1$. In the example, $P(\text{rain}) = 0.8$ gives $P(\text{no rain}) = 0.2$ — you knew the answer already; now it comes from the axioms.
2. $P(\varnothing) = 0$. $\varnothing$ is the complement of Ω; use the first one.
3. If $A \subseteq B$ then $P(A) \le P(B)$. Split B into the disjoint pieces A and $B \setminus A$: $P(B) = P(A) + P(B \setminus A) \ge P(A)$.
4. $P(A) \le 1$. $A \subseteq \Omega$; use the third one.
5. $P(A \cup B) = P(A) + P(B) - P(A \cap B)$. Split $A \cup B$ into the three disjoint pieces $A \setminus B$, $A \cap B$ and $B \setminus A$, then write out $P(A)$ and $P(B)$: the intersection is counted twice, so subtract it once.
6. The union bound, $P(A_1 \cup \cdots \cup A_n) \le \sum_i P(A_i)$. The fifth gives $P(A \cup B) \le P(A) + P(B)$; induct. Used constantly in "at least one of them happens" problems.

## An event is not a random variable

**A random variable is a function** $X: \Omega \to \mathbb{R}$ that maps each outcome to a number. The usual way to turn an event into one is the indicator:

$$X = \mathbf{1}_A = \begin{cases} 1 & \text{rain} \\ 0 & \text{no rain} \end{cases}$$

Then $P(X = 1) = P(A) = 0.8$ and $P(X = 0) = 0.2$: X is Bernoulli(0.8). And here is the line that powers half of all probability interview problems:

$$\mathbb{E}[\mathbf{1}_A] = 1 \cdot P(A) + 0 \cdot P(A^c) = P(A)$$

**The expectation of an indicator is the probability of the event.** It is the bridge from "events and chance" to "expected counts": a question of the form "on average, how many…" almost always splits the count into a sum of indicators and uses linearity of expectation. The variance comes for free:

$$\mathrm{Var}(\mathbf{1}_A) = P(A)\,\bigl(1 - P(A)\bigr) = 0.8 \times 0.2 = 0.16$$

## Which notation goes with what

| Object | Takes | Examples |
| --- | --- | --- |
| Event | P | $P(A)$, $P(A \cap B)$, $P(A \mid B)$ |
| Random variable | E and Var, or P of a specific value | $\mathbb{E}[X]$, $\mathrm{Var}(X)$, $P(X = k)$ |

Writing $P(X)$ when X is a random variable reads as a red flag in an interview. A small thing, and easiest to fix now.

Next, the random variable gets drawn: [drawing a distribution](distributions.en.md).

## Common interview questions

<details class="interview" markdown="1">
<summary>Flip two fair coins and let A be "at least one head". What is P(A)? Compute it two ways.</summary>

$\Omega = \{HH, HT, TH, TT\}$ and $A = \{HH, HT, TH\}$. Counting directly: $3/4$. Through the complement: $A^c = \{TT\}$, so $1 - 1/4 = 3/4$.

</details>

<details class="interview" markdown="1">
<summary>Continuing: what are the expectation and variance of the indicator 1_A?</summary>

$\mathbb{E}[\mathbf{1}_A] = P(A) = 3/4$ and $\mathrm{Var}(\mathbf{1}_A) = 3/4 \times 1/4 = 3/16$.

</details>

<details class="interview" markdown="1">
<summary>Using only the three axioms, derive P(A ∪ B) = P(A) + P(B) − P(A ∩ B).</summary>

Split $A \cup B$ into three disjoint pieces: $A \setminus B$, $A \cap B$ and $B \setminus A$. By axiom 3, $P(A \cup B)$ is the sum of the three; likewise $P(A) = P(A \setminus B) + P(A \cap B)$ and $P(B) = P(B \setminus A) + P(A \cap B)$. Substitute: $A \cap B$ is counted once too often, so subtract it.

</details>

<details class="interview" markdown="1">
<summary>Is "it rains tomorrow" a random variable? Why not write P(X)?</summary>

No — it is an event, a subset of Ω. A random variable is a function from Ω to the real numbers, such as the indicator $\mathbf{1}_A$. Events take P; random variables take E and Var, and $P(X = k)$ when you mean one particular value.

</details>
