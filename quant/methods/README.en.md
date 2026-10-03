# Problem-solving methods without a flash of inspiration

[中文](README.md) · **English** · [Review map](../README.en.md)

> Reading time: ~8 min · Logic, counting, and proof methods · Last reviewed: 2026-10

Many puzzles ask whether you can discard the story and find an invariant or unavoidable collision. The aim here is transferable methods, not memorized punchlines.

## 1 · Reduce the problem before guessing the answer

Try n=1,2,3. For an operation, list what changes after one step. Small examples reveal patterns and disprove conjectures; they do not prove a general claim.

If each move flips two coins, the number of heads changes by −2,0,2. Its **parity is invariant**. Starting with all tails, an odd number of heads is unreachable. No enumeration of operation sequences is needed.

## 2 · Pigeonholes: explain why a collision must exist

Put N objects into m boxes. Some box contains at least $\lceil N/m\rceil$ objects; otherwise the total is too small.

Among n+1 integers, two share a remainder modulo n, so their difference is divisible by n. The useful representation is the n possible remainders.

<details markdown="1">
<summary>Extend it: any sequence of n integers has a nonempty consecutive subsequence with sum divisible by n</summary>

Consider prefix sums $S_1,\dots,S_n$ modulo n. A zero remainder finishes the proof. Otherwise n prefixes occupy n−1 nonzero remainders, so two coincide. Their difference is a consecutive sum. Equivalently include $S_0=0$ and apply pigeonholes directly to n+1 prefixes.

</details>

## 3 · Induction: the transition is the proof

To prove $\sum_{j=1}^n j=n(n+1)/2$, check n=1. Assuming the statement for n, adding n+1 gives $(n+1)(n+2)/2$.

Use **strong induction** when the next case relies on arbitrary smaller cases. Every integer greater than 1 factors into primes: a prime needs no further work; a composite splits into two strictly smaller factors greater than 1. Apply the induction hypothesis to both. This proves existence, not uniqueness of factorization.

## 4 · Contradiction, contrapositive, and extremal arguments

| Method | Move | Useful for |
| --- | --- | --- |
| Contrapositive | Prove not-Q implies not-P instead of P implies Q | A conclusion that is awkward to manipulate directly |
| Contradiction | Negate the conclusion and derive an inconsistency | Impossibility, irrationality, uniqueness |
| Extremal element | Choose a largest, smallest, or shortest object | Finite structures and decreasing processes |

<details markdown="1">
<summary>Example: why is √2 irrational?</summary>

Suppose $\sqrt2=a/b$ with coprime positive integers a,b. Then $a^2=2b^2$, so a is even. Writing a=2c gives b²=2c², so b is also even, contradicting coprimality. Reducing the fraction first is essential.

</details>

## 5 · Pairing, double counting, and recurrences

Pairing can turn a sum into constant pairs. Double counting describes the same objects in two ways:

$$
\sum_{k=0}^n k\binom nk=n2^{n-1},\qquad n\ge1.
$$

The left side chooses a k-person group and its leader. The right side chooses one leader from n people, then independently includes or excludes everyone else. Both count groups with a designated leader.

For growing structures, derive a recurrence and initial conditions. Binary strings of length n without adjacent ones either end in 0, giving $a_{n-1}$ choices, or end in 01, giving $a_{n-2}$. Thus $a_n=a_{n-1}+a_{n-2}$ with $a_0=1,a_1=2$.

## 6 · Information lower bounds

A test with at most b outcomes produces at most $b^k$ leaves after k adaptive tests. Distinguishing N equally plausible candidates requires at least $\lceil\log_bN\rceil$ tests.

This is a **necessary bound, not a construction**. Allowed tests may not split candidates evenly. A balance scale has three outcomes, but an unknown heavy-or-light direction and access to reference weights change the state space and feasible tests.

## 7 · Adversarial and average-case models differ

Minimizing worst-case loss means $\min_{\text{strategy}}\max_{\text{outcome}}L$. Maximizing expected gain requires an outcome distribution. Do not silently make all cases equally likely when none was specified.

Explain the model, invariant or partition, proof, then boundary cases. A useful solution need not sound like a clever trick.

Continue: [Calculus and optimization](calculus-optimization.en.md). Reference: [MIT Mathematics for Computer Science](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/pages/readings/).
