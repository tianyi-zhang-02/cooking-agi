# Event proofs: why the decomposition works

[中文](event-proofs.md) · **English**

> Reading time: ~7 min · Prerequisite: [Probability axioms](README.en.md) · Last reviewed: 2026-10

Why can we not simply add probabilities for “at least one occurs”? The same outcome may have been counted several times. Often the first step is not choosing a formula, but separating overlapping sets.

## 1 · From the empty set to complements

**Use only the three axioms.** Write $\Omega$ as the disjoint union of $\Omega,\varnothing,\varnothing,\ldots$. Countable additivity and nonnegativity give $1=1+\sum_{n\ge1}P(\varnothing)$, which forces $P(\varnothing)=0$. Finite additivity then follows as a special case of countable additivity.

Since $\Omega=A\mathbin{\dot\cup}A^c$,

$$
P(A^c)=1-P(A).
$$

The symbol $\dot\cup$ emphasizes disjointness. **We can add directly because the pieces are disjoint, not because they are independent.**

## 2 · Monotonicity and inclusion–exclusion

If $A\subseteq B$, write $B=A\mathbin{\dot\cup}(B\setminus A)$. Then

$$
P(B)=P(A)+P(B\setminus A)\ge P(A).
$$

Two arbitrary events split into three pieces:

$$
A\cup B=(A\setminus B)\mathbin{\dot\cup}(A\cap B)\mathbin{\dot\cup}(B\setminus A).
$$

Expand $P(A)$ and $P(B)$ separately. Their intersection appears twice, giving inclusion–exclusion:

$$
P(A\cup B)=P(A)+P(B)-P(A\cap B).
$$

On a fair die, let $A=\{2,4,6\}$ and $B=\{4,5,6\}$. Adding gives 1, but their union is only $\{2,4,5,6\}$, with probability $2/3$. The overcount is exactly $\{4,6\}$.

<details markdown="1">
<summary>For 3 events, why add the triple intersection back?</summary>

After adding the three individual events and subtracting the three pairwise intersections, an outcome in all three has been counted $3-3=0$ times. Add it once:

$$
P(A\cup B\cup C)=P(A)+P(B)+P(C)-P(A\cap B)-P(A\cap C)-P(B\cap C)+P(A\cap B\cap C).
$$

Check the count of each outcome rather than memorizing alternating signs.

</details>

## 3 · The union bound does not need independence

**Theorem.** For finitely or countably many events,

$$
P\left(\bigcup_{n\ge1}A_n\right)\le\sum_{n\ge1}P(A_n).
$$

Remove outcomes already included in earlier events:

$$
B_1=A_1,\qquad B_n=A_n\setminus\bigcup_{k<n}A_k.
$$

The $B_n$ are disjoint, but their union is unchanged. Since $B_n\subseteq A_n$,

$$
P\left(\bigcup_n A_n\right)=\sum_nP(B_n)\le\sum_nP(A_n).
$$

If each of 20 checks has false-alarm probability at most $0.01$, the chance of at least one false alarm is at most $0.2$, regardless of dependence. This is a **bound**, not the actual false-alarm rate.

When every $A_n$ is the same event, the bound can be very loose. That is the cost of making fewer assumptions.

## 4 · Why increasing events allow a probability limit

**Theorem: continuity from below.** If $A_1\subseteq A_2\subseteq\cdots$ and $A=\bigcup_nA_n$, then $P(A_n)\to P(A)$.

<details markdown="1">
<summary>Proof: separate the new part at each step</summary>

Set $D_1=A_1$ and $D_n=A_n\setminus A_{n-1}$. These increments are disjoint, and

$$
P(A_n)=\sum_{k=1}^{n}P(D_k),\qquad P(A)=\sum_{k=1}^{\infty}P(D_k).
$$

An infinite series is the limit of its partial sums. Countable additivity is doing the work.

</details>

For decreasing events $A_n\downarrow A=\bigcap_nA_n$, apply this result to the complements to obtain $P(A_n)\to P(A)$. Total probability is finite, so no subtraction of infinities occurs.

**Application: why is a CDF right-continuous?** Fix $x$ and let $x_n\downarrow x$. The events $\{X\le x_n\}$ decrease to $\{X\le x\}$, so $F(x_n)\to F(x)$. A discrete CDF can jump, but its value agrees with the limit from the right.

## 5 · Optional: can errors occur infinitely often?

If $\sum_nP(A_n)<\infty$, the probability that $A_n$ occurs infinitely often is 0. This is the first Borel–Cantelli lemma; independence is not required.

<details markdown="1">
<summary>Derive it using the bound above</summary>

Infinitely many occurrences imply an occurrence after every index $N$. That event is contained in every $\bigcup_{n\ge N}A_n$, hence

$$
P(A_n\text{ infinitely often})\le P\left(\bigcup_{n\ge N}A_n\right)\le\sum_{n\ge N}P(A_n)\longrightarrow0.
$$

For example, $P(A_n)\le1/n^2$ suffices. A divergent sum alone does **not** give the converse: if every $A_n=A$ with $P(A)=1/2$, the sum diverges but the probability of infinitely many occurrences remains $1/2$.

</details>

## Next

Optional reference: [MIT 18.175: Borel–Cantelli and the strong law](https://ocw.mit.edu/courses/18-175-theory-of-probability-spring-2014/resources/mit18_175s14_lecture9/).

The repeated move is simple: **rewrite the sets before computing probabilities.** Next, turn events into numbers: [Expectation and variance proofs](expectation-proofs.en.md).

Foundational reference: [MIT 6.041: Probability models and axioms](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/resources/mit6_041f10_l01/). The Borel–Cantelli section is optional on a first pass.
