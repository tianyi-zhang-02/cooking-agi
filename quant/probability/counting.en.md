# Counting: decide what counts as a different outcome

[中文](counting.md) · **English** · [Review map](../README.en.md)

> Reading time: ~9 min · Prerequisite: events and equally likely outcomes · Last reviewed: 2026-10

“Choose 3” is not a complete model. Replacement and order change the answer. Most counting mistakes begin when the numerator and denominator use different rules.

## 1 · Ask four questions first

Are objects distinguishable? Are positions distinguishable? Are repetitions allowed? Does order matter? For 3 selections from 8 distinct books:

| Model | Count | Reason |
| --- | --- | --- |
| Ordered, no repetition | $8\cdot7\cdot6$ | Successively fewer choices |
| Unordered, no repetition | $\binom83=56$ | Each subset appeared $3!$ times |
| Ordered, repetition allowed | $8^3$ | Eight choices at every position |
| Unordered, repetition allowed | $\binom{10}3=120$ | Allocate 3 selections among 8 types |

The last count is not $8^3/3!$: AAA has one ordering, while ABC has six. **Division by a symmetry factor works only when every object has the same multiplicity.**

## 2 · What stars and bars actually represents

For nonnegative integer solutions of $x_1+\cdots+x_m=n$, arrange n stars and m−1 separators. Segment i contains $x_i$ stars. This is a reversible correspondence, so:

$$
\#\{x_i\ge0:\sum_i x_i=n\}=\binom{n+m-1}{m-1}.
$$

If every $x_i\ge1$, assign one first and distribute n−m remaining items: $\binom{n-1}{m-1}$, provided $n\ge m$.

<details markdown="1">
<summary>Try it: allocate 10 identical places to 3 groups, each receiving at least 2</summary>

Allocate 6 first. The remaining 4 give $\binom{4+3-1}{3-1}=15$ possibilities. Upper bounds do not disappear through one shift; inclusion–exclusion can remove allocations exceeding them.

</details>

## 3 · Sampling without replacement is not Binomial

Among N objects, K belong to a target class. Draw a uniformly chosen subset of size n. The target count X is Hypergeometric:

$$
P(X=k)=\frac{\binom Kk\binom{N-K}{n-k}}{\binom Nn}.
$$

Its support is $\max(0,n-N+K)\le k\le\min(n,K)$. Both numerator and denominator count unordered subsets.

Write $p=K/N$. Indicators for draw positions give $E[X]=np$. Success on one draw lowers the chance of success on the next, producing negative covariance.

<details markdown="1">
<summary>Derive the finite-population correction</summary>

Two distinct positions both succeed with probability $K(K-1)/(N(N-1))$, hence:

$$
\operatorname{Cov}(I_i,I_j)=-\frac{p(1-p)}{N-1},\qquad
\operatorname{Var}(X)=np(1-p)\frac{N-n}{N-1}.
$$

Assume $N>1$. At n=N the variance is zero, as it must be. A Binomial approximation becomes plausible when the sampled fraction is small.

</details>

## 4 · Birthdays and occupancy: an event is not a count

Let m independent birthdays be uniform over d days. For m≤d:

$$
P(\text{no collision})=\prod_{j=0}^{m-1}\left(1-\frac jd\right).
$$

For m>d it is zero. When j/d is small, $\log(1-u)\approx-u$ gives the approximation $\exp[-m(m-1)/(2d)]$. Accumulated higher-order terms limit its accuracy.

Think instead of m balls entering d boxes. An indicator for each nonempty box gives:

$$
E[\text{occupied boxes}]=d\left[1-\left(1-\frac1d\right)^m\right].
$$

This differs from the expected number of colliding pairs, $\binom m2/d$. Three balls in one box contribute one occupied box but three pairs.

## 5 · Inclusion–exclusion and derangements

For a uniform permutation of n elements, let $A_i$ mean position i remains fixed. Fixing a specified set of k positions leaves $(n-k)!$ permutations, so:

$$
P(\text{no fixed points})=\sum_{k=0}^n\frac{(-1)^k}{k!}.
$$

<details markdown="1">
<summary>Why does the kth term become 1/k!?</summary>

There are $\binom nk$ sets of k positions, and each intersection contains $(n-k)!$ permutations. Divide by n! to obtain $1/k!$. The answer tends to $e^{-1}$; it is not exactly that for finite n.

</details>

## 6 · Collecting every type: stages have different difficulty

Each independent draw is uniform over n types. After collecting k distinct types, a new one arrives with probability $(n-k)/n$. That stage has mean duration $n/(n-k)$. Add the stages:

$$
E[T]=n\sum_{j=1}^n\frac1j=nH_n.
$$

Linearity is enough; independence between stages is not needed for this expectation. With unequal type probabilities, the number already collected generally does not describe the state: which types are missing matters.

Check n=1: one draw. The final missing type alone takes n draws on average. See [expectation proofs](expectation-proofs.en.md) for the underlying technique.

Continue: [Deriving common distributions](distribution-toolkit.en.md). Background: [MIT Mathematics for Computer Science, counting chapters](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/pages/readings/).
