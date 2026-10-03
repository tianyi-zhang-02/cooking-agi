# Conditional expectation: fix one source of randomness

[中文](conditioning-proofs.md) · **English**

> Reading time: ~8 min · Prerequisites: [Conditioning](conditional.en.md), [Expectation](expectation-proofs.en.md) · Last reviewed: 2026-10

Several layers of randomness do not have to be handled at once. Fix one layer, solve the inner problem, then average over the outer layer. Total probability, total expectation, and total variance all use this move.

## 1 · Total probability and Bayes start with a partition

Let $A_1,A_2,\ldots$ be a finite or countable partition: disjoint events whose union is $\Omega$. Write conditional probabilities only for groups with $P(A_i)>0$; zero-probability groups contribute no intersection mass.

$$
P(B)=\sum_iP(B\cap A_i)=\sum_iP(B\mid A_i)P(A_i).
$$

The first equality is additivity; the second is the definition of conditioning. If $P(B)>0$, express the same intersection in the other direction:

$$
P(A_j\mid B)=\frac{P(B\mid A_j)P(A_j)}{\sum_iP(B\mid A_i)P(A_i)}.
$$

Bayes introduces no mysterious extra rule. It reverses which information is given.

## 2 · Different information means a different question

Draw 2 cards without replacement from a standard 52-card deck containing 4 aces. These two versions of “both are aces” are not interchangeable.

| Information given | Conditional probability | What the denominator counts |
| --- | --- | --- |
| The first card is an ace | $3/51=1/17$ | Remaining cards |
| At least one card is an ace | $\binom42/(\binom{52}2-\binom{48}2)=1/33$ | Unordered pairs containing an ace |

The second denominator is $198$ and its numerator is $6$. Neither calculation is cleverer: they condition on different information. If someone chooses a card to reveal, their selection rule must also enter the model.

## 3 · Total expectation: average within groups, then across them

First take discrete $Y$. Set $m(y)=\mathbb E[X\mid Y=y]$. The value $m(y)$ is a number, but $m(Y)=\mathbb E[X\mid Y]$ is still a random variable because the group is not yet known.

**Theorem.** If $\mathbb E|X|<\infty$,

$$
\mathbb E[\mathbb E[X\mid Y]]=\mathbb E[X].
$$

<details markdown="1">
<summary>Expand the discrete proof</summary>

Sum over positive-probability values of $y$:

$$
\begin{aligned}
\sum_y\mathbb E[X\mid Y=y]P(Y=y)
&=\sum_y\sum_xxP(X=x\mid Y=y)P(Y=y)\\
&=\sum_{x,y}xP(X=x,Y=y)\\
&=\mathbb E[X].
\end{aligned}
$$

Independence of $X,Y$ is not required. With a joint density, replace sums by integrals. The general statement follows from the definition of conditional expectation; its measure-theoretic construction is beyond this introduction.

</details>

**Example.** Choose one of two coins with equal probability, then flip that coin 10 times. Their head probabilities are $0.2$ and $0.8$. For the head count $S$, the conditional means are 2 and 8, so the overall mean is $(2+8)/2=5$.

The flips are independent **conditional on the chosen coin**, not unconditionally: observing many early heads changes your belief about which coin was selected.

## 4 · Total variance: where the variation comes from

**Theorem.** If $\mathbb E[X^2]<\infty$,

$$
\operatorname{Var}(X)=\mathbb E[\operatorname{Var}(X\mid Y)]+\operatorname{Var}(\mathbb E[X\mid Y]).
$$

Overall variation splits into average within-group variation and variation between group means.

<details markdown="1">
<summary>Proof: why does the cross term vanish?</summary>

Set $m(Y)=\mathbb E[X\mid Y]$ and $\mu=\mathbb E[X]$. Decompose the deviation:

$$
X-\mu=(X-m(Y))+(m(Y)-\mu).
$$

After squaring, the cross term vanishes because

$$
\mathbb E[(X-m(Y))(m(Y)-\mu)]
=\mathbb E[(m(Y)-\mu)\mathbb E[X-m(Y)\mid Y]]=0.
$$

Given $Y$, the factor $m(Y)-\mu$ is fixed, and the residual has conditional mean 0. The two squared terms give the two terms in total variance.

</details>

For the coin example, each conditional variance is $10\times0.2\times0.8=1.6$. The means 2 and 8 have variance 9 around their average 5. Total variance is therefore **10.6**, not the 2.5 of Binomial($10,0.5$). A mixture cannot generally be replaced by its average success probability.

## 5 · A tempting extra step that fails

Total expectation permits averaging in groups. It does not let arbitrary functions move through expectation. For example,

$$
\mathbb E[(\mathbb E[X\mid Y])^2]\ne(\mathbb E[X])^2
$$

in general. Their difference is exactly $\operatorname{Var}(\mathbb E[X\mid Y])$. Nor can a condition be dropped merely because variables are independent given $Y$.

<details markdown="1">
<summary>Exercise: if Y completely determines X, which variance term remains?</summary>

Then $\mathbb E[X\mid Y]=X$, and conditional variance is 0. All variation lies between values of $Y$, giving $\operatorname{Var}(X)=0+\operatorname{Var}(X)$. Conversely, if $X$ and $Y$ are independent, the conditional mean is constant and the second term is 0.

</details>

## Next

[Inequalities: useful bounds without an exact answer](inequalities.en.md). Further reference: [MIT 6.041: Conditional expectation and total variance](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/resources/mit6_041f10_l12/).
