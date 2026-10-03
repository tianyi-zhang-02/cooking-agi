# Probability exercises: choose the first step yourself

[中文](practice.md) · **English**

> Reading time: choose exercises individually · Prerequisite: [Review guide](study-guide.en.md) · Last reviewed: 2026-10

These exercises are not about difficulty for its own sake. Choose the right object, state the assumptions, and complete the argument. Hints and solutions expand separately; start with whichever topic feels least familiar.

## 1 · Do complements preserve independence?

Given independent events $A,B$, prove that $A^c$ and $B$ are independent, then that $A^c$ and $B^c$ are independent. Allow zero-probability events.

<details markdown="1">
<summary>Hint</summary>

Avoid dividing by an event probability. Write an intersection as a set difference and use the product definition.

</details>

<details markdown="1">
<summary>Solution</summary>

$P(A^c\cap B)=P(B)-P(A\cap B)=(1-P(A))P(B)=P(A^c)P(B)$. Thus $A^c,B$ are independent. Complement the second event in the same way to get $P(A^c\cap B^c)=P(A^c)P(B^c)$. No division is needed, so zero-probability cases are covered.

</details>

## 2 · What fails in “pairwise independent, so multiply all three”?

Flip two independent fair coins. Let $A$ be first-head, $B$ second-head, and $C$ matching outcomes. Check pairwise independence and determine whether they are mutually independent.

<details markdown="1">
<summary>Hint</summary>

Enumerate HH, HT, TH, TT and compare pairwise intersections with the triple intersection.

</details>

<details markdown="1">
<summary>Solution</summary>

Each event has probability $1/2$, and each pairwise intersection contains only HH, with probability $1/4$. The triple intersection also contains only HH, so its probability is $1/4$, not $(1/2)^3=1/8$. Mutual independence requires factorization for every finite subset, not just each pair.

</details>

## 3 · How many same-birthday pairs on average?

Assume $n$ independent birthdays, uniformly distributed over 365 days, ignoring leap days. Find the expected number of matching pairs. Is this the probability of at least one match?

<details markdown="1">
<summary>Hint</summary>

Assign an indicator to each pair rather than finding the full collision distribution.

</details>

<details markdown="1">
<summary>Solution</summary>

There are $\binom n2$ pairs, each matching with probability $1/365$, giving expectation $\binom n2/365$. This is an expected count, which can exceed 1, not a probability. If $N$ counts pairs, $\mathbf1_{\{N\ge1\}}\le N$, so the chance of at least one match is at most $\min(1,\mathbb E N)$.

</details>

## 4 · Derive geometric variance without recalling the formula

Independent trials succeed with probability $p\in(0,1]$. Let $T$ include the first successful trial. Given $\mathbb E[T]=1/p$, derive $\operatorname{Var}(T)$.

<details markdown="1">
<summary>Hint</summary>

Write $T=1+IT'$, where $I$ indicates first-trial failure and $T'$ is an independent remaining wait. Square and take expectations.

</details>

<details markdown="1">
<summary>Solution</summary>

Set $q=1-p$, $m=1/p$, and $s=\mathbb E[T^2]$. Since $I^2=I$, $s=1+2qm+qs$. Thus $s=(2-p)/p^2$ and variance is $s-m^2=q/p^2$.

Finiteness is justified by the geometric PMF: $\sum_{k\ge1}k^2pq^{k-1}$ converges. For $p=1$, $T=1$; otherwise use the ratio test. Rearranging the second-moment equation is therefore legitimate.

</details>

## 5 · Two layers of randomness are not an average coin

Choose with equal probability a coin having head probability $0.2$ or $0.8$, then flip that same coin independently 10 times. Find the mean and variance of the head count.

<details markdown="1">
<summary>Hint</summary>

Condition on the coin type. Do not discard the second term in total variance.

</details>

<details markdown="1">
<summary>Solution</summary>

Conditional means are 2 and 8; both conditional variances are 1.6. The overall mean is 5, and total variance is $1.6+((2-5)^2+(8-5)^2)/2=10.6$. This is not Binomial($10,0.5$). It revisits the [conditional expectation example](conditioning-proofs.en.md); try completing it without looking back.

</details>

## 6 · Maximum of uniform variables

Let $U_1,\ldots,U_n$ be independent Uniform$(0,1)$ variables. Find the CDF and mean of $M=\max_iU_i$.

<details markdown="1">
<summary>Hint</summary>

The maximum is at most x exactly when every value is at most x. Use the nonnegative tail integral for expectation.

</details>

<details markdown="1">
<summary>Solution</summary>

For $0\le x\le1$, $P(M\le x)=\prod_iP(U_i\le x)=x^n$. Outside the interval the CDF is 0 or 1. Hence

$$
\mathbb E[M]=\int_0^1(1-x^n)\,dx=\frac n{n+1}.
$$

Independence enters at the product. If every $U_i$ copies one variable, the maximum stays uniform and has mean $1/2$.

</details>

## 7 · Waiting patterns: H, HH, and HTH

For independent fair flips, find the mean first waiting times for H, HH, and HTH. Do not simply invert window probabilities.

<details markdown="1">
<summary>Hint</summary>

H is geometric. HH needs unfinished states 0 and H; HTH needs 0, H, and HT. Which suffix survives a failed attempt?

</details>

<details markdown="1">
<summary>Solution</summary>

H takes 2 flips on average. For HH, $E_0=1+(E_0+E_H)/2$ and $E_H=1+E_0/2$, giving 6. The HTH equations in [Markov chains](markov-chains.en.md) give 10. Reciprocal window probabilities are 2, 4, and 8; the last two fail. Nonoverlapping blocks justify finite expectations.

</details>

## 8 · Does a stable sample mean imply no bias?

Someone says, “My sample mean converged, so it must equal the true population mean I want.” What is missing?

<details markdown="1">
<summary>Hint</summary>

The LLN approaches the mean of which distribution? Are stability and lack of bias equivalent?

</details>

<details markdown="1">
<summary>Solution</summary>

Even iid samples converge toward the mean of the sampling distribution. Sampling only one subgroup does not remove selection bias as sample size grows. Define the target population and the sampling process before discussing error. Correlated data can be analyzed, but not by applying the iid proof unchanged.

</details>

## Use code to catch mistakes, not to replace proofs

The [exact checker](code/proof_checks.py) uses only the Python standard library. It enumerates card pairs and coins, verifies total expectation and variance, and builds pattern-waiting state equations solved with fractions. It uses neither random simulation nor external services.

Run from the repository root:

~~~bash
python3 quant/probability/code/proof_checks.py
~~~

Try head probability $1/3$ or target HHH. **Passing finite examples does not prove a general theorem.** The checks help detect missing ties, incorrect transitions, and numerical mistakes.

Return to [Proof techniques](proof-toolbox.en.md) or the [Review guide](study-guide.en.md) to revisit one step that gave you trouble.
