# Mixed review: remember the assumptions, not just the formula

[中文](review.md) · **English** · [Review map](README.en.md)

> Reading time: ~10 min · Work through it on paper or aloud · Last reviewed: 2026-10

Pick a few questions rather than rereading everything. Return to a chapter when you can name the step that is unclear. Explain first; speed can come later.

## 1 · A quick assumptions table

| Tool | Check first | Common confusion |
| --- | --- | --- |
| Equally likely counting | Finite outcomes really are equally likely | Random does not mean uniform |
| Linearity | Absolute integrability, or a valid nonnegative-sum version | Independence is unnecessary |
| Adding variances | Second moments and covariance terms | Identical distributions are insufficient |
| Bayes | Positive observation probability and explicit observation mechanism | Base rates and sampling rules matter |
| Multiplying MGFs | Independence and finiteness on the relevant domain | Having all moments is insufficient |
| Jensen | Convexity and suitable integrability | Convex versus concave changes direction |
| WLLN / CLT | The chosen version's independence and moment conditions | CLT does not promise n=30 suffices |
| Bounded stopping | Martingale, stopping time, bounded horizon | Eventual stopping alone is insufficient |
| Wald | Integrable iid increments, stopping time, E[T]<∞ | No future information in T |
| t interval | Exact version: iid Normal, nonzero variance | General samples need approximation |
| Quadratic Newton convergence | Near a smooth simple root, nonzero derivative | No global convergence guarantee |
| Risk-neutral valuation | Model, no-arbitrage, admissible replication conditions | q is not a real-world up probability |

These are the versions used here, not universally weakest assumptions. Return to the relevant chapter for details.

## 2 · Four changes-of-assumption questions

<details markdown="1">
<summary>1. Two Bernoulli trials each have success probability 1/2. Must at least one succeed with probability 3/4?</summary>

No. Independence gives 3/4; identical trials give 1/2; complementary trials give one. Marginals do not determine the joint distribution. See [conditioning and independence](probability/conditional.en.md).

</details>

<details markdown="1">
<summary>2. N=20, K=5, draw n=4 without replacement. Mean and variance of the target count?</summary>

Mean one, variance $4(1/4)(3/4)(16/19)=12/19$. With independent uniform replacement, variance becomes 3/4. Equal means do not imply equal fluctuation. See [counting](probability/counting.en.md).

</details>

<details markdown="1">
<summary>3. Can you stop a fair walk at its first +1 and immediately claim E[S_T]=0?</summary>

No. T is almost surely finite but has infinite expectation; the required limiting control fails. S_T is identically one, disproving an unrestricted stopping claim. See [martingales](processes/martingales.en.md).

</details>

<details markdown="1">
<summary>4. Why might random splits of time-indexed data look unusually good?</summary>

Nearby observations can share information, and future features or label construction can leak into training. A random split may evaluate a different task from future generalization. Specify information availability before splitting. Distinct rows alone do not establish absence of leakage. See [statistics](methods/statistics.en.md).

</details>

## 3 · Four short derivations

<details markdown="1">
<summary>5. Independent exponential waits with rates a,b: which arrives first?</summary>

$P(X<Y)=\int_0^\infty ae^{-at}e^{-bt}dt=a/(a+b)$, and the minimum is Exp(a+b). Require positive rates and independence. As a becomes large, X should almost always win. See [distributions](probability/distribution-toolkit.en.md).

</details>

<details markdown="1">
<summary>6. Why can a covariance matrix have no negative eigenvalues?</summary>

For every v, $v^\top\Sigma v=\operatorname{Var}(v^\top X)\ge0$. A negative-eigenvalue eigenvector would contradict this. Finite second moments are required. See [linear algebra](methods/linear-algebra.en.md).

</details>

<details markdown="1">
<summary>7. Why is the variance of waiting until success not generally equal to its mean?</summary>

Geometric on 1,2,… has mean 1/p and variance (1−p)/p². Equal mean and variance is a Poisson property. Derive the second moment by first-step recursion or differentiating a geometric series. At p=1 the wait is deterministically one with variance zero. See [foundational exercise 4](probability/practice.en.md).

</details>

<details markdown="1">
<summary>8. Stock 100→120/80, R=1.05, call payoff 20/0: why does the actual up probability not enter?</summary>

Half a share and borrowing 800/21 replicate both payoffs. Price is 250/21, determined by replication rather than forecasting. Changing the actual up probability does not change that idealized replication cost. See [no-arbitrage](finance/README.en.md).

</details>

## 4 · Four connections across chapters

<details markdown="1">
<summary>9. Why is the expected maximum of n Uniform(0,1) samples n/(n+1)?</summary>

The maximum is ≤x exactly when every observation is ≤x, giving CDF xⁿ and density nxⁿ⁻¹. Integrate $\int_0^1x\cdot nx^{n-1}dx=n/(n+1)$. At n=1 it is 1/2; it tends to one as n grows. See [order statistics](probability/joint-and-order.en.md).

</details>

<details markdown="1">
<summary>10. A Monte Carlo estimate varies a lot. What can you change?</summary>

Check the estimand, implementation, tails, and finite variance first. Then increase samples or exploit structure with controls, antithetics, stratification, or importance sampling. Compare error at equal computational budgets, not a favorable single seed. See [numerics](methods/numerical-and-coding.en.md).

</details>

<details markdown="1">
<summary>11. Why does d(W²) contain dt beyond 2W dW?</summary>

Squared increments accumulate rather than vanish: along equal partitions their sum converges in L² to elapsed time. The second-order Taylor term survives. This motivates and checks examples without replacing the general Itô proof. See [Itô](processes/brownian-ito.en.md).

</details>

<details markdown="1">
<summary>12. With at most three die rolls, why initially accept only 5 or 6?</summary>

One remaining roll is worth 3.5; two are worth 17/4=4.25. On the first roll compare x with 4.25. Accept 5 or 6, giving optimal value 14/3. See [dynamic programming](processes/dynamic-programming.en.md).

</details>

## 5 · Explain a solution in one minute

A useful order is:

“I assume… and define the target as…. Direct calculation is awkward, so I use… to decompose it. This step needs…. The result is…. At the boundary… it reduces to…, which is consistent. Removing… would invalidate this step.”

Replace “I remember the answer” with “I know which step simplifies the problem.” Once that works, change assumptions rather than reread the same solution.

These are general teaching examples and original variations, not undisclosed interview questions. Return to the [coverage map](README.en.md).
