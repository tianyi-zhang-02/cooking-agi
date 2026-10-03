# Numerics and coding: a correct formula still needs a stable implementation

[中文](numerical-and-coding.md) · **English** · [Review map](../README.en.md)

> Reading time: ~11 min · Prerequisites: calculus, expectation, variance, Python · Last reviewed: 2026-10

Where does numerical error come from? How do you detect nonconvergence? Did a simulation agree only because of a lucky random seed?

## 1 · Bisection versus Newton

For continuous f with opposite signs at [a,b], bisection keeps a sign-changing half-interval. After k steps its width is $(b-a)/2^k$, giving a transparent absolute error bound.

Newton finds a root of the local tangent:

$$
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}.
$$

Near a simple root, with sufficient smoothness and nonzero derivative, convergence can be quadratic. Far away or near a zero derivative it may fail. For $f(x)=x^3-2x+2$, starting at zero cycles between zero and one.

A practical hybrid maintains a bracket and falls back to bisection when a Newton step is unreliable or leaves it. Monitor bracket width, residual, and iteration limits.

## 2 · Differencing and integration: smaller h is not always better

The central difference $(f(x+h)-f(x-h))/(2h)$ has O(h²) truncation error with bounded local third derivative. Floating-point cancellation contributes roughly O(ε/h), so excessively small h can hurt.

Composite trapezoidal integration has O(h²) error with bounded second derivative; Simpson has O(h⁴) error with bounded fourth derivative and an even number of subintervals. Nonsmooth points and singular endpoints can invalidate these rates.

## 3 · Monte Carlo: uncertainty comes from variance

For iid samples estimating μ=E[g(X)]:

$$
\hat\mu_N=\frac1N\sum_{i=1}^Ng(X_i),\qquad
\operatorname{Var}(\hat\mu_N)=\frac{\operatorname{Var}(g(X))}{N}.
$$

With finite variance and an applicable CLT, estimate standard error by sample standard deviation/√N. Tenfold smaller error usually requires about a hundredfold larger sample. Correlated samples need covariance terms. Fixed seeds provide reproducibility, not correctness.

Importance sampling draws from q and weights by $g(x)p(x)/q(x)$. It requires support coverage wherever p|g| is nonzero and appropriate integrability. Unbiasedness does not guarantee low variance: weights may explode.

## 4 · Variance reduction changes the estimator

If E[Y]=ν is known, define $Z=X-c(Y-\nu)$. The mean remains E[X]; variance is quadratic in c:

$$
c^*=\frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(Y)}.
$$

Assume finite second moments and Var(Y)>0. Fixed c gives immediate unbiasedness. Estimating c on the same data needs extra care; an independent pilot or cross-fitting avoids simply assuming the fixed-c argument still applies.

<details markdown="1">
<summary>Example: estimate ∫₀¹x²dx using x as a control variate</summary>

For U~Uniform(0,1), X=U² and Y=U, Var(X)=4/45, Var(Y)=1/12, and Cov(X,Y)=1/12, so c*=1. Then Z=U²−U+1/2 has variance 1/180, one-sixteenth of the original variance, and mean 1/3.

The improvement comes from correlation, not merely introducing another variable.

</details>

Antithetic sampling pairs U with 1−U. It helps only with suitable negative correlation. For X=U², the paired mean also has variance 1/180, but uses two function evaluations; compare under equal computational cost.

## 5 · Online variance without subtracting huge numbers

Welford's update stores count, mean, and M2. For a new observation x:

$$
\delta=x-\mu_n,\quad
\mu_{n+1}=\mu_n+\frac{\delta}{n+1},\quad
M2_{n+1}=M2_n+\delta(x-\mu_{n+1}).
$$

For n≥2, sample variance is M2/(n−1). This is more robust to cancellation than mean-square minus squared-mean, but not exact arithmetic. A sliding window additionally removes old observations; all-history online statistics are not window statistics.

## 6 · State the data structure and its cost

| Need | Common structure | Complexity and limits |
| --- | --- | --- |
| Streaming median | Max-heap + min-heap | O(log n) insertion, O(1) query, O(n) storage |
| Online top-k | Size-k min-heap | O(log k) per item, O(k) storage |
| One uniform sample from a stream | Reservoir sampling | O(1) per item and storage |
| Fixed-width sliding maximum | Monotonic deque | O(n) total, O(w) storage |
| Comparison sorting | Merge / heap / quicksort | First two worst-case O(n log n); quicksort worst-case O(n²) |

<details markdown="1">
<summary>Prove reservoir sampling is uniform</summary>

At step t replace the retained item with probability 1/t. An old item had probability 1/(t−1) of being retained and survives with probability (t−1)/t, giving 1/t. The new item also has probability 1/t. Induction completes the proof. Uniform integer generation and endpoint conventions matter in code.

</details>

Continue coding practice in [algorithm methods](../../interview/leetcode.en.md). A C++ interview also needs object lifetimes, references/pointers, RAII, and iterator invalidation. Python algorithms do not replace those language semantics; this collection does not mark a C++ track complete.

## 7 · Run small checks

The [Python examples](../code/review_checks.py) include bisection, Welford, two-heap medians, one-period replication, and exact combinatorial checks. Start with hand-computed answers and invariants.

~~~bash
python3 quant/code/review_checks.py
~~~

Finite enumeration can catch mistakes but not prove general theorems. One simulation landing near a target does not establish correctness either.

Reference: [MIT 18.S096 mathematics and finance materials](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/pages/lecture-notes/). Numerical edge cases and executable checks are additional review material here.
