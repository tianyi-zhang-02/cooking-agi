# Poisson processes: counts and waiting times

[中文](README.md) · **English** · [Review map](../README.en.md)

> Reading time: ~8 min · Prerequisites: Poisson, Exponential, conditioning · Last reviewed: 2026-10

“How many arrivals in an hour?” and “How long until the next arrival?” are two views of one process. Connect them before applying the model.

## 1 · More than one Poisson marginal

A homogeneous Poisson process N(t) of rate λ>0 starts at N(0)=0, has independent increments over disjoint intervals, and a length-h increment is Poisson(λh). Thus:

$$
P(N(t)=k)=e^{-\lambda t}\frac{(\lambda t)^k}{k!}.
$$

λ has units of events per unit time. λt is a dimensionless expected count. Per-hour and per-minute rates are not interchangeable.

Knowing the marginal distribution of every N(t) does not itself establish independent increments.

## 2 · First arrival: calculate the probability of no arrivals

T₁>t exactly when no arrival has occurred by t:

$$
P(T_1>t)=P(N(t)=0)=e^{-\lambda t}.
$$

Thus T₁~Exp(λ). Successive interarrival times are iid Exp(λ), and the kth arrival Sₖ is Gamma(k,λ). Either exponential waiting or independent Poisson counts can construct the process.

Since $P(S_k\le t)=P(N(t)\ge k)$, differentiating the Poisson tail gives:

$$
f_{S_k}(t)=\frac{\lambda^k t^{k-1}e^{-\lambda t}}{(k-1)!},\qquad t>0.
$$

## 3 · Given n arrivals, when did they occur?

Conditional on N(T)=n, arrival times have the distribution of n sorted independent Uniform(0,T) samples. **The ordered arrival times themselves are not independent.**

<details markdown="1">
<summary>Why is the conditional joint density constant?</summary>

On $0<t_1<\cdots<t_n<T$, the density for n arrivals followed by no arrival until T is $\lambda^ne^{-\lambda T}$. Divide by $P(N(T)=n)=e^{-\lambda T}(\lambda T)^n/n!$ to obtain $n!/T^n$, the uniform order-statistic density. Hence $E[S_k\mid N(T)=n]=kT/(n+1)$.

</details>

## 4 · Superposition and thinning

Superposing independent processes of rates λ₁ and λ₂ produces rate λ₁+λ₂. Count MGFs multiply and increment independence remains.

Independently label each arrival A with probability p and B otherwise. The two labeled streams are independent Poisson processes of rates λp and λ(1−p). Their joint count PGF is:

$$
E[s^{N_A}u^{N_B}]
=\exp\{\lambda t[ps+(1-p)u-1]\}
=\exp[\lambda tp(s-1)]\exp[\lambda t(1-p)(u-1)].
$$

Factorization proves independence. Labels depending on congestion or previous events do not satisfy the assumptions automatically.

## 5 · When to change the model

| Observation | What to check |
| --- | --- |
| Rate changes over time | Nonhomogeneous Poisson uses integrated intensity |
| Events trigger later events | Independent increments may fail |
| Variance exceeds mean substantially | Random rates or clustering; first check time aggregation |
| Nearly regular arrival intervals | Exponential waiting may be unsuitable |
| General iid interarrival times | Renewal process, usually without memorylessness |

For a nonhomogeneous process, $\Lambda(t)=\int_0^t\lambda(s)ds$ and no-arrival probability is $e^{-\Lambda(t)}$. Waiting depends on the current time rather than a constant-rate memoryless law.

## 6 · One example, three quantities

At rate 3 per hour over 2 hours, expected count is 6, zero-count probability is e⁻⁶, and mean first wait is 1/3 hour. Given exactly 5 arrivals over those 2 hours, the expected second arrival is 2×2/6=2/3 hour.

The conditional mean happens here to equal the unconditional mean of the second arrival. That numerical coincidence does not make their distributions identical or justify ignoring the conditioning.

Continue: [Random walks and martingales](martingales.en.md). Reference: [MIT Poisson process notes](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/resources/mit6_262s11_chap02/).
