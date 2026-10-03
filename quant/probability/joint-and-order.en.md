# Joint distributions: draw the support before integrating

[中文](joint-and-order.md) · **English** · [Review map](../README.en.md)

> Reading time: ~10 min · Prerequisites: densities, conditioning, basic integration · Last reviewed: 2026-10

Knowing X and Y separately does not usually determine their joint behavior. Two arrival times can each be uniform while being independent—or exactly equal.

## 1 · Joint, marginal, and conditional densities

For joint density f(x,y), integrating out y gives the marginal. Fixing x and normalizing gives a conditional density:

$$
f_X(x)=\int f(x,y)\,dy,\qquad
f_{Y\mid X=x}(y)=\frac{f(x,y)}{f_X(x)},\quad f_X(x)>0.
$$

This is a density-based conditional distribution, not elementary conditioning by dividing by P(X=x)=0. Independence is equivalent to factorization into the marginal densities almost everywhere.

Take $f(x,y)=2$ on $0<x<y<1$, zero elsewhere. The triangle has area 1/2, so the density integrates to one. For fixed x, integrate y from x to 1: $f_X(x)=2(1-x)$.

<details markdown="1">
<summary>What is the distribution of Y given X=x?</summary>

Its density is $1/(1-x)$ on x<y<1: Uniform(x,1). Therefore $E[Y\mid X=x]=(1+x)/2$. The conditional distribution varies with x; constant density over its support does not imply independence.

</details>

## 2 · Adding variables: where convolution comes from

For independent continuous variables and Z=X+Y:

$$
f_Z(z)=\int_{-\infty}^{\infty}f_X(x)f_Y(z-x)\,dx.
$$

For each x, the other variable must lie near z−x. Integrate over valid x. For two independent Uniform(0,1) variables, the valid interval first grows, then shrinks:

$$
f_Z(z)=
\begin{cases}
z,&0<z<1,\\
2-z,&1\le z<2,\\
0,&\text{otherwise}.
\end{cases}
$$

Check that the area is one and $E[Z]=1$. Without independence, use $f_{X,Y}(x,z-x)$ rather than the product of marginals.

## 3 · Changes of variables need the Jacobian

Let X,Y be independent Exp(λ), S=X+Y and R=X/(X+Y). The inverse is X=RS, Y=(1−R)S, with S>0 and 0<R<1. The absolute Jacobian is S:

$$
f_{S,R}(s,r)=\lambda^2s e^{-\lambda s}\,\mathbf1_{\{s>0,\ 0<r<1\}}.
$$

This factors into Gamma(2,λ) and Uniform(0,1) densities, proving independence of S and R. Omitting S would break normalization.

For a many-to-one transformation, sum over inverse branches. Z=X² needs both square roots unless X is restricted to the positive half-line.

## 4 · The kth smallest: count observations to the left

For n iid observations with density f and CDF F, $X_{(k)}\le x$ means at least k observations are ≤x:

$$
P(X_{(k)}\le x)=\sum_{j=k}^n\binom njF(x)^j[1-F(x)]^{n-j}.
$$

<details markdown="1">
<summary>Turn the CDF into a density</summary>

Place one observation in a narrow interval, k−1 to its left, and n−k to its right:

$$
f_{X_{(k)}}(x)=\frac{n!}{(k-1)!(n-k)!}
F(x)^{k-1}[1-F(x)]^{n-k}f(x).
$$

A formal derivation differentiates the finite sum above, whose terms cancel. The iid assumption is essential.

</details>

For Uniform(0,1), this is Beta(k,n+1−k), giving $E[X_{(k)}]=k/(n+1)$. The expected maximum is n/(n+1), and the expected minimum is 1/(n+1). A Beta integral verifies these; an intuition about equal spacing is not a proof.

## 5 · Jointly Normal: when uncorrelated implies independent

For jointly Normal X,Y with nonzero standard deviations and |ρ|<1:

$$
Y\mid X=x\sim N\left(
\mu_Y+\rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X),
\ \sigma_Y^2(1-\rho^2)\right).
$$

Complete the square in y in the joint density. Alternatively represent $Y=\mu_Y+\rho(\sigma_Y/\sigma_X)(X-\mu_X)+\sigma_Y\sqrt{1-\rho^2}Z$, where Z is independent of X.

At ρ=0, the conditional distribution does not depend on x, giving independence. **Normal marginals alone do not establish joint Normality.** Handle the degenerate |ρ|=1 case separately.

## 6 · An often-missing sampling model

“A random chord of a circle” does not specify a distribution. Uniform endpoints and uniform midpoints produce different models. Define the sampling procedure before claiming equally likely outcomes.

<details markdown="1">
<summary>Exercise: independent Uniform(0,1) variables. Find E[|X−Y|].</summary>

Integrate over y<x and double by symmetry: $2\int_0^1\int_0^x(x-y)\,dy\,dx=1/3$. For P(X+Y≤1/2), use the right triangle with side length 1/2: its area is 1/8. Area ratios work because the joint density is uniform on the unit square.

</details>

Continue: [Statistical inference](../methods/statistics.en.md). Reference: [MIT 18.05 probability and statistics materials](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/pages/classes-reading-and-in-class-materials/).
