# Calculus: a zero derivative is only the beginning

[中文](calculus-optimization.md) · **English** · [Review map](../README.en.md)

> Reading time: ~10 min · Prerequisites: basic differentiation and integration · Last reviewed: 2026-10

The goal is to explain why a derivative answers the question, not merely calculate it. Start with assumptions, then connect theorems to optimization.

## 1 · Rolle and the mean value theorem

Suppose f is continuous on [a,b] and differentiable on (a,b). If f(a)=f(b), Rolle's theorem gives an interior c with f′(c)=0.

<details markdown="1">
<summary>Why must c exist?</summary>

A continuous function on a compact interval attains its extrema. Unless constant, at least one extremum lies inside because the endpoints agree. At a differentiable interior extremum, left and right difference quotients force derivative zero. For a constant function any interior point works.

</details>

Subtract the line joining the endpoints and apply Rolle to obtain:

$$
f'(c)=\frac{f(b)-f(a)}{b-a}.
$$

This proves that a nonnegative derivative throughout an interval implies a nondecreasing function. Watch assumptions: $|x|$ is not differentiable at zero.

## 2 · Why differentiation can undo integration

For continuous f, define $F(x)=\int_a^xf(t)\,dt$. The difference quotient is an average over a shrinking interval, tending to f(x). Thus F′=f: one version of the fundamental theorem of calculus.

Substitution reverses the chain rule; integration by parts reverses the product rule. For $\int_0^\infty xe^{-\lambda x}dx=1/\lambda^2$, λ>0, first integrate over [0,R], then take R→∞. An infinite endpoint needs a limit.

## 3 · Taylor expansions need remainder control

With sufficient smoothness, the second-order expansion with Lagrange remainder is:

$$
f(a+h)=f(a)+f'(a)h+\frac12f''(a)h^2+\frac16f'''(\xi)h^3.
$$

Here ξ lies between a and a+h. If |f'''|≤M, the error is at most M|h|³/6. **The order alone does not determine accuracy: derivative size and step size matter.**

For $\log(1+h)$ with |h|≤1/2, |f'''|≤16. The approximation $h-h^2/2$ therefore has error at most $8|h|^3/3$. Conservative, but guaranteed.

## 4 · A critical point need not minimize anything

For a global optimum on a closed interval, check stationary points, nondifferentiable points, and endpoints. On an unbounded domain, also inspect limiting behavior and whether an optimum is attained.

$f(x)=x^3$ has derivative zero at zero without an extremum. In multiple dimensions, $f(x,y)=x^2-y^2$ has a stationary saddle at the origin.

A twice-differentiable interior local minimum must have positive-semidefinite Hessian. Positive definiteness suffices for a strict local minimum. Semidefiniteness alone does not: even $x^3$ has second derivative zero at the origin.

## 5 · Convexity changes the problem

A convex function lies below its chords. A differentiable convex function satisfies:

$$
f(y)\ge f(x)+\nabla f(x)^\top(y-x).
$$

A zero gradient therefore gives a global minimum. Strict convexity gives at most one minimizer, not its existence: $e^x$ never attains its infimum over the real line.

On a convex open domain, a twice-continuously-differentiable function is convex exactly when its Hessian is everywhere positive semidefinite. Restricting to line segments reduces the proof to one dimension.

## 6 · Constraints and Lagrange multipliers

Minimize $x^2+y^2$ subject to x+y=1. Substituting y=1−x gives x=y=1/2 and value 1/2.

The Lagrangian $L=x^2+y^2+\lambda(x+y-1)$ gives equations 2x+λ=0, 2y+λ=0 and the constraint, with the same solution.

<details markdown="1">
<summary>Why parallel gradients? What changes for inequalities?</summary>

For a smooth equality constraint with nonzero constraint gradient, all feasible tangent directions must have zero directional derivative. The objective gradient is therefore normal to the constraint.

For differentiable convex problems under suitable conditions such as Slater's condition, KKT combines stationarity, primal feasibility, dual feasibility, and complementary slackness. For $g_i(x)\le0$, multipliers obey $\lambda_i\ge0$ and $\lambda_ig_i(x)=0$. KKT is not a blanket certificate of global optimality for nonconvex problems.

</details>

## 7 · Exchanging limits and derivatives needs justification

Pointwise convergence does not imply convergence of derivatives: $f_n(x)=\sin(nx)/n$ even converges uniformly to zero, but $f_n'(x)=\cos(nx)$ does not converge to zero.

L'Hôpital also has conditions: verify a 0/0 or ∞/∞ form, differentiability nearby, nonzero denominator derivative, and the relevant limit of the derivative ratio. Do not differentiate every quotient by habit.

Continue: [Linear algebra and least squares](linear-algebra.en.md). For the underlying calculus theorems, see [MIT Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/).
