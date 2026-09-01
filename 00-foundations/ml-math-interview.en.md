# ML interview mathematics: probability to estimators

[中文](ml-math-interview.md) · **English**

> Reading time: ~12 min · Type: interview reference · Last reviewed: 2026-09

## The five-question template

For every formula, be ready to explain its input and output, why it has that form, its
gradient, numerical failure modes, and the assumptions under which the conclusion
holds.

<details class="interview" markdown="1">
<summary>Quick learning: the common spine behind the seven formulas</summary>

**Quick memory**: Softmax, LSE, and CE connect normalization to likelihood; L1 and L2 connect geometry to priors; bias–variance decomposes generalization error; MLE and MAP trade data against prior information; BLUE is a minimum-variance statement among linear unbiased estimators.

**Interview answer**

> I define variables, shapes, and the objective first, then give the key gradient and stable implementation, and finally state theorem assumptions. CE has logit gradient $p-y$; L1 can create exact zeros because zero has a subgradient interval; Gauss–Markov needs linearity, unbiasedness, homoscedasticity, and uncorrelated errors, not Gaussian noise.

<details markdown="1">
<summary><b>Deep dive</b>: why is a proof strategy more useful than expanding every line?</summary>

The structure is what transfers: CE follows from negative log-likelihood; L1 sparsity follows from KKT or subgradient geometry; and BLUE writes any linear unbiased estimator as OLS plus a component orthogonal to $X$, then shows the extra covariance is positive semidefinite.

</details>
</details>

## Softmax, Cross-Entropy, and LogSumExp

$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad
\mathcal L_{\mathrm{CE}}=-\sum_i y_i\log p_i.$$

For one-hot class $c$,

$$\mathcal L=-z_c+\operatorname{LSE}(z),\qquad
\operatorname{LSE}(z)=\log\sum_j e^{z_j}.$$

The central derivatives are

$$\frac{\partial p_i}{\partial z_j}=p_i(\mathbf1_{i=j}-p_j),\qquad
\nabla\operatorname{LSE}(z)=\operatorname{softmax}(z),$$

$$\frac{\partial\mathcal L}{\partial z_i}=p_i-y_i.$$

For stability, with $m=\max_i z_i$ compute

$$\operatorname{LSE}(z)=m+\log\sum_i e^{z_i-m}.$$

Softmax is invariant to a common shift, so this is exact. Use fused `log_softmax` or
`cross_entropy` rather than taking logs of underflowed probabilities.

## Why Cross-Entropy is maximum likelihood

For independent labelled examples,

$$
\arg\max_\theta\prod_n p_\theta(y_n\mid x_n)
\iff
\arg\max_\theta\sum_n\log p_\theta(y_n\mid x_n)
\iff
\arg\min_\theta-\sum_n\log p_\theta(y_n\mid x_n).
$$

For one-hot classification, the final term is empirical Cross-Entropy. Thus minimizing
one-hot CE performs maximum conditional likelihood. Label smoothing changes the target
and is not the same MLE over the original hard labels.

## L1 and L2

$$\mathcal L_{L1}=\mathcal L_{\text{data}}+\lambda\|w\|_1,qquad
\mathcal L_{L2}=\mathcal L_{\text{data}}+\frac\lambda2\|w\|_2^2.$$

L2 has gradient $\lambda w$, whose shrinkage fades near zero. L1 has subgradient
$\lambda\operatorname{sign}(w)$ away from zero and the interval $[-\lambda,\lambda]$
at zero. For

$$\min_w\frac12(w-a)^2+\lambda|w|,$$

the solution is

$$w^*=\operatorname{sign}(a)\max(|a|-\lambda,0).$$

A whole interval $|a|\le\lambda$ maps to exact zero. The comparable L2 solution
$a/(1+\lambda)$ usually shrinks without becoming zero.

## Bias–variance

If $y=f(x)+\epsilon$, $\mathbb E[\epsilon]=0$, and
$\operatorname{Var}(\epsilon)=\sigma^2$, squared error decomposes as

$$
\mathbb E[(y-\hat f(x))^2]
=\sigma^2
+\big(\mathbb E[\hat f(x)]-f(x)\big)^2
+\mathbb E\big[(\hat f(x)-\mathbb E[\hat f(x)])^2\big].
$$

These are irreducible noise, squared bias, and variance. Regularization commonly
accepts additional bias for lower variance. The exact additive identity is specific
to squared loss.

## MLE, MAP, and priors

$$\hat\theta_{\mathrm{MLE}}=\arg\max_\theta p(D\mid\theta),$$

$$\hat\theta_{\mathrm{MAP}}
=\arg\max_\theta\big[\log p(D\mid\theta)+\log p(\theta)\big].$$

A Gaussian prior $p(w)\propto e^{-\lambda\|w\|_2^2/2}$ produces L2; a Laplace prior
$p(w)\propto e^{-\lambda\|w\|_1}$ produces L1. Both estimators depend on likelihood
assumptions, and MAP also depends on its prior.

## When least squares is BLUE

Here LSE means **Least Squares Estimator**, not LogSumExp. Under

$$y=X\beta+\epsilon,$$

OLS is

$$\hat\beta=(X^\top X)^{-1}X^\top y.$$

The Gauss–Markov conditions are linearity in parameters, full column rank of $X$,
$\mathbb E[\epsilon\mid X]=0$, and
$\operatorname{Var}(\epsilon\mid X)=\sigma^2I$. Normality is not required.

Let $A_0=(X^\top X)^{-1}X^\top$. Any other linear unbiased estimator is
$(A_0+C)y$ with $CX=0$, so

$$
\operatorname{Var}((A_0+C)y)-\operatorname{Var}(A_0y)
=\sigma^2CC^\top\succeq0.
$$

Therefore OLS has minimum covariance among linear unbiased estimators. With
heteroscedastic or correlated errors, it may remain unbiased but is no longer best.

## Self-check

1. Why is the CE gradient with respect to logits $p-y$?
2. Why does stable LogSumExp subtract the maximum without changing the result?
3. Why can L1 create exact zeros while L2 normally cannot?
4. Which priors correspond to L1 and L2?
5. Which assumptions make OLS BLUE, and why is normality unnecessary?

Next: [Hand-written formulas](hand-write-kit.en.md) · [Interview basics](interview-basics.en.md)
