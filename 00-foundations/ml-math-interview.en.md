# ML interview mathematics: from probability to estimators

[中文](ml-math-interview.md) · **English**

> Reading time: ~14 min · Type: interview quick reference · Last reviewed: 2026-09

<div class="lesson-recipe">
  <div><span>The problem</span><strong>Go beyond writing the formula: explain its inputs, gradient, numerical issues, and the conditions under which it holds</strong></div>
  <div><span>Prerequisites</span><strong>probability · derivatives · linear algebra</strong></div>
  <div><span>Core content</span><strong>CE / LSE · regularization · MLE / MAP · BLUE</strong></div>
  <div><span>Common mistakes</span><strong>Confusing the two LSEs; treating normality as a necessary condition for BLUE</strong></div>
</div>

These questions do not require pushing every proof from its first line to its last. A reliable order for the answer is: **what are the inputs and outputs → why is it defined this way → what is the gradient → where does it break numerically → which conditions does the conclusion depend on**.

<details class="interview" markdown="1">
<summary>Quick learning: the one spine the seven formulas actually follow</summary>

**Quick memory**: Softmax/LSE/CE are probability normalization and likelihood; L1/L2 are geometric constraints and priors; bias–variance is the bookkeeping of generalization error; MLE/MAP trade data against the prior; BLUE is the minimum-variance result among linear unbiased estimators.

**Interview answer**

> I first define the random variables, shapes, and objective, then give the key gradient and the stable implementation, and finally state the theorem assumptions explicitly. For example, the gradient of CE with respect to the logits is $p-y$; L1 has a subgradient interval at 0, so it can produce exact zeros; Gauss–Markov does not require normal errors, only linearity, unbiasedness, homoscedasticity, and uncorrelated errors.

<details markdown="1">
<summary><b>Deep dive</b>: why does the “proof strategy” matter more than expanding the algebra from scratch?</summary>

What an interview usually tests is whether you can point to the structure: CE comes from the negative log-likelihood; L1 sparsity is explained by KKT/subgradient conditions or by the geometry of the constraint; and BLUE writes any linear unbiased estimator as OLS plus a term orthogonal to $X$, then shows that the extra covariance is positive semidefinite.

</details>
</details>

### 1. Softmax, Cross-Entropy, and LogSumExp are one computation

Given logits $z\in\mathbb R^K$, Softmax returns a categorical distribution:

$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad \sum_i p_i=1.$$

Given a label distribution $y$, Cross-Entropy returns a scalar loss:

$$\mathcal L_{\mathrm{CE}}=-\sum_i y_i\log p_i.$$

If the correct class is $c$ and $y$ is one-hot:

$$\mathcal L=-\log p_c=-z_c+\operatorname{LSE}(z),\qquad
\operatorname{LSE}(z)=\log\sum_j e^{z_j}.$$

The three most important gradients are

$$\frac{\partial p_i}{\partial z_j}=p_i(\mathbf1_{i=j}-p_j),\qquad
\nabla_z\operatorname{LSE}(z)=\operatorname{softmax}(z),$$

$$\frac{\partial\mathcal L}{\partial z_i}=p_i-y_i.$$

Computing $e^{z_i}$ directly can overflow. First set $m=\max_i z_i$:

$$\operatorname{LSE}(z)=m+\log\sum_i e^{z_i-m}.$$

Softmax is invariant to a common shift, so subtracting $m$ does not change the answer. In
code, call `log_softmax` / `cross_entropy` directly rather than forming a tiny probability
first and then taking its log.

### 2. Why minimizing CE is equivalent to maximum likelihood

For independent examples, the conditional likelihood is:

$$L(\theta)=\prod_{n=1}^N p_\theta(y_n\mid x_n).$$

Hence:

$$
\arg\max_\theta\prod_n p_\theta(y_n\mid x_n)
\iff
\arg\max_\theta\sum_n\log p_\theta(y_n\mid x_n)
\iff
\arg\min_\theta-\sum_n\log p_\theta(y_n\mid x_n).
$$

For one-hot classification, the final expression is the empirical Cross-Entropy. The
precise statement is:

> Minimizing one-hot Cross-Entropy is equivalent to maximum conditional likelihood for the conditional categorical model.

This equivalence depends on the model interpreting its outputs as probabilities, and on
factorizing the joint likelihood of the training examples under the usual independence
assumption. After label smoothing, the objective is no longer the same MLE on the original
hard labels.

### 3. L1 and L2: why one is sparse and the other only shrinks

$$\mathcal L_{L1}=\mathcal L_{\text{data}}+\lambda\|w\|_1,\qquad
\mathcal L_{L2}=\mathcal L_{\text{data}}+\frac\lambda2\|w\|_2^2.$$

L2 has gradient $\lambda w$: the closer a parameter is to zero, the weaker the shrinkage.
Away from zero the L1 subgradient is $\lambda\operatorname{sign}(w)$; at zero it is the
interval $[-\lambda,\lambda]$.

The one-dimensional problem makes this clearest:

$$\min_w\frac12(w-a)^2+\lambda|w|.$$

The optimum is soft thresholding:

$$w^*=\operatorname{sign}(a)\max(|a|-\lambda,0).$$

When $|a|\le\lambda$, the subgradient at zero contains the optimality condition, so a whole
interval of inputs maps to an **exact zero**. The same data term paired with L2 gives
$w^*=a/(1+\lambda)$, which, unless $a=0$, normally only shrinks and never reaches zero.
Geometrically, the diamond-shaped boundary of the L1 constraint has corners, so level sets
are more likely to touch it on a coordinate axis; the L2 boundary is smooth.

### 4. Bias–variance is the bookkeeping of generalization error

If $y=f(x)+\epsilon$, $\mathbb E[\epsilon]=0$, and $\operatorname{Var}(\epsilon)=\sigma^2$,
then for squared error:

$$
\mathbb E[(y-\hat f(x))^2]
=\sigma^2
+\big(\mathbb E[\hat f(x)]-f(x)\big)^2
+\mathbb E\big[(\hat f(x)-\mathbb E[\hat f(x)])^2\big].
$$

The three terms are irreducible noise, Bias$^2$, and Variance. Bias asks whether the model
averaged over different training sets deviates systematically from the true function;
variance asks how much the model changes when the training set is swapped for another.

More flexible models usually lower bias and raise variance; regularization usually accepts
more bias in exchange for less variance. This standard additive decomposition depends on
squared loss. Classification admits a similar trade-off discussion, but the same identity
cannot be carried over without qualification.

### 5. MLE, MAP, and the probabilistic reading of regularization

Maximum Likelihood Estimation asks only which parameters best explain the data:

$$\hat\theta_{\mathrm{MLE}}=\arg\max_\theta p(D\mid\theta).$$

Maximum A Posteriori adds a prior:

$$\hat\theta_{\mathrm{MAP}}
=\arg\max_\theta p(D\mid\theta)p(\theta)
=\arg\max_\theta\big[\log p(D\mid\theta)+\log p(\theta)\big].$$

A Gaussian prior:

$$p(w)\propto e^{-\frac\lambda2\|w\|_2^2}$$

corresponds to an L2 penalty; a Laplace prior:

$$p(w)\propto e^{-\lambda\|w\|_1}$$

corresponds to an L1 penalty. Regularization can therefore be read as the negative
log-prior in MAP. Both MLE and MAP depend on the modelling assumptions in the likelihood;
MAP also depends on the prior. With a large amount of data and suitable conditions, the
relative influence of the prior often fades, but that does not automatically repair model
misspecification or non-identifiability.

### 6. When is least squares BLUE?

Here LSE means **Least Squares Estimator**, not LogSumExp. Under the linear model

$$y=X\beta+\epsilon$$

OLS is:

$$\hat\beta=(X^\top X)^{-1}X^\top y.$$

The Gauss–Markov conditions are: the model is linear in the parameters; $X$ has full column
rank; $\mathbb E[\epsilon\mid X]=0$; and
$\operatorname{Var}(\epsilon\mid X)=\sigma^2I$, that is, homoscedastic and uncorrelated errors.
**Normality is not a necessary condition for BLUE.**

Proof skeleton: let $A_0=(X^\top X)^{-1}X^\top$. Any other linear unbiased estimator can be
written as $(A_0+C)y$, and unbiasedness gives $CX=0$. The cross terms then vanish:

$$
\operatorname{Var}((A_0+C)y)-\operatorname{Var}(A_0y)
=\sigma^2CC^\top\succeq0.
$$

So OLS has the smallest covariance among linear unbiased estimators. Under
heteroscedasticity or correlated errors, OLS may still be unbiased given conditional mean
zero, but it is no longer best; when the covariance structure is known, consider GLS.

## Self-check

<div class="taste-check">
  <strong>If you really understand this, you should be able to explain:</strong>
  <ol>
    <li>Why is the gradient of CE with respect to the logits $p-y$, and why does the stable implementation use LogSumExp?</li>
    <li>Why does the subgradient of L1 at zero produce exact sparsity, while L2 usually does not?</li>
    <li>Which loss does the standard bias–variance identity depend on?</li>
    <li>Which regularizers do the Gaussian and Laplace priors correspond to?</li>
    <li>Which Gauss–Markov conditions does BLUE need, and why is normality not required?</li>
  </ol>
</div>

## Where to read next

- [Whiteboard hand-writing kit](hand-write-kit.en.md): turn these formulas into stable NumPy implementations
- [Interview basics](interview-basics.en.md): masks, normalization, gradient paths, and common coding questions
- [From linear models to neural networks](from-linear-to-neural.en.md): how these objectives enter model training
