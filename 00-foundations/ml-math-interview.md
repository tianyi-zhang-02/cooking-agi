# ML 数学面试主线：从概率到估计量

**中文** · [English](ml-math-interview.en.md)

> 阅读时间：约 14 分钟 · 类型：面试速查 · 最近审阅：2026-10

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>不只写出公式，还能解释输入、梯度、数值问题与成立条件</strong></div>
  <div><span>前置知识</span><strong>概率 · 导数 · 线性代数</strong></div>
  <div><span>核心内容</span><strong>CE / LSE · 正则化 · MLE / MAP · BLUE</strong></div>
  <div><span>常见错误</span><strong>混淆两个 LSE；把正态性当成 BLUE 的必要条件</strong></div>
</div>

这组题不需要把每个证明从第一行推到最后一行。稳定的回答顺序是：**输入输出是什么 → 为什么这样定义 → 梯度是什么 → 数值上哪里会坏 → 结论依赖哪些条件**。

<details class="interview" markdown="1">
<summary>快速学习：七个公式其实沿着哪条主线</summary>

**快速记忆**：Softmax/LSE/CE 连接概率归一化与似然；L1/L2 可以从几何约束和先验解释；Bias–Variance 分解泛化误差；MLE/MAP 说明数据与先验的权衡；BLUE 则给出线性无偏估计中的最小方差结论。

**面试回答**

> 我会先定义随机变量、张量形状和目标，再给核心梯度与稳定实现，最后说明定理成立的假设。比如 CE 对 logits 的梯度是 $p-y$；L1 在 0 处的次梯度是一个区间，因此可以产生精确的零；Gauss–Markov 不要求误差正态，只要求模型线性、$X$ 满列秩、误差条件均值为零、同方差且不相关。

<details markdown="1">
<summary><b>深挖</b>：为什么“证明思路”比从头展开代数更重要？</summary>

面试中通常需要说清证明的关键步骤：CE 来自负对数似然；L1 的稀疏性可以用 KKT 条件、次梯度或约束的几何形状解释；证明 BLUE 时，把任意线性无偏估计量写成 OLS 加一个与 $X$ 正交的项，再证明多出来的协方差矩阵是半正定的。

</details>
</details>

<div class="bilingual-note bilingual-intro">
  <span>逐概念双语 · CONCEPT-BY-CONCEPT</span>
  <p>卡片默认中文；点 <strong>English ↻</strong> 可在当前位置查看等价英文。</p>
</div>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 1. Softmax、Cross-Entropy 与 LogSumExp 是同一条计算

输入 logits $z\in\mathbb R^K$，Softmax 输出各类别的概率，构成类别分布（categorical distribution）：

$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad \sum_i p_i=1.$$

给定标签分布 $y$，Cross-Entropy 输出标量 loss：

$$\mathcal L_{\mathrm{CE}}=-\sum_i y_i\log p_i.$$

若正确类别为 $c$，$y$ 是 one-hot：

$$\mathcal L=-\log p_c=-z_c+\operatorname{LSE}(z),$$

$$\operatorname{LSE}(z)=\log\sum_j e^{z_j}.$$

三个最重要的梯度是：

$$\frac{\partial p_i}{\partial z_j}=p_i(\mathbf1_{i=j}-p_j),$$

$$\nabla_z\operatorname{LSE}(z)=\operatorname{softmax}(z),$$

$$\frac{\partial\mathcal L}{\partial z_i}=p_i-y_i.$$

直接算 $e^{z_i}$ 可能 overflow，先令 $m=\max_i z_i$：

$$\operatorname{LSE}(z)=m+\log\sum_i e^{z_i-m}.$$

Softmax 对整体平移不变，所以减去 $m$ 不改变答案。工程上应直接使用
`log_softmax` / `cross_entropy`，不要先形成极小概率再取 log。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">1. Softmax, Cross-Entropy, and LogSumExp are one computation</div>

Given logits $z\in\mathbb R^K$, Softmax returns a categorical distribution:

$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad \sum_i p_i=1.$$

For label distribution $y$, Cross-Entropy returns a scalar loss:

$$\mathcal L_{\mathrm{CE}}=-\sum_i y_i\log p_i.$$

For one-hot class $c$:

$$\mathcal L=-\log p_c=-z_c+\operatorname{LSE}(z),\qquad
\operatorname{LSE}(z)=\log\sum_j e^{z_j}.$$

The key derivatives are

$$\frac{\partial p_i}{\partial z_j}=p_i(\mathbf1_{i=j}-p_j),\qquad
\nabla_z\operatorname{LSE}(z)=\operatorname{softmax}(z),$$

$$\frac{\partial\mathcal L}{\partial z_i}=p_i-y_i.$$

Exponentials can overflow. With $m=\max_i z_i$, compute

$$\operatorname{LSE}(z)=m+\log\sum_i e^{z_i-m}.$$

Softmax is invariant to a common shift, so subtracting $m$ is exact. In code, use
`log_softmax` or `cross_entropy` rather than taking the log of an already underflowed
probability.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 2. 为什么最小化 CE 等价于最大似然

独立样本的条件似然（conditional likelihood）是：

$$L(\theta)=\prod_{n=1}^N p_\theta(y_n\mid x_n).$$

因此：

$$
\arg\max_\theta\prod_n p_\theta(y_n\mid x_n)
\iff
\arg\max_\theta\sum_n\log p_\theta(y_n\mid x_n)
\iff
\arg\min_\theta-\sum_n\log p_\theta(y_n\mid x_n).
$$

对于使用 one-hot 标签的分类任务，最后一项就是经验交叉熵。准确说法是：

> 最小化 one-hot 交叉熵，等价于对条件类别模型做最大条件似然估计。

这个等价关系要求把模型输出解释成概率，并按通常的独立性假设分解训练样本的联合似然。使用标签平滑（label smoothing）后，目标就不再是原始硬标签对应的那个最大似然估计（MLE）目标。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">2. Why minimizing Cross-Entropy is maximum likelihood</div>

For independent examples, the conditional likelihood is

$$L(\theta)=\prod_{n=1}^N p_\theta(y_n\mid x_n).$$

Hence

$$
\arg\max_\theta\prod_n p_\theta(y_n\mid x_n)
\iff
\arg\max_\theta\sum_n\log p_\theta(y_n\mid x_n)
\iff
\arg\min_\theta-\sum_n\log p_\theta(y_n\mid x_n).
$$

For one-hot classification, the final expression is empirical Cross-Entropy. More
precisely, minimizing one-hot CE performs maximum conditional likelihood for the
categorical model. This assumes probabilistic outputs and the usual factorization over
examples. Label smoothing changes the target and is no longer the same MLE on the
original hard labels.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 3. L1 与 L2：为什么一个稀疏、一个只收缩

$$\mathcal L_{L1}=\mathcal L_{\text{data}}+\lambda\|w\|_1,$$

$$\mathcal L_{L2}=\mathcal L_{\text{data}}+\frac\lambda2\|w\|_2^2.$$

L2 的梯度是 $\lambda w$，参数越接近零，收缩力也越小。L1 在非零处的 subgradient
是 $\lambda\operatorname{sign}(w)$，到了零点则是区间 $[-\lambda,\lambda]$。

一维问题最清楚：

$$\min_w\frac12(w-a)^2+\lambda|w|.$$

最优解是 soft thresholding：

$$w^*=\operatorname{sign}(a)\max(|a|-\lambda,0).$$

当 $|a|\le\lambda$，目标在 $w=0$ 处的次梯度集合 $[-a-\lambda,\,-a+\lambda]$ 包含 0，满足最优条件，因此整段输入都会映射成**精确的
零**。相同数据项配 L2 时：

$$w^*=\frac{a}{1+\lambda},$$

除非 $a=0$，否则通常只是缩小而不会归零。几何上，L1 constraint 的菱形边界有尖角，
等高线更容易在坐标轴上相切；L2 边界平滑。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">3. L1 and L2: exact sparsity versus smooth shrinkage</div>

$$\mathcal L_{L1}=\mathcal L_{\text{data}}+\lambda\|w\|_1,\qquad
\mathcal L_{L2}=\mathcal L_{\text{data}}+\frac\lambda2\|w\|_2^2.$$

L2 has gradient $\lambda w$, so its shrinkage fades near zero. Away from zero, the L1
subgradient is $\lambda\operatorname{sign}(w)$; at zero it is the interval
$[-\lambda,\lambda]$.

For

$$\min_w\frac12(w-a)^2+\lambda|w|,$$

the optimizer is the soft-thresholding operator

$$w^*=\operatorname{sign}(a)\max(|a|-\lambda,0).$$

When $|a|\le\lambda$, zero satisfies the subgradient condition, mapping a whole interval
of inputs to an exact zero. The comparable L2 solution is $a/(1+\lambda)$, which normally
shrinks without becoming zero. Geometrically, the L1 constraint has corners on the axes;
the L2 boundary is smooth.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 4. Bias–Variance 是泛化误差的分账

固定测试输入 $x$，设 $y=f(x)+\epsilon$，且 $\mathbb E[\epsilon\mid x]=0$、
$\operatorname{Var}(\epsilon\mid x)=\sigma^2$。假设测试噪声在给定 $x$ 时独立于训练集和训练过程的随机性，
并且噪声与预测都有有限二阶矩。于是对 squared error 有：

$$
\mathbb E[(y-\hat f(x))^2]
=\sigma^2
+\big(\mathbb E[\hat f(x)]-f(x)\big)^2
+\mathbb E\big[(\hat f(x)-\mathbb E[\hat f(x)])^2\big].
$$

式中所有期望都以这个固定的 $x$ 为条件：左边同时对训练随机性和新的测试噪声取平均；右边涉及 $\hat f(x)$ 的期望只对训练随机性取平均。上述独立性和条件零均值使交叉项 $\mathbb E[\epsilon(\hat f(x)-f(x))\mid x]=0$，这是分解成立的关键。

三项分别是不可约噪声（irreducible noise）、偏差平方（Bias$^2$）和方差。偏差衡量不同训练集上所得模型的平均预测是否系统性偏离真实函数；方差衡量换一份训练集后，模型预测会变化多大。

更复杂的模型通常降低偏差、提高方差；正则化通常以增加一些偏差为代价来降低方差。这个标准加法分解依赖平方损失。分类问题也可以讨论类似的取舍，但不能不加说明地照搬同一个等式。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">4. Bias–variance allocates generalization error</div>

Fix a test input $x$ and let $y=f(x)+\epsilon$, with $\mathbb E[\epsilon\mid x]=0$
and $\operatorname{Var}(\epsilon\mid x)=\sigma^2$. Assume that, conditional on $x$,
the test noise is independent of the training data and training randomness, and that
both the noise and prediction have finite second moments. Then under squared error

$$
\mathbb E[(y-\hat f(x))^2]
=\sigma^2
+\big(\mathbb E[\hat f(x)]-f(x)\big)^2
+\mathbb E\big[(\hat f(x)-\mathbb E[\hat f(x)])^2\big].
$$

All expectations here are conditional on this fixed $x$: the left side averages over
both training randomness and fresh test noise; expectations involving $\hat f(x)$ on
the right average only over training randomness. Independence and conditional zero-mean
noise give $\mathbb E[\epsilon(\hat f(x)-f(x))\mid x]=0$, so the cross term in the
expansion vanishes.

These are irreducible noise, squared bias, and variance. Bias measures systematic
error averaged over possible training sets; variance measures sensitivity to which
training set was sampled. More flexible models tend to reduce bias and increase
variance, while regularization often trades some bias for lower variance. The exact
additive identity above is a squared-loss result. Classification admits a similar
trade-off discussion, but the same identity cannot be carried over without qualification.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 5. MLE、MAP 与正则化的概率解释

最大似然估计（Maximum Likelihood Estimation，MLE）只问哪些参数最能解释数据：

$$\hat\theta_{\mathrm{MLE}}=\arg\max_\theta p(D\mid\theta).$$

最大后验估计（Maximum A Posteriori，MAP）在此基础上加入先验：

$$\hat\theta_{\mathrm{MAP}}
=\arg\max_\theta p(D\mid\theta)p(\theta)
=\arg\max_\theta\big[\log p(D\mid\theta)+\log p(\theta)\big].$$

高斯先验（Gaussian prior）：

$$p(w)\propto e^{-\frac\lambda2\|w\|_2^2}$$

对应 L2 惩罚；拉普拉斯先验（Laplace prior）：

$$p(w)\propto e^{-\lambda\|w\|_1}$$

对应 L1 惩罚。因此，这些正则化项可以解释为 MAP 中的负对数先验。MLE 和 MAP 都依赖似然函数的模型假设，MAP 还依赖先验。数据量很大且条件合适时，先验的相对影响通常会减弱，但这不会自动修复模型错设或不可辨识问题。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">5. MLE, MAP, and the probabilistic view of regularization</div>

Maximum likelihood chooses the parameters that best explain the data:

$$\hat\theta_{\mathrm{MLE}}=\arg\max_\theta p(D\mid\theta).$$

MAP adds a prior:

$$\hat\theta_{\mathrm{MAP}}
=\arg\max_\theta\big[\log p(D\mid\theta)+\log p(\theta)\big].$$

A Gaussian prior

$$p(w)\propto e^{-\frac\lambda2\|w\|_2^2}$$

produces an L2 penalty; a Laplace prior

$$p(w)\propto e^{-\lambda\|w\|_1}$$

produces L1. Regularization can therefore be read as a negative log-prior in MAP. Both
MLE and MAP depend on the likelihood assumptions, while MAP also depends on the prior.
With enough data and suitable conditions the prior often has less relative influence, but that does not repair
misspecification or non-identifiability automatically.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 6. Least Squares 什么时候是 BLUE

这里的 LSE 是 **Least Squares Estimator**，不是 LogSumExp。在线性模型

$$y=X\beta+\epsilon$$

下，OLS 为：

$$\hat\beta=(X^\top X)^{-1}X^\top y.$$

Gauss–Markov 条件是：模型对参数线性；$X$ 满列秩；
$\mathbb E[\epsilon\mid X]=0$；且
$\operatorname{Var}(\epsilon\mid X)=\sigma^2I$，即同方差、误差不相关。
**正态性不是 BLUE 的必要条件。**

证明骨架：令 $A_0=(X^\top X)^{-1}X^\top$。任意其他线性无偏估计量可以写成
$(A_0+C)y$，无偏条件给出 $CX=0$。于是交叉项消失：

$$
\operatorname{Var}((A_0+C)y)-\operatorname{Var}(A_0y)
=\sigma^2CC^\top\succeq0.
$$

所以 OLS 在线性无偏估计量中协方差最小。若存在异方差（heteroscedasticity）或相关误差，OLS 在条件零均值下仍可能无偏，但不再最优；已知协方差结构时，应考虑广义最小二乘（GLS）。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">6. When least squares is BLUE</div>

Here LSE means **Least Squares Estimator**, not LogSumExp. Under

$$y=X\beta+\epsilon,$$

OLS is

$$\hat\beta=(X^\top X)^{-1}X^\top y.$$

The Gauss–Markov conditions are linearity in parameters, full column rank of $X$,
$\mathbb E[\epsilon\mid X]=0$, and
$\operatorname{Var}(\epsilon\mid X)=\sigma^2I$: homoscedastic, uncorrelated errors.
Normality is **not** required for BLUE.

Proof sketch: let $A_0=(X^\top X)^{-1}X^\top$. Any other linear unbiased estimator can
be written as $(A_0+C)y$, and unbiasedness implies $CX=0$. Cross terms vanish, giving

$$
\operatorname{Var}((A_0+C)y)-\operatorname{Var}(A_0y)
=\sigma^2CC^\top\succeq0.
$$

Thus OLS has minimum covariance among linear unbiased estimators. With heteroscedastic
or correlated errors, OLS may remain unbiased under conditional mean zero but is no
longer best; known covariance structure motivates GLS.

</div>
</section>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>为什么 CE 对 logits 的梯度是 $p-y$，以及稳定实现为什么使用 LogSumExp？</li>
    <li>为什么 L1 的零点 subgradient 会产生精确稀疏，而 L2 通常不会？</li>
    <li>Bias–Variance 的标准等式依赖哪一种 loss？</li>
    <li>Gaussian / Laplace prior 分别对应什么正则化？</li>
    <li>BLUE 需要哪些 Gauss–Markov 条件，为什么不要求正态性？</li>
  </ol>
</div>

## 继续阅读

- [白板手写工具箱](hand-write-kit.md)：把这些公式写成稳定的 NumPy 实现
- [面试基础题](interview-basics.md)：mask、normalization、梯度通路和常见 coding 题
- [从线性模型到神经网络](from-linear-to-neural.md)：这些目标怎样进入模型训练
