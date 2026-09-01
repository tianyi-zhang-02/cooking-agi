# From linear models to neural networks

[中文](from-linear-to-neural.md) · **English**

<div class="lesson-recipe">
  <div><span>What we are making</span><strong>A nonlinear boundary from learned coordinates and a linear readout</strong></div>
  <div><span>Prerequisites</span><strong>linear model · sigmoid · cross-entropy · ReLU</strong></div>
  <div><span>Core technique</span><strong>feature maps · chain rule · backpropagation</strong></div>
  <div><span>Most common mistake</span><strong>Assuming depth or sigmoid alone creates nonlinear boundaries</strong></div>
</div>

## Quick learning: linear readouts and learned feature maps

<details class="interview" markdown="1">
<summary>The one-line spine, interview answer, and function-approximator deep dive</summary>

**Quick memory**: a linear model draws a hyperplane in a fixed feature space. A neural network learns new coordinates $\phi_\theta(x)$ through nonlinear layers and applies a linear readout there.

**Interview answer**

> Stacked linear layers without activations collapse into one matrix. Nonlinear activations such as ReLU let the network learn a piecewise nonlinear feature map that makes complex input-space boundaries linearly separable in hidden space. A linear final layer does not make the complete model linear.

<details markdown="1">
<summary><b>Deep dive</b>: why does universal approximation not mean “it will learn well”?</summary>

The theorem only says suitable parameters **exist** in a sufficiently rich family on a compact domain. It does not guarantee gradient descent finds them, finite data identifies them, the required network is efficient, extrapolation works, or the learned model generalizes. Expressivity, optimization, and generalization are separate questions.

</details>
</details>

## The central transition: learned feature maps

A neural network is a **learned change of coordinates** followed by **a linear classifier**. The last layer is always logistic regression; it just lives in a space the network invented.

## Linear regression: the boundary is a hyperplane

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b, \qquad \hat{\mathbf{w}} = (X^\top X)^{-1} X^\top \mathbf{y}$$

Used as a classifier, the decision surface is $\{\mathbf{x} : \mathbf{w}^\top \mathbf{x} + b = 0\}$ — a hyperplane. That is the only shape a linear model can draw.

## Do interaction terms make it nonlinear?

Two different meanings of "linear" collide here.

$$\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_1 x_2 + b$$

- **Linear in the parameters**: yes — statistics still calls this linear regression and the closed form above still applies.
- **Linear in the inputs**: no. Setting $\hat y = 0$ gives a hyperbola, not a line.

You hand-built a feature map $\phi(x_1,x_2) = (x_1, x_2, x_1x_2)$, cut a hyperplane in 3D, and the projection back down is curved. Polynomial regression and kernel SVMs are the same move: **choose $\phi$, then cut a straight line.** The catch is that you have to guess $\phi$.

## Logistic regression: the sigmoid buys gradients, not power

$$z = \mathbf{w}^\top\mathbf{x} + b, \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

The boundary is unchanged: $\sigma(z) = 0.5 \iff z = 0$. The sigmoid is monotone, so it cannot bend the surface — it only translates *distance from the surface* into a probability.

What it fixes is optimisation. With MSE on raw scores, a point at $\mathbf{w}^\top\mathbf{x} = 100$ with label 1 is already classified as well as possible, yet carries a residual of 99 and dominates the gradient. With cross-entropy:

$$\mathcal{L} = -\big[y\log\sigma(z) + (1-y)\log(1-\sigma(z))\big]$$

and using $\sigma'(z) = \sigma(z)(1-\sigma(z))$, the $\sigma'$ factor cancels exactly:

$$\frac{\partial\mathcal{L}}{\partial z} = \frac{s-y}{s(1-s)}\cdot s(1-s) = s - y, \qquad \nabla_{\mathbf{w}}\mathcal{L} = (\sigma(z)-y)\,\mathbf{x}$$

Prediction minus target. Correctly classified points fall silent; misclassified ones keep a full-strength gradient. (With MSE the surviving $\sigma'$ factor vanishes in saturation, so the *wrong* points are the ones that stop learning.)

![the sigmoid and its derivative](assets/sigmoid.svg)

$\sigma'$ peaks at $0.25$ and dies at both ends — the origin of **vanishing gradients**, and the reason ReLU replaced sigmoid in hidden layers.

```python
def fit_logistic(X, y, lr=0.1, steps=2000):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        err = sigmoid(X @ w + b) - y     # the entire derivation, one line
        w -= lr * (X.T @ err) / len(y)
        b -= lr * err.mean()
    return w, b
```

## Four common functions transform numbers for different reasons

$$
\boxed{
\begin{aligned}
\text{Sigmoid}&:\text{independent soft gate or binary probability}\\
\text{Softmax}&:\text{allocate a total probability of one across alternatives}\\
\text{ReLU}&:\text{introduce piecewise-linear nonlinearity in hidden layers}\\
\text{Tanh}&:\text{produce bounded state with positive and negative direction}
\end{aligned}}
$$

| Function | How it treats input | Output range | Sums to one? | Derivative / Jacobian | Typical role |
| --- | --- | --- | --- | --- | --- |
| Sigmoid | elementwise | $(0,1)$ | no | $\sigma(x)(1-\sigma(x))$ | binary/multi-label output, gates |
| Softmax | jointly normalizes a vector | each entry $(0,1)$ | **yes** | $p_i(\delta_{ij}-p_j)$ | multiclass output, attention, vocabulary distribution |
| ReLU | elementwise | $[0,\infty)$ | no | $\mathbf 1[x>0]$ (a chosen subgradient at zero) | hidden-layer nonlinearity |
| Tanh | elementwise | $(-1,1)$ | no | $1-\tanh^2(x)$ | RNN state, LSTM candidate |

The central distinction is not merely the output shape but **whether entries compete**. Sigmoid, ReLU, and tanh transform entries independently. Every softmax output depends on the whole vector, so increasing one logit changes the relative probabilities of the others.

<details class="interview" markdown="1">
<summary>Sigmoid: when should one logit become a probability or gate?</summary>

$$
\sigma(x)=\frac{1}{1+e^{-x}},\qquad \sigma'(x)=\sigma(x)(1-\sigma(x)).
$$

At $x=0$ the output is $0.5$; large negative inputs approach 0 and large positive inputs approach 1. A binary classifier interprets $\sigma(z)$ as a positive-class probability. A multi-label classifier applies one independent sigmoid per label, so person, car, and road may all be true. LSTM forget, input, and output gates also use it to control each channel between zero and one.

The cost is saturation: the derivative peaks at only $0.25$ and approaches zero for large $|x|$. Sigmoid therefore fits **output and gating semantics**, not ordinary hidden layers in modern deep networks.

</details>

<details class="interview" markdown="1">
<summary>Tanh: why does recurrent state need bounded values with both signs?</summary>

$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}},\qquad
\tanh'(x)=1-\tanh^2(x),\qquad
\tanh(x)=2\sigma(2x)-1.
$$

Tanh is zero-centered and bounds values in $(-1,1)$. A plain RNN uses

$$h_t=\tanh(W_xx_t+W_hh_{t-1}+b),$$

while an LSTM uses it for signed candidate memory. A sigmoid gate answers “how much passes?”; a tanh candidate answers “what signed content should be written?” Tanh still saturates at large magnitude, so it does not itself solve long-range vanishing gradients. The LSTM's additive cell-state path is the crucial part.

</details>

<details class="interview" markdown="1">
<summary>ReLU: how can a simple max stop deep layers collapsing into one linear map?</summary>

$$
\operatorname{ReLU}(x)=\max(0,x),\qquad
\operatorname{ReLU}'(x)=\begin{cases}0,&x<0\\1,&x>0.\end{cases}
$$

Without an activation, $W_2(W_1x)=(W_2W_1)x$, so an arbitrarily deep stack is still one linear map. ReLU activates different linear paths in different input regions, producing a piecewise-linear function. Its positive-side derivative is one, avoiding sigmoid/tanh saturation there, and computation is cheap.

If a unit remains negative, both output and gradient are zero and it may become a dying ReLU. Leaky ReLU preserves a small negative slope; modern Transformer FFNs more often use GELU, SiLU, or SwiGLU.

</details>

<details class="interview" markdown="1">
<summary>Softmax: why is it not an elementwise activation?</summary>

$$
p_i=\operatorname{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}},
\qquad \sum_i p_i=1.
$$

Softmax turns a complete logit vector into a categorical distribution. $[2,1,0]$ becomes approximately $[0.665,0.245,0.090]$. It serves mutually exclusive classification, each row of attention weights, and a language model's next-token distribution.

Because it is invariant to a shared shift, a stable implementation subtracts the maximum:

$$
\operatorname{softmax}(z)=\operatorname{softmax}(z-\max_j z_j).
$$

This prevents exponential overflow without changing the result. Softmax is not an argmax: several entries retain nonzero weight, although large logit gaps push it toward one-hot saturation.

</details>

### Sigmoid versus Softmax: independent labels or one competing choice

| Question | Output layer | Why |
| --- | --- | --- |
| “Which objects appear in this image?” | independent Sigmoid + BCE per class | person, car, and road may coexist; probabilities need not sum to one |
| “What is the image's one primary class?” | Softmax + categorical CE | classes compete and probabilities sum to one |
| “Is this the positive class?” | one-logit Sigmoid, or two-logit Softmax | in binary classification, two-class Softmax is a Sigmoid of the logit difference |

### Put all four back into their models

```text
LSTM gates                 → Sigmoid: how much passes in each channel
LSTM candidate / RNN state → Tanh: signed content to write
Transformer attention      → row-wise Softmax: allocate attention across keys
LM head                    → vocabulary Softmax: next-token distribution
Vanilla Transformer FFN    → ReLU; modern models often use GELU / SiLU / SwiGLU
```

One-line memory aid: **Sigmoid is an independent valve, Softmax distributes votes among candidates, ReLU cuts off the negative half-line, and Tanh compresses signed state into $(-1,1)$.**

## Neural networks: learn $\phi$ instead

$$\mathbf{h} = \phi(W_1\mathbf{x} + \mathbf{b}_1), \qquad \hat y = \sigma(\mathbf{w}_2^\top\mathbf{h} + b_2)$$

The second line is logistic regression with $\mathbf{h}$ in place of $\mathbf{x}$. Drop $\phi$ and $W_2(W_1\mathbf{x}) = (W_2W_1)\mathbf{x}$ — the whole stack collapses back to one linear map, at any depth.

Backprop is the chain rule reusing the same result:

$$\delta_2 = \hat y - y, \qquad \boldsymbol{\delta}_1 = (\delta_2\mathbf{w}_2)\odot\phi'(\mathbf{z}_1), \qquad \frac{\partial\mathcal{L}}{\partial W_1} = \boldsymbol{\delta}_1\mathbf{x}^\top$$

For ReLU, $\phi'(z) = \mathbb{1}[z>0]$: the gradient either passes through untouched or is cut off.

```python
def step(p, X, y, lr=0.05):
    z1 = X @ p["W1"].T + p["b1"]; h = np.maximum(z1, 0)      # forward
    yhat = sigmoid(h @ p["w2"] + p["b2"])
    d2 = (yhat - y) / len(y)                                  # backward
    d1 = np.outer(d2, p["w2"]) * (z1 > 0)                     # through the ReLU gate
    p["w2"] -= lr * (h.T @ d2);  p["b2"] -= lr * d2.sum()
    p["W1"] -= lr * (d1.T @ X);  p["b1"] -= lr * d1.sum(0)
```

## XOR: the evidence

| Model | Params | Accuracy |
| --- | --- | --- |
| A `Linear(2,1)` | 3 | 50.0% |
| B `Linear(2,8) → Linear(8,1)`, **no activation** | 33 | 50.0% |
| C `Linear(2,8) → ReLU → Linear(8,1)` | 33 | 100% |

![three decision boundaries](assets/decision-boundaries.svg)

**B and C are the same architecture with the same 33 parameters, one ReLU apart.** B's two matrices multiply out to a single $(1,2)$ row — identical in form to A, which has 3 parameters.

50% is not undertraining: on symmetric XOR the best any straight line can do is 50%, with loss pinned at $\ln 2 \approx 0.693$.

![input space warped into hidden space](assets/hidden-space.svg)

ReLU folds the plane along a crease until the two classes land on opposite sides of one straight line. The output layer is still just logistic regression — on coordinates that were learned rather than given.

<!-- widget:xor -->

## The pattern

| | Feature map $\phi$ | Final step |
| --- | --- | --- |
| Linear / logistic regression | none | linear classifier |
| Polynomial / kernel | you choose it | linear classifier |
| MLP | learned | linear classifier |
| CNN | learned, constrained to be translation-equivariant | linear classifier |
| Transformer | learned, $N$ layers of attention + FFN | linear classifier |

## Connecting to the Transformer

$$\mathbf{h} = \text{TransformerBlocks}(\text{Embed}(\mathbf{x})), \qquad p_i = \text{softmax}(W_\text{head}\mathbf{h})_i$$

Softmax generalises the sigmoid to many classes, and keeps the property that matters: with cross-entropy, $\partial\mathcal{L}/\partial z_i = p_i - y_i$ — still prediction minus target. `lm_head` is the linear classifier; every attention block beneath it exists to bend the space until the next token is linearly readable.

## Are they all function approximators in the end?

In the broad sense, yes. A supervised model searches a parameterized function family for some $f_\theta$ that approximates an unknown target mapping $f^*$ or conditional distribution $p^*$:

$$
\theta^*=\arg\min_\theta \frac{1}{N}\sum_{i=1}^{N}\ell\big(f_\theta(x_i),y_i\big).
$$

The key distinction is that **“the whole model is an approximator” does not mean “the whole model is a linear approximator.”**

| Name | Form | What is linear? |
| --- | --- | --- |
| Linear approximator | $\hat f(x)=\mathbf{w}^\top\mathbf{x}+b$ | linear in both the input and the parameters |
| Linear approximator over fixed bases | $\hat f(x)=\sum_j w_j\phi_j(x)$ | linear in $w_j$; possibly nonlinear in the raw input |
| Neural function approximator | $f_\theta(x)=W_L\phi(\cdots\phi(W_1x))$ | the final layer is often a linear readout, but the complete map is nonlinear |
| Language model | $p_\theta(x_t\mid x_{<t})$ | approximates a next-token conditional distribution, not one deterministic answer function |

Polynomial regression is a useful boundary case. After adding $x^2$, it is nonlinear in $x$ but still **linear in parameters**. A neural network goes one step further and learns the basis or feature map $\phi$ from data as well.

<details class="interview" markdown="1">
<summary>What does the universal approximation theorem actually guarantee?</summary>

With a suitable nonlinear activation and enough width, for any continuous target $f^*$ on a compact set $K$ and every $\varepsilon>0$, there **exists** a parameter setting such that

$$
\sup_{x\in K}\left|f_\theta(x)-f^*(x)\right|<\varepsilon.
$$

This establishes the function family's **expressivity**, but it does not guarantee that:

- gradient descent will find those parameters;
- finite training data identifies the right function;
- the required network is small or cheap to train;
- the model extrapolates outside the training distribution;
- low training loss implies good generalization.

“Universal approximator” is therefore an **existence result**, not a certificate that a particular model will learn well.

</details>

The same distinction applies to a Transformer. Its <code>lm_head</code> is linear, but the complete map

$$
x_{<t}\longmapsto \mathbf{h}_t\longmapsto
\operatorname{softmax}(W_{\text{head}}\mathbf{h}_t)
$$

contains attention, Softmax, MLP activations, and many compositions. The overall model is a highly nonlinear conditional-distribution approximator. What makes it useful is not only approximation capacity, but also **inductive bias, data, objective, optimization, and evaluation**.

## Self-check

<div class="taste-check">
  <strong>Explain without the summary table:</strong>
  <ol>
    <li>Why do stacked linear layers collapse into one linear layer?</li>
    <li>Does sigmoid plus cross-entropy improve expression or optimization?</li>
    <li>How does XOR demonstrate a learned coordinate transformation?</li>
    <li>Why does multi-label classification use Sigmoid while mutually exclusive multiclass classification uses Softmax?</li>
    <li>Why do LSTM gates use Sigmoid while candidate memory uses Tanh?</li>
    <li>Why is the <code>lm_head</code> a linear readout while the complete Transformer is not a linear approximator?</li>
    <li>What does the universal approximation theorem guarantee, and what does it not guarantee?</li>
  </ol>
</div>

## Where to read next

- [The Transformer architecture](transformer.en.md)
- [Post-training](../05-post-training/README.en.md)
- [Representation & memory](../02-memory/README.en.md)
