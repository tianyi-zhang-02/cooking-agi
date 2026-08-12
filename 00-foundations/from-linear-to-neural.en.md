# From linear models to neural networks

[中文](from-linear-to-neural.md) · **English**

## In one sentence

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

## Where to read next

- [The Transformer architecture](transformer.en.md)
- [Post-training](../05-post-training/README.en.md)
- [Representation & memory](../02-memory/README.en.md)
