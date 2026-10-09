# Activations and Initialization: Where Did the Gradient Go?

[中文](activation-and-initialization.md) · **English**

> Reading time: about 14 minutes · Level: foundations to advanced · Reviewed: 2026-10

“Sigmoid causes vanishing gradients, so avoid it” leaves out the location and objective. A hidden-layer sigmoid is not the same as a binary-classification output. Initialization also takes more than remembering the names Xavier and He.

Start with a confident wrong prediction, follow its gradient backward, then examine why weight scale must change with layer width.

## Hidden Layers and Output Layers Need Different Things

| Function | Computation / range | Useful role | Qualification |
| --- | --- | --- | --- |
| Sigmoid | $\sigma(z)=1/(1+e^{-z})$; `(0,1)` | Binary / independent multilabel probabilities, gates | Saturates at large magnitudes; bounded does not mean calibrated |
| Tanh | $\tanh(z)$; `(-1,1)` | Signed candidate states | Saturates too; an odd function does not make every output distribution zero-mean |
| ReLU | $\max(0,z)$ | Simple hidden layers | Zero local derivative on the negative side; no positive-side saturation is not a whole-network guarantee |
| Leaky ReLU / PReLU | $\max(0,z)+a\min(0,z)$ | Retain a negative-side gradient | Fixed versus learned $a$; neither universally beats ReLU |
| GELU / SiLU | $z\Phi(z)$ / $z\sigma(z)$ | Smooth hidden transformations | Not probability outputs; distinguish the activation from a gated FFN |
| Softmax | $p_i=e^{z_i}/\sum_j e^{z_j}$ | Exclusive classes / attention weights | Transforms a vector jointly, not each coordinate independently |

Composed linear layers remain affine. Activations change the function family; see the [XOR example](../from-linear-to-neural.en.md). Continue to [FFN / SwiGLU](../core/ffn-and-gates.en.md) for modern gated blocks. A fashionable hidden activation does not replace the constraints required at an output.

## Does a Confident Wrong Answer Have a Vanishing Gradient?

For binary classification, let $p=\sigma(z)$ and $y\in\{0,1\}$. Binary cross-entropy gives

$$\frac{\partial L_{BCE}}{\partial z}=p-y.$$

At $z=10,y=0$, the prediction is confidently wrong and the gradient is close to 1, not zero. Squared error on probabilities, $\frac12(p-y)^2$, instead gives

$$\frac{\partial L_{MSE}}{\partial z}=(p-y)p(1-p).$$

That extra factor is small in saturation. Use a stable logits loss rather than materializing probabilities and then taking logs. [BCEWithLogitsLoss](https://docs.pytorch.org/docs/main/generated/torch.nn.BCEWithLogitsLoss.html) combines the operations.

Similarly, softmax cross-entropy has logit gradient $p-\operatorname{onehot}(y)$. With logits `[1000, 10, 5]` and the second class correct, it is approximately `[1, -1, 0]`. A sharp softmax does not imply that the model cannot update. Attention uses softmax inside a different computation, so the output-loss conclusion cannot simply be transferred there.

```python
import math

def categorical_loss_gradient(logits, target):
    if not logits or not all(math.isfinite(value) for value in logits):
        raise ValueError("finite nonempty logits required")
    if type(target) is not int or not 0 <= target < len(logits):
        raise ValueError("invalid target")
    maximum = max(logits)
    shifted = [value - maximum for value in logits]
    total = sum(math.exp(value) for value in shifted)
    probabilities = [math.exp(value) / total for value in shifted]
    loss = math.log(total) - shifted[target]
    gradient = [value - (index == target) for index, value in enumerate(probabilities)]
    return loss, gradient
```

Adding one common constant to all logits changes neither result. Relative differences and temperature matter, not simply whether inputs are numerically small. This teaching implementation stabilizes ordinary finite inputs; it does not promise arbitrary floating-point-range support. PyTorch [CrossEntropyLoss](https://docs.pytorch.org/docs/main/generated/torch.nn.CrossEntropyLoss.html) also consumes logits directly.

## Deep Gradients Depend on the Whole Chain

For $h_\ell=\phi(W_\ell h_{\ell-1}+b_\ell)$, the local Jacobian is

$$J_\ell=\operatorname{diag}(\phi'(a_\ell))W_\ell.$$

The full derivative passes through a product of these matrices, not activation derivatives alone. If every operator norm is bounded by $c<1$, the product norm is at most $c^L$: a sufficient condition for decay. A largest singular value above 1 in one layer does not guarantee explosion, because subsequent layers can contract that direction.

A negative ReLU input on one example is not a permanently dead unit. The concern is a unit that stays negative for nearly all training examples without an update path that brings it back. Small weights alone do not make symmetric zero-mean inputs predominantly negative.

Inspect pre-activations, activations, and gradients by layer before blaming the learning rate, initialization, saturation, loss scale, or graph. Clipping limits large gradients; it cannot recover missing information.

## Xavier: Weight Scale Must Account for Width

Assume independent zero-mean weights with variance $s^2$, independent of inputs. For $z=\sum_{j=1}^{n}w_jh_j$ and common input second moment $q=\mathbb E[h_j^2]$, cross terms vanish and

$$\mathbb E[z^2]=n s^2q.$$

A linearized forward-scale requirement suggests $s^2\approx1/n_{in}$. Backpropagation gives a requirement involving $n_{out}$. A common Xavier compromise is

$$\operatorname{Var}(w)=\frac{2}{n_{in}+n_{out}}.$$

The normal-distribution standard deviation is its square root; the uniform bounds are $\pm\sqrt{6/(n_{in}+n_{out})}$. This is an approximate initialization analysis, not a guarantee throughout training. [Glorot and Bengio](https://proceedings.mlr.press/v9/glorot10a.html) examine activations, gradients, and Jacobians—not a universal theorem from which every optimizer follows.

## He: The Second Moment Halves, Not Necessarily the Variance

For symmetric $z$ and $r=\max(0,z)$,

$$\mathbb E[r^2]=\tfrac12\mathbb E[z^2].$$

However, $r$ generally has positive mean, so $\operatorname{Var}(r)=\mathbb E[r^2]-\mathbb E[r]^2$. If $z$ is equally likely to be `-1` or `1`, its variance is 1. ReLU produces `0` or `1`: second moment 0.5, but **variance 0.25**.

Combining this second-moment relation with the next layer's independent zero-mean weights gives the usual He fan-in variance $2/n_{in}$. For Leaky ReLU with fixed negative slope $a$,

$$\operatorname{Var}(w)=\frac{2}{(1+a^2)n_{in}}.$$

At width 128, the ReLU weight standard deviation is 0.125. Independence, symmetry, and approximately matched moments matter; residual paths, gates, and learned correlations change the real behavior. The [He et al. derivation](https://arxiv.org/html/1502.01852v1#S2.SS2) explicitly tracks the nonlinear output's second moment.

```python
import math

def initialization_std(fan_in, fan_out, kind, negative_slope=0.0):
    if any(type(width) is not int or width <= 0 for width in (fan_in, fan_out)):
        raise ValueError("positive integer widths required")
    if not math.isfinite(negative_slope) or negative_slope < 0:
        raise ValueError("nonnegative finite slope required")
    if kind == "xavier":
        return math.sqrt(2.0 / (fan_in + fan_out))
    if kind == "he":
        return math.sqrt(2.0 / ((1.0 + negative_slope ** 2) * fan_in))
    raise ValueError("unknown initialization")
```

## Checks Before a Real Run

- `Linear` weights usually have layout `[out_features, in_features]`. If your manual multiplication uses another convention, verify how the initializer determines fan-in / fan-out.
- Zero hidden weights preserve symmetry; zero biases usually do not create the same problem. Not every parameter must be random.
- Do not reinitialize a whole loaded pretrained model accidentally. Usually only newly added modules need initialization.
- Framework defaults need not equal the ReLU recipe you derived. Check [PyTorch initialization documentation](https://docs.pytorch.org/docs/main/nn.init.html) and the actual module.
- Measure scales after one forward pass, finite nonzero gradients after backward, then try overfitting a tiny batch. A reasonable starting point is not a successful training run.
