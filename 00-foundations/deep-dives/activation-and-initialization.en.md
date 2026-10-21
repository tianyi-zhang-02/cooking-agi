# Activations and Initialization: Where Did the Gradient Go?

[中文](activation-and-initialization.md) · **English**

> Reading time: about 17 minutes · Level: foundations to advanced · Last reviewed: 2026-10

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

That extra factor is small in saturation. Use a stable logits loss rather than materializing probabilities and then taking logs, so probabilities never round to exactly 0 or 1 first. [BCEWithLogitsLoss](https://docs.pytorch.org/docs/main/generated/torch.nn.BCEWithLogitsLoss.html) combines the operations.

<!-- widget:step-bce-vs-mse -->

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

$$J_\ell=\operatorname{diag}(\phi'(u_\ell))W_\ell,\qquad u_\ell=W_\ell h_{\ell-1}+b_\ell.$$

The full derivative passes through a product of these matrices, not activation derivatives alone. If every operator norm is bounded by $c<1$, the product norm is at most $c^L$: a sufficient condition for decay. A largest singular value above 1 in one layer does not guarantee explosion, because subsequent layers can contract that direction.

Shrink the chain to its simplest form: one neuron per layer, $h_\ell=\phi(w\,h_{\ell-1})$. Each $J_\ell$ is then a single number $w\,\phi'(u_\ell)$, and passing through 10 layers multiplies the gradient by 10 of these numbers:

| Factor per layer | Gradient multiplier after 10 layers |
| --- | ---: |
| Sigmoid, $w=1$, $\phi'\le0.25$ | At most $0.25^{10}\approx9.5\times10^{-7}$ |
| ReLU active region, $w=0.5$ | $0.5^{10}\approx9.8\times10^{-4}$ |
| ReLU active region, $w=1$ | $1$ |
| ReLU active region, $w=2$ | $2^{10}=1024$ |

The sigmoid derivative peaks at 0.25 (at $u=0$), so with $w=1$ each layer shrinks the gradient to at most a quarter. ReLU has derivative 1 in its active region, so the gradient depends entirely on $w$: a factor of 2 in the weight becomes a factor of $2^{10}$ after 10 layers. Initialization sets this per-layer factor. In a wide layer the single $w$ becomes a whole matrix, and the Xavier and He sections below work out how large it should be.

A negative ReLU input on one example is not a permanently dead unit. The concern is a unit that stays negative for nearly all training examples without an update path that brings it back. Small weights alone do not make symmetric zero-mean inputs predominantly negative.

Inspect pre-activations, activations, and gradients by layer before blaming the learning rate, initialization, saturation, loss scale, or graph. Clipping limits large gradients; it cannot recover missing information.

## Xavier: Weight Scale Must Account for Width

Assume independent zero-mean weights with variance $s^2$, independent of inputs. For $z=\sum_{j=1}^{n}w_jh_j$ and common input second moment $q=\mathbb E[h_j^2]$, cross terms vanish and

$$\mathbb E[z^2]=n s^2q.$$

A linearized forward-scale requirement suggests $s^2\approx1/n_{in}$. Backpropagation gives a requirement involving $n_{out}$. A common Xavier compromise is

$$\operatorname{Var}(w)=\frac{2}{n_{in}+n_{out}}.$$

The normal-distribution standard deviation is its square root; the uniform bounds are $\pm\sqrt{6/(n_{in}+n_{out})}$. This is an approximate initialization analysis, not a guarantee throughout training. [Glorot and Bengio](https://proceedings.mlr.press/v9/glorot10a.html) examine how activations, gradients, and layer Jacobians scale at initialization; the compromise above comes from that analysis.

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

Put the three scales into the same 20-layer, width-512 ReLU network and see how much gradient reaches the input. By the second-moment relation above, each layer multiplies the gradient's second moment by about $n s^2/2$, so the input-gradient RMS is about $(n s^2/2)^{10}$: $0.0256^{10}\approx1.2\times10^{-16}$ for standard deviation 0.01, $0.5^{10}\approx9.8\times10^{-4}$ for Xavier, and 1 for He. The code below reuses `initialization_std` from the previous block and requires PyTorch:

```python
import torch


def input_gradient_rms(weight_std, depth=20, width=512, seed=0):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(256, width, generator=generator, requires_grad=True)
    hidden = inputs
    for _ in range(depth):
        weight = torch.randn(width, width, generator=generator) * weight_std
        hidden = torch.relu(hidden @ weight)
    upstream = torch.randn(hidden.shape, generator=generator)
    (hidden * upstream).sum().backward()
    return inputs.grad.square().mean().sqrt().item()


small = input_gradient_rms(0.01)
xavier = input_gradient_rms(initialization_std(512, 512, "xavier"))
he = input_gradient_rms(initialization_std(512, 512, "he"))
assert 5e-17 < small < 5e-16
assert 5e-4 < xavier < 2e-3
assert 0.5 < he < 2
```

On PyTorch 2.8 / CPU, seed 0 gives about $1.1\times10^{-16}$, $9.2\times10^{-4}$, and 0.94; other seeds land in the same range. Xavier comes out small here because its derivation uses a linear approximation and does not account for ReLU zeroing half the inputs. This checks the moment of initialization only, not gradients later in training.

He only controls whether one layer preserves scale. Each Transformer or ResNet block also adds its branch back to a residual: $h_\ell=h_{\ell-1}+F_\ell(h_{\ell-1})$. If every branch preserves its input's second moment and is uncorrelated with the residual, the second moment doubles at every block: $2^{10}=1024$ times after 10 blocks, about 32 times in RMS. [Fixup §2](https://arxiv.org/abs/1901.09321) also shows that with standard initialization and no normalization, the output variance of such residual networks grows exponentially with depth. Two common fixes:

- **Add normalization:** the [GPT-2 report §2.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) moves layer normalization to the input of each sub-block and adds one more after the final self-attention block.
- **Scale residual-branch initialization with depth:** in the same section, GPT-2 multiplies the weights of residual layers by $1/\sqrt N$ at initialization, where $N$ is the number of residual layers. [Fixup §3](https://arxiv.org/abs/1901.09321) drops normalization, initializes the last layer of each residual branch to 0, and scales down the other layers inside the branch according to depth.

So in a deep network with residual paths, He alone is not enough: you also need to decide where normalization goes and how large the residual branches start.

## Checks Before a Real Run

- `Linear` weights usually have layout `[out_features, in_features]`. If your manual multiplication uses another convention, verify how the initializer determines fan-in / fan-out.
- Zero hidden weights preserve symmetry; zero biases usually do not create the same problem. Not every parameter must be random.
- Do not reinitialize a whole loaded pretrained model accidentally. Usually only newly added modules need initialization.
- Framework defaults need not equal the ReLU recipe you derived. Check [PyTorch initialization documentation](https://docs.pytorch.org/docs/main/nn.init.html) and the actual module.
- Measure scales after one forward pass, finite nonzero gradients after backward, then try overfitting a tiny batch. A reasonable starting point is not a successful training run.
