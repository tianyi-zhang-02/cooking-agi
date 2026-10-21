# Seven ML modules to implement by hand

[中文](hand-write-kit.md) · **English**

> Reading time: ~8 min · Type: quick reference and small experiments · Last reviewed: 2026-10

## What the seven implementations practice {#what-the-seven-formulas-are-checking}

These seven small implementations connect formulas to code: checking input shapes, spotting overflow, and understanding what a mask blocks. Start with something runnable, then check its boundaries with small examples rather than memorizing the formula alone.

The reference implementations in [`code/interview_kit.py`](code/) depend only on NumPy and include self-checks, including **a numerical gradient check of backpropagation**. Run them first:

```bash
python3 00-foundations/code/interview_kit.py
```

## What to check in each implementation {#the-seven-and-their-traps}

| What to write | The trap | The usual follow-up |
| --- | --- | --- |
| **softmax** | overflows without subtracting the max | "why is subtracting still correct?" — softmax is invariant to a shared shift; the common factor $e^c$ cancels between numerator and denominator, so subtracting the max leaves the mathematical result unchanged |
| **BCE from logits** | computing $p$ then its log underflows to $-\infty$ | "what's the gradient?" — $p-y$; the $\sigma'$ cancels |
| **LayerNorm** | `eps` goes inside the sqrt; variance divides by $N$, not $N-1$; this implementation normalizes the last dimension | `normalized_shape=(T,d)` also normalizes across time, unlike the per-token configuration |
| **attention + causal mask** | the $\sqrt{d_k}$; mask added **before** the softmax | "could you zero it after the softmax instead?" — no, it breaks normalization (two-score example in [multi-head attention](core/multi-head-attention.en.md#the-six-places-a-from-scratch-implementation-goes-wrong)) |
| **KV cache decode** | A single-token query can omit the causal mask when its cache contains only valid past and current positions | Padding, sliding windows, and multi-token decoding can still require masks |
| **top-k / top-p** | top-p must keep **the token that crosses the threshold**, and at least one | "what if the top probability already exceeds p?" — writing `cum <= p` deletes every token |
| **MLP forward + backward** | Define mean / sum reduction, then apply the chain rule | For ordinary ReLU, `z1 > 0` and `a1 > 0` are equivalent; do not assume this for other activations |

One small sampling detail is easy to miss: which two tokens does top-k keep from `[2, 2, 2, 1]` when k=2? This implementation breaks ties by token ID and keeps IDs 0 and 1. Filtering by “at least the kth score” would keep all three tied tokens instead. Either convention can be useful, but code that keeps ties does not implement a fixed count. Existing masks stay blocked, and `top_p=1` disables further truncation.

## Softmax: put the numbers in a workable range {#softmax-numerics}

Take logits `[1000, 1001, 1002]`. Their exponentials overflow, but the probabilities depend on their relative values. Subtract 1002 from each to get `[-2, -1, 0]`. Exponentiating and normalizing now gives approximately `[0.0900, 0.2447, 0.6652]`.

$$
p_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\qquad m=\max_j x_j.
$$

Dividing the numerator and denominator by $e^m$ leaves the mathematical result unchanged. The largest exponential is now 1 and the others are at most 1, preventing exponential overflow. **Subtracting the maximum does not solve every numerical problem**, though:

| Input or situation | What happens | What this implementation does |
| --- | --- | --- |
| `[1000, 1001, 1002]` | Direct exponentiation overflows | Subtract the maximum first |
| 65,536 zeros in FP16 | Every exponential is 1, but FP16 cannot hold 65,536: NumPy's pairwise sum overflows to `inf`, and one-by-one accumulation stalls at 2048, silently wrong | Compute and return at least FP32; do not immediately cast back to FP16 |
| `[0, -inf]` | The second position is masked | Return `[1, 0]` |
| `[-inf, -inf]` | No allowed position remains; the distribution is undefined | Raise an error, rather than invent a uniform distribution |
| `NaN` or `+inf` | The upstream computation needs checking | Raise an error; these are not mask values |

These are choices in the accompanying NumPy teaching implementation, not defaults shared by all frameworks. Integer inputs also require an exact conversion: casting `[2**60, 2**60+1]` to float64 loses the difference, so the helper rejects it rather than silently returning uniform probabilities. Higher precision cannot preserve arbitrarily small probabilities or repair logits that have already overflowed upstream.

### If you need log-probabilities, compute them directly {#log-softmax-example}

For `[0, -1000]`, the second probability underflows to zero in common floating-point formats. Taking its logarithm then gives `-inf`, while direct log-softmax still gives approximately `-1000`. A loss that needs a log-probability need not first represent that tiny probability.

```python
import numpy as np

logits = np.array([0.0, -1000.0], dtype=np.float64)
shifted = logits - logits.max()
weights = np.exp(shifted)
probabilities = weights / weights.sum()
with np.errstate(divide="ignore"):
    indirect = np.log(probabilities)
direct = shifted - np.log(weights.sum())
assert np.isneginf(indirect[1])
assert np.allclose(direct, [0.0, -1000.0])
```

[PyTorch's `log_softmax` documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.log_softmax.html) (2.14, checked October 2026) makes this distinction explicitly. For blockwise computation and attention without a stored probability row, continue to the [online softmax example](deep-dives/attention-kernels.en.md#online-output). Numerical range, output representation, and intermediate storage are related but separate concerns.

## Connect the forward and backward passes {#the-last-one-is-the-dividing-line}

The earlier modules can be tested separately; backpropagation connects them. Start with a binary-classification MLP with one ReLU hidden layer and one output logit. Sketch the forward pass, then walk backward through each shape and gradient.

Take mean BCE over $N$ examples, with hidden pre-activations $z_1$, activations $a_1=\max(z_1,0)$, and output logits $z_2$:

$$\frac{\partial \mathcal{L}}{\partial z_2} = \frac{\sigma(z_2)-y}{N} \quad\longrightarrow\quad \frac{\partial \mathcal{L}}{\partial W_2} = a_1^\top \frac{\partial \mathcal{L}}{\partial z_2} \quad\longrightarrow\quad \frac{\partial \mathcal{L}}{\partial z_1} = \left(\frac{\partial \mathcal{L}}{\partial z_2} W_2^\top\right)\odot \mathbb{1}[z_1 > 0]$$

**Keep the denominator**: divide all of $p-y$ by $N$. For $\frac{1}{N}\sum_i(p_i-y_i)^2$, the logit gradient is $2(p-y)p(1-p)/N$; the factor 2 disappears only if the loss includes $1/2$. For example, $N=2,p=0.8,y=1$ gives a mean-BCE gradient of -0.1 for that logit. Do not mix single-example and batch-mean formulas; see [interview basics](interview-basics.en.md).

After deriving the gradient, choose one parameter and compare it with a central difference:

$$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}$$

For a sufficiently smooth function, central-difference truncation error is $O(\epsilon^2)$ rather than $O(\epsilon)$ for a forward difference. Very small $\epsilon$ amplifies floating-point roundoff, and crossing a ReLU kink invalidates that smoothness argument. Use float64, try several step sizes, and avoid activations exactly at zero.

## How to practice {#how-to-practice}

Pick one module to start with; there is no need to implement all seven at once. For softmax, check that rows sum to approximately one, masked entries are zero, and adding a shared constant preserves the result within a tolerance. Refer back to the relevant section when you get stuck.

Once that feels familiar, try writing the tests first. Include large logits, partial masks, and entirely masked rows, not just typical inputs. That checks both the formula and what the code does with inputs it does not support.

## Where to read next {#where-to-read-next}

- [Interview basics: most of them ask the same thing](interview-basics.en.md): the conceptual half
- [Multi-head attention](core/multi-head-attention.en.md) · [Normalization](core/normalization.en.md) · [Decoding](core/decoding.en.md)
- [Reference implementations](code/): `interview_kit.py` pairs with this page; `attention_numpy.py` is the full multi-head version

## Quick learning: five questions to ask for every handwritten formula {#quick-learning-five-questions-to-ask-for-every-handwritten-formula}

<details class="interview" markdown="1">
<summary>A formula is not dictation: inputs, purpose, gradient, numerics, and assumptions</summary>

**Quick memory**

For every formula, ask: what are the inputs and outputs, why is it defined this way, what is the gradient, what can fail numerically, and under which assumptions is the claim valid?

**Interview answer**

> I do not stop at the formula. I define variables and shapes, derive it from a probabilistic or optimization objective, give the key gradient and stable implementation, and state the assumptions. For example, CE has logit gradient $p-y$, is implemented with LogSumExp, and equals maximum likelihood only for the corresponding conditional likelihood.

<details markdown="1">
<summary><b>Deep dive</b>: why are assumptions often the most discriminative part?</summary>

For the Gauss–Markov result in linear regression, check full column rank, zero conditional mean errors, and homoscedastic, uncorrelated errors—not just the phrase “best linear unbiased.” For another example, softmax is unchanged by a common shift, but LogSumExp increases by that same constant: its stable implementation must add the subtracted maximum back. These details tell you when a formula is safe to use.

</details>

</details>
