# Seven ML modules to implement by hand

[中文](hand-write-kit.md) · **English**

> Reading time: ~5 min · Type: quick reference · Last reviewed: 2026-10-09

## What the seven formulas are checking {#what-the-seven-formulas-are-checking}

These seven small implementations connect formulas to code: checking input shapes, spotting overflow, and understanding what a mask blocks. Start with something runnable, then check its boundaries with small examples rather than memorizing the formula alone.

The reference implementations in [`code/interview_kit.py`](code/) depend only on NumPy and include self-checks, including **a numerical gradient check of backpropagation**. Run them first:

```bash
python3 00-foundations/code/interview_kit.py
```

## The seven, and their traps {#the-seven-and-their-traps}

| What to write | The trap | The usual follow-up |
| --- | --- | --- |
| **softmax** | overflows without subtracting the max | "why is subtracting still correct?" — softmax is shift-invariant; numerator and denominator both pick up $e^c$ and it cancels, so subtracting the max is free |
| **BCE from logits** | computing $p$ then its log underflows to $-\infty$ | "what's the gradient?" — $p-y$; the $\sigma'$ cancels |
| **LayerNorm** | `eps` goes **inside** the sqrt; variance divides by $N$, not $N-1$; last dimension only | "why not BatchNorm?" |
| **attention + causal mask** | the $\sqrt{d_k}$; mask added **before** the softmax | "could you zero it after the softmax instead?" — no, it breaks normalization |
| **KV cache decode** | A single-token query can omit the causal mask when its cache contains only valid past and current positions | Padding, sliding windows, and multi-token decoding can still require masks |
| **top-k / top-p** | top-p must keep **the token that crosses the threshold**, and at least one | "what if the top probability already exceeds p?" — writing `cum <= p` deletes every token |
| **MLP forward + backward** | Define mean / sum reduction, then apply the chain rule | For ordinary ReLU, `z1 > 0` and `a1 > 0` are equivalent; do not assume this for other activations |

## The last one is the dividing line {#the-last-one-is-the-dividing-line}

The earlier modules can be tested separately; backpropagation connects them. Start with a binary-classification MLP with one ReLU hidden layer and one output logit. Sketch the forward pass, then walk backward through each shape and gradient.

Take mean BCE over $N$ examples, with hidden pre-activations $z_1$, activations $a_1=\max(z_1,0)$, and output logits $z_2$:

$$\frac{\partial \mathcal{L}}{\partial z_2} = \frac{\sigma(z_2)-y}{N} \quad\longrightarrow\quad \frac{\partial \mathcal{L}}{\partial W_2} = a_1^\top \frac{\partial \mathcal{L}}{\partial z_2} \quad\longrightarrow\quad \frac{\partial \mathcal{L}}{\partial z_1} = \left(\frac{\partial \mathcal{L}}{\partial z_2} W_2^\top\right)\odot \mathbb{1}[z_1 > 0]$$

**Keep the denominator**: divide all of $p-y$ by $N$. For $\frac{1}{N}\sum_i(p_i-y_i)^2$, the logit gradient is $2(p-y)p(1-p)/N$; the factor 2 disappears only if the loss includes $1/2$. For example, $N=2,p=0.8,y=1$ gives a mean-BCE gradient of -0.1 for that logit. Do not mix single-example and batch-mean formulas; see [interview basics](interview-basics.en.md).

After deriving the gradient, choose one parameter and compare it with a central difference:

$$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}$$

For a sufficiently smooth function, central-difference truncation error is $O(\epsilon^2)$ rather than $O(\epsilon)$ for a forward difference. Very small $\epsilon$ amplifies floating-point roundoff, and crossing a ReLU kink invalidates that smoothness argument. Use float64, try several step sizes, and avoid activations exactly at zero.

## How to practice {#how-to-practice}

**Don't read it. Write it.** Close this page, type it from scratch in an empty file, then check yourself against `interview_kit.py`'s self-tests. Wherever you stall is where you thought you knew it and didn't.

A harder version: **write the checks before the implementation.** If you can state that "softmax must produce rows summing to one and must be shift-invariant," you actually know what it is. The one whose criteria you can't state is the one where you only memorized the shape of the formula.

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

BLUE requires linearity, unbiasedness, homoscedastic uncorrelated errors; L1 sparsity relies on the nonsmooth kink and optimality conditions; stable LogSumExp relies on shift invariance. Many candidates can recall formulas. Knowing when a theorem stops applying demonstrates actual understanding.

</details>

</details>
