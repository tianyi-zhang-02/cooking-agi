# The seven things you may be asked to write on a whiteboard

[中文](hand-write-kit.md) · **English**

> Reading time: ~5 min · Type: quick reference · Last reviewed: 2026-08

## Quick learning: five questions to ask for every handwritten formula

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

## What the seven formulas are checking

These questions never test whether you remember the formula. They test **whether you know which line blows up**. Each one has a numerical or semantic trap, and the follow-up lands on exactly that line.

Reference implementations in [`code/interview_kit.py`](code/) — pure numpy, no dependencies, with 14 self-checks including **a gradient check of the backward pass against numerical differences**. Run it first:

```bash
python3 00-foundations/code/interview_kit.py
```

## The seven, and their traps

| What to write | The trap | The usual follow-up |
| --- | --- | --- |
| **softmax** | overflows without subtracting the max | "why is subtracting still correct?" — softmax is shift-invariant; numerator and denominator both pick up $e^c$ and it cancels, so subtracting the max is free |
| **BCE from logits** | computing $p$ then its log underflows to $-\infty$ | "what's the gradient?" — $p-y$; the $\sigma'$ cancels |
| **LayerNorm** | `eps` goes **inside** the sqrt; variance divides by $N$, not $N-1$; last dimension only | "why not BatchNorm?" |
| **attention + causal mask** | the $\sqrt{d_k}$; mask added **before** the softmax | "could you zero it after the softmax instead?" — no, it breaks normalization |
| **KV cache decode** | **no causal mask during decoding** | "why not?" — q is a single position and the cache is all past by construction |
| **top-k / top-p** | top-p must keep **the token that crosses the threshold**, and at least one | "what if the top probability already exceeds p?" — writing `cum <= p` deletes every token |
| **MLP forward + backward** | ReLU's derivative tests `z1 > 0`, not `a1 > 0` | "how do you know your backward is right?" — **a numerical gradient check** |

## The last one is the dividing line

The first six you can memorize. **Hand-writing backprop you can't** — it's the line between actually understanding the chain rule and knowing a framework's API, and it's the only one where the interviewer can tell from the *order* you write things.

The steps that matter:

$$\frac{\partial \mathcal{L}}{\partial z_2} = \big(\sigma(z_2) - y\big)\big/N \quad\longrightarrow\quad \frac{\partial \mathcal{L}}{\partial W_2} = a_1^\top \frac{\partial \mathcal{L}}{\partial z_2} \quad\longrightarrow\quad \frac{\partial \mathcal{L}}{\partial z_1} = \left(\frac{\partial \mathcal{L}}{\partial z_2} W_2^\top\right)\odot \mathbb{1}[z_1 > 0]$$

**Note the first term** — that's the BCE cancellation again. Swap in MSE and this line becomes $2(p-y)p(1-p)$, and everything downstream changes with it. That consequence is what [interview basics](interview-basics.en.md) is about.

**Then volunteer this**: "I'd normally run a numerical gradient check," and write it —

$$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}$$

Central difference, not forward difference — error $O(\epsilon^2)$ instead of $O(\epsilon)$. Saying that line does more for you than getting the gradient right, because **it shows you know how to verify yourself**.

## How to practice

**Don't read it. Write it.** Close this page, type it from scratch in an empty file, then check yourself against `interview_kit.py`'s self-tests. Wherever you stall is where you thought you knew it and didn't.

A harder version: **write the checks before the implementation.** If you can state that "softmax must produce rows summing to one and must be shift-invariant," you actually know what it is. The one whose criteria you can't state is the one where you only memorized the shape of the formula.

## Where to read next

- [Interview basics: most of them ask the same thing](interview-basics.en.md): the conceptual half
- [Multi-head attention](core/multi-head-attention.en.md) · [Normalization](core/normalization.en.md) · [Decoding](core/decoding.en.md)
- [Reference implementations](code/): `interview_kit.py` pairs with this page; `attention_numpy.py` is the full multi-head version
