# Sequence gradients, BPTT, and gates

[中文](recurrent-dynamics.md) · **English**

> Reading time: ~12 min · Level: advanced · Last reviewed: 2026-10

First replace each matrix with a scalar. Multiplying a gradient by 0.9 for 50 steps leaves about 0.005 of its original size; multiplying by 1.1 gives about 117 times the original. An RNN uses matrix products, but this calculation shows why modest local changes can become a serious optimization problem across a long sequence.

## The core question: why long-range dependencies are hard to learn {#the-core-question-why-long-range-dependencies-are-hard-to-learn}

The state update of a vanilla RNN is $h_t=f(a_t)$, where $a_t=W_hh_{t-1}+W_xx_t+b$. How an early state affects a much later loss is determined by a product of Jacobians:

$$J_k=\frac{\partial h_k}{\partial h_{k-1}}=\operatorname{diag}\!\big(f'(a_k)\big)W_h,
\qquad \frac{\partial h_T}{\partial h_t}=J_TJ_{T-1}\cdots J_{t+1}.$$

States are column vectors here. The earliest change passes through the rightmost $J_{t+1}$ first, then moves left. Matrices generally do not commute; reversing the order changes the calculation, even though a scalar example would hide the mistake.

If every Jacobian operator norm is bounded by one common $c<1$, the product norm is bounded by $c^{T-t}$. A largest singular value above 1 at one step does not guarantee explosion: gradient directions and later matrices matter. Long dependencies therefore involve both capacity and a long optimization path. Weights are shared, but Jacobians change with the input and state; they are not one fixed matrix.

## Dissection 1: BPTT is the chain rule after unrolling in time {#dissection-1-bptt-is-the-chain-rule-after-unrolling-in-time}

Backpropagation Through Time simply unrolls the recurrent cell with shared parameters, then accumulates, by ordinary backpropagation, each time step's gradient with respect to the same parameter:

$$\nabla_{W_h}\mathcal L=\sum_t \delta_t h_{t-1}^{\mathsf T},\qquad
\delta_t=\nabla_{a_t}\mathcal L.$$

$\delta_t$ already includes gradients arriving from later steps. The outer product accounts for this cell's direct use of the shared weight; expanding the dependence of $h_{t-1}$ on the weight again here would double-count it. [Pascanu et al. (2013), §§1.1–2.1](https://proceedings.mlr.press/v28/pascanu13.pdf) analyze these contributions through time.

Truncated BPTT cuts the computation graph every fixed number of steps, which lowers memory and latency, but the model can no longer assign credit through gradients to anything before the cut.

### Detaching keeps the value, not the gradient path {#truncated-example}

Use a minimal recurrence: $h_0=1$, $h_t=wh_{t-1}$, $w=0.5$, with a loss only at step 3: $\mathcal L=h_3^2/2$.

<figure class="worked-update" id="truncated-trace" lang="en">
<figcaption>The same forward result, different gradients</figcaption>
<ol>
<li><b>Run the recurrence</b><span>1 → 0.5 → 0.25 → 0.125; loss = 0.0078125.</span></li>
<li><b>Backpropagate through all steps</b><span>All three uses of w contribute. The gradient is 3w⁵ = 0.09375.</span></li>
<li><b>Detach after h₂</b><span>The next step still receives 0.25, now treated as a constant. The gradient is h₃ × h₂ = 0.03125.</span></li>
</ol>
</figure>

Detaching is not resetting memory. The next chunk can use the previous state value, but this loss no longer trains earlier computations through that path.

<details markdown="1">
<summary>Check both gradients with autograd</summary>

```python
import torch

def recurrence_gradient(truncate):
    weight = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    hidden = torch.tensor(1.0, dtype=torch.float64)
    for step in range(3):
        if truncate and step == 2:
            hidden = hidden.detach()
        hidden = weight * hidden
    loss = hidden.square() / 2
    loss.backward()
    return hidden.item(), loss.item(), weight.grad.item()

assert recurrence_gradient(False) == (0.125, 0.0078125, 0.09375)
assert recurrence_gradient(True) == (0.125, 0.0078125, 0.03125)
```

This isolates detaching: there is no optimizer step between chunks. If a real truncated-BPTT loop updates parameters after each chunk, also distinguish the parameters that produced a carried state from those used for the next computation.

</details>

## Dissection 2: what is really clever about the LSTM is the additive path {#dissection-2-what-is-really-clever-about-the-lstm-is-the-additive-path}

The core update of the cell state:

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$$

Along the direct cell path with gate values held fixed, the elementwise derivative is as follows (this is not the full derivative, which also includes paths from $c_{t-1}$ through $h_{t-1}$ into the gates and the candidate $\tilde c_t$):

$$\left.\frac{\partial c_t}{\partial c_{t-1}}\right|_{\text{fixed gates}}
=\operatorname{diag}(f_t).$$

The model can learn $f_t$ close to 1, so that the gradient need not pass through a saturated $\tanh(W_hh)$ at every step. Gates are not a mysterious memory module; they are **learnable controllers of gradient and information flow**.

## Training triage: check these first {#training-triage-check-these-first}

- gradient clipping handles explosion; it does not solve vanishing;
- orthogonal initialization can preserve norms through the recurrent weight matrix, but the activation derivative means the full Jacobian need not preserve them;
- a positive forget-gate bias encourages the model to keep memory early in training;
- packing / masking keeps padding from updating the hidden state;
- be explicit about whether state carries over across chunks or is reset for every sample.

## Evidence: how to show it is really remembering rather than guessing right by luck {#evidence-how-to-show-it-is-really-remembering-rather-than-guessing-right-by-luck}

1. How does accuracy change as the dependency distance grows from 8 to 64?
2. Log the hidden-state gradient norm at every time step: does it fall exponentially with distance?
3. Shuffle the early key tokens: does the output really change?
4. Is the forget gate persistently near 1? Inspect the input gate and candidate too; the forget gate alone does not establish pure state copying.

## GRU: one state, not merely one fewer LSTM gate {#gru}

A GRU maintains only $h_t$. A reset gate $r_t$ affects the candidate, while an update gate $z_t$ mixes the old state and candidate. In PyTorch's convention:

$$n_t=\tanh(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn})),\qquad
h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}.$$

Larger $z_t$ retains more old state; some texts define it in the opposite direction. With fixed gates, $h_{t-1}=0.8,n_t=-0.2,z_t=0.75$ gives $h_t=0.55$. Along the direct $z_t\odot h_{t-1}$ path alone, the derivative with respect to $h_{t-1}$ is $\operatorname{diag}(z_t)$; the total derivative also passes through the gates and candidate, so it is not simply $z_t$.

Implementation detail: $W(r\odot h)$ generally differs from $r\odot(Wh)$. For $h=[1,2],r=[1,0],W=[[1,1],[1,1]]$, they give `[1,1]` and `[3,0]`. This reset-placement difference exists between the original GRU and PyTorch; check it when transferring checkpoints. The [PyTorch 2.8 documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.GRU.html) gives both formulas; the October 2026 [2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html) is unchanged.

At equal width, a GRU typically has one fewer gate projection than a standard LSTM. Quality and speed still depend on task, sequence length, and kernels. Both recur through time and both have a gated additive path (the LSTM's $f_t\odot c_{t-1}$, the GRU's $z_t\odot h_{t-1}$); neither solves every long-range learning problem.

The corresponding experiment is in [`../code/sequence_torch.py`](../code/sequence_torch.py).

## Self-check {#self-check}

<div class="taste-check advanced">
  <strong>Without looking at the formulas, can you explain:</strong>
  <ol>
    <li>Why does the gradient problem come from a chain of Jacobians rather than from any single time step?</li>
    <li>Once gradient clipping has dealt with explosion, why has it not also dealt with vanishing?</li>
    <li>Which intervention distinguishes “the model really used the early token” from “the data happens to contain a shortcut”?</li>
  </ol>
</div>

## Quick learning: the minimal explanation of the long-range gradient problem {#quick-learning-the-minimal-explanation-of-the-long-range-gradient-problem}

<details class="interview" markdown="1">
<summary>BPTT, Jacobian products, and the LSTM additive path</summary>

**Quick memory**: a long-range dependency in an RNN passes through a product of Jacobians. Clipping can only stop explosion; it cannot recover gradients that have already vanished. When the forget gate is near 1, an LSTM's cell-state path is close to identity, so gradients attenuate less per step; the number of time steps is not reduced.

**Interview answer**

> BPTT unrolls the recurrence and sums each step's contribution to shared parameters. Gradients from a late loss to an early state involve an ordered product of Jacobians; attenuation or amplification depends on that product and the direction involved. An LSTM's gated additive path helps preserve signals, without guaranteeing that every long-range task can be learned.

<details markdown="1">
<summary><b>Deep dive</b>: how do you prove the model really remembered, rather than exploiting a shortcut?</summary>

Besides average accuracy, plot performance and gradient norm against dependency distance, intervene on the early key token, shuffle irrelevant local cues, and check whether the gates stay saturated for long stretches. Only when the prediction changes under a causal intervention on the memory can “solves the task” be separated from “really stores the distant information.”

</details>
</details>
