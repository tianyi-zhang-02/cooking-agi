# Sequence gradients, BPTT, and gates

[中文](recurrent-dynamics.md) · **English**

> Reading time: ~10 min · Level: advanced · Last reviewed: 2026-08

First replace each matrix with a scalar. Multiplying a gradient by 0.9 for 50 steps leaves about 0.005 of its original size; multiplying by 1.1 gives about 117 times the original. An RNN uses matrix products, but this calculation shows why modest local changes can become a serious optimization problem across a long sequence.

## The core question: why long-range dependencies are hard to learn {#the-core-question-why-long-range-dependencies-are-hard-to-learn}

The state update of a vanilla RNN is $h_t=f(a_t)$, where $a_t=W_hh_{t-1}+W_xx_t+b$. How an early state affects a much later loss is determined by a product of Jacobians:

$$\frac{\partial h_T}{\partial h_t}=\prod_{k=t+1}^{T}\frac{\partial h_k}{\partial h_{k-1}}
=\prod_{k=t+1}^{T}\text{diag}\!\big(f'(a_k)\big)W_h$$

If every Jacobian operator norm is bounded by one common $c<1$, the product norm is bounded by $c^{T-t}$. A largest singular value above 1 at one step does not guarantee explosion: gradient directions and later matrices matter. Long dependencies therefore involve both capacity and a long optimization path. Weights are shared, but Jacobians change with the input and state; they are not one fixed matrix.

## Dissection 1: BPTT is the chain rule after unrolling in time {#dissection-1-bptt-is-the-chain-rule-after-unrolling-in-time}

Backpropagation Through Time simply unrolls the recurrent cell with shared parameters, then accumulates, by ordinary backpropagation, each time step's gradient with respect to the same parameter:

$$\frac{\partial \mathcal L}{\partial W_h}=\sum_t \frac{\partial \mathcal L}{\partial a_t}\frac{\partial a_t}{\partial W_h}$$

Truncated BPTT cuts the computation graph every fixed number of steps, which lowers memory and latency, but the model can no longer assign credit through gradients to anything before the cut.

## Dissection 2: what is really clever about the LSTM is the additive path {#dissection-2-what-is-really-clever-about-the-lstm-is-the-additive-path}

The core update of the cell state:

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$$

Along the direct cell path with gate values held fixed, the elementwise derivative is below. The full derivative also includes paths through the hidden state and gates:

$$\frac{\partial c_t}{\partial c_{t-1}}=f_t$$

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

## GRU: One State, Not Merely One Fewer LSTM Gate {#gru}

A GRU maintains only $h_t$. A reset gate $r_t$ affects the candidate, while an update gate $z_t$ mixes the old state and candidate. In PyTorch's convention:

$$n_t=\tanh(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn})),\qquad
h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}.$$

Larger $z_t$ retains more old state; some texts define it in the opposite direction. With fixed gates, $h_{t-1}=0.8,n_t=-0.2,z_t=0.75$ gives $h_t=0.55$. The total derivative also passes through the gate and candidate, so it is not simply $z_t$.

Implementation detail: $W(r\odot h)$ generally differs from $r\odot(Wh)$. For $h=[1,2],r=[1,0],W=[[1,1],[1,1]]$, they give `[1,1]` and `[3,0]`. This reset-placement difference exists between the original GRU and PyTorch; check it when transferring checkpoints. The [official documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html) gives both formulas.

At equal width, a GRU typically has one fewer gate projection than a standard LSTM. Quality and speed still depend on task, sequence length, and kernels. Both recur through time; an LSTM's more direct cell-state path does not solve every long-range learning problem.

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

**Quick memory**: a long-range dependency in an RNN passes through a product of Jacobians. Clipping can only stop explosion; it cannot recover gradients that have already vanished. An LSTM shortens the effective optimization path with a near-identity cell-state path.

**Interview answer**

> BPTT unrolls the recurrence over time into a deep network, and the gradient of a shared parameter is the sum of the contributions from all time steps. The gradient from an early state to a late loss contains a product of many Jacobians, whose singular values determine exponential decay or growth. An LSTM uses an additive update and the forget gate to give gradients a more direct path.

<details markdown="1">
<summary><b>Deep dive</b>: how do you prove the model really remembered, rather than exploiting a shortcut?</summary>

Besides average accuracy, plot performance and gradient norm against dependency distance, intervene on the early key token, shuffle irrelevant local cues, and check whether the gates stay saturated for long stretches. Only when the prediction changes under a causal intervention on the memory can “solves the task” be separated from “really stores the distant information.”

</details>
</details>
