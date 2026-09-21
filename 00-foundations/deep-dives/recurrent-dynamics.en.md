# Sequence gradients, BPTT, and gates

[中文](recurrent-dynamics.md) · **English**

> Reading time: ~10 min · Level: advanced · Last reviewed: 2026-08

<div class="lesson-recipe advanced">
  <div><span>What we are dissecting</span><strong>Why long-range gradients vanish or explode</strong></div>
  <div><span>Prerequisites</span><strong>chain rule · matrix multiplication · RNN / LSTM forward pass</strong></div>
  <div><span>Main mechanism</span><strong>Jacobian products · BPTT · additive cell-state path</strong></div>
  <div><span>Evidence to demand</span><strong>The model really remembers rather than guessing right by luck</strong></div>
</div>

## Quick learning: the minimal explanation of the long-range gradient problem

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

## The core question: why long-range dependencies are hard to learn

The state update of a vanilla RNN is $h_t=f(a_t)$, where $a_t=W_hh_{t-1}+W_xx_t+b$. How an early state affects a much later loss is determined by a product of Jacobians:

$$\frac{\partial h_T}{\partial h_t}=\prod_{k=t+1}^{T}\frac{\partial h_k}{\partial h_{k-1}}
=\prod_{k=t+1}^{T}\text{diag}\!\big(f'(a_k)\big)W_h$$

If the typical singular values of these matrices are below 1, the gradient decays exponentially with distance; above 1, it explodes all the way. So “the model cannot remember things from long ago” is not only a story about representational capacity. It is first of all a story about **an optimization path that is too long**.

## Dissection 1: BPTT is the chain rule after unrolling in time

Backpropagation Through Time simply unrolls the recurrent cell with shared parameters, then accumulates, by ordinary backpropagation, each time step's gradient with respect to the same parameter:

$$\frac{\partial \mathcal L}{\partial W_h}=\sum_t \frac{\partial \mathcal L}{\partial a_t}\frac{\partial a_t}{\partial W_h}$$

Truncated BPTT cuts the computation graph every fixed number of steps, which lowers memory and latency, but the model can no longer assign credit through gradients to anything before the cut.

## Dissection 2: what is really clever about the LSTM is the additive path

The core update of the cell state:

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$$

Along the direct path,

$$\frac{\partial c_t}{\partial c_{t-1}}=f_t$$

The model can learn $f_t$ close to 1, so that the gradient need not pass through a saturated $\tanh(W_hh)$ at every step. Gates are not a mysterious memory module; they are **learnable controllers of gradient and information flow**.

## Training triage: check these first

- gradient clipping handles explosion; it does not solve vanishing;
- orthogonal initialization makes the recurrent Jacobian closer to norm-preserving at the start;
- a positive forget-gate bias encourages the model to keep memory early in training;
- packing / masking keeps padding from updating the hidden state;
- be explicit about whether state carries over across chunks or is reset for every sample.

## Evidence: how to show it is really remembering rather than guessing right by luck

1. How does accuracy change as the dependency distance grows from 8 to 64?
2. Log the hidden-state gradient norm at every time step: does it fall exponentially with distance?
3. Shuffle the early key tokens: does the output really change?
4. Is the LSTM forget gate saturated at 1 all the time, so that the model can only copy its state?

The corresponding experiment is in [`../code/sequence_torch.py`](../code/sequence_torch.py).

## Self-check

<div class="taste-check advanced">
  <strong>Without looking at the formulas, can you explain:</strong>
  <ol>
    <li>Why does the gradient problem come from a chain of Jacobians rather than from any single time step?</li>
    <li>Once gradient clipping has dealt with explosion, why has it not also dealt with vanishing?</li>
    <li>Which intervention distinguishes “the model really used the early token” from “the data happens to contain a shortcut”?</li>
  </ol>
</div>
