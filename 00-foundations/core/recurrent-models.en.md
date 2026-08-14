# RNN and LSTM

[中文](recurrent-models.md) · **English**

> Reading time: ~8 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div><span>What we are making</span><strong>A sequence position that carries the past forward</strong></div>
  <div><span>Ingredients</span><strong>current input · previous hidden state</strong></div>
  <div><span>Core technique</span><strong>shared update · LSTM gates · cell state</strong></div>
  <div><span>Most common failure</span><strong>long dependencies and sequential execution</strong></div>
</div>

## In one sentence

An RNN reads left to right with one shared update function and compresses the past into hidden state. An LSTM adds gates that learn what to write, preserve, and expose.

$$h_t=\tanh(W_xx_t+W_hh_{t-1}+b), \qquad y_t=W_oh_t$$

The same cell is unrolled $T$ times; parameters are shared across time.

## Why ordinary RNNs forget

An early state receives gradients through a long product of recurrent Jacobians. Repeated slight contraction drives the gradient toward zero; repeated expansion makes it explode. Both information and gradients must traverse one narrow state path.

## What LSTM adds

$$f_t=\sigma(W_f[x_t;h_{t-1}]+b_f), \qquad i_t=\sigma(W_i[x_t;h_{t-1}]+b_i)$$

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t, \qquad h_t=o_t\odot\tanh(c_t)$$

The forget gate preserves old memory, the input gate writes a candidate, and the output gate controls exposure. The additive cell-state path makes long gradient flow easier when $f_t$ stays near one.

## Defining limitations

1. Time cannot be parallelized because $h_t$ depends on $h_{t-1}$.
2. A long sequence is repeatedly compressed into fixed-size state.
3. Distant positions communicate through a path of length proportional to distance.

Run the NumPy forward pass in [`../code/sequence_numpy.py`](../code/sequence_numpy.py) and the trainable comparison in [`../code/sequence_torch.py`](../code/sequence_torch.py).

## Taste check

<div class="taste-check">
  <strong>You should now be able to explain:</strong>
  <ol>
    <li>Why is long memory an optimization problem, not only a capacity problem?</li>
    <li>What makes the LSTM cell-state path easier for gradients?</li>
    <li>Why does LSTM remain sequential?</li>
  </ol>
</div>

Next: [Seq2Seq](seq2seq.en.md).
