# Sequence gradients, BPTT, and gates

[中文](recurrent-dynamics.md) · **English**

> Reading time: ~10 min · Level: advanced · Last reviewed: 2026-08

For $h_t=f(W_hh_{t-1}+W_xx_t+b)$, the influence of an early state is a product of Jacobians:

$$\frac{\partial h_T}{\partial h_t}=\prod_{k=t+1}^{T}\text{diag}\!\big(f'(a_k)\big)W_h$$

Typical singular values below one cause exponential decay; values above one cause explosion. Long dependency learning is first a path-length and optimization problem.

Backpropagation Through Time unrolls the shared cell and accumulates each step's contribution to shared parameters. Truncated BPTT cuts the graph to reduce memory and latency, but removes direct credit assignment across the cut.

LSTM creates an additive cell path:

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t, \qquad \frac{\partial c_t}{\partial c_{t-1}}=f_t$$

When $f_t$ remains near one, information and gradients can travel without repeatedly crossing a saturated recurrent nonlinearity. Gates are learnable controllers of information and gradient flow.

Useful checks include gradient norms by distance, performance as dependency length grows, early-token interventions, and gate saturation. See [`../code/sequence_torch.py`](../code/sequence_torch.py).
