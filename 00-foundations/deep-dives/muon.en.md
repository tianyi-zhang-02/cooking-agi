# Muon: updating a matrix

[中文](muon.md) · **English**

> Checked: 2026-10-21 · Prerequisite: [momentum and AdamW](optimizers.en.md) · For a first pass, read the first 2 sections and the comparison table.

A linear layer has a weight matrix and a gradient matrix. AdamW adapts updates using each coordinate's history; Muon preserves matrix structure and changes the geometry of the update. **It changes parameter updates, not the loss or the attention computation.**

## Start with a 2 × 2 example

Suppose the momentum update is

$$M=\begin{bmatrix}6&0\\0&2\end{bmatrix}.$$

Both directions receive an update, but the first has 3 times the magnitude. Dividing the matrix by its norm shrinks both together; the ratio stays 3:1.

The idealized Muon operation differs. Write the singular value decomposition as $U\Sigma V^\top$, then replace each nonzero singular value with 1. Keeping only the corresponding singular vectors $U_r,V_r$ gives $U_rV_r^\top$. In this full-rank example the result is the identity, with a 1:1 ratio. Left and right singular vectors are preserved; the entries are not all set to 1. The [original author's explanation](https://kellerjordan.github.io/posts/muon/) develops this orthogonalization view.

| Operation | Result here | Directional ratio |
| --- | --- | --- |
| None | $\operatorname{diag}(6,2)$ | 3:1 |
| Frobenius normalization | $\operatorname{diag}(6,2)/\sqrt{40}$ | 3:1 |
| Idealized orthogonalization | $\operatorname{diag}(1,1)$ | 1:1 |

This explains the operation but does not establish that it improves training. A small singular value might describe a useful direction or mostly noise. Equalizing magnitudes does not by itself make the gradient more accurate.

For $\operatorname{diag}(2,0)$, our reference instead gives $\operatorname{diag}(1,0)$, not the identity: the zero direction stays zero. Multiplying all the returned $U,V$ vectors can include arbitrary completions associated with zero singular values. The code below uses a threshold to decide which singular values to retain.

## What happens in one update?

```text
loss → gradient matrix → momentum → approximate orthogonalization → scaling → weight update
                                                                                 ↑
                                                                       decoupled weight decay
```

A simplified version without Nesterov is

$$M_t=\mu M_{t-1}+G_t,$$
$$O_t\approx\operatorname{Polar}(M_t),\qquad
W_t=(1-\eta_t\lambda)W_{t-1}-\eta_t s(m,n)O_t.$$

Here $s(m,n)$ depends only on the matrix row and column counts. $\mu$ is momentum, $\lambda$ is weight decay, and $s$ is the implementation's scaling factor. Actual code may use Nesterov or an EMA momentum convention; mixing equations from different implementations does not establish step-by-step equivalence.

The [PyTorch Muon documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) lists these choices separately, including learning-rate adjustments. Record the library version and configuration rather than treating one page's defaults as a universal definition.

## What does Newton–Schulz approximate?

Running SVD every step is expensive. Muon instead uses matrix-multiplication-based Newton–Schulz iterations on the normalized update. Starting with $X_0=M/(\lVert M\rVert_F+\epsilon)$, one family of iterations is

$$A_k=X_kX_k^\top,\qquad
X_{k+1}=aX_k+(bA_k+cA_k^2)X_k.$$

The operation acts on the **update**, not on the weight matrix $W$. A rectangular matrix also cannot satisfy both identity conditions: full column rank can give $O^\top O=I$, while full row rank can give $OO^\top=I$.

An important detail in the [pinned original implementation](https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py): the quintic coefficients `(3.4445, -4.775, 2.0315)` prioritize useful behavior within a small iteration budget. **Repeating them does not guarantee exact convergence to the ideal polar factor.** Substitute a singular value of 1:

$$3.4445-4.775+2.0315=0.701.$$

One is not even a fixed point of that polynomial. “More iterations must converge exactly to 1” therefore cannot be the explanation. Approximate orthogonalization is not exact orthogonalization at arbitrary iteration counts.

<details markdown="1">
<summary>What happens to 6 and 2 after five steps?</summary>

For a diagonal matrix, we can check the polynomial one entry at a time. This calculation uses Python floats to illustrate the transformation, not the BF16 rounding of a production implementation:

```python
import math

diagonal = [6.0, 2.0]
scale = math.sqrt(sum(value * value for value in diagonal)) + 1e-7
transformed = [value / scale for value in diagonal]
for step in range(5):
    transformed = [3.4445 * value - 4.775 * value**3 + 2.0315 * value**5
                   for value in transformed]
assert math.isclose(transformed[0], 0.7530, abs_tol=1e-4)
assert math.isclose(transformed[1], 1.1337, abs_tol=1e-4)
```

The ratio is no longer 3:1, but it is not exactly 1:1 either. The SVD reference explains the geometry; the finite-step iteration is a specific algorithm and does not inherit every property of that reference.

</details>

### Check the geometry with SVD

This is an educational exact reference, not an efficient Muon implementation. It omits momentum and weight decay, requires PyTorch, and accepts real matrices only. For rank-deficient inputs it preserves the null space rather than inventing directions for zero singular values.

```python
import torch


def polar_reference(matrix):
    try:
        inferred = torch.as_tensor(matrix)
    except RuntimeError:
        inferred = torch.as_tensor(matrix, dtype=torch.float64)
    if torch.is_complex(inferred):
        raise ValueError("expected a real matrix, not complex values")
    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    if matrix.ndim != 2 or matrix.numel() == 0:
        raise ValueError("expected a nonempty real matrix")
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix must be finite")
    scale = matrix.abs().max()
    if scale == 0:
        return torch.zeros_like(matrix)
    left, singular, right = torch.linalg.svd(matrix / scale, full_matrices=False)
    threshold = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular.max()
    retained = (singular > threshold).to(matrix.dtype)
    return (left * retained) @ right


direction = polar_reference([[6.0, 0.0], [0.0, 2.0]])
assert torch.allclose(direction, torch.eye(2, dtype=torch.float64))
assert torch.count_nonzero(polar_reference(torch.zeros(2, 3))) == 0
```

Try `[[1, 1], [0, 1]]`: the result is not the elementwise sign. Near-zero singular values depend on precision and thresholds; this reference's threshold convention is not a compatibility guarantee for a production optimizer.

## Why isn't this just changing an optimizer name?

**First, scaling.** For a full-rank $m\times n$ matrix, the idealized $O$ has $\min(m,n)$ singular values equal to 1, so

$$\operatorname{RMS}(O)=\sqrt{\frac{\min(m,n)}{mn}}
=\frac{1}{\sqrt{\max(m,n)}}.$$

With the same learning rate, larger matrices receive smaller per-entry RMS updates. The [scaling paper](https://arxiv.org/abs/2502.16982) uses $0.2\sqrt{\max(m,n)}$ to match an empirical AdamW update scale. That constant is not a theorem; the original shape adjustment and RMS matching are different configurations.

For example, the `original` option in [PyTorch v2.9.0](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/_muon.py#L74-L86) uses $\sqrt{\max(1,m/n)}$. For ideal full-rank directions, before multiplying by the learning rate (reference values, not a claim that finite-step NS is exactly orthogonal):

| Matrix shape | Without scaling | After `original` | After RMS matching |
| --- | ---: | ---: | ---: |
| $8\times2$ | $1/\sqrt8\approx0.3536$ | 0.7071 | 0.2 |
| $2\times8$ | $1/\sqrt8\approx0.3536$ | 0.3536 | 0.2 |

As of October 2026, the stable [PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html) adds a third option, `spectral_unclamped`: the factor is $\sqrt{m/n}$, no longer clamped below at 1. The $8\times2$ row stays at 0.7071, while $2\times8$ becomes $0.3536\times0.5\approx0.1768$. Leaving the argument unset still means `original`.

In the [v2.9.0 implementation](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/_muon.py#L317-L320), weight decay uses the base learning rate, not the shape-adjusted rate. With $\eta=0.1,\lambda=0.2$, the decay term is $0.98W$; doubling the update direction does not change it to $0.96W$.

**Second, parameter groups.** The classical hybrid uses Muon for hidden matrices and AdamW for embeddings, output layers, biases, and normalization parameters. Embeddings are themselves matrices, so `ndim == 2` is not a sufficient selector. Deduplicate tied embedding/output weights by object identity to avoid two updates to the same parameter. The [original repository's usage guide](https://github.com/KellerJordan/Muon) distinguishes these groups.

**Finally, distributed execution.** Orthogonalizing matrix shards separately generally differs from orthogonalizing the full matrix before slicing. If the framework flattens or shards parameters, establish how logical matrices are reconstructed and what communication is needed; do not copy an elementwise AdamW sharding scheme unchanged.

<details id="distributed-step" markdown="1">
<summary>Why can't two devices each process one row?</summary>

Consider this small, constructed momentum matrix:

$$M=\begin{bmatrix}1&1\\0&1\end{bmatrix}.$$

Give each device one row. Ideal orthogonalization of each row simply normalizes its length. Putting those rows back together does not make them orthogonal. Processing the whole matrix gives a different result:

| Operation | Assembled update | Dot product of its rows |
| --- | --- | --- |
| Process each row separately | $\begin{bmatrix}1/\sqrt2&1/\sqrt2\\0&1\end{bmatrix}$ | $1/\sqrt2$ |
| Process the whole matrix | $\frac1{\sqrt5}\begin{bmatrix}2&1\\-1&2\end{bmatrix}$ | $0$ |

There is no network transfer or low-precision rounding in this example. The operations differ even with exact SVD. Using `polar_reference` from above:

```python
import math

momentum = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
whole_update = polar_reference(momentum)
row_updates = torch.cat([polar_reference(row) for row in momentum.split(1, dim=0)])
expected_whole = torch.tensor([[2.0, 1.0], [-1.0, 2.0]], dtype=torch.float64) / math.sqrt(5)
torch.testing.assert_close(whole_update, expected_whole)
assert math.isclose(torch.dot(row_updates[0], row_updates[1]).item(), 1 / math.sqrt(2))
assert not torch.allclose(row_updates, whole_update)
```

[Moonlight v1 §2.3, Algorithm 1](https://arxiv.org/html/2502.16982v1#S2.SS3) describes one implementation that retains full-matrix computation:

1. Reduce-scatter gradients across DP, then apply local momentum to each local shard.
2. Gather the required momentum-adjusted shards into a full matrix and run Newton–Schulz.
3. Keep the local update slice, update the local parameters, then all-gather the updated parameters.

**Orthogonalization itself is computation; the extra communication is the step-2 gather that rebuilds the full matrix.** TP needs additional handling. The report's communication ratios and negligible latency overhead belong to its configuration and cluster, not every training run. Measure reconstruction, temporary buffers, and overlap in your setup; no multi-GPU benchmark was run here.

</details>

## What is a meaningful AdamW comparison?

| Question | Measure | Common mistake |
| --- | --- | --- |
| Does it learn faster? | Validation loss at matched data/token budgets; reasonable tuning for both | Tune Muon but leave AdamW at defaults |
| Does it save time? | Wall-clock time to matched quality | Count steps but ignore matrix operations and communication |
| Does it save memory? | Optimizer state, master weights, gradients, activations, and temporary buffers separately | Call one fewer moment buffer a halving of total memory |
| Does it help fine-tuning? | Fixed base model, retained abilities, and target tasks | Extrapolate pretraining gains directly to small-data SFT/RL |
| Can it resume? | Momentum, parameter groups, scheduler, step count, and RNG state | Load weights alone and call it seamless continuation |

A practical starting experiment fixes a small model and data order, gives both optimizers the same number of learning-rate/weight-decay trials, and reports repeated-run validation curves, step time, total time, and peak memory. This is an experiment design; **this site has not run it to establish that Muon beats AdamW**.

## What needs qualification in 2026?

- “Embeddings must always use AdamW” is too broad. October 2026's [AF-Muon](https://arxiv.org/abs/2610.01395) studies specialized tied-embedding updates. It is a new design, not permission to apply classical Muon unchanged. Only its abstract and problem setting have been checked here; its experiments have not been reproduced.
- “Switch to Muon and scale up directly” is too optimistic. The [Kimi K2 report §2.1](https://arxiv.org/abs/2507.20534) (July 2025) found that in a run with 9B activated and 53B total parameters, plain Muon drove the maximum attention logit past 1,000 quickly. Its MuonClip checks each head's maximum logit after every update and, above a threshold τ (100 for K2), shrinks that head's query / key projection weights. K2 used it to pre-train on 15.5T tokens and reports no loss spike. By 2026 Muon is in production-scale training of this kind: the [DeepSeek-V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) also states that it uses Muon (embeddings and some other parameters stay on AdamW; see the [V4 note](../model-families/deepseek-v4.en.md)). At scale, monitor new failure signals such as logit growth.
- “Muon guarantees convergence” is also too broad. An August 2026 [convergence analysis](https://arxiv.org/abs/2608.04607) examines counterexamples and error conditions in specific stochastic optimization problems. That establishes neither universal failure in LLM training nor success without assumptions.

Understand the update geometry, then ask which operation a newer variant changes. That is more useful than memorizing claims that an optimizer is obsolete. Continue with [training memory](precision-and-memory.en.md) and [distributed training](../../06-systems/distributed-training.en.md) to place the costs in a complete training run.
