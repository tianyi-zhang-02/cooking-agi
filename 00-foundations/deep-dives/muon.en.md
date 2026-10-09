# Muon: updating a matrix

[中文](muon.md) · **English**

> Checked: 2026-10-09 · Prerequisite: [momentum and AdamW](optimizers.en.md) · For a first pass, read the first 2 sections and the comparison table.

A linear layer has a weight matrix and a gradient matrix. AdamW adapts updates using each coordinate's history; Muon preserves matrix structure and changes the geometry of the update. **It changes parameter updates, not the loss or the attention computation.**

## Start with a 2 × 2 example

Suppose the momentum update is

$$M=\begin{bmatrix}6&0\\0&2\end{bmatrix}.$$

Both directions receive an update, but the first has 3 times the magnitude. Dividing the matrix by its norm shrinks both together; the ratio stays 3:1.

The idealized Muon operation differs. Write the singular value decomposition as $U\Sigma V^\top$, then remove the magnitudes of the nonzero singular values, leaving $UV^\top$. In this full-rank example the result is the identity, with a 1:1 ratio. Left and right singular vectors are preserved; the entries are not all set to 1. The [original author's explanation](https://kellerjordan.github.io/posts/muon/) develops this orthogonalization view.

| Operation | Result here | Directional ratio |
| --- | --- | --- |
| None | $\operatorname{diag}(6,2)$ | 3:1 |
| Frobenius normalization | $\operatorname{diag}(6,2)/\sqrt{40}$ | 3:1 |
| Idealized orthogonalization | $\operatorname{diag}(1,1)$ | 1:1 |

This explains the operation, not a guarantee of better learning. A small singular value might describe a useful direction or mostly noise. Equalizing magnitudes does not make the gradient more accurate for free.

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

An important detail in the [original implementation](https://github.com/KellerJordan/Muon/blob/master/muon.py): the common quintic coefficients `(3.4445, -4.775, 2.0315)` prioritize useful behavior within a small iteration budget. **Repeating them does not guarantee exact convergence to $UV^\top$.** Substitute a singular value of 1:

$$3.4445-4.775+2.0315=0.701.$$

One is not even a fixed point of that polynomial. “More iterations must converge exactly to 1” therefore cannot be the explanation. Approximate orthogonalization is not exact orthogonalization at arbitrary iteration counts.

### Check the geometry with SVD

This is an educational exact reference, not an efficient Muon implementation. It omits momentum and weight decay, requires PyTorch, and accepts real matrices only. For rank-deficient inputs it preserves the null space rather than inventing directions for zero singular values.

```python
import torch


def polar_reference(matrix):
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

**Second, parameter groups.** The classical hybrid uses Muon for hidden matrices and AdamW for embeddings, output layers, biases, and normalization parameters. Embeddings are themselves matrices, so `ndim == 2` is not a sufficient selector. Deduplicate tied embedding/output weights by object identity to avoid two updates to the same parameter. The [original repository's usage guide](https://github.com/KellerJordan/Muon) distinguishes these groups.

**Finally, distribution.** Orthogonalizing matrix shards separately generally differs from orthogonalizing the full matrix before slicing. If the framework flattens or shards parameters, establish how logical matrices are reconstructed and what communication is needed; do not copy an elementwise AdamW sharding scheme unchanged.

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
- “Muon guarantees convergence” is also too broad. An August 2026 [convergence analysis](https://arxiv.org/abs/2608.04607) examines counterexamples and error conditions in specific stochastic optimization problems. That establishes neither universal failure in LLM training nor success without assumptions.

Understand the update geometry, then ask which operation a newer variant changes. That is more useful than memorizing claims that an optimizer is obsolete. Continue with [training memory](precision-and-memory.en.md) and [distributed training](../../06-systems/distributed-training.en.md) to place the costs in a complete training run.
