"""Multi-head attention in pure NumPy. No PyTorch, no autograd, no framework.

This is the version to be able to write on a whiteboard. Every line is either a
reshape, a matmul, or a softmax -- there is nothing else in it.

Run: python attention_numpy.py
"""

import numpy as np


# --------------------------------------------------------------------------- #
def softmax(x, axis=-1):
    """Numerically stable softmax.

    Subtracting the max changes nothing mathematically (exp(a-c)/sum(exp(a-c))
    equals exp(a)/sum(exp(a)) for any c) but keeps exp() from overflowing when
    the scores are large. Forgetting this is the classic interview slip.
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    """q: (B, H, Tq, d)   k, v: (B, H, Tk, d)   mask: True = blocked.

    Returns (out, weights) with out (B, H, Tq, d) and weights (B, H, Tq, Tk).
    """
    d = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d)      # (B, H, Tq, Tk)
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)      # BEFORE the softmax
    w = softmax(scores, axis=-1)
    return w @ v, w


def causal_mask(t):
    """True strictly above the diagonal: position i must not see j > i."""
    return np.triu(np.ones((t, t), dtype=bool), k=1)


# --------------------------------------------------------------------------- #
class MultiHeadAttention:
    """The whole module. Four weight matrices and two reshapes."""

    def __init__(self, d_model, n_head, seed=0):
        assert d_model % n_head == 0, "d_model must divide evenly into heads"
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.d_model, self.n_head = d_model, n_head
        self.d_head = d_model // n_head
        self.Wq, self.Wk, self.Wv, self.Wo = (
            rng.uniform(-s, s, (d_model, d_model)) for _ in range(4))

    def split_heads(self, x):
        """(B, T, d_model) -> (B, H, T, d_head).

        The trick that makes multi-head cheap: one (d_model, d_model) projection,
        then RESHAPE into heads. Mathematically identical to H separate small
        projections, but it is a single GEMM.
        """
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_head, self.d_head).transpose(0, 2, 1, 3)

    def merge_heads(self, x):
        """(B, H, T, d_head) -> (B, T, d_model). The inverse of split_heads."""
        b, h, t, d = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, h * d)

    def __call__(self, x, mask=None):
        q = self.split_heads(x @ self.Wq)
        k = self.split_heads(x @ self.Wk)
        v = self.split_heads(x @ self.Wv)
        out, w = scaled_dot_product_attention(q, k, v, mask)
        return self.merge_heads(out) @ self.Wo, w


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    B, T, D, H = 2, 6, 16, 4
    rng = np.random.default_rng(1)
    x = rng.normal(size=(B, T, D))

    mha = MultiHeadAttention(D, H)
    y, w = mha(x, mask=causal_mask(T))

    print(f"input   {x.shape}")
    print(f"output  {y.shape}   (same shape in, same shape out)")
    print(f"weights {w.shape}   = (batch, heads, queries, keys)\n")

    rows = w.sum(-1)
    print(f"every attention row sums to 1:  max deviation {np.abs(rows - 1).max():.2e}")

    upper = w[:, :, causal_mask(T)]
    print(f"nothing attends to the future:  max weight above diagonal {upper.max():.2e}")

    # first position can only see itself, so its output is one value vector
    print(f"row 0 is one-hot on itself:     {w[0, 0, 0].round(3)}")

    print("\nper-head shapes inside the module:")
    print(f"  after projection + split  (B, H, T, d_head) = "
          f"{mha.split_heads(x @ mha.Wq).shape}")
    print(f"  d_head = d_model / n_head = {D} / {H} = {mha.d_head}")
