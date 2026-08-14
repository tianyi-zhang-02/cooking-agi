"""Multi-head attention in PyTorch, twice, plus the library version.

Three implementations of the same function:

  1. MultiHeadAttention      -- written out, the version to reproduce in an interview
  2. F.scaled_dot_product_attention -- the fused kernel you would actually ship
  3. nn.MultiheadAttention   -- the library module

The script asserts all three agree to ~1e-6, and cross-checks against the pure
NumPy version in attention_numpy.py. Agreement is the point: a hand-rolled
attention that matches `nn.MultiheadAttention` bit-for-nearly-bit is a much
stronger claim than one that merely runs.

Run: python attention_torch.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
class MultiHeadAttention(nn.Module):
    """The from-scratch version. No nn.MultiheadAttention, no SDPA."""

    def __init__(self, d_model, n_head, bias=False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_head = n_head, d_model // n_head
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        def split(proj):                       # (B,T,C) -> (B,H,T,dh)
            return proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        q, k, v = split(self.wq), split(self.wk), split(self.wv)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # (B,H,T,T)
        if mask is not None:
            att = att.masked_fill(mask, float("-inf"))             # before softmax
        att = att.softmax(dim=-1)

        y = att @ v                                                # (B,H,T,dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)           # merge heads
        return self.wo(y), att


def causal_mask(t, device=None):
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)


# --------------------------------------------------------------------------- #
def sdpa_version(mha, x, mask):
    """Same weights, but let PyTorch fuse the attention itself."""
    B, T, C = x.shape
    split = lambda p: p(x).view(B, T, mha.n_head, mha.d_head).transpose(1, 2)
    q, k, v = split(mha.wq), split(mha.wk), split(mha.wv)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask if mask is not None else None)
    return mha.wo(y.transpose(1, 2).contiguous().view(B, T, C))


def library_version(mha, x, mask):
    """nn.MultiheadAttention, loaded with the very same weights.

    Its in_proj_weight is the [Wq; Wk; Wv] stack, which is why the from-scratch
    version above is usually written with three separate Linears -- they are the
    same parameters, just sliced differently.
    """
    d_model = mha.wq.weight.shape[0]
    ref = nn.MultiheadAttention(d_model, mha.n_head, bias=False, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat(
            [mha.wq.weight, mha.wk.weight, mha.wv.weight], dim=0))
        ref.out_proj.weight.copy_(mha.wo.weight)
    out, _ = ref(x, x, x, attn_mask=mask, need_weights=False)
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, H = 2, 7, 32, 4
    x = torch.randn(B, T, D)
    m = causal_mask(T)

    mha = MultiHeadAttention(D, H)
    mha.eval()
    with torch.no_grad():
        mine, att = mha(x, m)
        fused = sdpa_version(mha, x, m)
        lib = library_version(mha, x, m)

    print(f"input {tuple(x.shape)}  ->  output {tuple(mine.shape)}")
    print(f"attention weights {tuple(att.shape)} = (batch, heads, queries, keys)\n")

    d_fused = (mine - fused).abs().max().item()
    d_lib = (mine - lib).abs().max().item()
    print(f"  from-scratch vs F.scaled_dot_product_attention : {d_fused:.2e}")
    print(f"  from-scratch vs nn.MultiheadAttention          : {d_lib:.2e}")
    assert d_fused < 1e-5 and d_lib < 1e-5, "implementations disagree"

    rows = att.sum(-1)
    print(f"  every attention row sums to 1                  : "
          f"{(rows - 1).abs().max().item():.2e}")
    print(f"  weight on any future position                  : "
          f"{att.masked_select(m).max().item():.2e}")

    # the same numbers out of NumPy, with the weights carried across
    try:
        import numpy as np

        from attention_numpy import MultiHeadAttention as NPMHA
        from attention_numpy import causal_mask as np_causal

        npm = NPMHA(D, H)
        npm.Wq = mha.wq.weight.detach().numpy().T      # nn.Linear stores (out, in)
        npm.Wk = mha.wk.weight.detach().numpy().T
        npm.Wv = mha.wv.weight.detach().numpy().T
        npm.Wo = mha.wo.weight.detach().numpy().T
        np_out, _ = npm(x.numpy(), np_causal(T))
        d_np = np.abs(np_out - mine.numpy()).max()
        print(f"  from-scratch torch vs pure NumPy              : {d_np:.2e}")
        assert d_np < 1e-5
    except ImportError:
        print("  (numpy comparison skipped)")

    print("\nall implementations agree.")
