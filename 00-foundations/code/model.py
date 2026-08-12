"""Decoder-only Transformer, hand-rolled.

Deliberately avoids every batteries-included shortcut: no nn.MultiheadAttention,
no F.scaled_dot_product_attention, no nn.TransformerEncoderLayer. The only torch
building blocks used are nn.Linear / nn.Embedding / nn.Parameter and raw tensor ops.

Architecture is the modern GPT stack (Llama-flavored):

    tok_emb
    for each block:
        x = x + attn(rmsnorm(x))     # GQA + RoPE, causal
        x = x + mlp(rmsnorm(x))      # SwiGLU
    rmsnorm -> lm_head (weights tied to tok_emb)

Supports incremental decoding through a preallocated KV cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 8          # query heads
    n_kv_head: int = 2       # key/value heads; n_head % n_kv_head == 0 (GQA)
    d_model: int = 256
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    mlp_ratio: float = 8 / 3  # SwiGLU has 3 matrices, so shrink to keep params ~4*d^2

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head

    @property
    def hidden_dim(self) -> int:
        # round to a multiple of 64 the way production configs do
        return int(round(self.d_model * self.mlp_ratio / 64) * 64)


# --------------------------------------------------------------------------- #
# norm
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """LayerNorm without the mean subtraction (and without a bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accumulate the variance in fp32 even when the activations are bf16/fp16
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


# --------------------------------------------------------------------------- #
# rotary position embedding
# --------------------------------------------------------------------------- #
def build_rope_cache(head_dim: int, max_seq_len: int, theta: float):
    """cos/sin tables of shape (max_seq_len, head_dim // 2)."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels by an angle proportional to the position.

    x:        (B, H, T, head_dim)
    cos/sin:  (T, head_dim/2)

    Uses the split-half pairing (channel i pairs with i + d/2), i.e. the
    GPT-NeoX/Llama convention rather than the interleaved one from the paper.
    The two are related by a fixed permutation of the head channels, so they are
    equivalent as long as q and k agree -- but weights are NOT interchangeable.
    """
    x1, x2 = x.float().chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.type_as(x)


# --------------------------------------------------------------------------- #
# kv cache
# --------------------------------------------------------------------------- #
class KVCache:
    """Preallocated per-layer K/V ring-free cache. `pos` is owned by the model."""

    def __init__(self, cfg: Config, batch_size: int, device, dtype=torch.float32):
        shape = (batch_size, cfg.n_kv_head, cfg.max_seq_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.pos = 0  # number of tokens already written

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Append this step's k/v, return the full history including it."""
        t = k.shape[2]
        end = self.pos + t
        self.k[layer][:, :, self.pos:end] = k
        self.v[layer][:, :, self.pos:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


# --------------------------------------------------------------------------- #
# attention
# --------------------------------------------------------------------------- #
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv_head, T, hd) -> (B, n_kv_head * n_rep, T, hd), each kv head duplicated."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_rep = cfg.n_head // cfg.n_kv_head
        hd = cfg.head_dim

        self.wq = nn.Linear(cfg.d_model, cfg.n_head * hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.wo = nn.Linear(cfg.n_head * hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, mask, cache: KVCache | None = None):
        b, t, _ = x.shape
        hd = self.cfg.head_dim

        # (B, T, C) -> (B, H, T, hd)
        q = self.wq(x).view(b, t, self.cfg.n_head, hd).transpose(1, 2)
        k = self.wk(x).view(b, t, self.cfg.n_kv_head, hd).transpose(1, 2)
        v = self.wv(x).view(b, t, self.cfg.n_kv_head, hd).transpose(1, 2)

        # rotate q/k only -- v carries no positional information
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        # grouped-query: every group of n_rep query heads shares one kv head
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # (B, H, T, S)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att.float(), dim=-1).type_as(q)

        y = att @ v                                   # (B, H, T, hd)
        y = y.transpose(1, 2).reshape(b, t, -1)       # concat heads
        return self.wo(y)


# --------------------------------------------------------------------------- #
# mlp
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        h = cfg.hidden_dim
        self.w_gate = nn.Linear(cfg.d_model, h, bias=False)
        self.w_up = nn.Linear(cfg.d_model, h, bias=False)
        self.w_down = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, mask, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask, cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scale the residual-path output projections so the residual stream
        # variance stays ~constant with depth (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()  # tied, so counted once
        return n

    def forward(self, idx, targets=None, cache: KVCache | None = None):
        """idx: (B, T) int64. Returns (logits, loss)."""
        b, t = idx.shape
        start = cache.pos if cache is not None else 0
        assert start + t <= self.cfg.max_seq_len, "sequence longer than max_seq_len"

        cos = self.rope_cos[start:start + t]
        sin = self.rope_sin[start:start + t]

        # mask[i, j] = True means "query i must not see key j"
        # query i sits at absolute position start + i; keys run 0..start+t-1
        q_pos = torch.arange(start, start + t, device=idx.device)[:, None]
        k_pos = torch.arange(start + t, device=idx.device)[None, :]
        mask = k_pos > q_pos  # (T, S), broadcasts over batch and heads

        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, cos, sin, mask, cache)
        x = self.final_norm(x)

        if cache is not None:
            cache.pos += t  # advance once per forward, not once per layer

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """Prefill the prompt once, then decode one token at a time off the cache."""
        self.eval()
        cache = KVCache(self.cfg, idx.shape[0], idx.device, self.tok_emb.weight.dtype)
        logits, _ = self(idx, cache=cache)  # prefill
        out = idx
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :]
            if temperature == 0.0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            out = torch.cat([out, nxt], dim=1)
            logits, _ = self(nxt, cache=cache)  # single-token step
        return out
