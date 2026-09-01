"""The 2017 Transformer, as described in "Attention Is All You Need".

Faithful to the paper, not to what people build today:
  - encoder-decoder (6 + 6 layers), not decoder-only
  - POST-norm:  LayerNorm(x + Sublayer(x))     <- note the norm is OUTSIDE
  - fixed sinusoidal positional encoding added to the embeddings
  - plain multi-head attention: h=8 heads, all of them with their own K and V
  - FFN = Linear -> ReLU -> Linear, d_ff = 4 * d_model
  - embeddings scaled by sqrt(d_model), input/output embeddings weight-tied
  - dropout 0.1 on every sublayer output, on the embedding+PE sum, and on
    the attention weights

Mask convention throughout this file: `mask == True` means BLOCKED
(this matches PyTorch's bool attn_mask). Broadcasting shape is (B, H, T_q, T_k).

Base config from Table 3 of the paper: N=6, d_model=512, h=8, d_ff=2048, P_drop=0.1.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """N *independent* copies -- each layer has its own weights."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# --------------------------------------------------------------------------- #
# 1. attention
# --------------------------------------------------------------------------- #
def attention(q, k, v, mask=None, dropout: nn.Dropout | None = None):
    """Scaled dot-product attention -- eq. (1) of the paper.

    q: (B, H, T_q, d_k)
    k: (B, H, T_k, d_k)
    v: (B, H, T_k, d_v)
    -> out: (B, H, T_q, d_v),  attn: (B, H, T_q, T_k)
    """
    d_k = q.size(-1)

    # (B,H,T_q,d_k) @ (B,H,d_k,T_k) -> (B,H,T_q,T_k): every query dotted with every key
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    # why / sqrt(d_k): for unit-variance q,k the dot product of d_k terms has
    # variance d_k, and softmax over large-magnitude logits saturates -> tiny gradients.

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn = scores.softmax(dim=-1)  # each query's row sums to 1 over the keys
    if dropout is not None:
        attn = dropout(attn)

    return attn @ v, attn  # weighted average of the value vectors


class MultiHeadAttention(nn.Module):
    """h attention heads in parallel on d_k = d_model / h dims each -- eq. (2).

    Cheap trick that makes this look like one matmul: instead of h separate
    (d_model, d_k) projections, use one (d_model, d_model) projection and
    *reshape* the output into h heads. Mathematically identical, one GEMM.
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h        # 64 in the base model
        self.h = h
        self.w_q, self.w_k, self.w_v, self.w_o = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        self.attn = None               # kept around so we can look at it later

    def forward(self, query, key, value, mask=None):
        # NOTE query/key/value are three *different* tensors in cross-attention.
        # In self-attention the caller passes the same tensor three times.
        b = query.size(0)

        def split_heads(x, proj):
            # (B, T, d_model) -> (B, T, h, d_k) -> (B, h, T, d_k)
            return proj(x).view(b, -1, self.h, self.d_k).transpose(1, 2)

        q = split_heads(query, self.w_q)
        k = split_heads(key, self.w_k)
        v = split_heads(value, self.w_v)

        x, self.attn = attention(q, k, v, mask, self.dropout)

        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d_model): concat the heads
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.w_o(x)


# --------------------------------------------------------------------------- #
# 2. position-wise feed-forward
# --------------------------------------------------------------------------- #
class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, xW1 + b1)W2 + b2 -- eq. (3).

    "Position-wise" = the same MLP applied independently at every position;
    it is the only place in the block where positions do NOT talk to each other.
    Attention mixes across time, the FFN thinks within a position.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)   # 512 -> 2048
        self.w_2 = nn.Linear(d_ff, d_model)   # 2048 -> 512
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# --------------------------------------------------------------------------- #
# 3. embeddings + sinusoidal positions
# --------------------------------------------------------------------------- #
class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # the sqrt(d_model) scale-up matters: embeddings are init'd ~N(0, 1/d),
        # so without it the positional encoding (amplitude 1) would drown them out
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
       PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Fixed, not learned. Each channel is a sinusoid; wavelengths form a geometric
    series from 2*pi to 10000*2*pi, so the vector is a multi-scale binary-ish
    clock. The paper's motivation: PE(pos+k) is a linear function of PE(pos)
    (a rotation by a fixed angle), so relative offsets are learnable.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()            # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)   # even channels
        pe[:, 1::2] = torch.cos(pos * div)   # odd channels
        self.register_buffer("pe", pe.unsqueeze(0))                 # (1, max_len, d_model)

    def forward(self, x):
        # ADDED to the embedding, not concatenated -- position rides along in the
        # same d_model dims the token identity uses
        return self.dropout(x + self.pe[:, : x.size(1)].requires_grad_(False))


# --------------------------------------------------------------------------- #
# 4. the post-norm residual wrapper
# --------------------------------------------------------------------------- #
class SublayerConnection(nn.Module):
    """LayerNorm(x + Dropout(Sublayer(x))) -- section 3.1, "post-norm".

    This is the single biggest difference from every modern implementation,
    which does x + Sublayer(LayerNorm(x)) instead. Post-norm puts a LayerNorm
    on the residual highway itself, which is why the original post-norm setup NEEDS the
    warmup schedule: without warmup the early gradients through the stacked
    norms blow up and training diverges.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


# --------------------------------------------------------------------------- #
# 5. encoder / decoder layers
# --------------------------------------------------------------------------- #
class EncoderLayer(nn.Module):
    """self-attention -> feed-forward. Two sublayers."""

    def __init__(self, d_model, self_attn, ff, dropout):
        super().__init__()
        self.self_attn, self.ff = self_attn, ff
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, src_mask):
        # bidirectional: every source position sees every other (minus padding)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        return self.sublayer[1](x, self.ff)


class DecoderLayer(nn.Module):
    """masked self-attention -> cross-attention -> feed-forward. Three sublayers."""

    def __init__(self, d_model, self_attn, cross_attn, ff, dropout):
        super().__init__()
        self.self_attn, self.cross_attn, self.ff = self_attn, cross_attn, ff
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 1. look at what has been generated so far -- causally masked
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2. look at the source. THE key line: queries come from the decoder,
        #    keys and values from the encoder output ("memory"). This is the
        #    only place the two towers touch.
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.ff)


class Encoder(nn.Module):
    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x  # (B, S, d_model) -- the "memory"


class Decoder(nn.Module):
    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


# --------------------------------------------------------------------------- #
# 6. masks
# --------------------------------------------------------------------------- #
def pad_mask(seq, pad_idx: int):
    """(B, T) -> (B, 1, 1, T). True where the token is padding => blocked."""
    return (seq == pad_idx)[:, None, None, :]


def causal_mask(t: int, device):
    """(1, 1, T, T). True strictly above the diagonal => position i can't see j>i."""
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)[None, None]


# --------------------------------------------------------------------------- #
# 7. the whole thing
# --------------------------------------------------------------------------- #
class VanillaTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, n=6, d_model=512, d_ff=2048, h=8,
                 dropout=0.1, pad_idx=0, share_embeddings=True):
        super().__init__()
        self.pad_idx = pad_idx
        attn = MultiHeadAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        pe = PositionalEncoding(d_model, dropout)

        # copy.deepcopy everywhere: each layer gets fresh, independent weights
        self.encoder = Encoder(EncoderLayer(d_model, copy.deepcopy(attn),
                                            copy.deepcopy(ff), dropout), n)
        self.decoder = Decoder(DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn),
                                            copy.deepcopy(ff), dropout), n)
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), copy.deepcopy(pe))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), copy.deepcopy(pe))
        self.generator = nn.Linear(d_model, tgt_vocab)

        # section 3.4: share the two embedding matrices and the pre-softmax linear
        if share_embeddings:
            assert src_vocab == tgt_vocab
            self.tgt_embed[0].lut.weight = self.src_embed[0].lut.weight
            self.generator.weight = self.src_embed[0].lut.weight

        # Xavier init on every matrix (what the reference implementation does)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt):
        """src: (B, S) source ids.  tgt: (B, T) decoder INPUT ids (already shifted right).

        Returns logits (B, T, tgt_vocab).
        """
        src_m = pad_mask(src, self.pad_idx)                                   # (B,1,1,S)
        # decoder self-attn is blocked by EITHER padding OR being in the future
        tgt_m = pad_mask(tgt, self.pad_idx) | causal_mask(tgt.size(1), tgt.device)
        memory = self.encode(src, src_m)                                      # (B,S,d_model)
        out = self.decode(memory, src_m, tgt, tgt_m)                          # (B,T,d_model)
        return self.generator(out)

    @torch.no_grad()
    def greedy_decode(self, src, max_len: int, bos_idx: int):
        """Encode once, then extend the target one token at a time.

        No KV cache here -- the paper had none, and re-running the whole decoder
        each step is O(T^2) forwards. This is the O(T^3) version people actually
        wrote in 2017; caching is a later engineering addition.
        """
        self.eval()
        src_m = pad_mask(src, self.pad_idx)
        memory = self.encode(src, src_m)
        ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            tgt_m = causal_mask(ys.size(1), ys.device)
            out = self.decode(memory, src_m, ys, tgt_m)
            nxt = self.generator(out[:, -1]).argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
        return ys


# --------------------------------------------------------------------------- #
# 8. the training recipe (section 5.3)
# --------------------------------------------------------------------------- #
def noam_lr(step: int, d_model: int, warmup: int = 4000, factor: float = 1.0) -> float:
    """lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)

    Linear warmup then 1/sqrt(step) decay. Not optional for post-norm:
    it is the thing that keeps the first few thousand steps from diverging.
    """
    step = max(step, 1)
    return factor * d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def make_optimizer(model, d_model=512, warmup=4000, factor=1.0):
    """Adam with beta2=0.98 and eps=1e-9, exactly as in the paper.

    base lr MUST be 1.0: LambdaLR computes lr = base_lr * lambda(step), so a
    base of 0 pins the learning rate at 0 forever and the model quietly does
    not train at all.
    """
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: noam_lr(s, d_model, warmup, factor)
    )
    return opt, sched


def loss_fn(logits, target, pad_idx=0, smoothing=0.1):
    """Cross-entropy with label smoothing eps_ls = 0.1 (section 5.4).

    Hurts perplexity (the model is forced to be less confident) but improves
    BLEU. Padding positions are excluded.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1),
        ignore_index=pad_idx, label_smoothing=smoothing,
    )
