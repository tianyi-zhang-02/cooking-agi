"""Attention five ways, each a small change from the one before:
MHA -> causal -> GQA (grouped K/V heads) -> MLA (one cached latent) -> MLA with absorbed projections.

NumPy only, one sequence (no batch), no RoPE, no dropout. Every step is checked against the step
before it. The notes learn/ml-exercises/attention-ladder(.en).md show these functions line for line
(site/tests/test_attention_ladder.py keeps them in sync).

Run: python3 00-foundations/code/attention_ladder.py
"""
import numpy as np


def softmax(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def split(x, h):                                   # (T, h*d) -> (h, T, d)
    return x.reshape(x.shape[0], h, -1).transpose(1, 0, 2)


def merge(x):                                      # (h, T, d) -> (T, h*d)
    return x.transpose(1, 0, 2).reshape(x.shape[1], -1)


def future(T):                                     # True above the diagonal: a later token
    return np.triu(np.ones((T, T), dtype=bool), k=1)


def mha(x, Wq, Wk, Wv, Wo, h):
    q, k, v = split(x @ Wq, h), split(x @ Wk, h), split(x @ Wv, h)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    return merge(softmax(s) @ v) @ Wo


def causal_mha(x, Wq, Wk, Wv, Wo, h):
    q, k, v = split(x @ Wq, h), split(x @ Wk, h), split(x @ Wv, h)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    s = np.where(future(len(x)), -np.inf, s)       # block the future before the softmax
    return merge(softmax(s) @ v) @ Wo


def causal_gqa(x, Wq, Wk, Wv, Wo, h, g):
    q, k, v = split(x @ Wq, h), split(x @ Wk, g), split(x @ Wv, g)
    k, v = k.repeat(h // g, axis=0), v.repeat(h // g, axis=0)   # K/V head j serves query heads j*h/g ... (j+1)*h/g - 1
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    s = np.where(future(len(x)), -np.inf, s)
    return merge(softmax(s) @ v) @ Wo


def causal_mla(x, Wq, Wdkv, Wuk, Wuv, Wo, h):
    c = x @ Wdkv                                   # (T, d_c): the only thing cached
    q, k, v = split(x @ Wq, h), split(c @ Wuk, h), split(c @ Wuv, h)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    s = np.where(future(len(x)), -np.inf, s)
    return merge(softmax(s) @ v) @ Wo


def causal_mla_absorbed(x, Wq, Wdkv, Wuk, Wuv, Wo, h):
    c = x @ Wdkv                                   # (T, d_c)
    q = split(x @ Wq, h)                           # (h, T, d)
    Wuk_h, Wuv_h = split(Wuk, h), split(Wuv, h)    # (h, d_c, d) each
    s = (q @ Wuk_h.transpose(0, 2, 1)) @ c.T / np.sqrt(q.shape[-1])
    s = np.where(future(len(x)), -np.inf, s)
    return merge(softmax(s) @ c @ Wuv_h) @ Wo


def checks(seed=0, T=5, D=32, h=4, g=2, dc=16):
    """Each step against the one before. Returns {check: largest absolute difference}."""
    rng = np.random.default_rng(seed)
    d = D // h
    x = rng.normal(size=(T, D))
    Wq, Wk, Wv, Wo = (rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(4))
    Wk_g, Wv_g = (rng.normal(size=(D, g * d)) / np.sqrt(D) for _ in range(2))
    Wdkv = rng.normal(size=(D, dc)) / np.sqrt(D)
    Wuk, Wuv = (rng.normal(size=(dc, D)) / np.sqrt(dc) for _ in range(2))
    gap = lambda a, b: float(np.max(np.abs(a - b)))
    # GQA's K/V weights with each group's columns copied for every query head it serves
    shared = lambda W: np.concatenate([W[:, (i // (h // g)) * d:(i // (h // g) + 1) * d] for i in range(h)], axis=1)
    causal = causal_mha(x, Wq, Wk, Wv, Wo, h)
    return {
        "causal: row 0 equals attention over token 0 alone": gap(causal[0], mha(x[:1], Wq, Wk, Wv, Wo, h)[0]),
        "causal: tokens after t do not change row t": gap(causal[:3], causal_mha(x[:3], Wq, Wk, Wv, Wo, h)),
        "GQA with g = h is causal MHA": gap(causal_gqa(x, Wq, Wk, Wv, Wo, h, h), causal),
        "GQA is MHA with each K/V head's weights shared by h/g query heads": gap(
            causal_gqa(x, Wq, Wk_g, Wv_g, Wo, h, g), causal_mha(x, Wq, shared(Wk_g), shared(Wv_g), Wo, h)),
        "MLA is MHA with W_k = W_dkv W_uk and W_v = W_dkv W_uv": gap(
            causal_mla(x, Wq, Wdkv, Wuk, Wuv, Wo, h), causal_mha(x, Wq, Wdkv @ Wuk, Wdkv @ Wuv, Wo, h)),
        "MLA absorbed equals MLA": gap(
            causal_mla_absorbed(x, Wq, Wdkv, Wuk, Wuv, Wo, h), causal_mla(x, Wq, Wdkv, Wuk, Wuv, Wo, h)),
    }


def cache_per_token(h=4, d=8, g=2, dc=16):
    """Numbers kept per token for later tokens to read: K and V for every head, every group, or one latent."""
    return {"MHA": 2 * h * d, "GQA": 2 * g * d, "MLA": dc}


if __name__ == "__main__":
    for name, value in checks().items():
        print(f"{name:70s} {value:.1e}")
        assert value < 1e-12, name
    print("cache per token:", "  ".join(f"{k} {v}" for k, v in cache_per_token().items()))
