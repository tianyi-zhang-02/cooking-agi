"""Correctness checks for the hand-rolled Transformer.

These are the checks that actually catch bugs in a from-scratch implementation:
causality, cache/no-cache equivalence, RoPE's relative-position property, and
GQA head grouping. Run: python test_model.py
"""

import math

import torch

from model import Config, KVCache, Transformer, apply_rope, build_rope_cache, repeat_kv

torch.manual_seed(0)
DEV = "cpu"  # tests run on cpu so tolerances are deterministic
CFG = Config(vocab_size=64, n_layer=3, n_head=4, n_kv_head=2, d_model=64, max_seq_len=64)


def _model():
    torch.manual_seed(0)
    return Transformer(CFG).to(DEV).eval()


def test_causality():
    """Changing token s must not change logits at any position t < s."""
    m = _model()
    idx = torch.randint(0, CFG.vocab_size, (2, 16), device=DEV)
    base, _ = m(idx)

    s = 9
    idx2 = idx.clone()
    idx2[:, s] = (idx2[:, s] + 1) % CFG.vocab_size
    pert, _ = m(idx2)

    past = (base[:, :s] - pert[:, :s]).abs().max().item()
    future = (base[:, s:] - pert[:, s:]).abs().max().item()
    assert past == 0.0, f"information leaked backwards: {past}"
    assert future > 1e-4, "perturbation had no effect at all -- test is vacuous"
    print(f"  causality:        past delta {past:.2e} | future delta {future:.2e}  OK")


def test_kv_cache_matches_full_forward():
    """Incremental decode must reproduce the one-shot forward pass exactly."""
    m = _model()
    idx = torch.randint(0, CFG.vocab_size, (2, 12), device=DEV)
    full, _ = m(idx)

    cache = KVCache(CFG, idx.shape[0], DEV)
    # prefill on the first 5 tokens, then step the rest one at a time
    step_logits = [m(idx[:, :5], cache=cache)[0]]
    for t in range(5, idx.shape[1]):
        step_logits.append(m(idx[:, t:t + 1], cache=cache)[0])
    inc = torch.cat(step_logits, dim=1)

    err = (full - inc).abs().max().item()
    assert err < 1e-4, f"cache diverges from full forward: {err}"
    assert cache.pos == idx.shape[1], f"cache.pos = {cache.pos}, expected {idx.shape[1]}"
    print(f"  kv cache:         max |full - incremental| = {err:.2e}  OK")


def test_rope_is_relative():
    """<RoPE(q, i), RoPE(k, j)> must depend only on i - j."""
    hd = 32
    cos, sin = build_rope_cache(hd, 64, 10000.0)
    q = torch.randn(1, 1, 1, hd)
    k = torch.randn(1, 1, 1, hd)

    def score(i, j):
        qi = apply_rope(q, cos[i:i + 1], sin[i:i + 1])
        kj = apply_rope(k, cos[j:j + 1], sin[j:j + 1])
        return (qi * kj).sum().item()

    a, b = score(5, 2), score(20, 17)          # both offset 3
    c = score(20, 10)                          # offset 10
    assert abs(a - b) < 1e-4, f"same offset gave different scores: {a} vs {b}"
    assert abs(a - c) > 1e-3, "different offsets collapsed -- test is vacuous"
    # position 0 must be a no-op (angle 0)
    assert torch.allclose(apply_rope(q, cos[0:1], sin[0:1]), q, atol=1e-6)
    print(f"  rope relative:    score(5,2)={a:+.4f}  score(20,17)={b:+.4f}  score(20,10)={c:+.4f}  OK")


def test_gqa_grouping():
    """repeat_kv must duplicate each kv head n_rep times, contiguously."""
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).view(2, 3, 4, 5)
    r = repeat_kv(x, 2)
    assert r.shape == (2, 6, 4, 5)
    for h in range(3):
        assert torch.equal(r[:, 2 * h], x[:, h]) and torch.equal(r[:, 2 * h + 1], x[:, h])

    m = _model()
    a = m.blocks[0].attn
    assert a.wk.weight.shape[0] == CFG.n_kv_head * CFG.head_dim
    assert a.wq.weight.shape[0] == CFG.n_head * CFG.head_dim
    kv_frac = a.wk.weight.numel() / a.wq.weight.numel()
    print(f"  gqa:              {CFG.n_head} q heads / {CFG.n_kv_head} kv heads, "
          f"kv cache is {kv_frac:.0%} of MHA  OK")


def test_rmsnorm():
    m = _model()
    x = torch.randn(4, 7, CFG.d_model) * 3 + 1
    got = m.final_norm(x)
    want = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + CFG.norm_eps) * m.final_norm.weight
    assert torch.allclose(got, want, atol=1e-5)
    # unlike LayerNorm, RMSNorm does not remove the mean
    assert got.mean(-1).abs().max() > 1e-3
    print(f"  rmsnorm:          matches reference, mean not centered ({got.mean().item():+.3f})  OK")


def test_init_loss_is_uniform():
    """A freshly initialised LM should sit at ln(V) nats."""
    m = _model()
    idx = torch.randint(0, CFG.vocab_size, (8, 32), device=DEV)
    _, loss = m(idx[:, :-1], idx[:, 1:])
    expected = math.log(CFG.vocab_size)
    assert abs(loss.item() - expected) < 0.15, f"init loss {loss.item():.3f} vs ln(V)={expected:.3f}"
    print(f"  init loss:        {loss.item():.4f} vs ln(V) = {expected:.4f}  OK")


def test_weight_tying_and_shapes():
    m = _model()
    assert m.lm_head.weight is m.tok_emb.weight
    idx = torch.randint(0, CFG.vocab_size, (3, 11), device=DEV)
    logits, _ = m(idx)
    assert logits.shape == (3, 11, CFG.vocab_size)
    out = m.generate(idx, max_new_tokens=5, temperature=0.0)
    assert out.shape == (3, 16)
    print(f"  shapes/tying:     {m.n_params():,} params "
          f"({m.n_params(non_embedding=True):,} non-embedding)  OK")


if __name__ == "__main__":
    print("running checks on the hand-rolled transformer\n")
    for fn in [
        test_weight_tying_and_shapes,
        test_init_loss_is_uniform,
        test_rmsnorm,
        test_causality,
        test_kv_cache_matches_full_forward,
        test_rope_is_relative,
        test_gqa_grouping,
    ]:
        fn()
    print("\nall checks passed")
