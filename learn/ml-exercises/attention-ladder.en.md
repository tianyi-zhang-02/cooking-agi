# One attention, five ways: from MHA to MLA

[中文](attention-ladder.md) · **English**

> Reading time: ~10 min · Type: implementation practice · Last reviewed: 2026-11

Write multi-head attention once by hand, then change one thing at a time: add a causal mask; let several query heads share one set of K and V (GQA); finally, cache one small vector per token (MLA). Check every step against the one before it, so you can see which lines changed and what was saved.

The code uses only NumPy, handles one sequence (no batch), and leaves out RoPE and dropout. The full script, with every check below, is [`attention_ladder.py`](../../00-foundations/code/attention_ladder.py):

```bash
python3 00-foundations/code/attention_ladder.py
```

## Four shared helpers {#helpers}

All five versions use these. `split` cuts the last dimension of a projection into h heads and `merge` joins them back; `future` marks the positions after each token.

```python
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
```

## A few lines at a time {#five-ways}

The five versions are in order, each a few lines away from the one before. On the site, the lines that changed from the previous step are highlighted, and you can show only those. Under each step is how it is checked against the step before.

<details class="code-drop code-step" markdown="1">
<summary><b>MHA</b> · the start: every head has its own Q, K and V</summary>

```python
def mha(x, Wq, Wk, Wv, Wo, h):
    q, k, v = split(x @ Wq, h), split(x @ Wk, h), split(x @ Wv, h)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    return merge(softmax(s) @ v) @ Wo
```

The scale uses each head's dimension, `q.shape[-1]`, not the model dimension.

</details>

<details class="code-drop code-step" markdown="1">
<summary><b>+ causal</b> · one more line: set future positions to −inf before the softmax</summary>

```python
def causal_mha(x, Wq, Wk, Wv, Wo, h):
    q, k, v = split(x @ Wq, h), split(x @ Wk, h), split(x @ Wv, h)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    s = np.where(future(len(x)), -np.inf, s)       # block the future before the softmax
    return merge(softmax(s) @ v) @ Wo
```

Check: row 0 sees only token 0, so it matches attention over token 0 alone; cutting off the tokens after t leaves the first t rows unchanged. Both differences are 0.

</details>

<details class="code-drop code-step" markdown="1">
<summary><b>GQA</b> · only g heads of K and V, each shared by h/g query heads</summary>

```python
def causal_gqa(x, Wq, Wk, Wv, Wo, h, g):
    q, k, v = split(x @ Wq, h), split(x @ Wk, g), split(x @ Wv, g)
    k, v = k.repeat(h // g, axis=0), v.repeat(h // g, axis=0)   # K/V head j serves query heads j*h/g ... (j+1)*h/g - 1
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    s = np.where(future(len(x)), -np.inf, s)
    return merge(softmax(s) @ v) @ Wo
```

`Wk` and `Wv` now output only g·d values. Check: with g = h it is exactly causal MHA; with g < h it equals MHA whose K and V weights are copied from each group to the h/g heads it serves. Both differences are 0.

</details>

<details class="code-drop code-step" markdown="1">
<summary><b>MLA</b> · cache one small vector c per token; K and V are expanded from it</summary>

```python
def causal_mla(x, Wq, Wdkv, Wuk, Wuv, Wo, h):
    c = x @ Wdkv                                   # (T, d_c): the only thing cached
    q, k, v = split(x @ Wq, h), split(c @ Wuk, h), split(c @ Wuv, h)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    s = np.where(future(len(x)), -np.inf, s)
    return merge(softmax(s) @ v) @ Wo
```

Check: it equals MHA with `Wk` replaced by `Wdkv @ Wuk` and `Wv` by `Wdkv @ Wuv`; the largest difference is about 7e-16 (floating-point rounding). So this step changes how the K and V projections are parameterized: their rank is at most d_c. What it saves is cache; the attention computation is the same.

</details>

<details class="code-drop code-step" markdown="1">
<summary><b>MLA, absorbed</b> · never expand K or V: W_uk moves into the query, W_uv after the weights</summary>

```python
def causal_mla_absorbed(x, Wq, Wdkv, Wuk, Wuv, Wo, h):
    c = x @ Wdkv                                   # (T, d_c)
    q = split(x @ Wq, h)                           # (h, T, d)
    Wuk_h, Wuv_h = split(Wuk, h), split(Wuv, h)    # (h, d_c, d) each
    s = (q @ Wuk_h.transpose(0, 2, 1)) @ c.T / np.sqrt(q.shape[-1])
    s = np.where(future(len(x)), -np.inf, s)
    return merge(softmax(s) @ c @ Wuv_h) @ Wo
```

Projecting the query into the latent space and taking its dot product with c gives the same scores as before; the weighted sum is also taken over c, and `Wuv` is applied last. Check: the largest difference from the expanded version is about 6e-16. The derivation is in [MLA and sparse attention](../../00-foundations/deep-dives/latent-and-sparse-attention.en.md).

</details>

## How much does each token cache? {#cache}

When later tokens are generated, every earlier position has to leave something to read. For the versions above:

| Version | Cached per token | This example (h = 4, d = 8, g = 2, d_c = 16) |
| --- | --- | --- |
| MHA, causal | K and V for every head: 2·h·d | 64 |
| GQA | K and V for every group: 2·g·d | 32 |
| MLA | one latent: d_c | 16 |

These are checks of mathematical equivalence, not speed or memory measurements. Two more things to keep in mind:

- This MLA has no RoPE. RoPE rotates each key by its position, and that rotation sits between `Wuk` and the query, so the absorption no longer holds; DeepSeek-V2 adds a separate positional key for this reason, see [why RoPE needs separate handling](../../00-foundations/deep-dives/latent-and-sparse-attention.en.md#rope-absorption).
- In GQA, which query head reads which group of K and V must match the grouping used in training. `repeat` lets neighbouring heads share a group; `np.tile` would change the grouping (see the self-check below).

## Self-check {#self-check}

<details markdown="1">
<summary>In GQA, is replacing <code>k.repeat(h // g, axis=0)</code> with <code>np.tile(k, (h // g, 1, 1))</code> still correct?</summary>

Both run, but the grouping differs. `repeat` gives `[k0, k0, k1, k1]`, so query head i uses group i // (h/g); `tile` gives `[k0, k1, k0, k1]`, so head i uses group i % g. The weights were trained with one of these groupings, and switching gives a different model; the script's check against copying each group to the h/g heads it serves would fail.

</details>

<details markdown="1">
<summary>What happens if MLA's d_c is set to 2·h·d?</summary>

The cache is no longer smaller than MHA's. d_c is then no smaller than the model dimension, so c no longer limits the rank of K and V: the projections it can express are the same as plain MHA's, with one extra step. MLA saves cache only because d_c is much smaller than 2·h·d.

</details>

<details markdown="1">
<summary>Why is the absorbed score still divided by √d rather than √d_c?</summary>

Absorption only changes the order of the matrix products; the result is still each head's q·k, the same number as in the expanded version. The scale belongs to the head dimension d of that dot product, not to the latent dimension it passes through.

</details>
