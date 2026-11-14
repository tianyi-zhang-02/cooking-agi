# 同一个 attention 的五种写法：从 MHA 到 MLA

**中文** · [English](attention-ladder.en.md)

> 阅读时间：约 10 分钟 · 类型：手写练习 · 最近审阅：2026-11

先把 multi-head attention 手写一遍，然后每次只改一处：加上 causal mask；让几个 query 头共用一组 K、V（GQA）；最后每个 token 只缓存一个低维向量（MLA）。每改一步都和上一步对答案，看清改了哪几行、省下了什么。

代码只用 NumPy，处理一条序列（不带 batch），不含 RoPE 和 dropout。完整脚本和全部对照在 [`attention_ladder.py`](../../00-foundations/code/attention_ladder.py)，可以直接运行：

```bash
python3 00-foundations/code/attention_ladder.py
```

## 先写好四个公共函数 {#helpers}

五种写法都用到这四个函数。`split` 把投影结果的最后一维切成 h 个头，`merge` 再拼回去；`future` 标出每个 token 之后的位置。

```python
def softmax(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def split(x, h):                                   # (T, h*d) -> (h, T, d)
    return x.reshape(x.shape[0], h, -1).transpose(1, 0, 2)


def merge(x):                                      # (h, T, d) -> (T, h*d)
    return x.transpose(1, 0, 2).reshape(x.shape[1], -1)


def future(T):                                     # 对角线以上为 True：之后的 token
    return np.triu(np.ones((T, T), dtype=bool), k=1)
```

## 每次只改几行 {#five-ways}

五种写法按顺序排列，每一种只在上一种的基础上改几行。网页版会高亮相对上一步改动的行，也可以只看改动。每一步下面写着它和上一步怎么对答案。

<details class="code-drop code-step" markdown="1">
<summary><b>MHA</b> · 起点：每个头有自己的 Q、K、V</summary>

```python
def mha(x, Wq, Wk, Wv, Wo, h):
    q, k, v = split(x @ Wq, h), split(x @ Wk, h), split(x @ Wv, h)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    return merge(softmax(s) @ v) @ Wo
```

缩放用的是每个头的维度 `q.shape[-1]`，不是模型维度。

</details>

<details class="code-drop code-step" markdown="1">
<summary><b>+ causal</b> · 只多一行：softmax 之前把未来位置填成 −inf</summary>

```python
def causal_mha(x, Wq, Wk, Wv, Wo, h):
    q, k, v = split(x @ Wq, h), split(x @ Wk, h), split(x @ Wv, h)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    s = np.where(future(len(x)), -np.inf, s)       # softmax 之前屏蔽未来
    return merge(softmax(s) @ v) @ Wo
```

对照：第 0 行只看得到 token 0，和只拿 token 0 单独算的结果相同；把 t 之后的 token 截掉再算，前 t 行不变。两项的最大误差都是 0。

</details>

<details class="code-drop code-step" markdown="1">
<summary><b>GQA</b> · K、V 只有 g 个头，每个由 h/g 个 query 头共用</summary>

```python
def causal_gqa(x, Wq, Wk, Wv, Wo, h, g):
    q, k, v = split(x @ Wq, h), split(x @ Wk, g), split(x @ Wv, g)
    k, v = k.repeat(h // g, axis=0), v.repeat(h // g, axis=0)   # 第 j 组 K/V 给第 j*h/g 到 (j+1)*h/g - 1 个 query 头用
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    s = np.where(future(len(x)), -np.inf, s)
    return merge(softmax(s) @ v) @ Wo
```

`Wk`、`Wv` 的输出只有 g·d 维。对照：g = h 时和 causal MHA 完全相同；g < h 时，等于把每组 K、V 的权重复制给它负责的 h/g 个头之后的 MHA。最大误差都是 0。

</details>

<details class="code-drop code-step" markdown="1">
<summary><b>MLA</b> · 每个 token 只缓存一个低维向量 c，K、V 都从 c 展开</summary>

```python
def causal_mla(x, Wq, Wdkv, Wuk, Wuv, Wo, h):
    c = x @ Wdkv                                   # (T, d_c)：唯一需要缓存的东西
    q, k, v = split(x @ Wq, h), split(c @ Wuk, h), split(c @ Wuv, h)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    s = np.where(future(len(x)), -np.inf, s)
    return merge(softmax(s) @ v) @ Wo
```

对照：等于把 `Wk` 换成 `Wdkv @ Wuk`、`Wv` 换成 `Wdkv @ Wuv` 的 MHA，最大误差约 7e-16（浮点舍入）。所以这一步换的是 K、V 投影的参数化方式：它们的秩不超过 d_c。省下来的是缓存，attention 的算法没有变。

</details>

<details class="code-drop code-step" markdown="1">
<summary><b>MLA 吸收式</b> · 不展开 K、V：W_uk 并进 query，W_uv 放到加权之后</summary>

```python
def causal_mla_absorbed(x, Wq, Wdkv, Wuk, Wuv, Wo, h):
    c = x @ Wdkv                                   # (T, d_c)
    q = split(x @ Wq, h)                           # (h, T, d)
    Wuk_h, Wuv_h = split(Wuk, h), split(Wuv, h)    # 各为 (h, d_c, d)
    s = (q @ Wuk_h.transpose(0, 2, 1)) @ c.T / np.sqrt(q.shape[-1])
    s = np.where(future(len(x)), -np.inf, s)
    return merge(softmax(s) @ c @ Wuv_h) @ Wo
```

先把 query 投到 latent 空间再和 c 点积，分数和展开式相同；加权求和也先在 c 上做，最后再乘 `Wuv`。对照：和展开式的最大误差约 6e-16。推导见 [MLA 与稀疏注意力](../../00-foundations/deep-dives/latent-and-sparse-attention.md)。

</details>

## 每个 token 要缓存多少？ {#cache}

生成后面的 token 时，前面每个位置都要留下可读的东西。按上面的写法：

| 写法 | 每个 token 缓存 | 本例（h = 4，d = 8，g = 2，d_c = 16） |
| --- | --- | --- |
| MHA、causal | 每个头的 K 和 V：2·h·d | 64 |
| GQA | 每组的 K 和 V：2·g·d | 32 |
| MLA | 一个 latent：d_c | 16 |

这里只核对数学上的等价，不比较速度或显存。还有两点要注意：

- 本例的 MLA 不含 RoPE。RoPE 按位置旋转 key，旋转夹在 `Wuk` 和 query 之间，吸收就不再成立；DeepSeek-V2 因此另加一路带位置的 key，见[为什么 RoPE 要单独处理](../../00-foundations/deep-dives/latent-and-sparse-attention.md#rope-absorption)。
- GQA 里第几个 query 头用第几组 K、V，要和训练时的分组一致。`repeat` 让相邻的头共用一组；换成 `np.tile`，分组就变了（见下面的自检）。

## 自检 {#self-check}

<details markdown="1">
<summary>GQA 里把 <code>k.repeat(h // g, axis=0)</code> 换成 <code>np.tile(k, (h // g, 1, 1))</code>，结果还对吗？</summary>

两种都能跑，但分组不同。`repeat` 得到 `[k0, k0, k1, k1]`，第 i 个 query 头用第 i // (h/g) 组；`tile` 得到 `[k0, k1, k0, k1]`，第 i 个头用第 i % g 组。权重是按其中一种分组训练的，换一种就不是同一个模型了；脚本里「复制给它负责的 h/g 个头」那项对照会失败。

</details>

<details markdown="1">
<summary>把 MLA 的 d_c 设成 2·h·d，会怎样？</summary>

缓存不再比 MHA 小。d_c 也已经不小于模型维度，K、V 的秩不再受 c 限制，能表示的投影和普通 MHA 一样多，只是多算了一步。MLA 省缓存，靠的是 d_c 远小于 2·h·d。

</details>

<details markdown="1">
<summary>吸收式的分数为什么仍然除以 √d，而不是 √d_c？</summary>

吸收只改变矩阵乘法的顺序，算出来的仍是每个头的 q·k，和展开式是同一个数。缩放跟着这个点积所在的头维度 d 走，和中间经过的 latent 维度无关。

</details>
