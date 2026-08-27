"""白板上可能让你手写的那几个东西。纯 numpy，无依赖，自带自检。

    python3 interview_kit.py        # 跑全部自检

每个函数下面的注释写的是**这道题的坑在哪**——面试官追问的通常正是那一行。
"""

import numpy as np


# --------------------------------------------------------------------------- #
# 1. softmax
# --------------------------------------------------------------------------- #
def softmax(x, axis=-1):
    # 坑：不减 max 会溢出。logits 到 1000 时 exp(1000) = inf，结果全是 nan。
    # 为什么减了还对：softmax 对输入平移不变，softmax(x + c) == softmax(x)，
    # 分子分母同乘 e^c 约掉。所以减 max 是免费的——不改变结果，只改变数值范围。
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# --------------------------------------------------------------------------- #
# 2. 二分类交叉熵（从 logit 直接算）
# --------------------------------------------------------------------------- #
def bce_with_logits(z, y):
    """稳定形式：max(z,0) - z*y + log(1 + exp(-|z|))

    坑：先 p = sigmoid(z) 再 -y*log(p) 会在 p 下溢到 0 时得到 inf。
    上面这个恒等式对 z 的正负两侧分别化简得到，全程不出现 log(0)。
    log1p(exp(-|z|)) 里指数永远 ≤ 0，也不会溢出。
    """
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def bce_with_logits_grad(z, y):
    # 这就是那道题的答案：sigmoid 的导数被约掉了，只剩 p - y。
    return 1.0 / (1.0 + np.exp(-z)) - y


# --------------------------------------------------------------------------- #
# 3. LayerNorm
# --------------------------------------------------------------------------- #
def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)          # 坑一：有偏方差，除以 N 不是 N-1
    xhat = (x - mu) / np.sqrt(var + eps)         # 坑二：eps 在根号**里面**
    return gamma * xhat + beta                   # 坑三：只沿最后一维，与 batch 无关


# --------------------------------------------------------------------------- #
# 4. 缩放点积注意力 + 因果掩码
# --------------------------------------------------------------------------- #
def causal_mask(t):
    """True = 禁止。严格上三角：位置 i 不能看 j > i。"""
    return np.triu(np.ones((t, t), dtype=bool), k=1)


def attention(q, k, v, mask=None):
    """q,k,v: (..., T, d)。返回 (out, weights)。"""
    d = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d)   # 坑：除 √d，否则 softmax 饱和
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)   # 坑：softmax **之前**加 -inf
    w = softmax(scores, axis=-1)                   # 之后置零会破坏归一化
    return w @ v, w


# --------------------------------------------------------------------------- #
# 5. KV cache 解码循环
# --------------------------------------------------------------------------- #
def decode_with_kv_cache(steps, d, seed=0):
    """演示 cache 的形状与增长。返回每步的输出。

    最大的坑：**解码时不需要因果掩码。** q 只有一个位置（当前 token），
    而 cache 里全部是过去的 k/v——「只能看过去」由构造保证，不需要再掩。
    很多人手写时会习惯性把 mask 加上，那是训练时的事。

    复杂度：不带 cache 生成 T 个 token 是 O(T³)（每步重算整个前缀），
    带 cache 是 O(T²)。cache 的显存是 O(T·d·层数·2)——推理的真正瓶颈。
    """
    rng = np.random.default_rng(seed)
    k_cache = np.zeros((0, d))
    v_cache = np.zeros((0, d))
    outs = []
    for _ in range(steps):
        q = rng.normal(size=(1, d))                       # 当前 token 的 query
        k_new, v_new = rng.normal(size=(1, d)), rng.normal(size=(1, d))
        k_cache = np.concatenate([k_cache, k_new], axis=0)  # 追加，不重算
        v_cache = np.concatenate([v_cache, v_new], axis=0)
        out, _ = attention(q, k_cache, v_cache)            # 无 mask
        outs.append(out)
    return np.concatenate(outs, axis=0), k_cache.shape


# --------------------------------------------------------------------------- #
# 6. top-k / top-p 采样
# --------------------------------------------------------------------------- #
def filter_logits(logits, top_k=None, top_p=None, temperature=1.0):
    """返回过滤后的 logits（被剔除的位置为 -inf）。logits: (V,)"""
    logits = logits / temperature
    if top_k is not None:
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    if top_p is not None:
        order = np.argsort(-logits)                  # 降序
        probs = softmax(logits)[order]
        cum = np.cumsum(probs)
        # 坑：保留**跨过阈值的那一个**，且至少保留 1 个。
        # 写成 cum <= p 会在最大概率就 > p 时把所有 token 全删光。
        keep = cum - probs < top_p
        keep[0] = True
        drop = order[~keep]
        logits = logits.copy()
        logits[drop] = -np.inf
    return logits


# --------------------------------------------------------------------------- #
# 7. 一层 MLP 的前向 + 反向（纯 numpy，不用框架）
# --------------------------------------------------------------------------- #
def mlp_forward(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1
    a1 = np.maximum(z1, 0)            # ReLU
    z2 = (a1 @ W2 + b2).squeeze(-1)   # logit
    return z2, (x, z1, a1)


def mlp_backward(z2, y, cache, W2):
    """返回各参数的梯度。这题是「懂链式法则」和「会调 API」的分界线。"""
    x, z1, a1 = cache
    n = x.shape[0]
    dz2 = bce_with_logits_grad(z2, y) / n          # (N,) —— σ' 约掉了
    dW2 = a1.T @ dz2[:, None]                      # (H,1)
    db2 = dz2.sum()
    da1 = dz2[:, None] @ W2.T                      # (N,H)
    dz1 = da1 * (z1 > 0)                           # ReLU 的导数：坑在 z1>0 不是 a1>0
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return dW1, db1, dW2, db2


# --------------------------------------------------------------------------- #
def _check():
    rng = np.random.default_rng(0)
    ok = lambda name, cond: print(f"  {'✓' if cond else '✗'} {name}")

    # softmax
    x = rng.normal(size=(3, 5)) * 100
    s = softmax(x)
    ok("softmax 每行和为 1 且不溢出", np.allclose(s.sum(-1), 1) and np.isfinite(s).all())
    ok("softmax 平移不变", np.allclose(softmax(x), softmax(x + 12.3)))

    # BCE
    z = np.array([-500.0, -1.0, 0.0, 1.0, 500.0])
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    ok("BCE 在极端 logit 下有限", np.isfinite(bce_with_logits(z, y)).all())
    zs, ys = rng.normal(size=200), rng.integers(0, 2, 200).astype(float)
    p = 1 / (1 + np.exp(-zs))
    naive = -(ys * np.log(p) + (1 - ys) * np.log(1 - p))
    ok("BCE 与朴素实现在安全区一致", np.allclose(bce_with_logits(zs, ys), naive))
    num = (bce_with_logits(zs + 1e-6, ys) - bce_with_logits(zs - 1e-6, ys)) / 2e-6
    ok("BCE 梯度 = p - y", np.allclose(num, bce_with_logits_grad(zs, ys), atol=1e-5))

    # LayerNorm
    h = rng.normal(size=(4, 8)) * 5 + 3
    out = layer_norm(h, np.ones(8), np.zeros(8))
    ok("LN 输出零均值单位方差", np.allclose(out.mean(-1), 0, atol=1e-6)
       and np.allclose(out.var(-1), 1, atol=1e-3))

    # attention
    T, d = 6, 4
    q, k, v = (rng.normal(size=(T, d)) for _ in range(3))
    o, w = attention(q, k, v, causal_mask(T))
    ok("注意力每行和为 1", np.allclose(w.sum(-1), 1))
    ok("因果掩码：上三角全 0", np.allclose(w[np.triu_indices(T, 1)], 0))
    ok("第 1 行只看自己 = 1.0", np.isclose(w[0, 0], 1.0))

    # kv cache
    outs, shape = decode_with_kv_cache(5, d)
    ok("KV cache 长到 5 步", shape == (5, d) and outs.shape == (5, d))

    # top-k / top-p
    lg = np.array([3.0, 2.0, 1.0, 0.5, 0.1])
    ok("top-k 只留 k 个", np.isfinite(filter_logits(lg, top_k=2)).sum() == 2)
    kept = np.isfinite(filter_logits(lg, top_p=0.5)).sum()
    ok("top-p 至少留 1 个且跨过阈值", kept >= 1)
    spike = np.array([100.0, 0.0, 0.0])
    ok("top-p 在单峰下不会全删", np.isfinite(filter_logits(spike, top_p=0.5)).sum() >= 1)

    # mlp 梯度检查
    N, D, H = 6, 3, 5
    X = rng.normal(size=(N, D)); Y = rng.integers(0, 2, N).astype(float)
    W1, b1 = rng.normal(size=(D, H)) * .5, np.zeros(H)
    W2, b2 = rng.normal(size=(H, 1)) * .5, np.zeros(1)
    z2, cache = mlp_forward(X, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = mlp_backward(z2, Y, cache, W2)
    eps, worst = 1e-6, 0.0
    for i in range(D):
        for j in range(H):
            Wp = W1.copy(); Wp[i, j] += eps
            Wm = W1.copy(); Wm[i, j] -= eps
            lp = bce_with_logits(mlp_forward(X, Wp, b1, W2, b2)[0], Y).mean()
            lm = bce_with_logits(mlp_forward(X, Wm, b1, W2, b2)[0], Y).mean()
            worst = max(worst, abs((lp - lm) / (2 * eps) - dW1[i, j]))
    ok(f"MLP 反向 vs 数值梯度（最大偏差 {worst:.2e}）", worst < 1e-6)


if __name__ == "__main__":
    print("interview_kit 自检：")
    _check()
