# Attention Reference、Shape 与 Mask

**中文** · [English](01-reference-shapes-masks.en.md) · [返回项目](../../projects/01-attention-from-scratch.md)

> 阅读时间：约 5 分钟 · 难度：Intermediate · 时效性：Stable · 最近审阅：2026-08

## 只解决一个问题

怎样建立一个慢但可信的 attention correctness oracle？

```text
S = QKᵀ / √D
P = softmax(S + mask)
O = PV
```

Reference 应显式执行这些步骤，而不是调用 high-level attention API。这样中间 score、mask 后的值和 probability 都能检查。

## Shape contract

```text
Q [B, H, Nq, D]
K [B, Hkv, Nk, D]
V [B, Hkv, Nk, Dv]
O [B, H, Nq, Dv]
```

第一版设置 `H = Hkv`。GQA 中，多个 query heads 映射到较少 KV heads，mapping 必须显式定义。不要依靠 framework broadcasting 猜测语义。

Layout 也必须写清楚。`[B,H,N,D]` 与 `[B,N,H,D]` 的 logical shape 类似，但 stride、连续访问和 kernel mapping 不同。

## Stable softmax

每行使用：

```text
softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

缩放 `1/√D` 在 softmax 前应用。检查每行和接近 1，并对大 magnitude 输入检查 NaN/Inf。

## Mask contract

分开实现和测试：

- causal mask；
- padding/valid-length mask；
- arbitrary additive mask；
- causal + padding。

定义“整行都被 mask”时的行为。一个有限的大负数在不同 dtype 下可能表现不同，不能假设 FP32、FP16、BF16 完全一致。

## Test matrix

- `B=1` 与 `B>1`；
- single/multiple heads；
- `Nq != Nk`；
- `D != Dv`；
- sequence/head dimension 非常小或非 tile 整数倍；
- random、constant、large magnitude；
- no mask、causal、padding、combined。

固定 golden cases 便于 debug，随机 property tests 防止对单一输入过拟合。

## 动手验证

1. 保存 Q、K、V、scores、probabilities 和 output。
2. 让 CPU C++ 每个中间阶段分别与 oracle 对比。
3. 故意制造 off-by-one causal mask，确认测试会失败。
4. Permute heads，确认不同 head 不互相污染。

## 记住

优化版本永远与同一个独立 oracle 对比，不与前一个优化版本对比。否则一个早期错误会被所有后续实现继承。

下一篇：[从 CPU 到 tiled CUDA](02-cpu-to-tiled-cuda.md)
