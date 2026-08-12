# 手搓：能跑的实现

**中文** · [English](README.en.md)

这一章的所有说法都有对应的可执行代码。两版 Transformer 都**不调**
`nn.MultiheadAttention` 和 `F.scaled_dot_product_attention`，只用
`nn.Linear` / `nn.Embedding` / `nn.Parameter` 加裸张量运算。

只需要 PyTorch，CPU 就能跑完全部。

```bash
pip install torch
```

## 文件

| 文件 | 做什么 | 大概耗时 |
| --- | --- | --- |
| [`why_nonlinear.py`](why_nonlinear.py) | XOR 上跑三个模型，ASCII 画出决策边界和隐藏层空间 | ~20 秒 |
| [`make_figures.py`](make_figures.py) | 训练同样的模型，把 [`../assets/`](../assets/) 里的 SVG 图重新生成 | ~30 秒 |
| [`vanilla.py`](vanilla.py) | 2017 原版 encoder-decoder，严格照论文 | — |
| [`vanilla_demo.py`](vanilla_demo.py) | 张量形状追踪 → 训练「反转序列」→ 打印交叉注意力矩阵 | ~1 分钟 |
| [`model.py`](model.py) | 现代 decoder-only：RMSNorm + RoPE + GQA + SwiGLU + KV cache | — |
| [`test_model.py`](test_model.py) | 正确性验证（因果性、cache 等价、RoPE 相对性、GQA 分组） | ~5 秒 |
| [`train.py`](train.py) | 训练 `model.py`：默认是需要 induction head 的复制任务 | ~2 分钟 |

## 建议顺序

```bash
python why_nonlinear.py     # 为什么需要非线性
python test_model.py        # 手搓的 Transformer 是对的吗
python vanilla_demo.py      # encoder-decoder 和交叉注意力在干什么
python train.py             # 现代 decoder-only 真的能学会东西
```

`vanilla_demo.py` 学会「把序列反过来」之后，交叉注意力会自己长出反对角线——
这个任务的正确对齐是已知的，所以可以直接看图检查它学没学对：

```
        1  2  3  4  5  6  7  8   <- source (encoder)
  BOS                        @
    1                     @
    2                  @
    3               @
    4            #  :
    5         @
    6      @
    7   @
  ^ decoder step
```

## 为什么图是脚本生成的

[`make_figures.py`](make_figures.py) 里没有一条手画的线。每一条决策边界、每一个点、
每一条等高线都来自真实训练结果，所以图和正文的说法不可能对不上。改了模型就重跑一次，
图跟着更新。

## 那几个坑

从零实现 Transformer，最容易错的就是这四处，`test_model.py` 逐个验证：

1. **因果性** —— 改第 $t$ 个 token，位置 $< t$ 的 logits 必须**严格**不变（差 0.0，不是 1e-7）
2. **KV cache 等价性** —— 增量解码必须复现一次性前向的结果
3. **RoPE 相对性** —— $\langle R_i q, R_j k\rangle$ 只能依赖 $i-j$
4. **GQA 分组** —— `repeat_kv` 必须**连续**复制，否则 q 头和 kv 头配错组

还有两个不在测试里但同样常见的：

- `cache.pos` 每次前向只推进一次（在层循环**之后**），写进 `update()` 会翻 `n_layer` 倍
- `LambdaLR` 是拿 base_lr **乘** lambda 的；base 设成 0，学习率永远是 0，而 loss 因为 dropout 噪声看着还在动
