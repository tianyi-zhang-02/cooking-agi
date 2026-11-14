# 从零实现实验：从符号到生成

**中文** · [English](README.en.md)

> 类型：实验索引 · 运行环境：CPU 即可 · 最近审阅：2026-10

我喜欢把同一个东西写两遍。第一遍不用 PyTorch，逼自己看清每一个数字从哪里来；第二遍再交给 tensor、module 和 autograd，让它真的学起来。

高层 API 当然不是坏东西，但如果一开始就封装所有核心计算，后续 shape、mask 或 state 出错时，很难定位不收敛的原因。

如果对 view、broadcasting 或自动求导还不熟，先读 [PyTorch：从张量到训练](../pytorch/README.md)。那里用小例子把存储、梯度和完整训练循环接起来，再回这里实现模型。

## 第一阶段：用纯 Python / NumPy 验证计算

| 文件 | 依赖 | 验证什么 |
| --- | --- | --- |
| [`tokenizer_from_scratch.py`](tokenizer_from_scratch.py) | Python 标准库 | BPE merge rules、词表、encode/decode |
| [`sequence_numpy.py`](sequence_numpy.py) | NumPy | RNN 展开、LSTM gates、scaled dot-product attention、causal mask |

前两个文件不使用 autograd。每一个 hidden state、gate 和 attention weight 都能直接从公式追到数组，适合在还没被训练循环淹没之前把机制看清。

```bash
python tokenizer_from_scratch.py
python sequence_numpy.py
```

## 第二阶段：用 PyTorch 完成训练

| 文件 | 做什么 | 建议命令 |
| --- | --- | --- |
| [`sequence_torch.py`](sequence_torch.py) | 自行实现 RNN/LSTM cell；训练 delay-copy 或 encoder–decoder reversal | `python sequence_torch.py --model lstm --task reverse` |
| [`vanilla.py`](vanilla.py) | 2017 encoder–decoder Transformer | 由 `vanilla_demo.py` 调用 |
| [`vanilla_demo.py`](vanilla_demo.py) | 形状追踪、反转序列、cross-attention 对齐 | `python vanilla_demo.py` |
| [`model.py`](model.py) | 现代 decoder-only：RMSNorm + RoPE + GQA + SwiGLU + KV cache | 由测试和训练脚本调用 |
| [`test_model.py`](test_model.py) | 因果性、cache 等价、RoPE 相对性、GQA 分组 | `python test_model.py` |
| [`train.py`](train.py) | 训练 decoder-only 完成 induction / copy task | `python train.py` |

这些实现不调用 `nn.MultiheadAttention` 或 `F.scaled_dot_product_attention`。PyTorch 提供 tensor、自动求导，以及 Linear、Embedding、LayerNorm、cross-entropy 和优化器这类基础件；attention、mask 和整体结构都逐行写出。

单篇笔记还配有更小的检查脚本，从对应笔记的“动手验证”进入，例如 [`attention_numpy.py`](attention_numpy.py)（纯 NumPy 多头注意力）、[`attention_torch.py`](attention_torch.py)（手写版对照 `F.scaled_dot_product_attention` 与 `nn.MultiheadAttention`，所以不受上面那条限制）、[`norm_compare.py`](norm_compare.py)（三种归一化的统计轴与 batch size 1）和 [`attention_ladder.py`](attention_ladder.py)（MHA、causal、GQA、MLA 与吸收式 MLA，逐步对照）。

## 快速验证

```bash
python test_learning_path.py
python test_model.py
```

`test_learning_path.py` 检查 tokenizer round-trip、RNN/LSTM 形状、causal attention 的上三角严格为 0，以及 PyTorch 两种 cell 的输出契约。

## 建议运行顺序

```text
tokenizer_from_scratch.py
        ↓
sequence_numpy.py
        ↓
sequence_torch.py --task delay
        ↓
sequence_torch.py --task reverse
        ↓
vanilla_demo.py
        ↓
test_model.py → train.py
```

## 为什么保留两套实现

无框架版本适合回答“公式里的每个量来自哪里”；PyTorch 版本适合回答“参数怎样注册、梯度怎样流、训练循环怎样组织”。只看第一种会停在前向演示，只看第二种则容易把 shape、mask 和 state bug 误判成框架问题。

## Transformer 最容易“看起来能跑”的四个错

1. **因果性**：改第 $t$ 个 token，位置 $<t$ 的 logits 必须严格不变。
2. **KV cache 等价性**：增量解码必须复现一次性前向。
3. **RoPE 相对性**：attention score 的位置部分只依赖相对距离。
4. **GQA 分组**：query heads 必须映射到正确的 KV group。

[`make_figures.py`](make_figures.py) 用真实训练的小模型生成决策边界、隐藏空间等图；[`make_norm_figures.py`](make_norm_figures.py) 生成归一化轴示意图和实测的残差梯度图；[`make_arch_figures.py`](make_arch_figures.py) 画的是结构示意图（Transformer block、三处 attention、KV cache），不是测量结果。
