# 从零实现实验：从符号到生成

**中文** · [English](README.en.md)

> 类型：实验索引 · 运行环境：CPU 即可 · 最近审阅：2026-08

我喜欢把同一个东西写两遍。第一遍不用 PyTorch，逼自己看清每一个数字从哪里来；第二遍再交给 tensor、module 和 autograd，让它真的学起来。

高层 API 当然不是坏东西，但如果一开始就封装所有核心计算，后续 shape、mask 或 state 出错时，很难定位不收敛的原因。

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

这些实现不调用 `nn.MultiheadAttention` 或 `F.scaled_dot_product_attention`。PyTorch 只负责 tensor、参数管理和自动求导，模型结构仍然显式可见。

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

图表仍由 [`make_figures.py`](make_figures.py) 与 [`make_arch_figures.py`](make_arch_figures.py) 从实际计算生成，避免正文与实现漂移。
