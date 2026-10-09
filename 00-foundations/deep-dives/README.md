# 进阶阅读：每个组件到底改了什么？

**中文** · [English](README.en.md)

看懂 Transformer 之后，新名词很容易混在一起：FlashAttention、MLA、MoE、各种 gate，似乎都在说“更高效”。先别急着记名字，把它们放回计算过程，就能看出它们改的不是同一件事。

还不熟悉 Q、K、V 和 residual，可以先回[完整 Transformer](../transformer.md)。想顺着流程读，直接去[预训练](../../learn/pretraining/README.md)或[生成与推理](../../learn/inference/README.md)；这页适合用来区分机制。

## 先分清序列、层和参数

一句话里有很多 token，一个模型里有很多层。**沿序列找信息、沿深度组合信息、选择一部分参数来计算，是三个不同的动作。**

| 改动发生在哪里 | 读哪篇 | 用什么小例子分清 |
| --- | --- | --- |
| Attention 输出 | [Gated Attention](gated-attention.md) | 两个 head 的结果乘上不同门值；算完再少传，不等于少算 |
| 历史状态 | [Gated DeltaNet](gated-deltanet.md) | 写入一个新 key 后，原先存的关联怎样变化 |
| 局部模式查表 | [Engram](engram.md) | 相同短语查到同一条目，不同上下文决定用多少；不是聊天记忆 |
| 网络深度 | [Attention Residuals](attention-residuals.md) | 同一个 token 怎样组合前层输出，而不是多看几个 token |
| FFN 参数 | [MoE](../moe/README.md) | 哪些专家处理这个 token；省下的计算会不会换成通信 |
| 重复计算的次数 | [Looped Transformer](../looped/README.md) | 共享一套参数多算几轮；少参数不代表低延迟 |

这些组件可以组合，不是只能选一个。读具体配置时，仍要确认它放在哪层、哪个位置，以及有没有为这种组合重新训练。

想追进展，也先找它改的环节。[Muon](muon.md)接在优化器之后，[DFlash / DFlash 2](dflash.md)接在 KV cache 之后；前者讲参数更新，后者讲生成时的草拟与验证，不需要按发表年份一起读。

## 模型怎样学：从输入到一次更新

| 先弄清什么 | 阅读入口 | 接着核对什么 |
| --- | --- | --- |
| 文本怎样切开 | [BPE、WordPiece 与 Unigram](tokenizer-algorithms.md) | 词表、byte coverage 与模型权重是否兼容 |
| 序列梯度怎样传播 | [BPTT 与门控](recurrent-dynamics.md) | 乘法链为什么会衰减，LSTM 的加性路径改变了什么 |
| 为什么训练集好、验证集差 | [泛化与诊断](generalization.md) | 过拟合、泄漏与分布变化，别用同一种办法处理 |
| 初始值与梯度尺度 | [激活与初始化](activation-and-initialization.md) | Xavier / He 的假设，深层网络的尺度变化 |
| 梯度怎样变成参数变化 | [SGD 到 AdamW](optimizers.md) | 动量、二阶矩与 weight decay 各管什么 |
| 哪些位置提供监督 | [语言模型目标](language-model-objective.md) → [一次训练更新](training-step.md) | Shift、mask、有效 token 数与梯度累积 |
| 数据怎样送进训练 | [预训练流程](pretraining-pipeline.md) | 去重、混合、packing、验证集与断点恢复 |
| 同时预测多个位置 | [Multi-token prediction](multi-token-prediction.md) | 额外目标怎样对齐；训练收益与生成加速分开看 |
| 训练为什么占显存 | [精度与显存](precision-and-memory.md) | 权重、梯度、优化器状态、激活分别算账 |

需要把这些步骤写出来，配合 [PyTorch 四章](../pytorch/README.md)。它从存储和 shape 开始，最后跑一个完整训练循环，不要求先会分布式训练。

## 模型怎样生成：不要把几种“省”混起来

举个例子：上下文从 4K 变成 32K，有些方法让每个位置存得更少，有些方法让每次读取的位置更少，还有些只是改善数据搬运。它们影响的成本不同。

| 机制 | 主要改变什么 | 不要误读成什么 |
| --- | --- | --- |
| [KV cache、MQA / GQA](kv-cache-and-inference.md) | 复用历史 K/V；减少 K/V 头数 | 不再需要处理历史信息 |
| [RoPE、插值与 YaRN](position-and-context.md) | 位置信息怎样参与 attention | 上下文长度调大后，模型必然能用好远处证据 |
| [NoPE 与顺序信息](nope-and-order.md) | 不加显式位置编码时，顺序还能从哪里来 | 任意 checkpoint 都能直接关掉 RoPE |
| [FlashAttention](attention-kernels.md) | 精确 attention 的分块与访存 | 自动删掉一部分 attention 配对 |
| [PagedAttention](attention-kernels.md) | 多请求的 KV cache 如何分块存放 | 改变模型学习到的 attention 规则 |
| [MLA 与 sparse attention](latent-and-sparse-attention.md) | 压缩每个位置的状态，或选择读取哪些位置 | 所有方法都在做同一种压缩 |

比较时先固定任务、上下文长度、batch 和输出长度，再看显存、首 token 延迟与后续生成速度。只写“更快”，读者没法知道快在哪里。

## 最后回到模型报告

[模型家族精读](../model-families/README.md)把这些部件接回具体版本。读报告时可以做一张小记录：**改了什么 → 解决哪个瓶颈 → 对照怎么做 → 多付了什么代价**。

一个组件的算例说明它怎样工作；模型报告里的消融说明它在那套实验里有没有帮助。两者都要看，但不能互相替代。
