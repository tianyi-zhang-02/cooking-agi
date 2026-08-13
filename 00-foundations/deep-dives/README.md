# 进阶拆解：数学、目标与执行路径

**中文** · [English](README.en.md)

这一层不再重复架构名词，而是拆三个决定模型行为的机制：

- [序列梯度、BPTT 与门控](recurrent-dynamics.md)：信息为什么会忘，LSTM 的加法通路解决了什么。
- [Transformer 架构深拆](../transformer.md)：$Q/K/V$、mask、norm、RoPE、GQA、SwiGLU 与 KV cache。
- [语言模型目标、训练与生成](language-model-objective.md)：同一模型为什么有并行训练和串行 decode 两条路径。

建议每篇都带着一个可证伪问题读：改掉这个组件，哪个 invariant 应该首先坏掉？
