# 进阶拆解：数学、目标与执行路径

**中文** · [English](README.en.md)

必修部分负责把地图画出来，这一层专门处理那些“图看懂了，但我还是觉得哪里不对”的地方。我们不再重复架构名词，而是拆三个真正决定模型行为的机制：

- [序列梯度、BPTT 与门控](recurrent-dynamics.md)：信息为什么会忘，LSTM 的加法通路解决了什么。
- [Transformer 架构深拆](../transformer.md)：$Q/K/V$、mask、norm、RoPE、GQA、SwiGLU 与 KV cache。
- [语言模型目标、训练与生成](language-model-objective.md)：同一模型为什么有并行训练和串行 decode 两条路径。

我建议别把它们当补充阅读，而是带着一个可证伪的问题进去：**如果我把这个组件拿掉，哪个 invariant 应该先坏？** 能回答这个问题，才算真的知道它有什么用。
