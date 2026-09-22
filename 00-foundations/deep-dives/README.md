# 进阶精读：从机制到模型家族

**中文** · [English](README.en.md)

必修部分负责把地图画出来，进阶部分处理那些“图看懂了，但我还是觉得哪里不对”的地方。这里有两种读法：**往里钻机制，或者横向比较模型家族。**

先从机制开始：

- [序列梯度、BPTT 与门控](recurrent-dynamics.md)：信息为什么会忘，LSTM 的加法通路解决了什么。
- [Transformer 架构深拆](../transformer.md)：$Q/K/V$、mask、norm、RoPE、GQA、SwiGLU 与 KV cache。
- [语言模型目标、训练与生成](language-model-objective.md)：同一模型为什么有并行训练和串行 decode 两条路径。

我建议别把它们当补充阅读，而是带着一个可证伪的问题进去：**如果我把这个组件拿掉，哪个 invariant 应该先坏？** 能回答这个问题，才算真的知道它有什么用。

然后再横向读：

- [模型家族精读](../model-families/)：用同一组问题比较 Llama、Qwen、DeepSeek 和 Gemma。
- [MoE 专题](../moe/)：单独拆 router、负载均衡、shared expert 与系统代价。
- [Looped Transformer](../looped/)：理解参数共享、循环深度与自适应计算。

横向阅读时，别问“谁的组件更多”，而要问：**它在优化什么约束，能力从哪里来，成本又被转移到了哪里。**
