# 进阶精读：从机制到模型家族

**中文** · [English](README.en.md)

必修部分负责把地图画出来，进阶部分处理那些“图看懂了，但我还是觉得哪里不对”的地方。这里有两种读法：**往里钻机制，或者横向比较模型家族。**

先从机制开始：

- [序列梯度、BPTT 与门控](recurrent-dynamics.md)：信息为什么会忘，LSTM 的加法通路解决了什么。
- [Transformer 架构深拆](../transformer.md)：$Q/K/V$、mask、norm、RoPE、GQA、SwiGLU 与 KV cache。
- [语言模型目标、训练与生成](language-model-objective.md)：同一模型为什么有并行训练和串行 decode 两条路径。
- [一次语言模型训练](training-step.md)：用一个 batch 看清 label shift、3 种 mask、有效 token 分母与梯度累积。

读的时候可以带着一个具体问题：**如果拿掉这个组件，哪一步计算会先受影响，结果会怎么变？** 先写下自己的预测，再用公式或小实验检查。

然后再横向读：

- [模型家族精读](../model-families/)：用同一组问题比较 Llama、Qwen、DeepSeek 和 Gemma。
- [MoE 专题](../moe/)：单独拆 router、负载均衡、shared expert 与系统代价。
- [Looped Transformer](../looped/)：理解参数共享、循环深度与自适应计算。

比较不同模型时，不用数谁的组件更多。更值得问的是：**它想解决什么限制，哪项改动带来了提升，又多花了哪些资源。**
