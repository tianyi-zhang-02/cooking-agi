# Llama 精读：把 Dense Decoder 当作基线

**中文** · [English](llama.en.md)

> 阅读时间：约 9 分钟 · 类型：模型家族精读 · 最近审阅：2026-09

<div class="lesson-recipe advanced">
  <div><span>核心问题</span><strong>架构变化不大时，模型为什么还能变强？</strong></div>
  <div><span>重点组件</span><strong>Dense decoder · GQA · RoPE · RMSNorm · SwiGLU</strong></div>
  <div><span>训练主线</span><strong>大规模预训练 → SFT / 偏好优化 → 安全与工具使用</strong></div>
  <div><span>读完要会</span><strong>把架构、数据规模和 post-training 的贡献分开讲</strong></div>
</div>

## 一句话定位

读现代开源 LLM 时，可以先把 Llama 当作 **dense baseline**。它的看点不只是某个新组件，更在于较常见的 decoder-only 架构、大规模训练和 post-training 怎样配合。公开的模型也方便我们继续研究、部署和微调。

这正好提醒我们：模型能力不等于架构新颖度。即使 block 的结构看起来很熟悉，改善数据、训练稳定性和后训练，仍然可能让模型表现出明显不同的能力。

## 先看数据怎样流

```mermaid
flowchart LR
    A["Tokens"] --> B["Embedding + RoPE"]
    B --> C["RMSNorm → GQA"]
    C --> D["Residual"]
    D --> E["RMSNorm → SwiGLU FFN"]
    E --> F["Residual → logits"]
    F --> G["SFT / preference / safety"]
```

Llama 3 的主体仍是 dense Transformer：每个 token 都经过同一套 FFN 参数。GQA 减少 KV cache，RoPE 提供相对位置信号，RMSNorm 和 SwiGLU 已经成为现代 decoder 的常见组合。它们重要，但没有任何一个单独解释模型的全部能力。

## 真正值得抓住的三件事

1. **Dense 是清楚的对照组。** 没有 expert routing，参数路径更容易解释，也更方便做量化、微调和部署；代价是每个 token 都要经过完整的 FFN 计算。
2. **数据与规模是主角。** Llama 3 报告花了大量篇幅讨论数据混合、质量过滤、scaling 和训练稳定性。读报告时不要只停在架构表。
3. **Base 与 instruct 是两种研究对象。** 工具使用、拒答风格和偏好行为主要来自 post-training。不能看到 instruct model 更会用工具，就直接把提升归到 base model 的架构上。

## 取舍表

| 选择 | 得到什么 | 付出什么 |
| --- | --- | --- |
| Dense FFN | 路径简单、训练与部署成熟 | 每 token 激活全部 FFN 参数 |
| GQA | 更小的 KV cache 与更高解码吞吐 | KV 共享可能压缩 attention 容量 |
| 长上下文扩展 | 更长文档与多轮任务 | cache、训练数据和长上下文评估都更难 |
| 完整 post-training | 更好的指令、工具与安全行为 | 行为来源更难只靠 base benchmark 判断 |

## 我会怎样使用这个家族

Llama 是很好的实验基线：要研究数据、SFT、偏好优化、检索增强或 agent 行为时，先用它建立一个结构稳定的对照，再引入 MoE、MLA 或更复杂的推理训练。这样更容易判断提升究竟来自哪里。

## 自检

- 为什么说 Llama 的价值不能只从架构创新判断？
- GQA 省的是哪块内存，为什么主要影响 decode？
- 如果 instruct model 的工具使用变好了，你需要什么证据才能归因到架构？
- 什么时候 dense baseline 比一个更大的 MoE 更适合实验？

## 原始资料

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Meta AI：The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
