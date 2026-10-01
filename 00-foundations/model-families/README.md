# 模型家族精读：别只看参数表

**中文** · [English](README.en.md)

> 阅读时间：约 8 分钟 · 类型：阅读地图 · 最近审阅：2026-09

新模型发布时，大家通常先看参数量、上下文长度和 benchmark 分数。我更想弄明白：**它要解决什么问题，为什么选择这套架构，又是怎么训练出来的？**

这里用同一组问题读不同模型的技术报告，再对照公开配置和实验。下次遇到一个新模型，希望你能更快看出它改了哪里，哪些结论有依据，哪些还需要验证。

<div class="lesson-recipe advanced">
  <div><span>先看什么</span><strong>目标与约束，而不是参数量</strong></div>
  <div><span>再拆什么</span><strong>架构 · 数据与训练 · post-training · 推理系统 · 评估</strong></div>
  <div><span>最后比较什么</span><strong>效果为什么提高，计算开销又有什么变化</strong></div>
  <div><span>证据标准</span><strong>论文、公开 config、消融和可复现实验</strong></div>
</div>

## 先看架构图，再读具体模型

第一次读技术报告，可以先做[模型阅读练习](how-to-read.md)：跟着一个 token 看张量形状怎么变，算一次 KV cache 占多少空间，再分清哪些是实验结果、哪些是对结果的解释。以后读其他模型，也能沿用这个方法。

先打开 [Transformer 交互图解](../transformer-lab.md#tx-arch)，在不同家族之间切换。那张图回答“block 里哪里变了”；这一组精读继续回答“为什么变、怎么训、部署时需要多少资源”。

```mermaid
flowchart LR
    A["目标与约束"] --> B["架构选择"]
    B --> C["预训练数据与目标"]
    C --> D["Post-training"]
    D --> E["推理与服务成本"]
    E --> F["行为与评估"]
    F -. 新证据 .-> A
```

## 每个家族都问这六个问题

| 视角 | 要问的问题 | 容易掉进的坑 |
| --- | --- | --- |
| 目标 | 它优先解决能力、成本、长度、多模态，还是部署？ | 把所有改动都解释成“更强” |
| 架构 | attention、FFN、norm、位置编码和模态接口改了什么？ | 只记组件名，不看张量和数据流 |
| 训练 | 数据、目标、规模和 curriculum 怎样配合？ | 把架构收益和数据收益混在一起 |
| Post-training | SFT、偏好优化、RL、蒸馏分别改变什么行为？ | 用 base model 的结构解释 chat model 的行为 |
| 系统 | KV cache、激活参数、通信和延迟的账怎么算？ | 只看总参数，不看每 token 实际成本 |
| 证据 | 哪些结论有消融、外部评估或可复现结果？ | 用一张 leaderboard 代替机制证据 |

## 四个入口

<div class="curriculum-grid">
  <a class="curriculum-card" href="llama.md"><span class="card-step">Dense baseline</span><h3>Llama</h3><p>从较常见的 dense decoder 出发，分清架构、训练规模和 post-training 各自带来的提升。</p><b>开始精读 →</b></a>
  <a class="curriculum-card" href="qwen.md"><span class="card-step">Family design</span><h3>Qwen</h3><p>同一家族有 dense 和 MoE、大小不同的模型，也有 thinking 和 non-thinking 模式。可以对比不同预算下该怎么选。</p><b>开始精读 →</b></a>
  <a class="curriculum-card" href="deepseek.md"><span class="card-step">Co-design</span><h3>DeepSeek</h3><p>MLA、细粒度 MoE、训练系统和 reasoning post-training 需要放在一起看，才能理解它们如何配合。</p><b>开始精读 →</b></a>
  <a class="curriculum-card" href="gemma.md"><span class="card-step">Compact & multimodal</span><h3>Gemma</h3><p>看模型在支持长上下文和图像输入时，怎样通过 attention 和蒸馏控制部署开销。</p><b>开始精读 →</b></a>
</div>

## 按你的问题选择读法

- **想看架构**：Llama → DeepSeek。先熟悉稠密模型，再看 MLA 与 MoE 分别省下什么、增加什么开销。
- **想看后训练**：Llama → Qwen → DeepSeek。比较通用对齐、推理模式切换，以及用强化学习训练推理能力，各自在解决什么问题。
- **想补部署与多模态**：Gemma → Qwen。重点看模型大小、上下文长度和图像输入对显存、延迟及部署方式有什么影响。

## 读完后的自检

<div class="taste-check advanced">
  <strong>不要背“谁用了什么”，试着回答：</strong>
  <ol>
    <li>如果把模型名遮住，你能从 attention、FFN 和 post-training 看出它在优化什么吗？</li>
    <li>某个能力提升来自结构、数据、训练规模，还是 post-training？证据够不够把它们分开？</li>
    <li>省下计算、显存或延迟之后，通信、路由或数据处理的开销有没有增加？</li>
    <li>这套设计适合什么任务和负载？换个使用场景，哪里最可能先遇到瓶颈？</li>
  </ol>
</div>

这里只讨论公开信息。模型更新很快，具体配置请对照每篇末尾的原始报告和公开配置文件。
