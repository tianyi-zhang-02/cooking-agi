# Looped Transformer：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 3 分钟 · 难度：进阶 · 最近审阅：2026-10-09

## 面试常见问题

<details class="interview" markdown="1">
<summary>Looped Transformer 和 ALBERT 的参数共享有什么区别？</summary>

两者都共享参数。ALBERT 还结合 embedding 因式分解来压缩模型，不能把总参数变化全归于共享。Looped 模型研究在共享权重下调整执行深度；参数不变，不代表计算量、KV 或延迟不变，也不保证深度外推有效。

</details>

<details class="interview" markdown="1">
<summary>循环给模型增加了什么，又没有增加什么？</summary>

增加共享参数上的计算深度，不新增外部事实，也不新增每圈独立的权重。能否利用好这些计算是实验问题。约 2 bit/parameter 是合成知识实验的结果，不是容量定理；“多圈一定更准”也不成立。

</details>

<details class="interview" markdown="1">
<summary>Input injection 是什么？为什么需要？</summary>

每圈重新提供输入表示 $e$，如 Huginn 将 prelude 输出和当前状态送入 adapter。这样不必完全依赖循环状态保存输入。它是一种设计选择，不是所有循环模型稳定训练的必要或充分条件。

</details>

<details class="interview" markdown="1">
<summary>自适应深度有哪些做法？Ouro 的 early exit 在公开代码里省计算吗？</summary>

ACT、PonderNet、退出 gate 和深度 router 采用不同目标。需要分清条件停止概率、实际退出概率、输出选择和实际跳过计算。[自适应深度](adaptive-depth.md)里的 3 步算例给出平均 2.2 步；Ouro 的 `7ea635b` 快照则先执行全部配置圈数再选择输出，不能把选择深度直接当计算量。

</details>

<details class="interview" markdown="1">
<summary>Looped 模型的 KV cache 为什么会变大？能不能各圈共享？</summary>

各圈权重一样，收到的隐藏状态却不一样，所以完整缓存通常按（圈，层）区分。共享会改变计算，需要比较误差。不同论文的 checkpoint、prefill 规则和生成长度不完全相同；不能把协议差异当作严格复现失败。内存账与来源见[代价与局限](costs.md)。

</details>

<details class="interview" markdown="1">
<summary>Looping 和 chain-of-thought 是什么关系？</summary>

CoT 增加生成位置，looping 增加隐藏状态更新；两者能组合。带 KV cache 的逐 token 前向不重算全部前缀。模拟 CoT 的理论构造有额外结构与输入长度条件，不等于任意现成模型可以直接互换；可见文字也不保证解释忠实。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>为什么普通 Transformer 的深度和参数是绑在一起的，循环怎样把它们拆开？</li>
    <li>同样执行 12 次层计算，共享与独立参数的模型在哪些预算上不同？</li>
    <li>ACT、PonderNet、Ouro 的退出 gate 分别怎样决定一个 token 什么时候停？</li>
    <li>循环 4 圈的模型，KV cache 为什么默认是不循环时的 4 倍？</li>
    <li>哪些任务适合用 looped 模型，哪些不适合？</li>
  </ol>
</div>

## 参考论文

- [Universal Transformers](https://arxiv.org/abs/1807.03819)：跨步共享与 ACT
- [ALBERT](https://arxiv.org/abs/1909.11942)：跨层参数共享（对照）
- [PonderNet](https://arxiv.org/abs/2107.05407)：概率化的停止
- [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196)
- [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/abs/2311.12424)
- [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416)
- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)：Huginn
- [Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672)
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)
- [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741)：Ouro
- [Continuous Depth Batching](https://arxiv.org/abs/2608.09444)：各圈共享 KV 的复现
- [MELT](https://arxiv.org/abs/2605.07721)：免训练 KV 共享的评估
