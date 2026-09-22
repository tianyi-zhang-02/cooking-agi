# Looped Transformer：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 3 分钟 · 难度：进阶 · 最近审阅：2026-09

## 面试常见问题

<details class="interview" markdown="1">
<summary>Looped Transformer 和 ALBERT 的参数共享有什么区别？</summary>

两者都在层之间共享权重。ALBERT 的目的是减少参数，推理计算量和原来一样，层数也是固定的。Looped Transformer 的目的是把深度和计算变成可以调的：同一个 block 转更多圈，就得到更深的计算，参数不变。

</details>

<details class="interview" markdown="1">
<summary>为什么循环能提升推理，却不能提升知识容量？</summary>

推理需要串行步数，而每个 token 的串行步数由深度决定，循环正好增加深度。知识存在参数里，循环不增加参数：Ouro 测得循不循环都是每参数约 2 bit；Saunshi 等也发现同样计算量下 looped 模型的 perplexity 和记忆更差，但推理几乎追平更深的模型。

</details>

<details class="interview" markdown="1">
<summary>Input injection 是什么？为什么需要？</summary>

每一圈都把原始输入的 embedding $e$ 重新喂给循环块，比如 Huginn 用 adapter 把当前状态和 $e$ 拼起来。不这样做，输入只在开头出现一次，要靠隐藏状态一圈圈保存下来；圈数一多就容易丢，训练也更不稳定。

</details>

<details class="interview" markdown="1">
<summary>自适应深度有哪些做法？Ouro 的 early exit 在公开代码里省计算吗？</summary>

ACT（累计停止概率过阈值就停）、PonderNet（停止概率分布加几何先验的 KL，梯度无偏）、Ouro 的退出 gate（第一阶段带熵正则，第二阶段用损失改善作标签）、Mixture-of-Recursions（router 给每个 token 分配深度）。Ouro 公开的 Hugging Face 代码里 4 圈总是全部跑完，early exit 只决定读哪一圈的输出，不省计算。

</details>

<details class="interview" markdown="1">
<summary>Looped 模型的 KV cache 为什么会变大？能不能各圈共享？</summary>

每一圈都重新跑 attention，默认每个（圈，层）存一份 K/V，Ouro 1.4B 每个 token 有 96 个槽位，是不循环时的 4 倍。能不能共享取决于训练：Ouro 报告只在解码时复用最后一圈几乎无损，但两篇 2026 年的独立工作在 Ouro 上复现时掉得很厉害；Huginn 对共享就不敏感。不能默认它是免费的。

</details>

<details class="interview" markdown="1">
<summary>Looping 和 chain-of-thought 是什么关系？</summary>

两者都在增加串行计算步数。CoT 通过生成 token，每个 token 一轮前向，过程看得见；looping 在隐藏状态里多走几步，不产生 token，过程看不见。Saunshi 等证明 $m$ 步 CoT 可以用一个稍大的 block 循环 $m$ 次来模拟，所以 looped 模型也叫 latent reasoning。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>为什么普通 Transformer 的深度和参数是绑在一起的，循环怎样把它们拆开？</li>
    <li>一个 1 层模型循环 12 次能解加法，这说明了推理能力由什么决定？</li>
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
