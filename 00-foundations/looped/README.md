# Looped Transformer：把同一个 block 重复用

**中文** · [English](README.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>想要更深的计算，又不想多存参数</strong></div>
  <div><span>前置知识</span><strong>Transformer block · 残差流 · 自回归解码</strong></div>
  <div><span>核心机制</span><strong>同一组权重循环 L 次：深度和计算随 L 增长，参数不变</strong></div>
  <div><span>常见错误</span><strong>以为多转几圈就能多记知识；以为循环不会让 KV cache 变大</strong></div>
</div>

## 深度和参数，本来是绑在一起的

一个普通的 $L$ 层 Transformer，每一层都有自己的权重。想让模型对每个 token 做更多步串行计算，就只能加层，参数也跟着加。

## 把 block 循环起来

Looped Transformer 只存一个 $k$ 层的 block，把它反复用 $L$ 次：

$$h^{(t+1)} = f_\theta\big(h^{(t)},\, e\big), \qquad t = 0, 1, \ldots, L-1$$

$\theta$ 在每一圈都一样；$e$ 是输入的 embedding，很多设计会在每一圈把它重新喂进去（叫 input injection）。参数是 $k$ 层，有效深度和计算量是 $kL$ 层。

<!-- widget:tx-loop-unroll -->

## 老想法，新用法

- **Universal Transformer**（2018）：一个在所有位置、所有步之间共享的 transition function，再加上每个位置自己决定何时停下的 ACT。
- **ALBERT**（2019）也在层之间共享参数（默认全部共享），参数从 BERT-base 的 108M 降到 12M，但推理计算量不变。它的目的是省参数，不是把深度变成可调的旋钮。
- **最近两年**，循环被放进了大规模预训练：Huginn（3.5B）、Ouro（1.4B、2.6B）这样的 looped LM，把「转几圈」当作推理时可以加的计算量。

## 这一组怎么读

1. [为什么循环有助于推理](why-loops-reason.md)：串行步数、理论结果，以及和 CoT 的关系（可以动手拖）
2. [每个 token 该转几圈](adaptive-depth.md)：ACT、PonderNet、Ouro 的退出 gate、Mixture-of-Recursions（可以动手拖）
3. [现在的 looped LM 长什么样](models.md)：Huginn、Ouro、Relaxed Recursive Transformers
4. [代价与局限](costs.md)：延迟、KV cache、训练稳定性，以及它不能多记知识
5. [复习题](review.md)：面试题和自检
