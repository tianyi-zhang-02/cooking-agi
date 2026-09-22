# MoE：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 3 分钟 · 难度：进阶 · 最近审阅：2026-09

## 面试常见问题

<details class="interview" markdown="1">
<summary>总参数和激活参数分别由什么决定？Mixtral 8x7B 为什么不是 56B？</summary>

总参数随 expert 数 $N$ 增长，每个 token 的计算只随 $k$ 增长。Mixtral 8x7B 每层 8 个 expert、每个 token 走 2 个：只有 FFN 被复制，attention 和 embedding 只有一份，所以总参数是 46.7B，每个 token 激活 12.9B。

</details>

<details class="interview" markdown="1">
<summary>Router 怎么算门控权重？top-k 选择不可导，router 怎么学？</summary>

router 是一个线性层，给每个 expert 一个分数；留下最高的 $k$ 个，在这 $k$ 个里做 softmax 得到权重（DeepSeek-V3 用 sigmoid 再归一化）。「选谁」没有梯度，梯度通过被选中 expert 的门控权重 $g_i$ 传回 router；没被选中的 expert 从这个 token 拿不到梯度。

</details>

<details class="interview" markdown="1">
<summary>为什么需要负载均衡？Switch 的辅助 loss 为什么是 f_i 乘 P_i？</summary>

不均衡会自我强化：多拿 token 的 expert 被训练得更好，拿到更多 token，最后少数 expert 干了所有活。Switch 的辅助 loss 是 $\alpha N \sum f_i P_i$：$f_i$ 是真实负载比例但不可导，$P_i$ 是平均 router 概率、可导；两者相乘，梯度通过 $P_i$ 回传，力度由 $f_i$ 决定，完全均匀时取到最小值。

</details>

<details class="interview" markdown="1">
<summary>Capacity factor 是什么？超出容量的 token 去哪了？</summary>

每个 expert 最多收 $\frac{\text{token 数}}{N} \times \text{CF}$ 个 token，让每张卡的计算量固定。超出的 token 不会被删掉，而是跳过这层 expert，只走残差连接。Switch 发现 CF 在 1.0 到 1.25 效果更好；DeepSeek-V3 则不丢 token。

</details>

<details class="interview" markdown="1">
<summary>DeepSeek-V3 的 auxiliary-loss-free 是怎么做的？真的完全没有 balance loss 吗？</summary>

每个 expert 一个 bias，只在挑 top-k 时加到分数上，门控权重仍用原始分数；每步之后超载的减 $\gamma$、空闲的加 $\gamma$。这样没有额外梯度干扰语言模型 loss。但它仍保留一项很小的序列级 balance loss（$\alpha = 0.0001$），所以不是「完全没有」。

</details>

<details class="interview" markdown="1">
<summary>细粒度专家和共享专家各解决什么问题？</summary>

细粒度：把 expert 切小、多选几个，计算不变但组合数暴涨，专长分得更开。共享专家：所有 token 都经过，放通用知识，减少 routed expert 之间的重复。DeepSeek 两个都用；Qwen3 的 MoE 没有 shared expert。

</details>

<details class="interview" markdown="1">
<summary>MoE 省显存吗？部署时主要的代价是什么？</summary>

不省。任何 token 都可能用到任何 expert，全部参数都要放在显存里（或分片放在多卡上）。主要代价是显存、两次 all-to-all 通信，以及大 batch 下几乎所有 expert 都会被读到。省的是每个 token 的计算。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>为什么 dense 模型的参数量和每个 token 的计算量是绑在一起的，MoE 又是怎么把它们拆开的？</li>
    <li>top-k 之后在谁上面做归一化？没被选中的 expert 会从这个 token 拿到梯度吗？</li>
    <li>不做负载均衡会发生什么？辅助 loss 和「只调 bias」各自的代价是什么？</li>
    <li>超出 capacity 的 token 最后去了哪里？</li>
    <li>为什么说 MoE 省的是计算不是显存？batch 大小怎样改变解码时的账？</li>
  </ol>
</div>

## 参考论文

- [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)：sparsely-gated MoE 与 noisy top-k
- [GShard](https://arxiv.org/abs/2006.16668)：top-2 路由、expert capacity、expert parallelism
- [Switch Transformers](https://arxiv.org/abs/2101.03961)：top-1、辅助 loss、capacity factor
- [ST-MoE](https://arxiv.org/abs/2202.08906)：router z-loss
- [Expert Choice Routing](https://arxiv.org/abs/2202.09368)：expert 选 token
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)：细粒度专家与共享专家
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)：auxiliary-loss-free 均衡、node-limited routing
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [gpt-oss model card](https://arxiv.org/abs/2508.10925)
