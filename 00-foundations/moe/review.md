# MoE：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 4 分钟 · 难度：进阶 · 最近审阅：2026-10-09

## 面试常见问题

<details class="interview" markdown="1">
<summary>总参数和激活参数分别由什么决定？Mixtral 8x7B 为什么不是 56B？</summary>

固定 expert 大小时，总 expert 参数随 N 增长，每个 token 的主要 expert 计算随 k 增长；router 仍需为 N 个 experts 打分。Mixtral 8x7B 复制的是 FFN，不是 8 个完整模型，因此总参数为 46.7B、每 token 激活 12.9B，而不是直接算 8×7B。

</details>

<details class="interview" markdown="1">
<summary>Router 怎么算门控权重？top-k 选择不可导，router 怎么学？</summary>

常见 router 用线性层打分，再按 top-k 选择。Mixtral 在选中集合里归一化；Switch 的 top-1 则保留全体 softmax 中的概率，不把唯一门值改成 1。普通 autograd 不对离散索引求导，任务梯度通过连续门值回传。未选中的 expert 参数没有该 token 的任务梯度，但对应 router logit 可能通过全体 softmax 或辅助目标得到梯度，两者要分开说。

</details>

<details class="interview" markdown="1">
<summary>为什么需要负载均衡？Switch 的辅助 loss 为什么是 f_i 乘 P_i？</summary>

更多 token 带来更多训练机会，可能形成自我强化的不均衡，也可能让部分设备等待。Switch 的辅助 loss 是 $\alpha N \sum f_i P_i$：$f_i$ 是不可导的分配比例，$P_i$ 是可导的平均 router 概率。梯度通过 $P_i$ 回传；完全均匀时值为 $\alpha$，但这不是所有路由分布下的严格下界。

</details>

<details class="interview" markdown="1">
<summary>Capacity factor 是什么？超出容量的 token 去哪了？</summary>

在 Switch 的 top-1 设定里，上限按 $\frac{\text{token 数}}{N}\times\text{CF}$ 设置，再按实现取整。它限制每个 expert 的容量，不保证每张卡实际耗时一样。溢出的 expert 分支可被跳过，但 token 仍沿残差传播；top-k 或 dropless 实现的处理方式可能不同，不能把一个 capacity 公式套到所有模型。

</details>

<details class="interview" markdown="1">
<summary>DeepSeek-V3 的 auxiliary-loss-free 是怎么做的？真的完全没有 balance loss 吗？</summary>

Bias 只用于挑 top-k，门值仍来自原始 affinity；超载的 expert 降低 bias，负载偏低的提高。这个更新不靠辅助 loss 的梯度，但会改变选中集合，因而也可能改变输出与任务梯度。V3 仍保留很小的序列级 balance loss（$\alpha=0.0001$），所以不是完全没有辅助目标。

</details>

<details class="interview" markdown="1">
<summary>细粒度专家和共享专家各解决什么问题？</summary>

细粒度专家允许在相近的 expert 矩阵计算预算下，组合更多小专家；组合更多不保证分工更好，通信也未必不变。共享专家为所有 token 提供共同的计算路径，减少重复学习是设计目的，不是预先指定知识归属。DeepSeek-V3 两者都有；Qwen3-235B-A22B / 30B-A3B 没有 shared expert。

</details>

<details class="interview" markdown="1">
<summary>MoE 省显存吗？部署时主要的代价是什么？</summary>

稀疏激活不会自动按 k/N 缩减模型权重存储。全部参数仍需存放在某处，常驻 GPU、跨卡分片或 offload 都有不同代价。Expert parallelism 通常还要派发与回收 token；大 batch 可能用到很多专家。是否比某个 dense baseline 更省显存或更快，要说明质量、精度、batch 与部署方式。

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
