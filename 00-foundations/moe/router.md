# MoE：router 怎样选 expert

**中文** · [English](router.en.md)

> 阅读时间：约 3 分钟 · 难度：进阶 · 最近审阅：2026-10-09

## 最常见的 router，从一个线性层开始

一种常见 router 拿 token 的隐藏状态 $x$ 乘一个 $N \times d$ 的矩阵，得到 $N$ 个分数：$s = W_r x$。留下最高的 $k$ 个，只在这几个里面归一化，得到门控权重；其他 router 可以有分组、额外偏置或不同打分规则：

$$g_i = \frac{e^{s_i}}{\sum_{j \in \mathrm{TopK}} e^{s_j}}, \qquad i \in \mathrm{TopK}$$

这和「先把 $N$ 个全做一遍 softmax，再把留下的重新归一化」完全等价。Mixtral 和 gpt-oss 都这么做；DeepSeek-V3 改用 sigmoid 打分，再在选中的几个里归一化。

例如分数为 `[2, 1, 0]`，选前两个再归一化，权重约为 `[0.731, 0.269]`。但这不是所有 MoE 的通用规定：[Switch Transformer](https://www.jmlr.org/papers/v23/21-0998.html) 的 top-1 保留全体 softmax 中被选专家的概率，不把它重新归一化成 1；否则这个门值就无法给 router 提供任务梯度。图里展示的是选中后归一化的版本。

<!-- widget:tx-moe-router -->

## top-1、top-2 还是 top-8

- **top-1**：Switch Transformer。每个 token 只走一个 expert；其他条件相同时，expert 计算较少。
- **top-2**：GShard、Mixtral。每个 token 可组合两个 expert 的输出，多了一条路径，也多了计算。
- **top-4**：gpt-oss。
- **top-8**：DeepSeek-V3、Qwen3。它们把 expert 切得更细（[细粒度那篇](fine-grained-and-shared.md)讲），所以要多挑几个。

固定 expert 宽度时，$k$ 越大，每个 token 的 expert 计算通常越多；跨卡流量还取决于专家放在哪里。跨模型比较不能只看 $k$：top-8 的小 expert 未必比 top-2 的大 expert 更贵，也不保证更稳定。

## 为什么早期要加噪声

最早的 sparsely-gated MoE（Shazeer 等，2017）在打分上加了可学习幅度的高斯噪声，再取 top-k：

$$H_i = (xW_g)_i + \mathcal N(0,1)\cdot \mathrm{Softplus}\big((xW_{\text{noise}})_i\big)$$

[加噪声](https://arxiv.org/abs/1701.06538)让接近选择边界的 expert 有机会入选，有助于探索，但一次扰动不保证换人，也不保证训练后负载均衡。图里的分数只是演示数据；切换噪声可以观察边界附近的选择怎样变化，不代表某个词实际会被哪个 expert 处理。

## top-k 不可导，router 怎么学

常见 top-k 实现不会对“选中了谁”这个离散索引反传；梯度通过选中门值 $g_i$ 回到 router。未选中的 expert **参数**通常没有这个 token 的任务梯度，但未选中的 router logit 是否有梯度，要看归一化是否涉及它，以及有没有辅助目标。比如全体 softmax 的分母包含其他 logits；选中后再归一化则有不同的梯度关系。

一个可能的问题是：早期多分到 token 的 expert 训练机会更多，之后又更常被选中。下一篇讨论怎样发现和缓解这种失衡，而不是假定它必然发生。

## Token 选 expert，还是 expert 选 token

上面都是 token 挑 expert。[Expert Choice](https://arxiv.org/abs/2202.09368) 把它倒过来：每个 expert 从一批 token 中挑固定数量。槽位足够时，分配数可以按构造保持均衡，但各卡耗时未必相同；同一 token 也可能被多选或完全不选。论文报告了特定训练设置下的收敛速度收益，不能直接当作任意服务的提速。

自回归任务还要多问一句：候选集合里有没有未来 token？如果一个早期 token 是否入选取决于后面的 token，选择操作本身就可能泄漏未来信息。训练时要限制候选范围；部署时还需检查 batch 组成是否改变单条请求的行为。
