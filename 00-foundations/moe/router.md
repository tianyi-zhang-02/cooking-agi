# MoE：router 怎样选 expert

**中文** · [English](router.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

## Router 就是一个线性层

Router 把 token 的隐藏状态 $x$ 乘上一个 $N \times d$ 的矩阵，得到 $N$ 个分数：$s = W_r x$。然后留下分数最高的 $k$ 个，只在这 $k$ 个里面做归一化，得到门控权重：

$$g_i = \frac{e^{s_i}}{\sum_{j \in \mathrm{TopK}} e^{s_j}}, \qquad i \in \mathrm{TopK}$$

这和「先对全部 $N$ 个做 softmax，再在留下的里面重新归一化」是同一件事。Mixtral 和 gpt-oss 都是这样做的；DeepSeek-V3 换成了 sigmoid 打分，再在选中的 expert 里归一化。

<!-- widget:tx-moe-router -->

## top-1、top-2 还是 top-8

- **top-1**：Switch Transformer。每个 token 只走一个 expert，最省计算和通信。
- **top-2**：GShard、Mixtral。两个 expert 的组合比一个更稳。
- **top-4**：gpt-oss。
- **top-8**：DeepSeek-V3、Qwen3。它们的 expert 切得更细（下一篇讲），所以要多选几个。

$k$ 越大，每个 token 的计算和跨卡通信越多，但可以组合的专长也越多。

## 为什么早期要加噪声

最早的 sparsely-gated MoE（Shazeer 等，2017）在打分上加了可学习幅度的高斯噪声，再取 top-k：

$$H_i = (xW_g)_i + \mathcal N(0,1)\cdot \mathrm{Softplus}\big((xW_{\text{noise}})_i\big)$$

分数接近的 expert 会轮流被选中，负载因此更分散，每个 expert 也都有机会被训练到。上图里对「的」这种功能词加一次噪声，就能看到被选中的 expert 换了。

## top-k 不可导，router 怎么学

「选谁」这一步是离散的，没有梯度。梯度走的是门控权重：输出是 $g_i E_i(x)$ 的和，所以被选中的 expert 帮了多大忙，会通过 $g_i$ 反传给 router。没被选中的 expert，这个 token 对它没有任何梯度。

这就埋下了一个问题：一开始多拿到一点 token 的 expert 会被训练得更好，于是拿到更多 token。这是下一篇负载均衡要解决的事。

## Token 选 expert，还是 expert 选 token

上面都是每个 token 选自己的 top-k。Expert Choice（Zhou 等，2022）反过来：每个 expert 从一批 token 里挑自己最想要的固定数量。负载天然均衡，一个 token 可能被好几个 expert 选中，也可能一个都没有；论文报告训练收敛比 Switch 和 GShard 快两倍以上。代价是「谁被选」取决于同一批里的其他 token，放到逐 token 的自回归解码里比较别扭。
