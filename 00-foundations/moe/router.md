# MoE：router 怎样选 expert

**中文** · [English](router.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

## Router 就是一个线性层

router 就是拿 token 的隐藏状态 $x$ 乘一个 $N \times d$ 的矩阵，得到 $N$ 个分数：$s = W_r x$。留下最高的 $k$ 个，只在这几个里面归一化，就是门控权重：

$$g_i = \frac{e^{s_i}}{\sum_{j \in \mathrm{TopK}} e^{s_j}}, \qquad i \in \mathrm{TopK}$$

这和「先把 $N$ 个全做一遍 softmax，再把留下的重新归一化」完全等价。Mixtral 和 gpt-oss 都这么做；DeepSeek-V3 改用 sigmoid 打分，再在选中的几个里归一化。

<!-- widget:tx-moe-router -->

## top-1、top-2 还是 top-8

- **top-1**：Switch Transformer。每个 token 只走一个 expert，最省计算和通信。
- **top-2**：GShard、Mixtral。两个 expert 的组合比一个更稳。
- **top-4**：gpt-oss。
- **top-8**：DeepSeek-V3、Qwen3。它们把 expert 切得更细（[细粒度那篇](fine-grained-and-shared.md)讲），所以要多挑几个。

$k$ 越大，每个 token 的计算和跨卡通信越多，能凑出来的专长组合也越多。

## 为什么早期要加噪声

最早的 sparsely-gated MoE（Shazeer 等，2017）在打分上加了可学习幅度的高斯噪声，再取 top-k：

$$H_i = (xW_g)_i + \mathcal N(0,1)\cdot \mathrm{Softplus}\big((xW_{\text{noise}})_i\big)$$

分数接近的几个 expert 会轮流上场，负载分得更开，每个 expert 都有机会被训练到。上面的图里选「的」这种功能词，加一次噪声，就能看到选中的 expert 换了人。

## top-k 不可导，router 怎么学

「选谁」是离散的，没有梯度。梯度走的是门控权重：输出是若干个 $g_i E_i(x)$ 相加，被选中的 expert 帮了多少忙，就顺着 $g_i$ 传回 router。没被选中的 expert，这个 token 给不了它任何梯度。

问题就埋在这里：一开始多分到几个 token 的 expert，会被训练得更好，于是分到更多 token。下一篇的负载均衡，就是冲着这件事去的。

## Token 选 expert，还是 expert 选 token

上面都是 token 挑 expert。Expert Choice（Zhou 等，2022）把它倒过来：每个 expert 从一批 token 里挑固定数量、自己最想要的。负载天生就是均的，代价是一个 token 可能被好几个 expert 同时挑走，也可能没人要它；论文里训练收敛比 Switch 和 GShard 快两倍以上。另一个麻烦是，谁被挑中取决于同一批里还有哪些 token，放进一个字一个字往外吐的自回归解码里就很别扭。
