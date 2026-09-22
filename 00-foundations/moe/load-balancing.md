# MoE：负载均衡

**中文** · [English](load-balancing.en.md)

> 阅读时间：约 3 分钟 · 难度：进阶 · 最近审阅：2026-09

## 不管它，就会塌缩

router 只从被选中的 expert 那里拿到梯度。一个 expert 碰巧多拿到一些 token，就被训练得更好，下次得分更高，拿到的 token 更多；其余 expert 越来越少被选，几乎学不到东西。结果是花了 $N$ 份参数，真正在用的只有几份。

下面是一个玩具模拟，可以切换三种做法看负载怎么变：

<!-- widget:tx-moe-balance -->

## Capacity factor：每个 expert 最多收多少 token

训练时为了让每张卡的计算量固定，Switch Transformer 给每个 expert 设了上限：

$$\text{capacity} = \frac{\text{一批里的 token 数}}{N} \times \text{capacity factor}$$

超出上限的 token **不会被删掉**，而是跳过这一层的 expert，只走残差连接进入下一层。Switch 试过 1.0、1.25 和 2.0，发现 1.0 到 1.25 效果更好；ST-MoE 训练时用 1.25、评估时用 2.0。DeepSeek-V3 则不丢任何 token。

## 辅助 loss：把不均衡直接写进损失

Switch Transformer 的辅助 loss：

$$\mathcal L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i P_i$$

- $f_i$：这一批里 argmax 落在 expert $i$ 的 token 比例，是一个计数，不可导；
- $P_i$：这一批 token 分给 expert $i$ 的平均 router 概率，可导；
- $\alpha = 10^{-2}$。

两者乘起来，梯度通过 $P_i$ 传回 router，而推多用力由真实负载 $f_i$ 决定。完全均匀时 $f_i = P_i = 1/N$，损失取到最小值 $\alpha$。GShard 用的是同样的思路（用平均门控值代替不可导的计数），最早的 sparsely-gated MoE 则用了 importance 和 load 两个变异系数损失。

## Router z-loss：管数值，不管均衡

ST-MoE 另加了一项 router z-loss，惩罚过大的 router logit：

$$L_z = \frac{1}{B}\sum_{i=1}^{B}\Big(\log \sum_{j=1}^{N} e^{x^{(i)}_j}\Big)^2$$

系数 $c_z = 0.001$。它的目的是训练稳定、减少低精度下的舍入误差，不是负载均衡。

## 不加辅助 loss：只调 bias

辅助 loss 的梯度和语言模型 loss 作用在同一组 router 分数上，两者会互相拉扯。DeepSeek-V3 换了一个办法：

1. 每个 expert 有一个 bias $b_i$，**只在挑 top-k 时**加到分数上；
2. 门控权重仍然用原始分数算，所以 bias 不改变输出的加权方式，也不产生梯度；
3. 每一步之后，超载的 expert 把 $b_i$ 减 $\gamma$，空闲的加 $\gamma$。前 14.3T token 用 $\gamma = 0.001$，最后 500B token 设为 0。

它并不是完全没有 balance loss：DeepSeek-V3 仍保留一项很小的序列级 balance loss（$\alpha = 0.0001$），防止单条序列内部极端不均衡。

另一种思路是 Qwen3 的 global-batch balance loss：在全局 batch 而不是每个 micro-batch 上算均衡，允许 expert 在局部更偏科。
