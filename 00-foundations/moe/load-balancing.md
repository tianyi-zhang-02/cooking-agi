# MoE：负载均衡

**中文** · [English](load-balancing.en.md)

> 阅读时间：约 3 分钟 · 难度：进阶 · 最近审阅：2026-09

## 不管它，就会塌缩

router 只能从被选中的 expert 那里拿到梯度。某个 expert 碰巧多分到一些 token，就练得更好，下次分数更高，分到的 token 更多；剩下的越来越少被选中，几乎学不到东西。最后是掏了 $N$ 份参数的钱，真正干活的只有几份。

下面是个玩具模拟，切换三种做法，看负载怎么变：

<!-- widget:tx-moe-balance -->

## Capacity factor：每个 expert 最多收多少 token

为了让每张卡的计算量固定，Switch Transformer 给每个 expert 设了一个上限：

$$\text{capacity} = \frac{\text{一批里的 token 数}}{N} \times \text{capacity factor}$$

超出上限的 token **不会被扔掉**，只是跳过这一层的 expert，顺着残差连接直接去下一层。Switch 试过 1.0、1.25、2.0，发现 1.0 到 1.25 更好；ST-MoE 训练用 1.25、评估用 2.0；DeepSeek-V3 干脆一个 token 都不丢。

## 辅助 loss：把不均衡直接写进损失

Switch Transformer 的辅助 loss：

$$\mathcal L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i P_i$$

- $f_i$：这一批里 argmax 落在 expert $i$ 上的 token 比例，本质是个计数，不可导；
- $P_i$：这一批 token 分给 expert $i$ 的平均 router 概率，可导；
- $\alpha = 10^{-2}$。

两个乘在一起：梯度顺着 $P_i$ 传回 router，推多狠则由真实负载 $f_i$ 说了算。完全均匀时 $f_i = P_i = 1/N$，损失落到最小值 $\alpha$。GShard 是同一套思路（用平均门控值顶替不可导的计数）；最早的 sparsely-gated MoE 则用了 importance 和 load 两个变异系数损失。

## Router z-loss：管数值，不管均衡

ST-MoE 另加了一项 router z-loss，惩罚过大的 router logit：

$$L_z = \frac{1}{B}\sum_{i=1}^{B}\Big(\log \sum_{j=1}^{N} e^{x^{(i)}_j}\Big)^2$$

系数 $c_z = 0.001$。它管的是训练稳定和低精度下的舍入误差，跟负载均衡没关系。

## 不加辅助 loss：只调 bias

辅助 loss 的梯度和语言模型 loss 落在同一组 router 分数上，两边会互相拉扯。DeepSeek-V3 换了个办法：

1. 每个 expert 有一个 bias $b_i$，**只在挑 top-k 时**加到分数上；
2. 门控权重还是用原始分数算，所以 bias 既不改变输出怎么加权，也不产生梯度；
3. 每步结束，超载的 expert 把 $b_i$ 调低 $\gamma$，空闲的调高 $\gamma$：前 14.3T token 用 $\gamma = 0.001$，最后 500B token 设成 0。

说它完全没有 balance loss 并不准确：DeepSeek-V3 还留了一项很小的序列级 balance loss（$\alpha = 0.0001$），防止单条序列内部失衡得太离谱。

Qwen3 走的是另一条路：global-batch balance loss，在全局 batch 上算均衡，而不是每个 micro-batch 都算，这样 expert 在局部可以更偏科。
