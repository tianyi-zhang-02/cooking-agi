# LatentMoE：专家一定要和主干一样宽吗？

**中文** · [English](latent-moe.en.md)

> 阅读时间：约 14 分钟 · 难度：进阶 · 最近审阅：2026-10-10 · 前置：[Router](router.md)、[细粒度与共享专家](fine-grained-and-shared.md)

一个 token 要交给 8 个专家处理，就得把它的表示送到这些专家那里。即使每个专家只算一点，搬数据和读权重也可能很贵。既然专家不一定需要主干的全部维度，能不能先把表示缩窄，再交给它们？

这就是 LatentMoE 的出发点。**缩窄的是 routed experts 的输入和输出，不是整条模型主干。** 如果只想知道它省在哪里，先看下面的路径图和预算表；稳定性与负载均衡放在后半篇。

## 缩窄的是哪一段？ {#paths}

<figure class="worked-update worked-update--pairs" aria-label="LatentMoE 的完整宽度与低维路径">
  <figcaption><strong>同一个输入，走两条路</strong><span>用 128 → 64 → 128 的教学尺寸看数据怎么走。</span></figcaption>
  <ol>
    <li><small>共享路径 · 所有 token 都走</small><strong>128 → 128</strong><span>输入经过共享专家（shared experts），输出仍是 128 维。这部分计算没有缩窄。</span></li>
    <li><small>路由路径 · 只走选中的专家</small><strong>128 → 64 → 128</strong><span>先下投影到 64 维，再分发、计算和聚合，最后上投影回 128 维，与共享输出相加。</span></li>
  </ol>
</figure>

还有一条容易被图省略的线：**router 可以直接读取原来的 128 维输入来选专家**，不必读取压缩后的 64 维。把“路由时搬运的表示”和“router 打分的输入”分开。[NVIDIA 的原始说明](https://research.nvidia.com/labs/nemotron/LatentMoE/)明确区分了这两件事。

用列向量写一个基础版本，$D$ 是下投影，$U$ 是上投影，$\mathcal T(x)$ 是选中的专家集合：

$$
\begin{gathered}
z=Dx,\\
u=\sum_{i\in\mathcal T(x)}g_i(x)E_i(z),\\
y=S(x)+Uu.
\end{gathered}
$$

$S$ 表示共享专家的总输出；这里没写外层 residual。主干宽度 $d$、潜在宽度 $\ell$ 和专家内部的 FFN 宽度 $f$ 是三个不同的尺寸：$E_i$ 做的是 $\ell\rightarrow f\rightarrow\ell$，不是把所有维度都除以 2。

## 别把三种“变小”混在一起 {#three-widths}

| 改动 | 缩小什么 | 其他条件固定时，直接改变什么 |
| --- | --- | --- |
| 细粒度专家 | FFN 中间宽度 $f$ | 每个 expert 的内部计算；输入输出仍可保持 $d$ |
| LatentMoE | Routed path 的接口宽度 $\ell$ | 专家矩阵大小、分发表示的宽度；另加上下投影 |
| MLA | Attention 历史 KV 的表示 | 每个历史位置需要缓存什么 |

所以 LatentMoE 和 MLA 可以同时出现，但 LatentMoE 本身不会把 attention 的 KV cache 压小。想看 MLA 为什么还要分离 RoPE，接着读[矩阵吸收与位置分支](../deep-dives/latent-and-sparse-attention.md#rope-absorption)。

## 省下来的预算，真的够多开几个专家吗？ {#budget}

用一组小尺寸，把省掉和新增的开销算清楚：$d=128,f=256$，8 个 SwiGLU 专家，每个 token 选 2 个。忽略 bias，专家权重数是 $3df$；LatentMoE 改成 $3\ell f$，另外有 $2d\ell$ 个上下投影权重。

| 教学方案 | Routed experts | $\ell$ | 专家与投影总参数 | 每 token 主要 MAC |
| --- | --- | --- | --- | --- |
| 普通 MoE | 8 选 2 | 128 | 786,432 | 196,608 |
| 只缩窄接口 | 8 选 2 | 64 | 409,600 | 114,688 |
| 再增加专家 | 16 选 4 | 64 | 802,816 | 212,992 |

第三行的**专家部分**与第一行预算相同，但两次投影仍要付钱。总参数约多 2.1%，主要 MAC 约多 8.3%，不能写成“完全免费地翻倍”。表中未计 router、shared experts、归一化、激活函数或 attention；1 MAC 按乘和加分开计约为 2 FLOPs。

<details markdown="1">
<summary>用 Python 重算参数、MAC 和分发字节</summary>

```python
def latent_budget(width, latent, intermediate, experts, top_k, tokens, item_bytes):
    values = (width, latent, intermediate, experts, top_k, tokens, item_bytes)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Expected positive integer dimensions and counts")
    if latent > width or top_k > experts:
        raise ValueError("Invalid latent width or expert selection")
    projections = 0 if latent == width else 2 * width * latent
    per_expert = 3 * latent * intermediate
    return {
        "parameters": experts * per_expert + projections,
        "mac_per_token": top_k * per_expert + projections,
        "slot_bytes": 2 * tokens * top_k * latent * item_bytes,
    }

base = latent_budget(128, 128, 256, 8, 2, 32, 2)
small = latent_budget(128, 64, 256, 8, 2, 32, 2)
more = latent_budget(128, 64, 256, 16, 4, 32, 2)
assert [row["parameters"] for row in (base, small, more)] == [786432, 409600, 802816]
assert [row["mac_per_token"] for row in (base, small, more)] == [196608, 114688, 212992]
assert [row["slot_bytes"] for row in (base, small, more)] == [32768, 16384, 32768]
```

`slot_bytes` 按每个 token–expert 槽位分别发送输入、返回输出来记账。32 个 token、每元素 2 bytes 时，前后两程分别合计 32、16、32 KiB。这不是网卡上的实测流量：同卡专家、本地计算、同目的地合并、padding、元数据和通信拓扑都会改变网络账本。`latent == width` 在这个教学函数中表示普通 MoE，省去两次投影。

</details>

把宽度减半却把 top-k 翻倍，槽位字节会回到原值。是否更划算，要看同等开销能否得到更好的质量，而不是只看“压缩了几倍”。原始 [LatentMoE 论文](https://arxiv.org/html/2601.18089v1)也区分了算量与访存、通信约束；本页数字是独立构造的预算例子，不是论文测速。

## Kimi K3：缩窄以后，还要处理什么？ {#kimi-k3}

K3 使用 896 个 routed experts、每 token 选 16 个。[官方发布说明](https://www.kimi.com/news/kimi-k3-open-source)把这一设计称为 Stable LatentMoE。对照 [报告 §2.3](https://arxiv.org/html/2607.24653v1#S2.SS3)，主干宽度为 7168，routed path 为 3584，2 个 shared experts 保持完整宽度；router 从完整输入打分。不要把“分发 3584 维向量”误写成“router 必须读 3584 维”。

它在聚合结果与上投影之间加入 RMSNorm，并用 SiTU-GLU 控制专家内部激活范围；负载调节则用 Quantile Balancing。下面分开看：怎样控制输出尺度，以及怎样让专家都有活可做。

### 为什么归一化的位置不能随便挪？ {#normalization}

先看一个不涉及训练的小反例。两个专家输出 `[2, 0]`、`[0, 1]`，各占一半。忽略 epsilon 和可学习增益：

| 计算顺序 | 输出 |
| --- | --- |
| 先加权，再 RMSNorm | 约 `[1.265, 0.632]` |
| 各自 RMSNorm，再加权 | 约 `[0.707, 0.707]` |

第一种保留了原来的 2:1 方向比例，第二种变成 1:1。**RMSNorm 不是能从求和里随便提出来的线性投影。** 若实现与论文的顺序不同，shape 可能完全对，结果却已不是同一个模型。

SiTU 的直觉也不复杂：两个很大的数相乘，更容易出现极端激活；先对两条支路做平滑限幅，再相乘。它不是把所有中间量强行裁成常数，也不保证整个网络永不溢出。可学习的投影仍能放大输出；饱和区的梯度也可能变小。

## Quantile Balancing：这次统计，下次使用 {#quantile-balancing}

6 个 token、3 个专家、top-1，一共只有 6 次分配，平均目标就是每专家 2 次。把分数直方图或平均负载看一眼，还不够说明该怎么调 bias：我们还得知道，每个专家离被选中差多少。

K3 的规则用当前带 bias 的第 $k+1$ 名分数作门槛，再按各专家的分数差估计下一步 bias；混合权重仍来自原 affinity。**新 bias 下一步才使用，推理时冻结。** 全局统计使用直方图近似分位数。[报告 §2.3.3](https://arxiv.org/html/2607.24653v1#S2.SS3.SSS3)

这里有两个不该跳过的限制。按旧门槛算出的分位数，不保证改完所有 bias 后重新做 top-k 仍恰好均分；下一批数据也可能变了。其次，“aux-loss-free”不等于 bias 通过普通任务梯度更新，它仍有额外的状态更新规则。

<details markdown="1">
<summary>只看一个专家，怎样从分数差找到门槛？</summary>

假设它对 6 个 token 的“原 affinity − 当前入选门槛”分别是 `[0.31, 0.16, 0.07, -0.08, -0.19, -0.34]`。想在**固定这组门槛**下留下 2 个，就把 bias 设为 `-0.07`：严格大于 0 的只剩前 2 项。

```python
def fixed_cutoff_bias(margins, target):
    import math

    if not margins or any(not math.isfinite(value) for value in margins):
        raise ValueError("Expected finite margins")
    if type(target) is not int or not 0 < target < len(margins):
        raise ValueError("Expected an interior integer target")
    ordered = sorted(margins, reverse=True)
    return -ordered[target]

margins = [0.31, 0.16, 0.07, -0.08, -0.19, -0.34]
bias = fixed_cutoff_bias(margins, 2)
assert sum(margin + bias > 0 for margin in margins) == 2
```

这是严格阈值下的次序统计量示例，不是完整 QB。并列分数可能使留下的数量少于目标；$Tk/N$ 不是整数时也不能要求每个专家完全相同。实际实现还要定义分位数插值、histogram bins、跨 rank 汇总与状态恢复。减去所有 bias 的共同均值不改变排序，但把 bias 混进 gate 权重会改变计算。

最需要防住的是时间顺序：先拿本批所有 token 的统计改 bias，再回头路由本批，早期 token 就可能受后面 token 影响。应记录本步实际使用的 bias，完成统计后更新下一步状态，并随 checkpoint 保存。

</details>

## 该怎么验证它值不值得用？ {#validation}

先确定是在修哪个瓶颈。少量 token 的 decode 可能主要在读权重；大批量、跨多卡时，也可能等在 all-to-all 或忙碌的专家上。不能只靠 top-k 和参数量判断。

| 要验证什么 | 做一个什么对照 | 别漏掉什么 |
| --- | --- | --- |
| 低维路径是否丢信息 | 固定数据与训练预算，扫描 $\ell$ | 分领域质量、稀有样本，而不只有平均 loss |
| 收益来自压缩还是更多专家 | 分别比较上表 3 个方案 | Router 与上下投影的额外成本 |
| 稳定组件是否有效 | 分开改变 norm、激活、均衡更新 | 梯度、激活尾部、丢 token、负载和质量 |
| 服务是否真变快 | 同硬件、精度、batch 和长度测速 | Prefill/decode 分开，吞吐与 p95 时延一起看 |

截至本次核对，Nemotron 3 Super 也采用 LatentMoE，但具体配置不同：其报告列出 $4096\rightarrow1024$ 的 routed path。[Nemotron 3 Super §2.1.1](https://arxiv.org/html/2604.12374v1#S2.SS1.SSS1) 这说明压缩比例是设计选择，不是 LatentMoE 定义的一部分。原报告的收益属于各自实验，本页只验证算例与数学关系，没有复现模型训练或多卡性能。

如果现在能说清“主干没缩窄、router 输入与通信载荷不同、额外投影要算成本”，已经抓住了主线。之后再读[系统代价](systems.md)，把这些尺寸换成具体设备上的存储和时间。
