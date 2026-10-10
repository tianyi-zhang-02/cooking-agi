# MoE：router 怎样选 expert

**中文** · [English](router.en.md)

> 阅读时间：约 10 分钟 · 难度：进阶 · 最近审阅：2026-10-09

一个 token 来到 MoE 层，不会把所有专家都跑一遍。Router 先选出几个，再把它们的输出按权重加起来。难点不只是“取最大的几个数”：同一个 token 会走多条分支，结果要加回原来的位置，梯度也要沿这些分支传回去。

只想先理解机制，可以看前两节和[分发算例](#dispatch-example)；准备写代码，再展开实现和梯度检查。这里先讲单机，再讨论跨设备时为什么还要限制分组。

## 最常见的 router，从一个线性层开始

最简单的做法是给每个专家算一个分数。输入是 token 的隐藏状态 $x$，不是 token ID；因此同一个词出现在不同上下文里，也可能去不同专家。把 $x$ 乘一个 $N \times d$ 的矩阵，就得到 $N$ 个分数：$s = W_r x$。其中 $N$ 是专家数，$d$ 是隐藏维度。

$W_r$ 的第 $i$ 行负责“给专家 $i$ 打分”，不是专家自己的 FFN 权重。Router 选人，expert 才做主要的特征变换；这两套参数分别学习。

一种常见规则是留下最高的 $k$ 个，只在这几个里面归一化，得到门控权重（gate weights）。其他 router 可以有分组、额外偏置或不同打分规则：

$$g_i = \frac{e^{s_i}}{\sum_{j \in \mathrm{TopK}} e^{s_j}}, \qquad i \in \mathrm{TopK}$$

这和「先把 $N$ 个全做一遍 softmax，再把留下的重新归一化」完全等价。Mixtral 和 gpt-oss 都这么做；DeepSeek-V3 改用 sigmoid 打分，再在选中的几个里归一化。

例如分数为 `[2, 1, 0]`，选前两个再归一化，权重约为 `[0.731, 0.269]`。但这不是所有 MoE 的通用规定：[Switch Transformer](https://www.jmlr.org/papers/v23/21-0998.html) 的 top-1 保留全体 softmax 中被选专家的概率，不把它重新归一化成 1；否则这个门值就无法给 router 提供任务梯度。图里展示的是选中后归一化的版本。

这些数常被叫作“概率”，但 **top-k 不等于按概率抽签**。在没有噪声、且分数不并列时，同一组分数会确定地选中同一组专家；门值随后决定每个输出占多大权重。

<!-- widget:tx-moe-router -->

## top-1、top-2 还是 top-8

- **top-1**：Switch Transformer。每个 token 只走一个 expert；其他条件相同时，expert 计算较少。
- **top-2**：GShard、Mixtral。每个 token 可组合两个 expert 的输出，多了一条路径，也多了计算。
- **top-4**：gpt-oss。
- **top-8**：DeepSeek-V3、Qwen3。它们把 expert 切得更细（[细粒度那篇](fine-grained-and-shared.md)讲），所以要多挑几个。

固定 expert 宽度时，$k$ 越大，每个 token 的 expert 计算通常越多；跨卡流量还取决于专家放在哪里。跨模型比较不能只看 $k$：top-8 的小 expert 未必比 top-2 的大 expert 更贵，也不保证更稳定。

## 为什么早期要加噪声

最早的 sparsely-gated MoE（Shazeer 等，2017）在打分上加了可学习幅度的高斯噪声，再取 top-k：

$$\begin{aligned}
\sigma_i &= \mathrm{Softplus}\big((xW_{\text{noise}})_i\big),\\
\epsilon_i &\sim \mathcal N(0,1),\\
H_i &= (xW_g)_i+\epsilon_i\sigma_i.
\end{aligned}$$

$\sigma_i$ 决定噪声幅度，$\epsilon_i$ 是本次抽到的噪声，$H_i$ 才是拿去选专家的分数。

[加噪声](https://arxiv.org/abs/1701.06538)让接近选择边界的 expert 有机会入选，有助于探索，但一次扰动不保证换人，也不保证训练后负载均衡。图里的分数只是演示数据；切换噪声可以观察边界附近的选择怎样变化，不代表某个词实际会被哪个 expert 处理。

## top-k 不可导，router 怎么学

常见 top-k 实现不会对“选中了谁”这个离散索引反传；梯度通过选中门值 $g_i$ 回到 router。未选中的 expert **参数**通常没有这个 token 的任务梯度，但未选中的 router logit 是否有梯度，要看归一化是否涉及它，以及有没有辅助目标。比如全体 softmax 的分母包含其他 logits；选中后再归一化则有不同的梯度关系。

一个可能的问题是：早期多分到 token 的 expert 训练机会更多，之后又更常被选中。下一篇讨论怎样发现和缓解这种失衡，而不是假定它必然发生。

## 选好以后，token 怎么过去，输出怎么回来？ {#dispatch-example}

先不考虑多卡。用两个 token、三个专家，手工指定下面的输出；数字只用于算清流程，不是模型测量：

| 输入位置 | 选中的专家 | 权重 | 该专家的输出 |
| --- | --- | --- | --- |
| token 0 | expert 0 | 0.75 | `[2, 0]` |
| token 0 | expert 2 | 0.25 | `[0, 4]` |
| token 1 | expert 1 | 0.60 | `[1, 3]` |
| token 1 | expert 2 | 0.40 | `[5, 1]` |

按专家整理输入时，expert 0 收到 token 0，expert 1 收到 token 1，expert 2 收到两个 token。这一步叫 **dispatch**。专家算完后，还要记得每一行来自哪个 token，再加权放回去：

$$\begin{aligned}
y_0 &= 0.75[2,0]+0.25[0,4]\\
&= [1.5,1],\\
y_1 &= 0.60[1,3]+0.40[5,1]\\
&= [2.6,2.2].
\end{aligned}$$

注意是**相加，不是覆盖，也不是拼接**。如果把 expert 2 的结果直接赋给 `output[token_ids]`，前面专家的贡献就丢了。这个错误不会必然报 shape 错，却会把模型改成另一种计算。

<details markdown="1">
<summary>看一个能反向传播的 PyTorch 实现</summary>

下面只实现 routed FFN 分支，输入和输出都是 `[tokens, width]`；外层 residual、shared expert 和 padding 处理不在这个函数里。每个专家是一个输入输出同宽的 PyTorch 模块。

```python
def sparse_moe(hidden, logits, experts, top_k, normalization="selected"):
    if hidden.ndim != 2 or logits.shape != (hidden.shape[0], len(experts)):
        raise ValueError("Expected [tokens, width] inputs and one logit per expert")
    indices, weights = selected_gates(logits, top_k, normalization)
    output = torch.zeros_like(hidden)
    for expert_id, expert in enumerate(experts):
        token_ids, slots = torch.where(indices == expert_id)
        if token_ids.numel() == 0:
            continue
        expert_inputs = hidden.index_select(0, token_ids)
        expert_outputs = expert(expert_inputs)
        weighted = expert_outputs * weights[token_ids, slots, None]
        output = output.index_add(0, token_ids, weighted)
    return output
```

`indices` 和 `weights` 都是 `[tokens, top_k]`。`token_ids` 找到原始位置，`slots` 找到这个专家是该 token 的第几个选择。`index_select` 取输入，`index_add` 把输出累加回去。`selected_gates` 在[完整代码](../code/moe_routing.py)里：可选择只对入选 logits 做 softmax，或保留全体 softmax 中的门值。

这是方便检查的 **dropless 单机教学实现**：不丢分配，不做 capacity 限制。Python 循环、逐专家索引和新的输出张量都不是高吞吐方案。实际实现可能排序 token、做 grouped GEMM，再通过 all-to-all 跨卡收发；这些优化仍要保持相同的分发与合并语义。CUDA 上累加还可能有数值非确定性，见 [PyTorch `index_add_` 说明](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.index_add_.html)。这里的测试在 CPU 上运行。

</details>

## 怎么知道实现没把梯度弄丢？ {#routing-gradients}

先做一个慢但容易信任的对照：让所有专家都计算全部输入，把未选专家的权重设为 0，再求和。对没有 dropout 等随机性、也没有跨 token 耦合的专家，这应该与稀疏分发得到相同输出；输入、router 和专家参数的梯度也应一致。它是测试用的 **dense oracle**，不是部署方式。

还可以只看一个 token，刻意留出一个从未入选的专家：

| 门值规则 | Router 的任务梯度 | 未入选专家的参数 |
| --- | --- | --- |
| top-2，入选后归一化 | 通过两个门值传回；当前选集外的 logits 不参与分母 | 这个 token 不给它任务梯度 |
| top-1，入选后归一化 | 唯一权重恒为 1，这条门值路径的梯度为 0 | 同上 |
| top-1，保留全体 softmax 概率 | 选中的概率不是常数，分母还涉及其他 logits | 同上；不要把 logit 梯度当成专家参数梯度 |

表里说的是**当前 token、当前任务 loss、固定选集附近**。其他 token、辅助 loss、共享参数都会改变实际训练中的梯度；即使门值可导，特定输出也可能让梯度恰好为 0。

有限差分检查要避开 top-k 的并列和切换边界。比如第二名和第三名几乎相等，微小扰动就可能换专家，此时不能拿同一条光滑分支的导数解释两边。PyTorch 也不保证 [`topk` 对并列元素返回稳定索引](https://docs.pytorch.org/docs/2.8/generated/torch.topk.html)。

## 跨设备时，为什么还要先选组？ {#group-limited-routing}

如果 top-2 恰好选中两张卡上的专家，一份 token 表示就要发往两个目的地。把候选专家限制在少数设备组内，可以减少目的地，但也可能放弃原本得分更高的专家。

看一个手工例子：A 组的分数是 `[0.90, 0.10]`，B 组是 `[0.80, 0.79]`，每个 token 选两个专家。不限组时选 `0.90` 和 `0.80`；如果先按组内最大值选一个组，就会选 A，最后只能用 `0.90` 和 `0.10`。少访问一组是收益，放弃 `0.80` 是代价，不能仅凭通信减少就断言质量不变。

[DeepSeek-V2 §2.2.2](https://arxiv.org/html/2405.04434v5#S2.SS2.SSS2)采用设备范围限制；[V3 §2.1.2](https://arxiv.org/html/2412.19437v1#S2.SS1.SSS2)的节点评分则汇总每节点若干个最高 affinity，不能把所有分组路由都写成“取组内最大值”。同一个例子若按组内前两个分数之和选组，B 的 `1.59` 就超过 A 的 `1.00`。组评分规则本身会改变选择。

完整代码的 `group_limited_topk` 把组数、保留组数和组内评分宽度分开。候选不足时直接报错；屏蔽的组用负无穷而不是 0，避免输入是负 logits 时被错误选中。它只检查选择逻辑，没有模拟网络通信，更不是完整 V2 / V3 路由器。

在仓库根目录运行下面的 CPU 演示，会检查输出 shape 和反传，再比较两种均衡统计：

```bash
python3 00-foundations/code/moe_routing.py
```

接下来读[负载均衡](load-balancing.md#sequence-balance)：选出来的专家数相同，不代表每个专家分到的活相同。

## Token 选 expert，还是 expert 选 token

上面都是 token 挑 expert。[Expert Choice](https://arxiv.org/abs/2202.09368) 把它倒过来：每个 expert 从一批 token 中挑固定数量。槽位足够时，分配数可以按构造保持均衡，但各卡耗时未必相同；同一 token 也可能被多选或完全不选。论文报告了特定训练设置下的收敛速度收益，不能直接当作任意服务的提速。

自回归任务还要多问一句：候选集合里有没有未来 token？如果一个早期 token 是否入选取决于后面的 token，选择操作本身就可能泄漏未来信息。训练时要限制候选范围；部署时还需检查 batch 组成是否改变单条请求的行为。
