# 张量并行：两张卡怎样算同一层？

**中文** · [English](tensor-parallel.en.md)

> 最近核对：2026-10 · 前置：[矩阵与 shape](../00-foundations/pytorch/README.md)、[多卡训练](distributed-training.md)

把一层交给两张 GPU，难的不是把权重切成两半，而是切完以后，下一步还算不算得下去。每张卡手上的结果，是完整答案的一部分，还是还需要相加的半成品？这决定了什么时候必须通信。

先用一个很小的 FFN 算一遍，再看 attention 和反向传播。不熟悉分布式框架也没关系：前半篇只需要矩阵乘法；想对照实现，可以展开后面的 PyTorch 例子。

## 先算对，再决定怎么传 {#two-ranks}

假设输入是 `[2, −1]`，第一层有 4 个输出，第二层把它们变回 2 个。为了方便手算，激活函数用 ReLU，暂时不加 bias 和 dropout。

第一层按**输出通道**分给两张卡；第二层按对应的**输入通道**分。这里称为列切、行切，都是针对公式 `输入 @ 权重` 中的权重矩阵说的。

<figure class="worked-update">
<ol>
<li><small>01 · 相同输入</small><strong>两卡都拿到 [2, −1]</strong><span>rank 0 负责前 2 个中间通道，rank 1 负责后 2 个。不是一张卡只读输入的一个数字。</span></li>
<li><small>02 · 各自计算</small><strong>[1, 0] 与 [0, 4]</strong><span>各卡算完第一层和 ReLU，直接接上自己的第二层权重。中间不必拼回 4 个通道。</span></li>
<li><small>03 · 相加，不是拼接</small><strong>[1, 2] + [8, 4]</strong><span>两卡得到的都是 2 维部分和。AllReduce 求和后，每卡都有完整输出 [9, 6]。</span></li>
</ol>
<figcaption>rank 是进程在通信组里的编号。这里为了讲清楚，让一个 rank 对应一张 GPU；图中所有数字都是教学算例。</figcaption>
</figure>

具体权重如下。两张卡各持有一个 $A_i$ 和一个 $B_i$：

$$
A_0=\begin{bmatrix}1&0\\1&1\end{bmatrix},\quad
A_1=\begin{bmatrix}-1&2\\0&0\end{bmatrix}.
$$

$$
B_0=\begin{bmatrix}1&2\\0&1\end{bmatrix},\quad
B_1=\begin{bmatrix}1&0\\2&1\end{bmatrix}.
$$

rank 0：`[2, −1] @ A₀ = [1, −1]`，ReLU 后是 `[1, 0]`，再乘 `B₀` 得 `[1, 2]`。rank 1：先得到 `[−2, 4]`，ReLU 后是 `[0, 4]`，再乘 `B₁` 得 `[8, 4]`。合起来就是 `[9, 6]`。

如果误用了拼接，会得到 4 维的 `[1, 2, 8, 4]`；如果误用了平均，会得到 `[4.5, 3]`。它们都不是原来那层的输出。

## 列切接行切，省掉了什么？ {#column-row}

设 $N$ 是本轮参与计算的 token 数，$H$ 是 hidden size，$F$ 是 FFN 中间宽度，先假设 $F$ 能被 2 整除。

| 张量 | 完整 shape | 每卡有什么 |
| --- | --- | --- |
| 输入 $X$ | $N\times H$ | 同一份完整输入 |
| 第一层 $A$ | $H\times F$ | $H\times(F/2)$ 的列分片 |
| 激活 $U$ | $N\times F$ | $N\times(F/2)$ 的通道分片 |
| 第二层 $B$ | $F\times H$ | $(F/2)\times H$ 的行分片 |
| 局部输出 $Z_i$ | $N\times H$ | 同一 shape，但只是部分和 |

每卡独立算 $U_i=\phi(XA_i)$、$Z_i=U_iB_i$，最后得到

$$Y=Z_0+Z_1.$$

关键在中间的 $U$：下一层恰好只需要本卡已有的通道，所以不用 AllGather。原始 [Megatron-LM §3](https://arxiv.org/html/1909.08053v4#S3)采用这种配对，让 FFN 的前向只在末尾归约一次。

反过来，若第一层先沿输入维切分，同一个中间通道会得到两份部分和。必须先加起来，才能过非线性。用一个数字就能看出问题：

$$\operatorname{ReLU}(3-4)=0,$$

$$\operatorname{ReLU}(3)+\operatorname{ReLU}(-4)=3.$$

所以“先行切”不是数学上不允许，而是需要在激活前先合并部分和；之后怎样布置第二层，还会影响新的通信。列切接行切是一种省通信的安排，不是所有矩阵都必须遵守的口诀。

SwiGLU 也要保持通道配对：同一个中间通道的 gate 和 up 分支应在本卡完成逐元素相乘，再接对应的 down 分片。不要把两个分支随意切开，以为维度对上就行。

## AllGather 和 AllReduce 到底差在哪？ {#collectives}

先问两件事：**值要不要相加？结果谁需要？** 下表中的归约都取 SUM；通信原语本身也可以支持其他运算。

| 操作 | 两个 rank 的小例子 | 完成后 |
| --- | --- | --- |
| Broadcast | root 有 `[2, 5]` | 每卡都有 `[2, 5]` |
| Scatter | root 有 `[2, 5, 7, 9]` | rank 0 得 `[2, 5]`，rank 1 得 `[7, 9]` |
| Gather | 各卡有 `[2, 5]`、`[7, 9]` | 只在 root 拼成 `[2, 5, 7, 9]` |
| AllGather | 同上 | 每卡都有拼好的 4 个值 |
| Reduce | 各卡有 `[2, 5]`、`[7, 9]` | 只在 root 得到和 `[9, 14]` |
| AllReduce | 同上 | 每卡都有和 `[9, 14]` |
| ReduceScatter | 同上 | 求和后再分开：rank 0 得 `[9]`，rank 1 得 `[14]` |

AllGather 没有替你求和，AllReduce 也不会自动把通道拼成长向量。ReduceScatter 再 AllGather，可以得到与 AllReduce 相同的归约结果，但这不要求实际实现一定用两次独立 API 调用。[NCCL 通信说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)定义的是这些结果语义；ring、tree 等是实现它们的算法。

## 反向传播为什么还要通信？ {#backward}

设输出梯度为 $G=\partial L/\partial Y$。因为 $Y$ 是局部输出的和，每个 $Z_i$ 都接收同一个 $G$。随后每卡可以独立求第二层权重梯度、中间激活梯度，以及自己的第一层权重梯度：

$$\nabla B_i=U_i^\top G,$$

$$D_i=(GB_i^\top)\odot\phi'(XA_i),$$

$$\nabla A_i=X^\top D_i.$$

但是对**输入**的梯度还没齐。输入参与了两卡的计算，两条路径的贡献都要算进去：

$$\nabla X=D_0A_0^\top+D_1A_1^\top.$$

这就是 FFN 反向里的那次 AllReduce。上面算例若取 `loss = output.sum()`，两卡的输入梯度贡献分别为 `[3, 3]` 和 `[6, 0]`，相加才得到 `[9, 3]`。

注意这里是在把**同一次计算的不同路径相加**，不是把两批不同样本的梯度取平均。不要直接套用数据并行的平均系数。

<details markdown="1">
<summary>用 PyTorch 核对输出和梯度（CPU 即可）</summary>

代码把两个 rank 的运算放在同一进程里，以便逐项比较。它检查代数，不模拟 NCCL、网络或多卡性能。权重按数学公式的方向保存；`torch.nn.Linear.weight` 实际存储为 `[out_features, in_features]`，切真实参数时要留意这个转置。

```python
import torch

def split_ffn(inputs, first_weight, second_weight, parts=2):
    if type(parts) is not int or parts < 1:
        raise ValueError("parts must be a positive integer")
    if any(tensor.ndim != 2 for tensor in (inputs, first_weight, second_weight)):
        raise ValueError("expected matrices")
    if inputs.shape[1] != first_weight.shape[0]:
        raise ValueError("input width does not match the first layer")
    if first_weight.shape[1] != second_weight.shape[0]:
        raise ValueError("intermediate widths must match")
    if first_weight.shape[1] == 0 or first_weight.shape[1] % parts:
        raise ValueError("intermediate width must divide evenly into parts")
    first_shards = first_weight.chunk(parts, dim=1)
    second_shards = second_weight.chunk(parts, dim=0)
    partials = [torch.relu(inputs @ first) @ second
                for first, second in zip(first_shards, second_shards)]
    return torch.stack(partials).sum(dim=0)

inputs = torch.tensor([[2., -1.]], dtype=torch.float64, requires_grad=True)
first_weight = torch.tensor([[1., 0., -1., 2.], [1., 1., 0., 0.]],
                            dtype=torch.float64, requires_grad=True)
second_weight = torch.tensor([[1., 2.], [0., 1.], [1., 0.], [2., 1.]],
                             dtype=torch.float64, requires_grad=True)
parameters = (inputs, first_weight, second_weight)
full = torch.relu(inputs @ first_weight) @ second_weight
split = split_ffn(*parameters)
torch.testing.assert_close(split, full)
torch.testing.assert_close(split, torch.tensor([[9., 6.]], dtype=torch.float64))
full_grads = torch.autograd.grad(full.sum(), parameters)
split_grads = torch.autograd.grad(split.sum(), parameters)
for full_grad, split_grad in zip(full_grads, split_grads):
    torch.testing.assert_close(full_grad, split_grad)

with torch.no_grad():
    upstream = torch.ones_like(full)
    input_contributions = []
    for first, second in zip(first_weight.chunk(2, 1), second_weight.chunk(2, 0)):
        preactivation = inputs @ first
        local_gradient = (upstream @ second.T) * (preactivation > 0)
        input_contributions.append(local_gradient @ first.T)
    combined = torch.stack(input_contributions).sum(0)
    torch.testing.assert_close(combined, full_grads[0])
    torch.testing.assert_close(combined, torch.tensor([[9., 3.]], dtype=torch.float64))
```

真实实现还要处理 dtype、设备、随机数、bias 和通信。比如第二层 bias 应在部分和归约后加一次；每卡先加完整 bias 再求和，会把它重复加进去。

</details>

## 一层 Decoder：什么时候是“四次”？ {#decoder-count}

先限定范围：普通 dense decoder block、标准多头注意力（MHA）、TP 组内有完整输入副本、无 sequence/context parallel、无激活重算。只数 block 内的逻辑 collective，不数网络包、DP 梯度同步、词表输出层或跨流水线 stage 的传输。

Attention 里可以按完整 head 分：Q/K/V 投影生成本卡的 heads，本卡完成它们的 attention，输出投影沿输入通道切。比如 8 heads、TP=2，每卡算 4 heads，并不是强制“一卡一头”。每个 head 仍能访问原来允许的上下文，并保留同样的 attention mask；切的是 heads，不是把前后半段文本隔开。输出投影产生部分和，再相加。

| block 内的位置 | 前向 | 反向 |
| --- | --- | --- |
| Attention 分支 | 输出投影后求和 1 次 | 对输入的梯度贡献求和 1 次 |
| FFN 分支 | 第二层后求和 1 次 | 对输入的梯度贡献求和 1 次 |
| 合计 | 2 次 | 2 次 |

因此这个配置训练时一共 4 次，纯前向推理是 2 次。不能把“每层 4 次”直接乘进推理延迟，也不能把一次 AllReduce 当成一次网络传输。

为什么不是前向归约的地方，反向再归约一次？这里下游的完整输出与梯度在 TP 组内是副本，不是几份独立损失。重复求和会把同一贡献多算。Megatron 的 [mapping 实现](https://github.com/NVIDIA/Megatron-LM/blob/4603a836261fb39fd0050342d58dc66aecdf6f74/megatron/core/tensor_parallel/mappings.py)明确区分“前向不动、反向归约”和“前向归约、反向不动”的区域边界。不能不看张量布局，就给所有通信套同一种反向规则。

## 加上 SP，为什么 profiler 看起来又不一样？ {#sequence-parallel}

普通 TP 仍可能在每卡保存同一份残差和归一化激活。Megatron 风格的 sequence parallel（SP）把这些区域沿 token 维分开，到列并行计算前再 AllGather，行并行输出后用 ReduceScatter 留下本卡的 token 分片。

以前向 FFN 为例：

<figure class="worked-update">
<ol>
<li><small>入口 · AllGather</small><strong>收齐本轮 token</strong><span>输入从 N/2 × H 的 token 分片，变为 N × H，供本卡的列分片计算。</span></li>
<li><small>中间 · 本地计算</small><strong>只算本卡通道</strong><span>中间激活仍是 N × F/2。token 分片和通道分片不是同一个维度。</span></li>
<li><small>出口 · ReduceScatter</small><strong>求和后只留本卡 token</strong><span>局部输出先归约，再按 token 分片，各卡回到 N/2 × H。</span></li>
</ol>
<figcaption>这是一个明确的布局示例，不是所有框架的 SP 定义。SP 本身也不等于把 attention 的上下文依赖切断。</figcaption>
</figure>

配对的反向会交换 AllGather / ReduceScatter 的角色；实现可以把输入梯度通信与权重梯度计算重叠。看 [固定版本的线性层](https://github.com/NVIDIA/Megatron-LM/blob/4603a836261fb39fd0050342d58dc66aecdf6f74/megatron/core/tensor_parallel/layers.py) 时，把 `sequence_parallel`、`gather_output` 和 `input_is_parallel` 连起来看，别只搜一个 AllReduce 名字。该版本于 2026-10-09 核对，其他 backend 或更新后的实现需要重新确认。

GQA / MQA 还要检查 KV heads 如何分配；头数少时可能有复制。MoE 会引入 expert dispatch，context parallel 会交换上下文相关信息，激活重算会重跑部分前向。因此“四次”是理解一种基线的起点，不是所有 Decoder 的固定属性。

## 配置以前，先把这几件事画出来 {#before-scaling}

选一个 block，标清每步输入的 shape、哪个维度被切、结果是分片还是部分和。然后对照单卡，比较输出、输入梯度、权重梯度与一次更新。先关 dropout 做基线，再检查真实随机数策略；还要确认 bias 和 residual 没有重复相加。

正确以后再看性能：每次传多少字节，什么时候必须等，矩阵切小以后是否还算得高效。**少一个 collective 不一定更快，多几张卡也不一定更快。** 如果你关心的是“卡数相同，该用 TP 还是副本”，接着看[部署选择](distributed-training.md#eight-gpus)；如果在排查首 token 很慢，转到[请求与调度](llm-serving.md)。
