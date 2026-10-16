# MoE：训练和推理的系统代价

**中文** · [English](systems.en.md)

> 阅读时间：约 6 分钟 · 难度：进阶 · 最近审阅：2026-10

每个 token 只算 8 个专家，为什么部署起来还是不省事？因为“算得少”只是一笔账。另外两笔是：token 要在卡之间搬来搬去，全部专家的权重都得放得下。

**MoE 把每 token 的计算压下去，代价转成了通信和显存；划不划算，要看互联带宽、显存总量和 batch 大小。** 下面用 DeepSeek-V3 的尺寸各算一笔。

## Expert parallelism 与两次 all-to-all

模型一大，expert 就得分散在不同的卡上。一层 MoE 的前向于是变成三步（GShard 的做法）：

1. **dispatch**：一次 all-to-all，把每个 token 发到它选中的 expert 所在的卡；
2. 各卡算自己的 expert；
3. **combine**：再一次 all-to-all，把结果送回 token 原来所在的卡。

这是 expert parallelism 的典型逻辑流程，不代表任何部署都必须发两次跨卡集合通信；专家都在本地时不需要这一步。跨卡流量还取决于选中的专家落在哪里、是否合并同节点发送。排查慢 step 时，分别看派发、expert 计算、回收及重叠程度。一次集合通信要多久，怎样用带宽估算，见[性能：多卡通信](../../12-infra/performance.md#communication)。

### 算一笔：一个 token 要发多少字节？ {#dispatch-bytes}

每个 token 要把自己的 hidden state 发给选中的每个专家，所以一层单向的载荷是

$$
\text{bytes} = k \times d \times s .
$$

$k$ 是每 token 选几个专家，$d$ 是 hidden size，$s$ 是每个数几字节。用 DeepSeek-V3 的 $k=8$、$d=7168$，按 BF16 的 $s=2$：

$$
8 \times 7168 \times 2 = 114{,}688\ \text{B} = 112\ \text{KiB}.
$$

这是不合并同节点发送时的数，combine 回来还要再走一遍。一个 micro-batch 有 4,096 个 token 时，单层单向就是 448 MiB。实际系统会压这笔账：DeepSeek-V3 把 dispatch 前的激活量化成 FP8（$s=1$，减半），combine 仍用 BF16；同一节点上的几个专家只收一份，所以跨节点那段最多 4 份，FP8 下是 $4 \times 7168 \times 1$ B = 28 KiB。[V3 §3.2.2、§3.3.3](https://arxiv.org/html/2412.19437v2#S3.SS2)

## 限制一个 token 跨几台机器

跨机器的带宽比机器内部低得多：V3 报告里 H800 集群的数是节点内 NVLink 160 GB/s、跨节点 IB 50 GB/s，差约 3.2 倍。NVLink 域有多大、各代 GPU 的带宽，见 [NVIDIA 各代 GPU](../../12-infra/nvidia-generations.md)。DeepSeek-V3 的做法是 node-limited routing：一个 token 最多只发往 4 个节点，节点按「这个节点上最高的几个 expert 分数之和」来挑。top-8 的选择保住了，跨节点的流量也压住了。

## 显存：装得下才谈得上稀疏

全 GPU 常驻时，要为全部权重安排存储，而不是只按激活参数分配。DeepSeek-V3 的主模型为 671B，包含 MTP 模块的发布权重还多约 14B；gpt-oss 的 expert 权重使用 MXFP4，约 4.25 bit/参数，其他张量不全是这个精度。Offload 可以把部分权重放到 CPU，但带宽和缓存命中就会进入延迟账。具体型号来源见[模型报告列表](review.md)。

## 解码时，batch 大小会改变账

小 batch 解码常受到内存带宽限制（为什么 batch 变大会走向算力受限，见[性能：decode 的 batch](../../12-infra/performance.md#decode-batching)），但长上下文的 KV 读取、跨卡通信和较大的矩阵计算，也可能成为瓶颈，最好用 profiler 确认。

- **batch 小的时候**，一步可能只用少数 expert，读取权重较少；小矩阵效率、派发开销仍可能抵消收益。
- **batch 大的时候**，如果路由分散，更多 expert 会被用到，读权重可能接近总量，但可以摊给更多 token；偏斜路由则可能继续只用少数 expert。

### 算一笔：一步解码会碰到多少专家？ {#experts-touched}

一层有 $N$ 个 routed experts，每个 token 选 $k$ 个。假设路由均匀、各 token 互相独立，某个专家没被一个 token 选中的概率是 $1-k/N$，一步里 $B$ 个 token 都没选它的概率是 $(1-k/N)^B$。所以这一步要读的专家比例约为

$$
1-\left(1-\frac{k}{N}\right)^{B}.
$$

用 V3 的 $N=256$、$k=8$：

| 一步的 token 数 $B$ | 碰到的专家比例 |
| --- | --- |
| 1 | 3.1% |
| 32 | 63.8% |
| 128 | 98.3% |

```python
def experts_touched(experts, top_k, batch):
    if not 0 < top_k <= experts or batch < 0:
        raise ValueError("need 0 < top_k <= experts and batch >= 0")
    return 1 - (1 - top_k / experts) ** batch

assert round(experts_touched(256, 8, 1), 3) == 0.031
assert round(experts_touched(256, 8, 32), 3) == 0.638
assert round(experts_touched(256, 8, 128), 3) == 0.983
```

单条请求每步在每层只读约 3% 的专家权重；batch 到 128 时，几乎每个专家都要读一遍，这时每份权重摊给的 token 也多了。路由偏斜时，比例会低于这个均匀假设。所以“每个 token 只算一小部分”这个好处，在不同 batch 下兑现的方式并不一样；再加上 all-to-all，真实吞吐还得看具体怎么部署。

## 数值精度

router 的 softmax 对数值很敏感。Switch Transformer 让 router 单独用 fp32 算，其余部分照旧低精度；[负载均衡那篇](load-balancing.md)讲的 z-loss，针对的也是这个问题。

## 小结

MoE 部署起来要比同等质量的 dense 模型省，通常得同时满足几个条件：

- **权重放得下**：按总参数准备显存，或者能接受 offload 带来的搬运延迟。
- **通信盖得住**：dispatch 和 combine 的时间能和专家计算重叠，或者专家大多在同一个高带宽域里，比如一个 NVLink 域。
- **batch 够大**：一步里的 token 足够多，把每份读进来的专家权重摊开；batch 太小时，读得少，却也摊不开派发开销。

反过来，单卡、显存紧、追求最低延迟的小 batch 部署，这几条大多不满足，dense 模型往往更省心。具体到某次部署，用 profiler 看派发、专家计算和回收各占多少时间，再下结论。
