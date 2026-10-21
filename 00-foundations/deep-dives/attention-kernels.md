# FlashAttention 和 PagedAttention，分别省了什么？

**中文** · [English](attention-kernels.en.md)

> 最近审阅：2026-10 · 前置：[多头注意力](../core/multi-head-attention.md)、[KV cache](kv-cache-and-inference.md)

两者名字很像，但不是同一道题。**FlashAttention 关心计算 attention 时搬了多少数据；PagedAttention 关心很多请求同时生成时，KV cache 怎样放。**一个主要改变计算与读写方式，另一个主要改变缓存的组织方式。

先各算一个小例子，再讨论它们什么时候有用。

## 一行 attention，不一定要存下整行概率

固定一个 query，设它与 3 个 keys 的分数为 $[\log1,\log2,\log3]$，对应的标量 values 为 $[10,20,0]$。这是方便手算的例子，真实 value 是向量。

$$
\begin{aligned}
o&=\frac{1\cdot10+2\cdot20+3\cdot0}{1+2+3}\\
 &=\frac{50}{6}\approx8.3333.
\end{aligned}
$$

直接做法先存分数、softmax 概率，再乘 values。若每个 query 都这样展开，普通实现可能产生 $T\times T$ 的中间矩阵。我们其实只需要最后的加权结果，不一定要把所有中间概率写到显存里。

## Online softmax：新块来了，旧统计怎样调整？

已经读过的分数最大值记为 $m$，保存两个量：

$$
\begin{aligned}
\ell&=\sum_j e^{s_j-m},\\
u&=\sum_j e^{s_j-m}v_j.
\end{aligned}
$$

最终输出是 $u/\ell$。新块的最大分数是 $m_b$，更新到 $m'=\max(m,m_b)$ 时，旧统计要乘 $a=e^{m-m'}$：

$$
\begin{aligned}
\ell'&=a\ell+\sum_{j\in b}e^{s_j-m'},\\
u'&=au+\sum_{j\in b}e^{s_j-m'}v_j.
\end{aligned}
$$

不是“每块分别 softmax，再平均结果”。块的归一化分母不同，直接平均会算错。

| 读到哪里 | $m$ | $\ell$ | $u$ |
| --- | --- | ---: | ---: |
| 前两个位置 | $\log2$ | 1.5 | 25 |
| 加入第三个位置 | $\log3$ | 2 | $50/3$ |

第二步把旧统计乘 $2/3$，再加新块。最后仍得到 $25/3$。第三个 value 虽然是 0，分母还是增加了，不能把这个位置忽略。

### Online 不代表概率可以立刻定下来 {#online-output}

只看到前两个位置时，概率是 $[1/3,2/3]$；第三个位置加入后，前两项就变成了 $[1/6,2/6]$。后面还有多少分数没读到，分母就还没定下来。

因此，online softmax 能一遍维护最大值和归一化分母，**不代表读到一个元素就能输出它最终的概率**。[2018 年的原论文](https://arxiv.org/html/1805.02867v2#S3)算法 3 仍有一个输出阶段。若要返回完整概率向量，还要重新读输入或保留所需数据；attention 则可以同时积累加权分子 $u$，最后只输出 $u/\ell$。这才是本例省下整行概率存储的原因。

<details markdown="1">
<summary>两个块已经分别算完，怎样合起来？</summary>

块 A、B 各保存 $(m_A,\ell_A,u_A)$ 和 $(m_B,\ell_B,u_B)$。先选共同的最大值 $m=\max(m_A,m_B)$，再把两边换到同一个尺度：

$$
\begin{aligned}
\ell&=e^{m_A-m}\ell_A+e^{m_B-m}\ell_B,\\
u&=e^{m_A-m}u_A+e^{m_B-m}u_B.
\end{aligned}
$$

比如本例前两项的加权结果是 $50/3$，第三项单独算是 0。直接把两个输出平均会得到 $25/3$，这次恰好与正确答案相同，因为两块的未归一化权重和都是 3。**这个巧合不能当作算法。** 换成分数 $[\log1,\log1,\log8]$、values $[10,20,0]$，两块平均变成 7.5，正确结果却是 3。合并统计量才会保留两块不同的权重。

实数运算中这些统计可以按不同顺序合并；浮点求和仍有舍入差异。整块都被 mask 时直接跳过，避免算 $e^{-\infty-(-\infty)}$；整行都被 mask 则没有可归一化的分布。

</details>

<details markdown="1">
<summary>少一遍扫描，具体省在哪里？</summary>

按原论文算法 2 / 3 的计数，safe softmax 分别扫描输入来找最大值、求分母、算输出：每个元素读三次、写一次。Online normalizer 合并前两遍，变成读两次、写一次。这里数的是算法中的内存访问，**不是 CPU 与 GPU 往返传输的次数**。

两者时间复杂度仍是 $O(n)$。实际 kernel 能否把数据留在寄存器或共享内存、是否与后续算子融合，还会改变真正的 HBM 流量和耗时；不能拿扫描次数直接推算加速比。

</details>

## 用几十行代码核对，而不是先写 GPU kernel

下面是单行、标量 value 的教学实现。它验证数学，不模拟 GPU 性能。被 mask 的位置用负无穷表示，可以跳过；NaN、正无穷和可见位置的非有限 value 则报错。整行都无效时也明确报错，而不是默默输出 NaN。

```python
import math

def streaming_attention(scores, values, block_size):
    if (len(scores) != len(values) or not isinstance(block_size, int)
            or isinstance(block_size, bool) or block_size < 1):
        raise ValueError("Invalid shape or block size")
    if any(score != -math.inf and not math.isfinite(score) for score in scores):
        raise ValueError("Scores must be finite or -inf for masked positions")
    if any(not math.isfinite(value) for score, value in zip(scores, values)
           if score != -math.inf):
        raise ValueError("Visible values must be finite")
    running_max = -math.inf
    normalizer = 0.0
    numerator = 0.0
    for start in range(0, len(scores), block_size):
        pairs = [(score, value) for score, value in
                 zip(scores[start:start + block_size], values[start:start + block_size])
                 if score != -math.inf]
        if not pairs:
            continue
        new_max = max(running_max, max(score for score, value in pairs))
        correction = math.exp(running_max - new_max)
        normalizer *= correction
        numerator *= correction
        for score, value in pairs:
            weight = math.exp(score - new_max)
            normalizer += weight
            numerator += weight * value
        running_max = new_max
    if normalizer == 0:
        raise ValueError("No valid keys")
    return numerator / normalizer

scores = [math.log(1), math.log(2), math.log(3)]
assert math.isclose(streaming_attention(scores, [10, 20, 0], 2), 25 / 3)
```

可以把 block size 改成 1 或 3，再给所有分数加 1000。正确的归一化结果应保持一致；浮点误差按容差比较。实际实现还需处理向量 values、批次、不同 mask、dropout 和梯度。

## FlashAttention 的收益，不等于少算所有配对

[FlashAttention](https://arxiv.org/abs/2205.14135)将分块计算与这些统计量结合，尽量在较快的片上存储中处理块，避免把完整 attention 中间矩阵反复写入、读出 HBM。反向阶段也可以重算部分量。片上存储和 HBM 的容量、带宽差多少，见 [GPU 架构的内存层级](../../12-infra/gpu-architecture.md)；分块读进 shared memory 再反复使用的写法，见 [CUDA 的 tiling](../../12-infra/cuda.md#tiling)。

对 dense attention，它仍计算允许的 query–key 配对，通常没有把二次方配对数变成线性。数学上的 exact attention 也不承诺不同运算顺序下逐 bit 相同。

短序列、特殊 mask、不同 head dimension 或不匹配的硬件，收益会不同。测试应同时比较输出和梯度，并分别测 prefill、decode；不要把某个 kernel 的速度提升直接当作端到端提升。

## FA2 到 FA4 为什么还要继续改？

不再存整张矩阵以后，GPU 也未必已经忙满。计算怎样分给线程、不同操作能否重叠，仍会影响速度。

| 版本 | 主要改动 | 别误会成什么 |
| --- | --- | --- |
| FA1 | 分块与 online softmax，减少 HBM 中间量读写 | 少看一部分 keys |
| FA2 | 减少非矩阵乘操作，改进线程块与 warp 之间的工作划分 | 换一个近似 attention 目标 |
| FA3 | 面向 Hopper，把异步搬运、矩阵乘与 softmax 更好地重叠；另有 FP8 路径 | 所有硬件都会同样受益，或只能算 FP8 |
| FA4 | 论文面向 Blackwell（B200 / GB200）：全异步矩阵乘流水线和更大的块；软件模拟 exp，只在必要时 rescale；反向用 tensor memory 和 2-CTA 矩阵乘减少 shared memory 读写 | 论文在 B200 上报告的加速比，换到 Hopper 或你的形状上同样成立 |

[FA2](https://arxiv.org/abs/2307.08691)中一个具体改动是让不同 warps 负责不同 Query 行，减少交换部分结果的工作。这里的 warp 间通信主要涉及**片上 shared memory**，不要和 GPU 显存 HBM 混在一起；几种内存谁能看见、放在哪里，见 [CUDA 的内存空间](../../12-infra/cuda.md#memory-spaces)。

[FA3](https://arxiv.org/abs/2407.08608)进一步利用 Hopper 的异步执行能力（TMA 等异步搬运，见 [NVIDIA 各代 GPU](../../12-infra/nvidia-generations.md)）。它同时讨论 FP16 和 FP8；采用 FP8 时，量化误差要单独测，不能拿“exact attention”当作没有数值误差的保证。

[FA4](https://arxiv.org/abs/2603.05451)（2026-03）针对 Blackwell。这一代的 Tensor Core 吞吐翻倍，shared memory 带宽和算指数的单元却没跟着涨，瓶颈又回到了矩阵乘以外的部分，所以它主要减少 exp、rescale 和 shared memory 流量。作者报告：B200 上 BF16 最高比 cuDNN 9.13 快 1.3 倍、比 Triton 快 2.7 倍。这是作者自报的 kernel 数字，不是端到端结果。实现全部用嵌在 Python 里的 CuTe-DSL 写成。

发布的代码比论文覆盖得更宽：截至 2026-10，[官方 README](https://github.com/Dao-AILab/flash-attention/blob/e7a6402a9ff099de5eb628ce207a88a3f0aa5378/README.md#flashattention-4-cutedsl) 写明 FA4 针对 Hopper 和 Blackwell（如 H100、B200）优化，用 `pip install flash-attn-4` 单独安装。上面的加速数字来自 B200；在 Hopper 上选 FA3 还是 FA4，要用自己的形状实测。

判断要不要用某个版本，先看当前硬件、形状和 dtype 支不支持，再测端到端收益。

## PagedAttention：逻辑连续，不要求物理连续

假设一个物理块能存 4 个 token 的 KV。请求 A 有 5 个 tokens，请求 B 有 3 个：

```text
A 的逻辑块 0 → 物理块 7：[A0 A1 A2 A3]
A 的逻辑块 1 → 物理块 2：[A4 -- -- --]
B 的逻辑块 0 → 物理块 5：[B0 B1 B2 --]
```

共分配 3 块、12 个槽，使用 8 个槽，尾块空了 4 个槽。分块没有消灭所有浪费，但避免了每个请求都预留最大长度的大块连续内存。

A 再把 3 个新 tokens 送入模型、形成它们的 K/V 后，可填满已有尾块；再追加一个位置才需要新块。请求结束后，引用计数允许时释放对应块。两个请求若共享相同前缀与模型计算条件，可以共享相应缓存；写入共享尾块时需要 copy-on-write 或等价机制，不能把别人的前缀改掉。

[PagedAttention 论文](https://arxiv.org/abs/2309.06180)讨论了这种块映射与共享。这里的块号是自拟示意，不是 vLLM API。缓存是否可共享，还要核对权重、adapter、位置与多模态输入等条件，而不只是文本相同。

上图的尾块空槽叫**内部碎片（internal fragmentation）**。另一种情况是空闲空间散在几处，总量够，却放不下一段要求连续的大缓存，这是**外部碎片（external fragmentation）**。所有块大小相同，所以分页消除了外部碎片；按需分配小块，又把内部碎片限制在每个请求最后一个未填满的块里。块表和这些尾块空槽仍有成本。

还要分清论文里的共享机制与当前框架实际开放的功能。[vLLM 的 prefix caching 说明](https://docs.vllm.ai/en/latest/design/prefix_caching/)目前按完整块复用缓存，并把前缀、token IDs 及 adapter / 多模态信息等纳入缓存键。不能据此假定它会复用任意半满尾块。

<span id="cache-lifecycle"></span>

## 请求结束，不代表缓存立刻消失

按本次核对的 vLLM V1 `10cc2f6`，活跃请求持有引用；最后一个引用释放后，缓存块可进入空闲队列，但内容仍可能被再次命中。新分配复用该块时会移除旧映射。这是容量驱动的回收，不是“过 10 分钟自动失效”的 TTL。[固定版本的 block pool](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/block_pool.py#L729)分别处理缓存块与未缓存块，不能把所有空闲块都理解成同一条纯 LRU 队列。

区分这三件事：**请求不再持有、缓存不能再命中、GPU 内存归还给系统**。它们不是同一时刻，也不是同一个操作；预分配的池可以一直占着显存供后续使用。取消请求时这些引用怎样释放，见[请求生命周期](../../06-systems/llm-serving.md#request-lifecycle)。

拿一个已计算好的前缀块看就更直观：

| 发生的事 | 活跃引用数 | 这个块处于什么状态？ |
| --- | ---: | --- |
| 请求 A 持有它 | 1 | 仍在使用，不能给不相关内容覆盖 |
| 请求 B 命中同一前缀 | 2 | 两个请求共享，不是复制两份 KV |
| A 结束 | 1 | B 还在使用 |
| B 也结束 | 0 | 可以被回收，但旧缓存仍可能命中 |
| 新请求命中，或分配器复用它 | 1 | 前者复用旧内容；后者先撤销旧映射，再存新内容 |

例如共用前缀有 7 个 token，块大小为 4，那么完整块路径至多复用前 4 个，不是 7 个。若第 2 个 token 改了，第一个块已经不匹配，后面的同样文字也不能单独拼上去：那些 KV 是在另一个前缀下算出来的。这里讲普通 full attention；混合 attention 的 cache group、hash 粒度与物理块可能不一一对应，要按实际配置检查。

## vLLM 和 SGLang，不是“谁更细就谁更快”

两者的索引结构（块哈希对 radix 树）、默认粒度和回收策略，对比见[推理框架一篇](../../06-systems/inference-frameworks.md#prefix)。树节点能拆分，不代表所有配置都按单 token 复用。这里只拿上面 7 个 token 的共同前缀，算一笔页对齐的账。本次核对 SGLang `436d73d`：普通模型默认走 [UnifiedRadixCache](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/mem_cache/registry.py#L142)，page size [默认是 1](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/arg_groups/overrides.py#L1282)，只有部分后端或模型会调大；`page_size > 1` 时，[匹配前先做页对齐](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L840)。page size=1 时可以保留 7；page size=4 则向下对齐到 4。若 vLLM 使用基础默认值 16 作为块大小，同一个例子凑不满一个完整块，命中为 0。它说明配置会改变复用粒度，不说明哪个系统更快。

所以不写“高并发一定选 vLLM、agent 一定选 SGLang”。先拿自己的请求回放，固定模型和预算；同一框架换 backend，也可能比换框架带来更大的差别。同一批请求该核对哪些条件、报告哪些延迟指标，见[推理框架的对照实验](../../06-systems/inference-frameworks.md#benchmark)。这里核对的是机制与公开实现，没有运行 serving 性能对比。

## 分页、连续组批、分块 prefill 不是同一个开关

| 机制 | 管哪件事 | 一个直接的问题 |
| --- | --- | --- |
| PagedAttention | KV 放在哪里 | 请求变长时，能否不搬整段缓存？ |
| Continuous batching | 下一步让哪些请求一起跑 | 一个请求结束了，新请求能否马上补位？ |
| Chunked prefill | 长 prompt 一次处理多少 | 新来的长 prompt 会不会让正在输出的人等太久？ |

假设只能同时处理 2 条请求，A、B、C 还分别需要 4、1、2 步 decode。为了只看调度，**暂不计 prefill、到达时间和每步耗时差别**。

| Decode 轮次 | 固定一批跑完再换下一批 | 每轮允许补位 |
| --- | --- | --- |
| 1 | A + B，B 完成 | A + B，B 完成 |
| 2 | A | A + C |
| 3 | A | A + C，C 完成 |
| 4 | A，A 完成 | A，A 完成 |
| 5–6 | C，直到完成 | 已完成 |

补位把这个理想例子从 6 轮变成 4 轮，**不等于实测延迟下降三分之一**。真实请求还需要 prefill，也会争抢带宽与缓存。Chunked prefill 让长 prompt 分几次处理，代价是首 token 可能更晚；能换来更平稳的 decode。[vLLM 的调优文档](https://docs.vllm.ai/en/latest/configuration/optimization/)介绍了这类 TTFT / ITL 取舍。这里的 prefill chunk 大小也不是 KV 物理块大小。

## 两种优化怎样放在一起看？

| 问题 | 更相关的机制 | 仍需测量 |
| --- | --- | --- |
| Prefill 的中间 attention 矩阵太大 | FlashAttention | 峰值显存、前向/反向误差、耗时 |
| 并发请求长短不同，缓存碎片大 | PagedAttention | 块利用率、吞吐、调度开销 |
| 每条请求的 KV 本体就很大 | GQA / MQA / MLA 等架构 | 质量、缓存大小、实际 kernel |
| 只看少量历史能否更快 | 稀疏注意力 | 被遗漏的信息、选择开销 |

这些方法可以组合，但组合效果取决于实现。要报告硬件、dtype、序列长度、batch 和计时边界；先确认瓶颈，再选优化，比把名字全打开更可靠。
