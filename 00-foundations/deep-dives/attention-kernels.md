# FlashAttention 和 PagedAttention，分别省了什么？

**中文** · [English](attention-kernels.en.md)

> 最近审阅：2026-10 · 前置：[多头注意力](../core/multi-head-attention.md)、[KV cache](kv-cache-and-inference.md)

两者名字很像，但不是同一道题。FlashAttention 关心计算 attention 时搬了多少数据；PagedAttention 关心很多请求同时生成时，KV cache 怎样放。一个主要改变计算与读写方式，另一个主要改变缓存的组织方式。

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

## 用几十行代码核对，而不是先写 GPU kernel

下面是单行、标量 value 的教学实现。它验证数学，不模拟 GPU 性能。被 mask 的位置用负无穷表示；整行都无效时明确报错，而不是默默输出 NaN。

```python
import math

def streaming_attention(scores, values, block_size):
    if len(scores) != len(values) or block_size < 1:
        raise ValueError("Invalid shape or block size")
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

[FlashAttention](https://arxiv.org/abs/2205.14135)将分块计算与这些统计量结合，尽量在较快的片上存储中处理块，避免把完整 attention 中间矩阵反复写入、读出 HBM。反向阶段也可以重算部分量。

对 dense attention，它仍计算允许的 query–key 配对，通常没有把二次方配对数变成线性。数学上的 exact attention 也不承诺不同运算顺序下逐 bit 相同。

短序列、特殊 mask、不同 head dimension 或不匹配的硬件，收益会不同。测试应同时比较输出和梯度，并分别测 prefill、decode；不要把某个 kernel 的速度提升直接当作端到端提升。

## FA2、FA3 为什么还要继续改？

不再存整张矩阵以后，GPU 也未必已经忙满。计算怎样分给线程、不同操作能否重叠，仍会影响速度。

| 版本 | 主要改动 | 别误会成什么 |
| --- | --- | --- |
| FA1 | 分块与 online softmax，减少 HBM 中间量读写 | 少看一部分 keys |
| FA2 | 减少非矩阵乘操作，改进线程块与 warp 之间的工作划分 | 换一个近似 attention 目标 |
| FA3 | 面向 Hopper，把异步搬运、矩阵乘与 softmax 更好地重叠；另有 FP8 路径 | 所有硬件都会同样受益，或只能算 FP8 |

[FA2](https://arxiv.org/abs/2307.08691)中一个具体改动是让不同 warps 负责不同 Query 行，减少交换部分结果的工作。这里的 warp 间通信主要涉及**片上 shared memory**，不要和 GPU 显存 HBM 混在一起。

[FA3](https://arxiv.org/abs/2407.08608)进一步利用 Hopper 的异步执行能力。它同时讨论 FP16 和 FP8；采用 FP8 时，量化误差要单独测，不能拿“exact attention”当作没有数值误差的保证。判断要不要用某个版本，先看当前硬件、形状和 dtype 支不支持，再测端到端收益。

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

上图的尾块空槽叫**内部碎片（internal fragmentation）**。另一种情况是空闲空间散在几处，总量够，却放不下一段要求连续的大缓存，这是**外部碎片（external fragmentation）**。分页主要缓解后一种问题；块表和未填满的尾块仍有成本。

还要分清论文里的共享机制与当前框架实际开放的功能。[vLLM 的 prefix caching 说明](https://docs.vllm.ai/en/latest/design/prefix_caching/)目前按完整块复用缓存，并把前缀、token IDs 及 adapter / 多模态信息等纳入缓存键。不能据此假定它会复用任意半满尾块。

## 请求结束，不代表缓存立刻消失

截至 2026-10-09 核对的 vLLM V1 设计，活跃请求持有引用；最后一个引用释放后，块可进入空闲队列，但内容仍可能被再次命中。新分配覆盖该块时才需要移除旧映射。这里主要是容量驱动的 LRU 回收，不是“过 10 分钟自动失效”的 TTL。[缓存生命周期](https://docs.vllm.ai/en/latest/design/prefix_caching/)

区分这三件事：**请求不再持有、缓存不能再命中、GPU 内存归还给系统**。它们不是同一时刻，也不是同一个操作；预分配的池可以一直占着显存供后续使用。

例如共用前缀有 7 个 token，块大小为 4，那么完整块路径至多复用前 4 个，不是 7 个。若第 2 个 token 改了，第一个块已经不匹配，后面的同样文字也不能单独拼上去：那些 KV 是在另一个前缀下算出来的。

## vLLM 和 SGLang，不是“谁更细就谁更快”

vLLM 用带前缀信息的块哈希；SGLang 的 RadixAttention 用树组织共享路径。但树节点能拆分，不代表所有配置都按单 token 复用。2026-10-09 查看 [SGLang RadixCache 实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)，`page_size > 1` 的路径会做页对齐。比较时需要记录 commit、cache backend、page size 和 attention 类型，不能只记框架名字。

| 同一批请求，要检查什么？ | 为什么 |
| --- | --- |
| token IDs、模板、adapter 与图片是否一致 | 文本相似不等于可复用同一计算 |
| 请求落在哪个副本、是否 warm-up | 缓存不一定跨副本共享 |
| 共享前缀长度与块/页大小 | 逻辑重合不等于实际命中长度 |
| 权重更新、租户隔离与主动 reset | 命中必须先正确，再谈收益 |
| TTFT、输出间隔、吞吐与尾延迟 | 命中率高也可能排队很久 |

所以不写“高并发一定选 vLLM、agent 一定选 SGLang”。先拿自己的请求回放，固定模型和预算；同一框架换 backend，也可能比换框架带来更大的差别。这里核对的是机制与公开实现，没有运行 serving 性能对比。

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
