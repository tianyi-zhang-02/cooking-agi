# KV cache：少算了什么，又多存了什么？

**中文** · [English](kv-cache-and-inference.en.md)

> 最近审阅：2026-10 · 前置：[多头注意力](../core/multi-head-attention.md)、[Decoder-only](../core/decoder-only.md)

模型刚生成一个词，下一步为什么还要重新读整段话？其实不必从头算一遍。过去位置的部分计算可以留着，这就是 KV cache 的起点。

但“有缓存所以很快”还不够。长对话会吃掉更多显存；很多人同时请求时，瓶颈又可能换地方。这里沿着一次很短的生成，把计算、存储和几个常见优化分开。

## 先看 3 个输入 token

假设 prompt 已经切成 `[A, B, C]`。字母只是 token 的占位符，不是实际分词结果。模型先并行处理这 3 个位置，利用 C 位置的输出预测下一个 token D。

| 阶段 | 本次送入的 token | 每层缓存结束时有哪些位置 | 用哪个输出做预测 |
| --- | --- | --- | --- |
| Prefill | A、B、C | A、B、C 的 K/V | C 的输出预测 D |
| 第 1 次 decode | D | A、B、C、D 的 K/V | D 的输出预测 E |
| 第 2 次 decode | E | A、B、C、D、E 的 K/V | E 的输出预测 F |

刚采样出 D 时，D 的 K/V 还没算；把 D 再送入模型，才会形成它在各层的缓存。这个时间顺序写反，很容易让代码重复处理一个 token，或漏掉一个位置。

## 旧位置为什么可以不变？

在标准 causal decoder 里，A 只能看 A，B 只能看 A、B。后面加上 D，不会改变 A、B、C 原本能看到的内容。固定权重、位置编码与前缀，在推理模式下，旧位置各层的 K/V 就可以复用。普通双向 encoder 不满足这个条件：新 token 会影响之前位置的表示，不能直接套这个结论。[Transformers 的缓存说明](https://huggingface.co/docs/transformers/cache_explanation)沿着这条因果关系解释了复用过程。

在某一层，处理新位置 $t$ 时：

$$
\begin{aligned}
K_{\le t}&=[K_{<t};k_t],\\
V_{\le t}&=[V_{<t};v_t].
\end{aligned}
$$

$$
o_t=\operatorname{softmax}\left(\frac{q_tK_{\le t}^{\top}}{\sqrt{d_h}}\right)V_{\le t}.
$$

这里只有当前 Query $q_t$，但 Key 和 Value 包括全部可见历史。已经处理过的 Query 不参与新位置这行 attention，所以通常不用缓存它们。

这不是把答案缓存下来，也没有跳过当前 token 的 FFN 或其他层。**省掉的是旧位置的重复计算，付出的是保存各层 K/V 的空间。**

## 单步变便宜，不代表长上下文免费

单个 query head 在长度为 $t$ 的历史上，需要计算大约 $t$ 个匹配分数，attention 的这部分工作随 $t$ 增加。它不再为所有旧 query 重做整个 $t\times t$ 矩阵，但也不是每步固定成本。

从长度为 $P$ 的 prompt 开始，再处理 $N$ 个新 token，仅累计 query–key 匹配数量就约为：

$$
\sum_{j=1}^{N}(P+j)=NP+\frac{N(N+1)}2.
$$

这里没算 prefill、投影、FFN 和硬件并行。它只提醒我们：越往后生成，历史越长；KV cache 没有消除全注意力读取历史的代价。

## 显存账可以自己算

对各层结构相同、未分片、没有窗口淘汰和缓存量化的 decoder，保存相同长度序列的 KV 大小为：

$$
M_{KV}=2\,B\,L\,T\,H_{KV}\,d_h\,b.
$$

| 符号 | 意思 |
| --- | --- |
| 2 | Key 和 Value 各一份 |
| $B$ | 同时缓存的序列数 |
| $L$ | 层数 |
| $T$ | 每条序列已缓存的长度 |
| $H_{KV}$ | 每层 KV head 数，不一定等于 Query head 数 |
| $d_h$ | 每个 head 的维度 |
| $b$ | 每个元素占几个字节 |

取一组假设配置：32 层、32 个 Query heads、8 个 KV heads、head dimension 128、缓存 8192 tokens、BF16 每元素 2 bytes。

```python
def kv_bytes(batch, layers, tokens, kv_heads, head_dim, bytes_per_value=2):
    return 2 * batch * layers * tokens * kv_heads * head_dim * bytes_per_value

one_sequence = kv_bytes(1, 32, 8192, 8, 128)
eight_sequences = kv_bytes(8, 32, 8192, 8, 128)
print(one_sequence / 2**30, eight_sequences / 2**30)
```

结果为 **1 GiB、8 GiB**。这是 KV 本体，不含模型权重、工作区、分配器余量。如果改为 32 个 KV heads，单条就变成 4 GiB。不同长度时应按每条的实际长度求和；预分配、padding、分片或复制又会改变实际占用。

## 权重放得下，为什么一接请求就 OOM？ {#inference-budget}

“7B 模型用 BF16，大约 14 GB，24 GiB 的卡应该够了吧？”只加载权重时可能够，但用户一来，KV、临时张量和 kernel 工作区也要占地方。输入越长、同时服务的人越多，差距就越明显。

先列预算，不急着猜卡数。假设模型恰好有 70 亿个参数，全用 BF16；设备有 24 GiB 可见显存。为了方便计算，**假设**给运行时留 2 GiB，再保留 2 GiB 安全余量：

<figure class="worked-update worked-update--pairs">
<ol>
<li><small>权重</small><strong>约 13.04 GiB</strong><span>70 亿 × 2 bytes = 14 GB。GB 与 GiB 的换算要一致。</span></li>
<li><small>运行时预算</small><strong>2 GiB</strong><span>暂定给激活、工作区、图捕获等使用；真实值要测。</span></li>
<li><small>安全余量</small><strong>2 GiB</strong><span>不是一定已分配的张量，而是暂不承诺给请求的容量。</span></li>
<li><small>留给 KV</small><strong>约 6.96 GiB</strong><span>24 − 13.04 − 2 − 2。还要考虑分页和具体模型的状态布局。</span></li>
</ol>
<figcaption>这是一份假设预算，不是 7B 模型在某张显卡上的实测占用。</figcaption>
</figure>

若 KV 恰好采用前面的 32 层、8 个 KV heads 配置，每条缓存 8192 tokens 占 1 GiB，那么这个预算最多装 6 条这样的独立缓存，装不下 8 条。长度翻倍到 16384，单条变成 2 GiB，最多 3 条。这里的长度包括**已经送入模型的 prompt 和生成位置**，不能只按输入长度估算长回答。

2 GiB 运行时占用只是算例参数。实际应在目标模型、batch、prefill chunk 和 graph 配置下测量；不能给所有模型套一个固定常数。量化权重也未必同时量化 KV，还可能多出 scales、zero-points 和未量化模块。[vLLM 的内存配置说明](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)可作为部署时的检查入口。

### 不同长度、分页和共享前缀怎么算？ {#paged-budget}

前面的统一长度公式很方便，但服务里的请求通常不等长。先对每条缓存按块向上取整，再合计。假设每块 4 个 token，长度是 5、3、8：需要 2 + 1 + 2 = 5 块，也就是 20 个槽；实际只有 16 个位置，4 个槽空着。

<details markdown="1">
<summary>换几个长度，自己算 KV 占用（Python）</summary>

```python
def paged_kv_bytes(lengths, layers=32, kv_heads=8, head_dim=128,
                   bytes_per_value=2, block_tokens=16):
    lengths = list(lengths)
    dimensions = (layers, kv_heads, head_dim, bytes_per_value, block_tokens)
    if any(type(value) is not int or value < 1 for value in dimensions):
        raise ValueError("Dimensions must be positive integers")
    if any(type(length) is not int or length < 0 for length in lengths):
        raise ValueError("Lengths must be nonnegative integers")
    blocks = sum((length + block_tokens - 1) // block_tokens for length in lengths)
    per_token = 2 * layers * kv_heads * head_dim * bytes_per_value
    return blocks * block_tokens * per_token

kv_budget = 24 * 2**30 - 7_000_000_000 * 2 - 4 * 2**30
assert paged_kv_bytes([8192] * 6) <= kv_budget
assert paged_kv_bytes([8192] * 7) > kv_budget
assert paged_kv_bytes([16384] * 3) <= kv_budget
assert paged_kv_bytes([16384] * 4) > kv_budget
```

这个函数只算无共享、相同 full-attention 层的 KV，不是完整显存预测器。

</details>

共享以后要数**唯一物理块**，不能再直接加每个请求的逻辑长度。例如两条 8-token 缓存，块大小为 4，第一块共用、第二块各自独立：逻辑上是 16 个位置，物理上只需 3 块、12 个槽。若运行时另外保留了无人使用但可复用的旧前缀，这些也占池子，不能当作已经归还给系统。

Full attention 的缓存常随长度增长；sliding-window 层可以只保留窗口所需状态，混合模型则要按层类型分别记账。不能把统一的 $L\times T$ 公式直接套给所有模型。[Hybrid KV cache 的设计说明](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)专门处理这些差异。

### 多卡以后，不是所有东西都除以卡数 {#kv-per-rank}

先只看前面单条 1 GiB 的 KV，假设切分均匀，忽略通信 buffer：

| 放置方式 | 每个 rank 保存什么 | 这个例子的 KV |
| --- | --- | ---: |
| TP=2，8 个 KV heads 均分 | 每层的 4 个 KV heads | 每 rank 0.5 GiB |
| PP=2，32 层均分 | 16 层的全部 KV heads | 每 stage 0.5 GiB |
| DP=2，每副本各接一条请求 | 各自完整的 32 层、8 个 KV heads | 每副本 1 GiB，合计 2 GiB |

还有一个容易漏掉的情况：某个 MQA 模型只有 1 个 KV head，在采用 KV 复制的 TP=8 实现中，8 个 ranks 可能各留一份这个 head，而不是再把它切成八分之一。是否复制要看模型实现与 backend，不能只看 `tensor_parallel_size`。

这些数字只回答 KV 怎么放，不保证模型的权重、工作区也恰好同比例缩小。评估能不能部署，要检查**最紧张的那个 rank**，而不是把所有卡的剩余显存简单相加。

## MHA、GQA、MQA：共享哪一部分？

```text
MHA：Q1 → K1,V1    Q2 → K2,V2    Q3 → K3,V3    Q4 → K4,V4
GQA：Q1,Q2 → K1,V1              Q3,Q4 → K2,V2
MQA：Q1,Q2,Q3,Q4 → K1,V1
```

这是 4 个 Query heads 的示意图。GQA 和 MQA 减少的是 KV 组数；每个 Query head 仍有自己的查询和 attention 输出。缓存可以更小，但参数共享也改变了模型的表示方式，不能把已训练 MHA 模型随手删几个 KV heads，就假定效果不变。[GQA 论文](https://arxiv.org/abs/2305.13245)给出了从 MHA checkpoint 继续训练的方案与实验。

接着用前面的 32 个 Query heads 算：从 32 个 KV heads 改为 8 个，在层数、长度、head dimension 和 dtype 不变时，KV 本体剩 **8/32 = 1/4**，也就是缩小为原来的四分之一。不要把“缩小 4 倍”写成“剩下 4 倍”。这里减少的是缓存，不代表整个模型显存也下降 75%。

KV head 数是模型架构的选择，不必等于 GPU 数量；tensor parallel 怎么切分、是否复制，还会影响每卡占用。教学代码为了方便可能用 `repeat_interleave` 把共享 K/V 展开到 Query head 数，但这种显式复制本身并没有展示省显存。高效实现需要保留共享存储，并由算子处理分组访问。

## FlashAttention 和 PagedAttention 又在省什么？

| 方法 | 主要改变 | 没有承诺的事 |
| --- | --- | --- |
| KV cache | 复用旧位置的 K/V | 不再读取历史 |
| GQA / MQA | 多个 Query heads 共享 K/V | 质量必然与 MHA 相同 |
| FlashAttention | 分块计算，减少中间 attention 数据的显存读写 | 把所有全注意力配对数变成线性 |
| PagedAttention | 按块管理 KV 存储，改善碎片和共享 | 消除 K/V 本身占用，或改变模型的学习目标 |

[FlashAttention](https://arxiv.org/abs/2205.14135)计算数学上相同的 attention，不需要先把整张分数/概率矩阵写到显存再读回来；浮点运算顺序改变时，不要求逐 bit 一致。[PagedAttention](https://arxiv.org/abs/2309.06180)主要解决多请求服务时的缓存管理。它们并不互斥，但具体能否组合、收益多大，取决于实现、硬件和工作负载。

## 缓存打开后结果不一样，先查什么？

1. **位置**：第 4 个位置不能又从位置 0 开始。RoPE 的位置必须延续，而不是每次 decode 重置。
2. **Mask**：缓存有 3 个位置、新输入有 2 个位置时，attention 是 2 × 5，不是 2 × 2。第一个新位置不能看到第二个新位置。
3. **前缀**：改了 system prompt、图片输入或 adapter，旧缓存不一定还能用。相同文字但不同模型权重也不算相同前缀。
4. **确定性**：先用 `eval()` 关掉 dropout，比较 logits，再比较采样文本。两次随机采样不同，不能直接证明 cache 出错。
5. **生命周期**：请求结束要回收；批次重排时缓存与序列身份必须一起移动，不能把甲的历史接给乙。

最有用的测试不是“能生成一句话”，而是固定一段输入，把**完整前向**和**prefill 后逐步追加**在对应位置的 logits 做容差比较。本站 [decoder 测试](../code/test_model.py)已经有缓存等价性测试，可以接着看实现。

## 要验证优化有效，至少分开报这些

| 观察项 | 更接近什么问题 |
| --- | --- |
| 首 token 延迟（TTFT） | 用户等多久才开始看到回答，包括排队和 prefill |
| 逐 token 间隔（ITL） | 开始回答以后，输出是否流畅 |
| 吞吐量 | 系统一段时间能处理多少请求或 token |
| 峰值显存与最大并发 | 在给定长度下能同时服务多少人 |

固定模型、输入/输出长度分布、并发与硬件，再比较。短 prompt 的单请求实验不能替代长对话的并发测试。优化前先知道是在等计算、搬权重、读 KV，还是排队；否则换了一个更酷的 kernel，也可能没有解决真正的瓶颈。

想回头看这些张量怎样产生，读[注意力](../core/multi-head-attention.md)；想算训练而非推理的内存，接着看 [LoRA 与 QLoRA](../../05-post-training/lora-and-qlora.md)。
