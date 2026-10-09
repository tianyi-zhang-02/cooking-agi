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
K_{\le t}=[K_{<t};k_t],\qquad V_{\le t}=[V_{<t};v_t],
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
