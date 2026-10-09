# 向量太多，怎么找得快又不漏太多？

**中文** · [English](vector-indexes.en.md)

> 阅读时间：约 10 分钟 · 最近审阅：2026-10

双塔已经把问题和文档变成了向量。现在库里有 1000 万条，每条 768 维：每次提问，都要和它们比一遍吗？

可以，这叫精确检索。问题是，算完需要多久、机器装不装得下。近似最近邻检索允许少漏一点，以换取速度或存储空间。这里的“漏”，先指漏掉同一距离函数下的近邻，**不是漏掉所有真正相关的答案**。表示学错了，精确检索也救不了。

先读过[双塔与训练目标](dual-encoder.md)会更容易理解。下面的数字都是教学算例，不是某个产品的性能测试。

## 1. 先把内存和比较标准算明白

只存 float32 向量，需要：

$$
10^7\times768\times4=30{,}720{,}000{,}000\ \text{bytes}\approx28.61\ \text{GiB}.
$$

这还没有算 ID、索引结构、元数据、复制副本和查询缓冲区。所谓“压缩 48 倍”，如果只算向量编码，就不能拿来当整个服务省了 48 倍内存。

比较索引前，先固定 embedding 版本、候选库、过滤条件和距离函数。余弦相似度不是普通点积；对非零向量先做单位归一化，才有：

$$
\|\hat q-\hat x\|_2^2=2-2\hat q^\top\hat x.
$$

这时最大点积和最小平方 L2 的排序一致。零向量要另外处理；如果训练目标本来就使用向量范数，不要为了套余弦检索悄悄改掉它。[Faiss 距离说明](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)

## 2. IVF：先选几个区域，再仔细找

IVF 先把库向量分到若干个簇，每个簇保存自己的向量列表。查询先找近的簇，再扫描选中的列表。`nlist` 是总簇数，`nprobe` 是一次查几个簇。[Faiss 索引说明](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

假设有 100 万条、1000 个簇，且每簇刚好 1000 条。查 10 个簇，就扫描约 1 万条，而不是 100 万条。除此之外还要支付找簇、维护 top-k 等成本。真实数据的簇大小不均匀，所以 `nprobe / nlist` 只是粗略扫描比例，不能直接换算成加速倍数。

```text
同一查询 q
  ├─ nprobe = 1 → 扫描最近的簇 A
  └─ nprobe = 2 → 扫描簇 A、B
                         ↑
               真正最近的向量可能在 B 的边缘
```

簇中心最近，不代表簇内一定有最近的点。多查几个簇能减少这类遗漏，但会更慢。IVF-Flat 仍存原始向量：它主要少算一些距离，并没有顺便把向量压小。

## 3. PQ：存编号，但距离不是编号的差

乘积量化（PQ）把一个向量切成若干段，每段使用自己的码本，用最近的码字代表它。比如把 768 维分成 64 段，每段用 256 个码字，则每段编号占 8 bit，向量编码共 64 bytes。原始 float32 是 3072 bytes，**仅编码部分**缩小 48 倍。码本由训练样本拟合，不是随便切几段就完成了量化。

查询时常用 ADC：查询保留原值，库向量用码字重建值估计距离。对平方 L2，可以把各段距离相加；SDC 则连查询也量化。它们的区别是距离估计中哪一侧被量化，不是“一个查表、一个不查表”。[PQ 原论文 §II–III](https://paper-notes.zhjwpku.com/assets/pdfs/Product_Quantization_for_Nearest_Neighbor_Search.pdf)

把维度缩到 4，亲手算一次。查询是 `[1, 2 | 3, 4]`，两个子空间的码本分别有 3 个码字。以下代码只演示 ADC，不负责训练码本或选择最近的编码；示意编号不按 bit 打包。

```python
def adc_squared_distance(query_parts, codebooks, codes):
    if not (len(query_parts) == len(codebooks) == len(codes)):
        raise ValueError("one query part and code per codebook required")
    total = 0.0
    for query_part, codebook, code in zip(query_parts, codebooks, codes):
        if not isinstance(code, int) or not 0 <= code < len(codebook):
            raise ValueError("invalid code")
        center = codebook[code]
        if len(query_part) != len(center):
            raise ValueError("dimension mismatch")
        total += sum((value - centroid) ** 2
                     for value, centroid in zip(query_part, center))
    return total


query_parts = [[1.0, 2.0], [3.0, 4.0]]
codebooks = [
    [[0.0, 0.0], [1.0, 1.0], [4.0, 4.0]],
    [[0.0, 0.0], [3.0, 3.0], [5.0, 5.0]],
]
assert adc_squared_distance(query_parts, codebooks, [1, 1]) == 2.0
assert adc_squared_distance(query_parts, codebooks, [2, 1]) == 14.0
```

编号 `[1, 1]` 表示重建向量 `[1, 1, 3, 3]`，距离就是 `0² + 1² + 0² + 1² = 2`。实际扫描会先为查询建立“子空间 × 码字”的距离表，再按编码查表求和。

**编号本身没有几何意义。** 把某个码本的编号重新排列，同时更新库里的编码，重建向量和 ADC 距离都不变；直接比较编号的汉明距离却可能变。需要汉明预过滤时，可以研究专门学习编码排列的 [Polysemous Codes](https://arxiv.org/abs/1609.01882)，不能把普通 PQ 编号直接当二进制哈希。

## 4. IVF-PQ：两种近似，两个误差来源

IVF 负责少找几个区域，PQ 负责少存一些数。常见的 IVFADC 还会编码向量相对粗簇中心的**残差**，不只是机械地把两个算法串起来。[PQ 原论文 §IV](https://paper-notes.zhjwpku.com/assets/pdfs/Product_Quantization_for_Nearest_Neighbor_Search.pdf)

| 出错位置 | 一个具体例子 | 怎么分开检查 |
| --- | --- | --- |
| 没有扫描到 | 好文档在没选中的簇里 | 固定编码，增加 `nprobe` |
| 距离估错了 | 两个候选的近似距离顺序反了 | 固定扫描范围，对比原始向量的分数 |
| 最后仍不相关 | 精确近邻也是主题相似、答非所问 | 看相关性标注与 embedding，不只调索引 |

先取一个较大的近似候选集，再用原始向量重算距离，能修正集合**内部**的排序，不能找回从没入选的向量。而且原始向量必须存放在某处，读取它们也有成本。这里的精确距离重排，不等于[联合阅读 query 和文档的 cross-encoder](hybrid-and-reranking.md)。

## 5. Flat、HNSW、IVF-PQ 怎么选？

[HNSW](https://arxiv.org/abs/1603.09320)用分层邻接图寻找近邻：先在稀疏的上层移动，再在下层扩大搜索。它没有要求把所有向量压成 PQ 码；图结构本身也占内存。

| 方案 | 先试它的理由 | 不能省略的检查 |
| --- | --- | --- |
| Flat | 数据量不大、批量计算合适，或者需要精确基线 | 全扫描延迟、吞吐与内存；精确不代表实现一定最慢 |
| HNSW | 想在内存中做低延迟搜索 | 图的额外内存、构建成本、搜索宽度；删除行为依实现而定 |
| IVF-Flat | 想单独观察“少扫一些区域”的影响 | 簇是否偏斜、`nprobe` 与尾延迟 |
| IVF-PQ | 向量存储成为主要限制 | 码长、量化误差、训练样本代表性，以及原始向量重排成本 |

这是选实验起点，不是规模到某个阈值就必须换算法。硬件、并发、过滤比例和更新频率都能改变结果。

## 6. 别把 ANN recall 当作业务 recall

设精确 top-10 是 A，近似 top-10 是 B。`|A ∩ B| / 10` 衡量索引保住了多少精确近邻；相关性 Recall@10 则以标注相关文档为参照。前者很高，只能说明“近似得不错”，不能证明“表示得好”。

建议先固定查询集与库快照，再扫索引参数，同时记录：

- ANN recall、真实相关性指标；
- p50 / p95 / p99 延迟、吞吐、峰值内存和构建时间；
- 严格过滤后的返回数量、稀有主题和短查询的表现；
- embedding 更新、文档新增或删除后，索引是否仍与元数据版本一致。

权限过滤还会改变可搜索集合。不能先找到未授权原文发给模型，再靠 prompt 要求它不要透露。下一篇看[RAG 怎样保住证据、又怎样检查答案](rag-evidence.md)。
