# TF-IDF 与 BM25：关键词检索到底怎么算分？

**中文** · [English](tfidf-and-bm25.en.md)

> 阅读时间：约 12 分钟 · 前置知识：频次、对数、加权求和 · 最近审阅：2026-10-09

搜一段报错时，你大概不希望系统只返回一篇“主题很像”的文章。错误码、函数名、版本号，少一个字都可能是另一回事。关键词检索保留了这些字面线索；不过，命中两个词和重复命中同一个词，该怎么比较？

先不看公式，想两个小变化。一篇笔记把 `cache` 连写 20 次，不该因此比真正解释报错的笔记有用 20 倍；两篇都只提了一次 `error`，一句话的报错记录和整本操作手册提供的线索强度也未必相同。BM25 把“重复的收益会递减”和“文档有长有短”都放进了打分，**但它并没有理解哪篇真的解决了问题**。

只想先弄懂区别，可以读这段直觉和[参数对照](#parameter-choice)。想看分数怎么来，下面用 4 段短文本完整算一遍；要接到自己的系统，再看[实现与排错](#implementation)。这是[混合检索](hybrid-and-reranking.md)的前置解释，不要求先学 embedding。

## 1. 先分清：一个词出现几次，还是几篇里出现过？

查询是 `cache error`。为了让每一步都能手算，把文档简化成下面这些已经分好的词，不考虑语序、标点和大小写。

| 文档 | 分词结果 | 长度 | `cache` 次数 | `error` 次数 |
| --- | --- | --- | --- | --- |
| A | `cache cache error` | 3 | 2 | 1 |
| B | `cache error` | 2 | 1 | 1 |
| C | `cache guide setup notes` | 4 | 1 | 0 |
| D | `release notes` | 2 | 0 | 0 |

词频（term frequency, TF）看一篇内部：`cache` 在 A 中出现 2 次。文档频率（document frequency, DF）看整个库：有 3 篇包含 `cache`，所以它的 DF 是 3，**不是总共出现的 4 次**。

这里文档数 $N=4$，平均长度 $\overline L=(3+2+4+2)/4=2.75$。这些量都取自同一个语料快照。换一批文档、改一次切块方式，统计量也会变。

## 2. TF-IDF：常见词少加一点，少见词多加一点

TF-IDF 把词频和逆文档频率（inverse document frequency, IDF）相乘：前者看这篇提了多少次，后者看这个词在库里有多常见。先选一个简单版本，使用自然对数：

$$
\operatorname{idf}(t)=\ln\frac{N}{\operatorname{df}(t)},\qquad
w(t,d)=f(t,d)\operatorname{idf}(t).
$$

$f(t,d)$ 是词 $t$ 在文档 $d$ 的次数。查询中的不同词各算一次，把对应权重相加；不在文档中的词贡献 0，不在整个词表中的查询词也忽略。TF-IDF 是一类权重约定，不是唯一一种打分函数。[IR 教材的定义](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html)可以用来核对这里的乘积形式。

`cache` 的 IDF 为 $\ln(4/3)\approx0.2877$；`error` 为 $\ln(4/2)\approx0.6931$。因此 A 得分 $2\times0.2877+0.6931\approx1.2685$，B 是 $0.9808$。

| 文档 | `cache` 的贡献 | `error` 的贡献 | TF-IDF 求和 |
| --- | --- | --- | --- |
| A | 0.5754 | 0.6931 | **1.2685** |
| B | 0.2877 | 0.6931 | 0.9808 |
| C | 0.2877 | 0 | 0.2877 |
| D | 0 | 0 | 0 |

A 靠多写一次 `cache` 排在了 B 前面。这不一定错，但重复 20 次，真的应该得到 20 倍的贡献吗？

一种改法是把正词频换成 $1+\ln f$，未命中仍是 0；这是 [sublinear TF](https://nlp.stanford.edu/IR-book/html/htmledition/sublinear-tf-scaling-1.html)。另一种做法把查询和文档都转成权重向量，再做 cosine normalization。**后者不等于上面的权重求和**：如果两边都带 IDF，点积的单词贡献包含 IDF 的平方，还要除以向量范数。比较实现前，先对齐公式，别只核对“都叫 TF-IDF”。

## 3. BM25：重复有用，但不会一直等比例加分 {#bm25-score}

下面采用带正值 IDF 的 BM25：

$$
\operatorname{idf}_{+}(t)=
\ln\left(1+\frac{N-\operatorname{df}(t)+0.5}{\operatorname{df}(t)+0.5}\right),
$$

$$
\operatorname{score}(q,d)=\sum_{t\in\operatorname{unique}(q)}
\operatorname{idf}_{+}(t)
\frac{f(t,d)(k_1+1)}{f(t,d)+k_1\left(1-b+bL_d/\overline L\right)}.
$$

取 $k_1=1.2,b=0.75$。IDF 约定与参数默认值可对照 [Lucene 10.3.1](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/search/similarities/BM25Similarity.html)；这里是教学公式，不承诺逐位复现 Lucene 的完整 scorer、字段统计或长度编码。

公式可以拆成两件事：

- **词频饱和（term-frequency saturation）**：先固定长度修正项，$f$ 越大，继续重复带来的增量越小。
- **长度归一化（length normalization）**：词频相同的情况下，较长的文档通常会被适当降权。一个词在 10 个词的短记录里出现，和在几千词的综述里出现，证据强度未必相同。

先只看词频部分，固定 $L_d/\overline L=1$：

| 词频 $f$ | 原始 TF | $f(k_1+1)/(f+k_1)$，$k_1=1.2$ |
| --- | --- | --- |
| 1 | 1 | 1.0000 |
| 2 | 2 | 1.3750 |
| 4 | 4 | 1.6923 |
| 8 | 8 | 1.9130 |

同一长度下，这一因子的极限是 $k_1+1=2.2$。这是把两个影响分开看的控制例子；真的往文档后面添词，还会同时改变文档长度，不能只改 $f$。

回到那 4 篇文档：`cache` 的新 IDF 为 $0.3567$，`error` 为 $0.6931$。A 的长度修正项是 $1-0.75+0.75\times3/2.75\approx1.0682$，所以它的 `cache` 贡献为：

$$
0.3567\times\frac{2\times2.2}{2+1.2\times1.0682}\approx0.4782.
$$

| 文档 | `cache` 的贡献 | `error` 的贡献 | BM25 求和 |
| --- | --- | --- | --- |
| A | 0.4782 | 0.6683 | 1.1465 |
| B | 0.4015 | 0.7802 | **1.1817** |
| C | 0.3008 | 0 | 0.3008 |
| D | 0 | 0 | 0 |

这次 B 略高于 A：同样覆盖两个查询词，B 更短，A 多写一次 `cache` 的收益则被压小了。**这个排序只是公式的结果，不是“B 一定更相关”的人工标签。** 它们可能一个是无用的日志，一个才有修复方案；内容是否有用，仍要另外判断。

<span id="4"></span>

## 4. 两个参数怎么选，哪些结论别混在一起？ {#parameter-choice}

| 改什么 | 直接影响 | 要小心什么 |
| --- | --- | --- |
| $k_1=0$ | 命中的词只加一次 IDF，不再按词频加权 | 未命中项应直接返回 0，别算出 $0/0$ |
| 增大 $k_1$ | 在固定长度比下，允许重复次数产生更大差异 | 不是越大越好，要看开发集 |
| $b=0$ | 关闭长度修正 | 词频饱和仍在，不能叫“退化成原始 TF” |
| $b=1$ | 长度修正项变为 $L_d/\overline L$ | 不代表长文一定更差，词频和命中的词也在变 |

还有一个容易踩的坑：有的 BM25 公式没有 IDF 外层的 `1 +`。在那个约定下，一个词出现在超过一半的文档里，IDF 可能为负；我们的 `cache` 就会遇到。这是公式约定不同，不一定是 bug。[IR 教材的 BM25 一节](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html)同时讨论了这些形式。

不要把 BM25 分数当相关性概率，也不要直接和 embedding 的余弦分数相加。下一篇的 [RRF](hybrid-and-reranking.md#rank-fusion)就是一种不要求原始分数同尺度的融合办法。

## 5. 写出来：先验证计算，再考虑索引 {#implementation}

单个词的贡献只需要几行。参数检查与语料统计由外层的 `LexicalIndex` 负责；这个函数假定输入已合法，且出现过的词对应正的平均文档长度。

```python
def bm25_term(frequency, length, average_length, inverse_frequency, k1, length_weight):
    if frequency == 0:
        return 0.0
    length_factor = 1 - length_weight + length_weight * length / average_length
    return inverse_frequency * frequency * (k1 + 1) / (frequency + k1 * length_factor)
```

代码里用 `length_weight` 表示公式的 $b$，避免把这个参数和其他权重混淆。完整的标准库实现是 [lexical_retrieval.py](code/lexical_retrieval.py)，从仓库根目录执行：

```bash
python3 04-search/code/lexical_retrieval.py
```

```text
TF-IDF: [1.2685, 0.9808, 0.2877, 0.0]
BM25: [1.1465, 1.1817, 0.3008, 0.0]
```

代码接收**已经分词的字符串序列**，不负责分词、不自动转小写。示例里的 `.split()` 只是为了复算上面的英文词表，不适用于中文搜索。重复的查询词去重；空文档得 0 分，但计入文档数和平均长度；空库返回空列表，全空文档也不会除以 0。真实引擎对缺失字段、空字段的统计约定可能不同，接入时需要单独核对。

建统计表扫描全部 $T$ 个词，期望时间为 $O(T)$；一次查询扫描 $N$ 篇文档，对每篇检查 $Q$ 个不同查询词，期望时间为 $O(NQ)$，这里假设哈希表查找的期望代价为常数。词频表占用与所有“文档–不同词”组合数同阶。这是方便检查的参考实现，**不是大规模搜索引擎**。

实际查询通常从倒排索引（inverted index）读候选：`cache → A:2, B:1, C:1`，`error → A:1, B:1`。这样不必为了命中词扫描 D。但高频词的 postings 仍可能很长，Top-K、过滤、剪枝和索引更新各有代价，不能简单说“有倒排就 O(1)”。

## 6. 接到真实数据，优先查这几件事

| 现象 | 先查什么 | 一个小验证 |
| --- | --- | --- |
| 明明有同样的报错，却搜不到 | 文档和查询的 analyzer 是否一致 | 打印 `ERR_42`、`C++`、中文词的实际 tokens |
| 切块后分数和排名变了 | 文档数、DF、平均长度都可能变 | 固定查询，比较切块前后的逐词贡献 |
| 长教程总被压在短便签下面 | 长度修正是否过重、正文是否混入导航垃圾 | 分长度切片比较，并扫 $b$；不要只看全局均值 |
| 精确词命中了，但回答仍错 | 命中不等于内容可信、当前或能回答问题 | 看版本、权限、证据段，再看 reranker / RAG |
| 换个说法就完全没有结果 | 字面重叠不足 | 加 dense 路线，固定总候选数比较独有相关结果 |

稀有词也可能是拼写错误，不天然等于重要线索。停用词、同义词、字段权重都要看任务；像错误码这样的字段，随手拆开或统一清洗反而可能丢掉最有用的部分。

评估时保留一份固定的查询与相关性判断，单独看精确标识符、自然语言改写、中文和长文档等切片。开发集选 $k_1,b$，留出的测试集只做最终比较。检索库可以按任务要求包含待检索文档；不要用测试集的相关性标签反复调参数。

## 7. 再往哪里看？

- 词不同、意思接近怎么办：[双塔检索](dual-encoder.md)。
- 两路怎样合在一起：[混合检索与重排](hybrid-and-reranking.md)。
- 找到之后怎样形成可核对的回答：[RAG 的证据链](rag-evidence.md)。
- 想确认自己理解了：把查询改成 `cache cache error unknown`。本实现应与 `cache error` 同分。接着设 $b=0$，看排序为什么变化；再设 $k_1=0$，A、B 应打平。

以上例子是教学语料，不是产品效果或模型对比结果。标准公式之外的统计与工程约定，最终以你实际使用的引擎版本为准。
