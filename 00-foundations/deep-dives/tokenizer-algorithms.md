# 分词算法：BPE 到底学到了什么？

**中文** · [English](tokenizer-algorithms.en.md)

> 最近审阅：2026-10 · 前置：[文本到 ID](../core/tokenization.md)

你给模型一个没见过的人名，它通常不会直接报“这个词不存在”。原因不是 tokenizer 理解了人名，而是它能把新字符串拆成已有的小片段。拆得合理不合理、长不长，是另一回事。

这一篇把“切成小片段”讲具体：先手算 BPE，再比较 WordPiece 和 Unigram，最后看为什么一个很小的分词改动也可能让模型出错。例子里的词表和数字都是教学用的，不对应某个商业模型。

## 先把三件事分开

| 你看到的名字 | 它回答什么问题 | 不代表什么 |
| --- | --- | --- |
| 字符、字节（byte） | 从哪些基础符号开始？ | 最终一个 token 一定就是一个基础符号 |
| BPE、WordPiece、Unigram | 如何学习词表、如何切分？ | 是某种神经网络结构 |
| SentencePiece、Tokenizers 等工具 | 怎样实现并保存这套处理？ | 工具名就是唯一算法名 |

例如，byte-level BPE 从字节开始，合并后一个 token 可以包含多个字节。SentencePiece 支持 BPE 和 Unigram，不能看到 `.model` 文件就断言它用了哪一种。[SentencePiece 官方说明](https://github.com/google/sentencepiece)列出了它支持的算法和归一化行为。

## 手算两轮 BPE

先忽略空格、词尾标记和标点，只对下面两种字符串做合并，不允许跨字符串合并：

| 字符串 | 在训练语料里出现几次 | 初始切分 |
| --- | ---: | --- |
| `rain` | 3 | `r a i n` |
| `rail` | 2 | `r a i l` |

统计的是**出现次数**，不是不同字符串的数量。`(r, a)` 和 `(a, i)` 各出现 5 次，`(i, n)` 出现 3 次，`(i, l)` 出现 2 次。频数相同时必须有确定的规则；这个例子约定先选 `(r, a)`。

| 轮次 | 本轮合并 | `rain` 变成 | `rail` 变成 |
| --- | --- | --- | --- |
| 0 | 尚未合并 | `r a i n` | `r a i l` |
| 1 | `r + a → ra` | `ra i n` | `ra i l` |
| 2 | `ra + i → rai` | `rai n` | `rai l` |

第 2 轮必须重新统计相邻符号；不能一直沿用最初的字符对计数。编码新字符串 `rair` 时，依次应用学到的规则，得到 `rai r`。没有规则要求把整个新词放进词表。

还有个实现细节：**合并不是删除基础字符。** 词表仍然要保留 `r`、`a` 等符号和中间合并出的 `ra`，否则新组合可能切得出来，却找不到对应 ID。本站的 [TinyBPE](../code/tokenizer_from_scratch.py) 可以直接验证：

```python
from tokenizer_from_scratch import TinyBPE

tokenizer = TinyBPE()
tokenizer.train(["rain rain rain rail rail"], num_merges=4)
pieces = tokenizer.tokenize("rair")
ids = tokenizer.encode("rair")
assert tokenizer.token_to_id["<unk>"] not in ids
assert tokenizer.decode_tokens(pieces) == "rair"
print(pieces, ids)
```

从 `00-foundations/code/` 运行。这个实现额外使用 `▁` 标记词首，所以具体 merge 顺序不必等于上表。它还会折叠连续空白，不是生产级、逐字节可逆的 tokenizer。原始 BPE 子词方法见 [Sennrich 等人的论文](https://arxiv.org/abs/1508.07909)。

## “字节级”为什么还可以有很长的 token？

```text
文字「中」
    ↓ UTF-8 编码
E4 B8 AD                  3 个字节
    ↓ 已学到的合并规则
[E4] [B8] [AD]            可能还没有合并
[E4 B8 AD]                也可能已合成 1 个 token
```

上面展示的是两种可能的词表，不是说每次随机选一种切法。固定 tokenizer 配置后，普通的确定性编码应得到固定结果。

如果保留完整的 256 个基础字节，并且编码前没有丢失信息，就能表示任意 UTF-8 字符串，而不需要为每个新字符新增词表项。但一个孤立 token 的字节不一定能单独解码成合法字符：通常要把整段 token 拼回字节串，再解码。

字符级 BPE 则没有天然获得这项保证。基础字母表没收录的新字符，仍可能需要 `<unk>` 或 byte fallback。**“能表示输入”也不等于“模型理解输入”**：罕见文本被拆得很碎，模型仍可能处理得不好。

## WordPiece：编码时不回放 merge 表

WordPiece 常用最长匹配：从当前位置开始，找词表里最长的合法片段，再继续处理剩余部分。以一个假设词表为例：

```text
词表里有：play, ##ing, ##er
playing → play + ##ing
player  → play + ##er
```

`##` 在这里表示“不是词首”，不是原文真的有井号。BERT 风格实现若无法完整分解一个词，可能把整个词变成 `[UNK]`；具体处理要看实现。

WordPiece **训练**与**编码**别混在一起。教材常用 `freq(a,b)/(freq(a)freq(b))` 解释候选合并，但不能把它称为所有 WordPiece 实现的统一规范。Hugging Face 的[教学实现](https://huggingface.co/learn/llm-course/chapter6/6)也明确说明：原始训练实现未完整公开，这是一种根据文献做的重建。编码阶段的最长匹配，与 BPE 按 merge 优先级合并，是更应该记住的区别。

## Unigram：比较完整切分的得分

Unigram 给词表里的片段分配概率。在一个简化模型里，一种切分的概率是各片段概率的乘积；取对数后变成求和。

假设词表的概率为 `a: 0.3`、`b: 0.2`、`ab: 0.5`。对 `ab` 有两种切法：

| 切法 | 得分 |
| --- | ---: |
| `ab` | 0.5 |
| `a b` | 0.3 × 0.2 = 0.06 |

最可能的切法是 `ab`。而在模型训练的似然里，如果把所有合法切分求和，这个字符串对应的量是 `0.5 + 0.06 = 0.56`，不是只取最大的 `0.5`。**寻找最佳切分**和**对潜在切分求和**是不同的计算。

对长字符串，最佳切分可以用动态规划，不用真的枚举所有组合。训练时通常从较大的候选词表出发，估计概率、裁掉贡献较小的片段；这和 BPE 不断加入合并项的方向不同。[Unigram / Subword Regularization 论文](https://arxiv.org/abs/1804.10959)还讨论了训练时采样多种切分，避免模型只依赖一种边界。

## 方法对照：别只背三个名字

| 方法 | 主要保存什么 | 确定性编码怎么做 | 最容易误解 |
| --- | --- | --- | --- |
| BPE | 词表和有优先级的合并规则 | 从基础符号按规则合并 | 高频片段不一定是词根 |
| WordPiece | 词表、词内标记等配置 | 贪心最长合法匹配 | 训练打分不等于编码过程 |
| Unigram | 片段及其得分 | 用动态规划找最高分切分 | 一个字符串可以有多种合法切分 |

这些算法都不是“越新越好”。语种覆盖、训练文本、词表大小、边界处理，会共同影响结果。换 tokenizer 后，应在相同文本上比较 token 数、未知输入、可逆性和下游任务，而不是只看词表有多大。

## 为什么不是 token 越少越好？

假设同一批文本，A 分成 200 个 token，B 分成 300 个。朴素全注意力的单头分数矩阵从 40,000 项变成 90,000 项，是 2.25 倍；但**整次推理不会因此必然慢 2.25 倍**，还要看投影、FFN、batch 和实际 attention kernel。

更大的词表也有代价。若输入 embedding 维度为 4096，词表增加 32,000 项，就多出 131,072,000 个参数。输入输出权重若不共享，输出头还可能再增加同样规模。少预测几步和多存一大张表，需要一起算。

不同 tokenizer 下，per-token perplexity 也不能直接横比：预测单位变了。先固定 tokenizer，或明确按同一份文本的字节等单位换算，并报告预处理和归一化方式。

## 接模型以前，检查这几条

- **可逆性**：中文、emoji、连续空格、换行各试一次。归一化改了什么，要知道。
- **ID 对齐**：词表即使大小一样，ID 排列也可能不同。旧 embedding 的第 42 行不会自动知道新 ID 42 的意思。
- **特殊 token**：模板已经加了 BOS/EOS 后，再编码会不会加第二次？
- **训练标签**：哪些 token 计入 loss？padding mask、causal mask、loss mask 不是同一个东西。
- **预算**：在你自己的文本上统计长度分布，不凭“英文平均几个字母”估算所有语言。

接下来读[一次训练怎么走](training-step.md)，把这些 ID 接到 labels 和 loss；如果想知道序列变长后具体占哪块显存，读 [KV cache 与推理成本](kv-cache-and-inference.md)。
