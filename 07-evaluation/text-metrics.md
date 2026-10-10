# BLEU、ROUGE、编辑距离：分数到底在比较什么？

**中文** · [English](text-metrics.en.md)

> 最近审阅：2026-10-10 · 先读：[分层评估](evaluation-stack.md)

参考答案写“阀门已打开”，模型回答“阀门未打开”。大部分字都对上了，意思却反了。文本指标仍然有用，只是得先弄清它在数什么：重合的词、顺序、需要修改的次数，还是模型给原文的概率。

## 先选比较对象

| 指标 | 比较什么 | 适合看什么 | 不直接回答什么 |
| --- | --- | --- | --- |
| BLEU | 输出中有多少 n-gram 能在参考里匹配 | 翻译等固定协议下的语料级比较 | 事实和语义是否正确 |
| ROUGE-N | 参考里的 n-gram 被覆盖多少 | 摘要的内容覆盖线索 | 覆盖的内容是否被否定或误解 |
| ROUGE-L | 最长公共子序列 | 重合与相对顺序 | 完整的语义关系 |
| 编辑距离 / CER / WER | 替换、删除、插入次数 | OCR、语音转录 | 同义改写是否合理 |
| Perplexity | 模型给固定测试文本的概率 | 语言建模拟合 | 自由生成的回答是否有用 |

下面用已经分好词的短句手算。这里的 `split()` 只是英文算例的约定，不是中文评估的分词方案。

## BLEU：不能靠重复或少说刷高分

BLEU 使用截断计数（clipped counts）：一个 n-gram 的匹配次数不能超过参考中的次数。候选写 3 次 `red`，参考只写 1 次，最多记 1 次匹配。多参考时，对每个 n-gram 取各参考中的最大次数，不把所有参考简单拼起来。[原论文 §2](https://aclanthology.org/P02-1040.pdf)

单参考下，设候选长度为 $c$、参考长度为 $r$，$p_n$ 是第 $n$ 阶的截断精确率：

$$
\begin{gathered}
\operatorname{BLEU}_N=\\
BP\exp\left(\frac1N\sum_{n=1}^N\log p_n\right),\\
BP=\exp\left(\min(0,1-r/c)\right).
\end{gathered}
$$

空候选另行定义为 0。长度惩罚（brevity penalty）是为了避免模型只说最容易匹配的几个词。

参考为 `the bus leaves at ten`，候选为 `the bus leaves`。候选的 unigram 和 bigram 都完全匹配，但只交出了 3/5 的长度。教学版 BLEU-2 为 $e^{1-5/3}\approx0.5134$，不是 1。

换成 `the bus leaves at nine`：长度相同，unigram 匹配 4/5，bigram 匹配 3/4，BLEU-2 为 $\sqrt{0.8\times0.75}\approx0.7746$。时间错了，分数却更高。若任务就是查发车时间，字段正确率才是直接指标。

### 可以运行的小版本

```python
from collections import Counter
import math

def ngram_counts(tokens, order):
    if type(order) is not int or order < 1:
        raise ValueError("order must be a positive integer")
    return Counter(tuple(tokens[start:start + order])
                   for start in range(len(tokens) - order + 1))

def bleu_example(candidate, reference, max_order=2):
    if type(max_order) is not int or max_order < 1:
        raise ValueError("max_order must be a positive integer")
    if not reference:
        raise ValueError("reference must not be empty")
    if not candidate:
        return 0.0
    precisions = []
    for order in range(1, max_order + 1):
        proposed = ngram_counts(candidate, order)
        expected = ngram_counts(reference, order)
        total = sum(proposed.values())
        matches = sum((proposed & expected).values())
        if total == 0 or matches == 0:
            return 0.0
        precisions.append(matches / total)
    penalty = math.exp(min(0.0, 1 - len(reference) / len(candidate)))
    return penalty * math.exp(sum(map(math.log, precisions)) / max_order)

reference = "the bus leaves at ten".split()
assert math.isclose(bleu_example("the bus leaves".split(), reference), math.exp(-2 / 3))
assert math.isclose(bleu_example("the bus leaves at nine".split(), reference), math.sqrt(0.6))
assert bleu_example(["red", "red", "red"], ["red", "blue"], 1) == 1 / 3
```

这段只实现单参考、单句、等权、无平滑的教学版本。某阶没有匹配就返回 0；生产使用还要明确最大阶数、平滑、多参考和短句处理。语料 BLEU 先累加各句计数再计算，不是把每句话的 BLEU 简单平均。[SacreBLEU](https://github.com/mjpost/sacrebleu)会记录 tokenizer、大小写、平滑等配置签名，适合留下可比较的报告。

## ROUGE：覆盖了词，不一定覆盖了意思

单参考 ROUGE-N recall 是匹配的 n-gram 数除以参考 n-gram 数。也可以同时报告 precision 与 F1；只写“ROUGE 高了”无法确定用了哪一种。[原论文](https://aclanthology.org/W04-1013.pdf)

参考 `the valve is open`，候选 `the valve is not open`。ROUGE-1 recall 为 1，precision 为 4/5，F1 为 8/9。多了一个否定词，结论就错了，但词覆盖很高。

ROUGE-L 用最长公共子序列（LCS），不要求连续，却要保持相对顺序。例如 `we ship today` 和 `today we ship` 的 LCS 长度为 2，不是 3。它比词袋多看了顺序，仍不理解“谁否定了什么”。句级 ROUGE-L 与摘要级 `rougeLsum` 的处理也不应混用。

### 子序列与编辑距离一起算

下面两个算法都做前缀动态规划，时间为 $O(mn)$，滚动数组空间为 $O(n)$；$n$ 是第二个序列长度。它们的目标不同：LCS 找保留的顺序，编辑距离找改动成本。

```python
def lcs_length(left, right):
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for column, right_token in enumerate(right, start=1):
            current.append(previous[column - 1] + 1 if left_token == right_token
                           else max(previous[column], current[-1]))
        previous = current
    return previous[-1]

def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for row, left_token in enumerate(left, start=1):
        current = [row]
        for column, right_token in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (left_token != right_token)))
        previous = current
    return previous[-1]

assert lcs_length("we ship today".split(), "today we ship".split()) == 2
assert edit_distance("we ship today".split(), "today we ship".split()) == 2
assert edit_distance("1280", "1230") == 1
assert lcs_length([], ["word"]) == 0
assert edit_distance([], ["one", "two"]) == 2
```

这里的替换、插入、删除成本都为 1。转录 `1280` 为 `1230` 只错 1 个字符，CER 是 1/4；但金额字段完全匹配得分为 0。两者回答的是不同问题。

WER 通常除以参考词数，CER 除以参考字符数。插入很多内容时，错误率可以超过 100%，不是代码一定写错；参考为空时分母为 0，需要另定策略，而不是悄悄加一个 epsilon。当字符包含组合音符或 emoji 时，还要约定按 Unicode code point 还是用户看到的 grapheme 计数。

## Perplexity：评的是测试原文，不是模型自己的作文

假设测试文本里接下来的两个 token 是 `bus` 和 `leaves`。我们把真实前文交给模型，检查它给这两个**实际出现的 token** 分了多少概率；不是让它先写一段话，再评价自己的输出。

| 有效目标 token | 给真实 token 的概率 | 负对数概率（NLL，自然对数） |
| --- | --- | --- |
| `bus` | 0.5 | 约 0.693 |
| `leaves` | 0.25 | 约 1.386 |

平均 NLL 约为 1.040，再取指数得到 **PPL ≈ 2.828**。模型给原文的概率越低，这个分数越高。公式只是把这两步推广到 $T$ 个有效目标：

$$
\begin{gathered}
\operatorname{PPL}=\\
\exp\left(-\frac1T\sum_{t=1}^T\log p(x_t\mid x_{<t})\right).
\end{gathered}
$$

也可以把它看作“真实 token 概率的几何平均”的倒数，不是准确率的倒数。如果每个目标的概率都恰好为 $1/4$，PPL 就是 4；一般情况下，不要把 2.828 理解成每个位置真的有 2.828 个等可能选项。用自然对数就配 `exp`；用以 2 为底的对数，就配 $2^{\text{平均损失}}$。

**PPL 下降，说明模型更能预测这份测试文本，不等于回答一定更有用。** 数据、tokenizer、上下文和计分位置变了，数字就可能不可比。下面两个细节尤其容易让结果看起来合理、实际算错。

### 长短 batch，不能各算一票 {#ppl-aggregation}

假设 batch A 有 2 个有效目标，PPL 为 2；batch B 有 8 个有效目标，PPL 为 8。直接平均得到 5，但 B 明明包含更多要预测的 token。

正确做法是先累计 NLL 和目标数，最后只取一次指数：

$$
\begin{gathered}
\operatorname{PPL}_{\text{all}}=\\
\exp\!\left(\frac{2\ln2+8\ln8}{10}\right)\\
\approx6.063.
\end{gathered}
$$

做分布式评估也是一样：各卡交回 **NLL 总和与有效 token 数**，分别求和，再相除、取指数。不要直接平均各卡的 PPL。

### 滑动窗口：前文可以重复读，目标只计一次 {#ppl-windows}

模型一次只能读 4 个 token，而测试序列有 6 个。下面不加 BOS，第一项 A 没有前文，不计分。字母只是 token 占位符：

| 输入窗口 | 只作上下文、不计分 | 本次计分的目标 | 有效目标数 |
| --- | --- | --- | --- |
| `A B C D` | A | B、C、D | 3 |
| `C D E F` | C、D | E、F | 2 |

第二个窗口保留 C、D，让 E、F 有前文可读，但不再给 C、D 计分。整段总共计 5 个目标，而不是 6 个或两个窗口的长度之和。窗口步长变小，通常能给目标更多上下文，但会增加 forward 次数；因此评测时也要记下窗口长度、步长、BOS/EOS 和文档边界的处理方式。[上下文窗口说明](https://huggingface.co/docs/transformers/perplexity)

<details markdown="1">
<summary>代码里怎样避免 shift 后少算一个 token？</summary>

下面采用常见的 causal LM 约定：输入和 labels 原本对齐，位置 $t$ 的 logits 预测位置 $t+1$ 的标签，`-100` 表示该目标不计分。**在 shift 之后数有效标签**，就不用猜“是不是每行都该减 1”。

```python
import math
import torch
import torch.nn.functional as functional


def causal_nll_totals(logits, labels):
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected aligned [batch, length, vocab] logits and labels")
    shifted_labels = labels[:, 1:]
    count = int((shifted_labels != -100).sum())
    if count == 0:
        raise ValueError("No scored targets after the causal shift")
    total = functional.cross_entropy(
        logits[:, :-1, :].double().reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1), ignore_index=-100, reduction="sum",
    )
    return total, count


uniform_logits = torch.zeros(1, 4, 6)
first_labels = torch.tensor([[-100, 1, 2, 3]])
second_labels = torch.tensor([[-100, -100, 4, 5]])
first_total, first_count = causal_nll_totals(uniform_logits, first_labels)
second_total, second_count = causal_nll_totals(uniform_logits, second_labels)
assert (first_count, second_count) == (3, 2)
average_nll = (first_total + second_total) / (first_count + second_count)
assert math.isclose(math.exp(float(average_nll)), 6.0)
```

这是计分逻辑的教学例子，不是模型评测结果。真实评测还需关闭 dropout、使用正确的 causal / padding mask。Labels 中的 `-100` 只影响 loss，不会阻止模型读取某个输入。若左侧有 padding，首个实际 token 又没有 BOS 或有效前驱，也要排除“padding 位置预测首个 token”这一项；packing 则需处理跨文档边界。

第二行原本就只有 2 个有效标签，而且都在 shift 后保留。此时若再机械地“有效标签数减 batch size”，就会错算成 1。检查实际传进 cross-entropy 的 labels，比照抄分母公式可靠。

</details>

最后，BERT 的 masked-token 分数不是同一个自回归概率分解。逐个遮住 token 可以构造 [pseudo-perplexity](https://aclanthology.org/2020.acl-main.240/)，但左右文条件和计算成本不同，不应拿它与这里的 PPL 直接排名。PPL 适合检查语言建模拟合；事实、偏好和任务成功率还要各自评估。

## 到了 2026 年，还用这些指标吗？

会用，但不把它们当成通用的模型质量分。经典指标的定义没有因为出现新模型而失效；变的是任务、评估数据与所需证据。

| 实际任务 | 可以保留的检查 | 还需要什么 |
| --- | --- | --- |
| OCR、语音转录 | CER / WER | 关键字段、漏行、说话人和时间戳 |
| 翻译 | 固定协议的 BLEU / chrF | 语义质量、专有名词、人工抽查 |
| 摘要 | ROUGE 与长度 | 事实一致性、遗漏、引用支持 |
| RAG 问答 | Exact match / 字段检查 | 检索证据、归因、不可回答识别 |
| Agent | 最终文本可作辅助 | 实际状态变化、工具执行和副作用 |

语义指标如 [BERTScore](https://arxiv.org/abs/1904.09675)利用上下文表示比较匹配程度；[COMET](https://aclanthology.org/2020.emnlp-main.213/)学习翻译质量评价。它们比词面重合多看一些东西，但仍依赖模型、训练数据和评估域，不能自动替代事实核查。LLM judge 也要单独[校准](llm-as-a-judge/calibration.md)。

实际报告里，把测试集版本、分词与归一化、参考数量、实现版本、聚合方式和失败样例放在一起。两次分数差很小，还应看配对不确定性；不要仅凭一个小数点变化宣布模型变好了。下一篇：[指标稳健性](metric-robustness.md)。
