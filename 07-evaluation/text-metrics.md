# BLEU、ROUGE、编辑距离：分数到底在比较什么？

**中文** · [English](text-metrics.en.md)

> 最近审阅：2026-10-08 · 先读：[分层评估](evaluation-stack.md)

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
\operatorname{BLEU}_N=BP\exp\left(\frac1N\sum_{n=1}^N\log p_n\right),
\qquad BP=\exp\left(\min(0,1-r/c)\right).
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

给定固定测试序列，按每个有效目标 token 的负对数概率取平均，再取指数：

$$
\operatorname{PPL}=\exp\left(-\frac1T\sum_{t=1}^T\log p(x_t\mid x_{<t})\right).
$$

假设 2 个目标的概率分别为 0.5 和 0.25，PPL 为 $\sqrt8\approx2.828$。它不是“准确率 1/2.828”，也不是字面上每个位置都存在 2.828 个同等可能选项。

比较 PPL 时，固定数据、tokenizer、上下文窗口与有效目标。不同 tokenizer 改变计数单位，数值不能直接横比。滑动窗口可以给后面的 token 更多上下文，但重叠部分不能重复计入目标；不同长度 batch 的平均 loss 也应按有效 token 数加权。[实现说明](https://huggingface.co/docs/transformers/perplexity)

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
