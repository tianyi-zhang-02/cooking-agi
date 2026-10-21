# 分数涨了，究竟能说明什么？

**中文** · [English](metric-robustness.en.md)

新模型 Recall 高了 3 个点，你准备宣布有效。先等一下：候选池一样吗？新旧模型测的是同一批 query 吗？少数用户有没有退化？换一个随机种子，还会是这个结果吗？

先算几个小例子，看看分数是怎么来的，再决定要用什么统计方法。所有数字都是教学数据，不是模型实测。只想先看懂，可以读[人群混合](#population-mix)和[同样涨分的两种情况](#paired-binary)；想自己算，再看[配对 bootstrap](#paired-bootstrap)。

## 1. 同一个 Recall 名字，可能不是同一道题

设 query 的已标注正例集合为 $P$，系统前 $K$ 个结果为 $R_K$。标准 Recall@K 是：

$$
\mathrm{Recall@K}=\frac{|P\cap R_K|}{|P|},\qquad |P|>0.
$$

假设有 8 个已知正例，前 3 条全是正例。Recall@3 是 $3/8$，不是 1。有些实现把分母换成 $\min(|P|,K)$，此时会得到 1。后者可以作为一个明确命名的归一化指标，但不能和前者不加说明地比较。

| 要先核对什么 | 为什么会改变结论 |
| --- | --- |
| 没有正例的 query 怎么处理 | 排除、记零、单独统计会改变平均值；要报告覆盖率 |
| 按 query 平均，还是先合并命中数 | Macro 给每个 query 同等权重；micro 更偏向正例多的 query |
| 候选池怎么构造 | 全库、随机采样池和困难负例池不是同一个任务 |
| 正例从哪里来 | 曝光点击、人工标注和 teacher 判断测的不是完全相同的东西 |

只有一个正例、其余都是采样对照时，Recall@K 退化为“正例有没有进前 K”。随机排序下，若有 $n$ 个对照，命中概率为 $\min(K,n+1)/(n+1)$。$K=10$ 时，19 个对照对应 50%，999 个对照对应 1%。这里不讨论模型优劣，只说明候选池本身就能改变难度。

更麻烦的是，对照里可能还有未标注的相关内容。模型把它们排上来，在现有尺子上也可能吃亏。所以结果应该写成“在这一套候选池和标注下的 Recall”，不要直接等同于真实满意度。

## 2. 平均值是按谁的比例算的？ {#population-mix}

一个虚构评估里，90% 是熟悉用户，10% 是新用户：

| 人群 | 旧模型 | 新模型 | 变化 |
| --- | --- | --- | --- |
| 熟悉用户 | 80% | 85% | +5 个百分点 |
| 新用户 | 40% | 30% | −10 个百分点 |
| 按 90% / 10% 汇总 | 76% | 79.5% | +3.5 个百分点 |

总体变好了，新用户却更差。这两个结论可以同时成立。若我们关心的是两类用户各占一半的目标人群，旧模型为 60%，新模型为 57.5%，结论就反过来了。

```text
新用户占 10%：旧模型 76.0% → 新模型 79.5%  （+3.5 个百分点）
新用户占 50%：旧模型 60.0% → 新模型 57.5%  （−2.5 个百分点）
每组内的表现完全没变，变的是我们在回答谁的问题。
```

不要为了让结论好看才选权重。原始人群比例、目标使用人群、重点保护的切片，要事先约定。分层汇总 $\bar m=\sum_g w_gm_g$ 里的 $w_g$ 是问题定义的一部分。

匹配抽样（matched sampling）可以让某些已测量条件更可比，但不会自动让不同人群成为随机对照。匹配了正例数量，仍可能没有匹配任务难度；小样本切片的高分，也不能直接解释成某一设计带来的因果效果。

## 3. 多高才算不是运气？先用同一批样本比 {#paired-comparison}

同一批 query 同时跑旧模型和新模型，记录每个 query 的差值 $d_i=m_i^{\mathrm{new}}-m_i^{\mathrm{old}}$。先看差值分布，再看平均。只比较两组无关样本的均值，会把题目难度变化也混进来。

### 都从 70% 涨到 80%，区别在哪里？ {#paired-binary}

如果每题只判对错，可以先把结果分成四类。下面是两种虚构的比较，每种都有 200 道题，每道题都保留新旧结果：

<div class="worked-table" markdown="1">

| 逐题发生了什么 | 比较 A | 比较 B |
| --- | --- | --- |
| 新旧都答对 | 100 | 140 |
| 原来对，现在错 | 40 | 0 |
| 原来错，现在对 | 60 | 20 |
| 新旧都答错 | 0 | 40 |
| 正确率变化 | 70% → 80% | 70% → 80% |

</div>

A 改对了 60 题，也弄错了原本会的 40 题；B 改对了 20 题，在这批题里没有退步。**净增都是 20 题，但变化的分布不同。** 只保存 70% → 80%，这层区别就丢了。

对于“这两个固定模型的正确率是否相同”，可以用成对二分类的 McNemar 检验。它关注的是两类不一致结果：改善多少题，退步多少题。B 的变化更集中在改善方向；A 则有更多来回变化。检验还需要题目对之间独立、样本来自同一目标分布、判分规则固定等前提，不能把同一个用户的一百条近似问题当成一百份独立证据。

<details markdown="1">
<summary>想算一下：为什么不只看净增的 20 题？</summary>

把改善数记为 $g$，退步数记为 $\ell$。两种情况的正确率差都是 $(g-\ell)/200=0.10$，但不一致的题数 $g+\ell$ 分别是 100 和 20。

在新旧正确率相同的零假设及上述抽样条件下，给定不一致的题数后，改善数服从 $G\sim\mathrm{Binomial}(g+\ell,1/2)$。直觉上，变化如果没有偏向哪一个模型，不一致结果朝两个方向出现的机会就相同。对称的二项分布给出双侧精确 p 值：

$$
p=\min\left(1,\;2\sum_{j=0}^{\min(g,\ell)}\binom{g+\ell}{j}2^{-(g+\ell)}\right).
$$

下面直接用 Python 标准库算这张小表。输入是题数，不接受负数、小数或布尔值：

```python
from math import comb

def exact_mcnemar(gained, lost):
    if any(type(count) is not int or count < 0 for count in (gained, lost)):
        raise ValueError("Counts must be nonnegative integers")
    disagreements = gained + lost
    if disagreements == 0:
        return 1.0
    tail_count = sum(comb(disagreements, count)
                     for count in range(min(gained, lost) + 1))
    return min(1.0, 2 * tail_count / (2 ** disagreements))

comparison_a = exact_mcnemar(60, 40)
comparison_b = exact_mcnemar(20, 0)
assert abs(comparison_a - 0.0568879336409808) < 1e-12
assert comparison_b == 2 / (2 ** 20)
print(f"A: {comparison_a:.6f}; B: {comparison_b:.8f}")
```

结果约为 A：0.056888，B：0.00000191。同样的涨幅，在这个检验下支持“正确率不同”的证据强弱不同。p 值不是“新模型没用的概率”，也不是收益大小；B 更小，不代表它比 A 多提高了准确率。[statsmodels 的 McNemar 文档](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)说明了精确二项检验与卡方近似的区别；也可用 [SciPy 的双侧 binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)核对。

两边每题都判得一样时，代码约定返回 1；没有观察到不同结果，不等于证明两模型在所有任务上等价。检验也不替你决定可否上线：退步是否涉及关键任务、时延增加多少，还要另外看。

</details>

### 分数不只是对错时，再看配对 bootstrap {#paired-bootstrap}

Recall 这样的逐题分数可以是 0.4、0.6，不适合直接当作 McNemar 的对错计数。我们要估计的仍然是平均差值，只是换一种办法看它对样本变化有多敏感。

下面是一个最小的配对 bootstrap：每次重采样 query，同时保留它的新旧配对。它估计的是**给定这两次模型输出时，测试样本组成带来的不确定性**。

```python
import random
from statistics import mean

old_scores = [0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3]
new_scores = [0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6]
differences = [new - old for old, new in zip(old_scores, new_scores)]
generator = random.Random(7)
estimates = sorted(mean(generator.choices(differences, k=len(differences)))
                   for _ in range(4000))
lower, upper = estimates[100], estimates[3899]
assert abs(mean(differences) - 0.05) < 1e-12
assert lower <= 0 <= upper
print(round(mean(differences), 3), round(lower, 3), round(upper, 3))
```

均值高了 0.05，但这个八个样本的示例区间仍覆盖 0，不能有把握地宣布稳定提升。这也不证明两个模型等价；数据可能还不足以分辨。这里用近似的 percentile 区间演示，实际分析可以参考 [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) 的配对设置与区间方法，检查小样本、偏态和退化情况。

如果同一个用户贡献了很多 query，它们可能相关，应按用户等合理单位做 cluster bootstrap，再按事先选择的 query 加权或用户加权方式计算指标。按日志行随便重采样，往往把相关行当成了更多独立证据。

Bootstrap 也不能替你测训练方差。要比较训练方法，还需要独立训练种子；一个 checkpoint 在同一测试集上反复跑，不等于重新训练了多次。若 judge 也有随机性，还要单独检查评分重复性。

## 4. 一次看二十个指标，总能碰到几个涨的

看更多切片有助于找问题，但从几十次比较中挑最大的涨幅，很容易高估效果。可以先约定一个主要指标、几项不能明显退化的指标，以及少量有理由关注的切片；其余发现记作探索性结果，再用独立留出数据确认。

不是每个结果都要塞进“显著／不显著”二选一。还要报效应大小、区间、样本量和实际代价。一个很小但统计显著的提升，未必值得两倍延迟；没达到显著性，也不等于证明没有价值。

## 5. 评分器本身变了怎么办？

模型版本、prompt、参考答案和任务分布都会影响 LLM judge。新系统的回答变长了，judge 分数变高，可能是内容更完整，也可能是评分器偏爱长答案。需要拿固定的人审锚点复查，按任务类型记录错误，而不是只看整体一致率。

评估集也不能无限次当开发集使用。记录哪些样本用来调 rubric、选阈值和改模型，最后留一部分未参与这些决策的数据。训练 teacher 和最终 evaluator 相同并不必然无效，但不能把二者的一致当成独立证据；需要外部参考检验共同盲点。

更具体的偏差例子见 [LLM-as-a-Judge](llm-as-a-judge/README.md)。如果还不确定失败发生在检索、上下文还是生成，回到[分层评估](evaluation-stack.md)先拆开查。

想把这些检查做成能运行的小项目，可以接着看[评估报告与 CI](../practice/web-lab/eval-report.md)。其中的 CI 按预先约定的规则拦截回归；检查通过，只能说明满足了这些规则，不能证明模型整体更好。
