# 奖励模型：一题有多个回答，怎样训练排序？

**中文** · [English](reward-ranking.en.md)

> 最近审阅：2026-11 · 前置：[偏好与奖励模型](where-preferences-come-from.md)

同一个问题给出四份回答，标注者从好到差排一次，就能拆出六个偏好对。数据看起来增加了六倍，但这不是六次独立判断：回答还是那四份，问题还是同一个。

这篇沿着一份排序看三件事：怎么计算 loss、不同问题该占多大权重、最后的分数能拿来比较什么。先算小例子，再讨论数据和实现；不是给某个奖励模型推荐一套通用最优配置。

## 多比较几对，为什么会改变题目的权重？ {#prompt-weighting}

假设问题甲只有两个候选，产生 1 个偏好对；问题乙有四个候选，产生 $\binom42=6$ 个偏好对。若把七对混在一起取平均，乙会占 $6/7$ 的权重。

假设甲的 pair loss 为 0.2，乙的六个 pair loss 都为 0.8：

| 聚合方式 | 计算 | 结果 |
| --- | --- | ---: |
| 每个偏好对等权 | $(0.2+6\times0.8)/7$ | 0.714286 |
| 每道题先平均，再平均题目 | $(0.2+0.8)/2$ | 0.5 |

哪个更合理取决于目标。若每道题代表一次同等重要的用户需求，题目等权更容易解释；若有意让某类题或高置信度标注占更大权重，也可以明确加权。不要让“恰好多采了几个候选”悄悄决定训练分布。

对第 $i$ 道题的已标注偏好对集合 $\mathcal P_i$，一种题目等权目标是

$$
L_{\mathrm{pair}}=
\frac1N\sum_{i=1}^N\frac1{|\mathcal P_i|}
\sum_{(w,l)\in\mathcal P_i}\operatorname{softplus}(r_{i,l}-r_{i,w}).
$$

这里用 $\operatorname{softplus}(z)=\log(1+e^z)$ 写 Bradley–Terry 的负对数似然。没有标注对的题目不能直接除以零；应该跳过并记数，或放进另外的监督任务。

[InstructGPT（2022），§3.5](https://arxiv.org/html/2203.02155v1#S3.SS5)报告过拆散相关比较导致过拟合的问题，并把同一 prompt 的所有比较放在一个 batch element 里。论文的 RM 损失（式 1）也在每题内部除以 $\binom K2$，正是上表第二行的题目等权。意思不是“整个 batch 只能有一题”，也不是“全库的题必须一起训练”。这个组织方式还能让同一候选的 reward forward 被多对比较复用。论文里的改善是其数据和训练配置下的结果，不能当成所有 RM 的保证。

## 三份回答，完整排序的概率怎么算？ {#listwise}

假设一个修报错的问题有 A、B、C 三份回答，标注为 $A\succ B\succ C$。模型给出的分数暂定为

$$
r_A=\ln4,\qquad r_B=\ln2,\qquad r_C=0.
$$

这只是便于手算的输出，不是给三份回答预先规定正确分数。Plackett–Luce（PL）模型逐个选出下一名：先从三个里选第一，再从剩下两个里选第二。

<div class="drl-flow" aria-label="三份回答的完整排序概率">
<span>从 A、B、C 选 A<br><small>4 / (4 + 2 + 1)</small></span><b>×</b><span>从 B、C 选 B<br><small>2 / (2 + 1)</small></span><b>×</b><span>只剩 C<br><small>概率 1</small></span>
</div>

因此

$$
P(A\succ B\succ C)=\frac47\frac23=\frac8{21}\approx0.380952,
\qquad L_{\mathrm{PL}}=-\log\frac8{21}\approx0.965081.
$$

对于候选集合内一个完整、无并列的排序 $\tau$：

$$
P(\tau\mid x,y_{1:K})
=\prod_{k=1}^{K}
\frac{e^{r(x,y_{\tau(k)})}}
{\sum_{j=k}^{K}e^{r(x,y_{\tau(j)})}}.
$$

它是**给定这一组候选的排序概率**，不是模型生成这些文本的概率。当 $K=2$，公式变成 Bradley–Terry。PL 的条件还包括：两个候选的相对选择比由它们的分数决定，不随其他候选变化；强烈的上下文效应、多维取舍或群体偏好未必符合它。[DPO 论文附录 A.3](https://arxiv.org/html/2305.18290v2#A1.SS3)也用这一定义推导了排名监督下的目标。搜索和推荐里的排序学习早就用过同一个损失：[ListMLE（Xia 等，2008）§4.3.1](https://icml.cc/Conferences/2008/papers/167.pdf)把它称为似然损失，即真实排序在 PL 模型下的负对数似然。

## 拆成所有偏好对，与完整排序并不等价 {#pairs-versus-ranking}

同一排序可以拆成 A 胜 B、A 胜 C、B 胜 C。对应的成对概率是

$$
\frac46,\quad \frac45,\quad \frac23.
$$

把它们相乘得到 $16/45\approx0.355556$；负对数之和约 1.034074，不等于 PL 的 0.965081。再按三对取平均，结果为 0.344691，只是又换了归一化。

| 训练方式 | 用到的标签 | 这里的 loss | 怎样解释 |
| --- | --- | ---: | --- |
| 两两比较取平均 | 三个 pair | 0.344691 | 平均每对的负对数似然 |
| 两两比较求和 | 三个 pair | 1.034074 | 合并相关比较的目标 |
| PL 完整排序 | 一个完整 rank | 0.965081 | 该完整排序的负对数似然 |

不能看 0.344691 更小，就说第一个方法更好。三个数统计的单位不同。

PL 对六种可能排序的概率加起来为 1；把每个排序对应的三个 pair 概率相乘，再对六种排序求和，一般不等于 1。两两比较仍然是合理的训练选择，只是这些相关事件的概率乘积，不自动成为完整排序的归一化概率模型。

两种做法估计的是不是同一组参数？[Zhu、Jiao 与 Jordan（2023）§4.1 定理 4.3](https://arxiv.org/html/2301.11270v5#S4.SS1)在 PL 模型成立、奖励对已知特征线性的条件下证明：完整排序的最大似然，和 InstructGPT 那样拆成 $K(K-1)/2$ 对的估计都会收敛到真实参数；完整排序的渐近方差更小。拆开不算错，只是同样多的标注能提供的信息没有用足。模型假设不成立时，这个比较也要重新检查。

## 一个排序怎样推动三个分数？ {#gradient}

PL loss 可以写成数值更稳定的形式：

$$
L_{\mathrm{PL}}=\sum_{k=1}^{K-1}
\left[
\operatorname{logsumexp}_{j\ge k}r_{\tau(j)}-r_{\tau(k)}
\right].
$$

每一步的梯度是“剩余候选上的 softmax，减去本步被选中的 one-hot”。本例：

| 分数 | 第一步贡献 | 第二步贡献 | 总梯度 |
| --- | ---: | ---: | ---: |
| $r_A$ | $4/7-1$ | 0 | $-3/7$ |
| $r_B$ | $2/7$ | $2/3-1$ | $-1/21$ |
| $r_C$ | $1/7$ | $1/3$ | $10/21$ |

直接把三个分数当独立变量做梯度下降，会抬高 A、略抬高 B、压低 C。B 虽然输了第一名，仍胜过 C，不是所有“非第一名”都要往下压。

三个梯度加起来为 0，对应共同平移不改变排序概率。实际奖励网络共享参数，更新后每个输出怎样移动，还要经过网络的 Jacobian；不能把这里的独立分数更新当成真实神经网络的保证。

下面是 loss 的核心计算；输入检查、梯度与测试在[配套程序](deep-rl/code/reward_learning_checks.py)中。分数先减最大值，避免直接计算大数的指数。

```python
import math

def ranking_loss(ordered_scores):
    loss = 0.0
    for position in range(len(ordered_scores) - 1):
        remaining = ordered_scores[position:]
        maximum = max(remaining)
        shifted = [score - maximum for score in remaining]
        loss += math.log(sum(math.exp(score) for score in shifted)) - shifted[0]
    return loss

print(ranking_loss([math.log(4), math.log(2), 0]))
```

## 标签没有说的，不要替它补出来 {#partial-rankings}

| 实际标注 | 能监督什么 | 不要擅自做什么 |
| --- | --- | --- |
| 只选 A 为最佳 | A 胜过已展示的其他候选；或 PL 的首选项 | 给剩余候选编造 B 胜 C 的顺序 |
| A、B 都高于 C，A/B 未比较 | A 胜 C、B 胜 C；或对允许的完整排序求和 | 随机排 A/B，再当成确定标签 |
| A 与 B 真正打平 | 保留 tie 标签，并选择支持平局的模型或明确的损失约定 | 把“不知道”和“同样好”混成同一种标签 |
| A、B 都不合格 | 记录绝对不合格信息，必要时再记录相对偏好 | 把被选中的那份当作合格示范 |

比如只知道 A、B 都在 C 前面，PL 对这个事件的概率是

$$
P(A\succ B\succ C)+P(B\succ A\succ C)
=\frac8{21}+\frac27\frac45=\frac{64}{105}.
$$

这表示 A/B 的顺序没有观察到，不是定义了“并列”如何产生。标准的无并列 PL 不能直接表达真实平局；需要额外建模或明确改用什么监督规则。

## 为什么不能把不同题的 reward 放一起排？ {#score-offsets}

问题甲的两个回答给分 2、1；问题乙给分 102、101。两题内部都预测优选回答以 $\sigma(1)\approx0.731$ 的概率胜出。但如果把四份回答混在一起取最高分，只会选中乙。

对每题单独加一个 $c(x)$，BT 和 PL 概率都不变：

$$
r'(x,y)=r(x,y)+c(x).
$$

所以训练标签没有决定跨题的绝对零点。参数共享可能带来某种分数习惯，却不等于这套习惯已经被跨题质量标签验证过。若要做全局过滤、设统一门槛或比较不同任务，必须另做校准与切片检查；题内排序正确不够。

同理，一道题若只比较 A/B 和 C/D，两组之间没有连起来，单看这些标签也无法决定 A 与 C 谁更好。补比较时，连接原本分开的候选群，可能比重复同一对更有信息。

## 数据怎样存，验证集怎样切？ {#data-and-eval}

先保留排序任务的原始结构，再派生训练偏好对。例如：

```json
{
  "prompt_id": "debug-017",
  "candidate_ids": ["answer-a", "answer-b", "answer-c"],
  "ordered_groups": [["answer-a"], ["answer-b"], ["answer-c"]],
  "rubric_version": "debug-v2",
  "generator_snapshot": "policy-demo-v1",
  "annotation_batch": "review-03"
}
```

候选正文与生成参数另按 ID 保存；有多位标注者时，也保留匿名标注标识、分歧和标签状态。`ordered_groups` 是这个教学数据的约定，不是某个框架通用格式。不要靠数组里碰巧的存储顺序表示偏好。

| 检查 | 具体做法 | 能发现什么 |
| --- | --- | --- |
| 数据泄漏 | 先按 prompt／近重复任务组划分集合，再拆偏好对 | 同一回答通过另一对组合进入验证集 |
| 权重 | 打印每题候选数、pair 数与最终 loss 权重 | 候选多的题无意中主导梯度 |
| 候选复用 | 每个候选在本次计算中取一个分数，再组成多对 | 重复 forward、同一候选的随机输出不一致 |
| 标注一致性 | 交换展示位置，查看分歧与缺失标签 | 位置偏差、强行排序、含义不清的规则 |
| 分布变化 | 额外评估当前策略生成的回答 | 旧候选上准确，新策略上失准 |
| 真实效果 | 用任务测试、独立人工复核和失败切片 | 会预测偏好，但不一定会判断事实或代码正确性 |

训练前也要固定“一个样本”指什么：一个 pair、一次完整排序，还是一道题汇总的多次标注。候选数、标注者数、每题权重最好分别记录，之后才说得清一次改数据到底改变了什么。

## 复算与下一步 {#checks}

```bash
python3 05-post-training/deep-rl/code/reward_learning_checks.py
python3 -m unittest discover -s site/tests -p test_reward_learning_checks.py
```

测试核对 PL 的六种排序归一化、$K=2$ 时退化成 BT、梯度的有限差分，以及大分数、共同平移和题目加权。数值下溢可能让极小排序概率显示成 0，所以训练应保留稳定的 log loss，不要先算概率再取 log。

接下来可以看 [DPO 如何把奖励改写成策略概率比](after-ppo.md#dpo-derivation)，或回到[偏好数据与奖励失真](where-preferences-come-from.md)。数据组织、目标函数和评估需要一起核对，不是换一个 loss 名字就能解决全部问题。
