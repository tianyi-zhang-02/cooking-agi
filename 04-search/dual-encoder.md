# 双塔检索：为什么把查询和文档分开算？

**中文** · [English](dual-encoder.en.md)

假设你在做一个课程资料搜索器，库里有一百万段笔记。用户问“为什么梯度会消失”，你想找到讨论连续相乘和 RNN 的段落，而不是只找出包含“梯度”两个字的文章。

最直接的办法，是把问题和每一段笔记一起交给模型判断。但每次搜索都做一百万次这样的比较，很难等得起。双塔的出发点很朴素：**文档大部分时候没变，能不能提前算好，只在用户提问时算问题？**

只需要知道向量点积，就可以跟完下面的例子。想动手的话，跳到[手算一次训练](#training-step)；已经用过向量库，可以直接看[索引和模型为什么会对不上](#index-version)。数字都是教学用的。

## 1. 分开编码，换来什么？

查询编码器把问题变成向量 $\mathbf q=f_\theta(x)$，文档编码器把段落变成 $\mathbf d=g_\phi(y)$。两边可以共享参数，也可以使用不同参数；“双塔”描述计算路径，不一定是两套完全不同的模型。

$$
s(x,y)=\mathbf q^\top\mathbf d.
$$

如果先把两个向量都除以各自的长度，点积就是余弦相似度。归一化之后，方向决定分数；不归一化时，向量的长度也会影响排名。这不是一个可以随便切换的服务端选项，要和训练约定一致。

```text
文档更新时：文档 → 分段 → 文档编码器 → 向量索引
用户提问时：问题 → 查询编码器 → 近邻搜索 → 少量候选 → 重排
```

双塔省掉的是每次请求都重新编码全部文档，不是让检索完全免费。暴力扫描 $N$ 个 $d$ 维向量仍要约 $O(Nd)$ 的乘加；近似最近邻搜索（ANN）用索引换取更快的查找，同时允许漏掉部分真正的近邻。索引的延迟、内存和召回率要实际测，不能一概写成 $O(\log N)$。

| 做法 | 可以提前算什么 | 每次请求还要算什么 | 损失或代价 |
| --- | --- | --- | --- |
| 双塔 | 文档向量 | 查询向量、近邻搜索 | 查询看不到文档的细节再决定怎么编码 |
| Cross-encoder | 通常不能缓存完整的配对分数 | 查询与每个候选联合编码 | 细粒度比较更贵，通常只用于较小候选集 |
| Late interaction | 文档的多个 token 向量 | 查询 token 与候选 token 的匹配 | 保留更多细节，但索引更大，匹配也更复杂 |

[DPR](https://arxiv.org/abs/2004.04906) 是学习式双编码器的经典例子。[ColBERT](https://arxiv.org/abs/2004.12832) 则展示了 late interaction：不是先把整段文档压成一个点，再做一次点积。

## 2. “相似”是怎么学出来的？

单纯拿一个语言模型做平均池化，不会自动得到适合任务的几何空间。训练要告诉模型：对这个查询，哪些文档应该比当前对照文档更靠前。

一个常见目标是单正例的对比损失：

$$
L=-\log\frac{\exp(s^+/\tau)}{\exp(s^+/\tau)+\sum_{j=1}^{B-1}\exp(s_j^-/\tau)}.
$$

$s^+$ 是正例分数，$s_j^-$ 是当前批次里的对照分数，$\tau>0$ 是温度。分母不是“世界上所有不相关文档”，只是这次训练拿来比较的一组候选。模型学的是在这组比较里区分正例。

### 有正负例，不等于在做逐对二分类 {#infonce-and-ce}

还是搜“为什么梯度会消失”。这次训练拿来 3 段笔记：连续相乘与 RNN、怎样设置学习率、怎样保存 checkpoint。正例在第 0 个位置。**单正例 InfoNCE 问的是“这 3 个里面，应该选哪一个”，不是分别问 3 次“相关吗”。** 因此它是候选索引上的多类交叉熵；候选换了，“类别”的具体文档也会换，不是永远固定的主题分类。

用一组没有偏向的分数最容易看出差别：

| 分数都为 0 | 输出 | 监督 |
| --- | --- | --- |
| 候选 softmax | `[1/3, 1/3, 1/3]`，相加为 1 | 正例索引 `0` |
| 每对独立 sigmoid | `[1/2, 1/2, 1/2]`，不要求相加为 1 | 二值标签 `[1, 0, 0]` |

前者的 loss 是 $\log 3\approx1.099$。后者若对 3 对取平均，BCE 是 $\log 2\approx0.693$；若求和则是 $3\log 2$。不能比较这两个数字来判断哪种目标更好，它们回答的问题和归一化方式不同。公式可对照 [InfoNCE 原论文](https://arxiv.org/abs/1807.03748)和 PyTorch 2.8 的 [CrossEntropyLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html)、[BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html)。

再做一个不用算公式的检查：保持原来 3 个候选的分数不变，添一个同样 0 分的候选。Softmax 中正例的概率会从 $1/3$ 变成 $1/4$；独立 sigmoid 给原来每一对的概率仍是 $1/2$。**候选 softmax 的输出依赖这次跟谁比较，不是用户喜欢某篇文档的绝对概率。**

实现 CE 时直接传入温度缩放后的 logits。先 softmax 再把概率当 logits 传进去，会多做一次 softmax，改变目标。只想理解目标的区别，读到这里就可以；下面沿用 3 个候选，算一遍它怎样更新。

## 3. 手算一次训练 {#training-step}

暂时令温度为 1。一个正例和两个对照的分数是 $[2,1,0]$，softmax 后大约是 $[0.665,0.245,0.090]$。损失是 $-\log(0.665)\approx0.408$。

记温度缩放后的分数为 $z_j=s_j/\tau$，则：

$$
\frac{\partial L}{\partial z_j}=p_j-\mathbb 1[j=+].
$$

三个梯度约为 $[-0.335,0.245,0.090]$。如果把分数当成独立变量，梯度下降会抬高正例、压低对照，而且更用力地压那个得分较高的对照。真正训练 encoder 时，多个分数共用参数，单次更新不保证每个分数都按这个方向变化。对原始分数 $s_j$ 的梯度还要乘 $1/\tau$。

```python
import math

def contrastive_step(scores, temperature=1.0):
    if not scores or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Need scores and a finite positive temperature")
    scaled = [score / temperature for score in scores]
    if not all(math.isfinite(value) for value in scaled):
        raise ValueError("Scaled scores must be finite")
    maximum = max(scaled)
    shifted = [value - maximum for value in scaled]
    weights = [math.exp(value) for value in shifted]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    loss = math.log(total) - shifted[0]
    gradients = [(probability - int(index == 0)) / temperature
                 for index, probability in enumerate(probabilities)]
    return loss, probabilities, gradients

loss, probabilities, gradients = contrastive_step([2, 1, 0])
assert abs(loss - 0.407606) < 1e-5
assert gradients[0] < 0 < gradients[1]
assert abs(sum(gradients)) < 1e-12
assert math.isclose(contrastive_step([1e20, 1e20])[0], math.log(2))
```

最后一个检查故意给两个候选相同的巨大分数。它们各占一半概率，loss 应该是 $\log 2$，不是 0。先把所有分数减去最大值，再直接用平移后的数算 loss；不要把大数加回去又减掉，让小的 loss 被浮点舍入吞掉。

现在把第二篇文档改成另一篇同样能解释梯度消失的文章，但仍把它标成负例。代码一点也不会报错，梯度却会把它推远。这就是假负例（false negative）：**数值计算正确，不代表监督正确。**

## 4. 多个正例和困难负例，怎么处理？

In-batch negatives 很便宜：其他查询的正例可以复用为本查询的对照。但同一个主题可能对应多篇好文档。先检查重复文档和已知多正例；否则批次越大，碰到假负例的机会也可能越大。

随机抽来的文档、别的用户点过的内容、没被曝光的内容，都不自动等于当前查询下已确认不相关。它们可能是训练用的对照，但这个假设要写清楚，不能把“没有正标签”说成“用户拒绝过”。

如果已知正例集合 $P$，一种设计是最大化它们的总概率：

$$
L_{\mathrm{set}}=-\log\sum_{j\in P}p_j.
$$

它允许模型把大部分概率放在其中一个正例上。另一种设计是分别照顾每个正例：$L_{\mathrm{avg}}=-|P|^{-1}\sum_{j\in P}\log p_j$。两者并不等价：前者关注“至少找到一个”，后者要求每个已知正例都有概率。选择哪个，要看你的任务，不是看公式哪个更短。

困难负例（hard negative）也不是越难越好。一个排得很靠前的“负例”，可能是模型分不清，也可能是标签漏了。可以检查一小批高分错误，再决定要不要增加难度。无论如何，都要记录负例从哪里来、何时挖掘、用的哪个模型；相关背景见[反馈与训练目标](../01-data-and-feedback/feedback-to-objectives.md)。

评估也要固定候选池与标注口径。增加负例数量或改变采样方法，会改变任务难度；此时 loss 或 Recall 的变化，不能全部归因于模型变好或变坏。

## 5. 模型升级了，为什么反而搜不到？ {#index-version}

假设训练更新了两个编码器，却只上线新的查询编码器，索引里还是旧文档向量。维度都一样，API 也正常，但两个空间未必还对齐。除非训练明确约束兼容旧空间，否则“同样是 768 维”不代表可以混用。

一次可复现的检索版本，至少要把编码器 checkpoint、tokenizer、文本清洗、分段规则、池化、归一化、相似度和索引快照对应起来。发布时先用一小批固定 query 验证，再决定重建索引或保留兼容路径。

排查时分两步，别一开始就调 ANN：

1. **精确搜索也找不到**：先查表示、训练数据、分段和文档是否入库。
2. **精确搜索能找到，ANN 找不到**：再查索引参数、量化、过滤和检索预算。

“ANN 召回率”衡量近似搜索相对精确近邻丢了多少；“相关文档 Recall”衡量结果里有多少已标注的相关内容。两个指标不能互相替代，精确找到一堆不相关近邻也不算任务成功。

## 6. 单向量不够时，先别急着堆塔

一条查询同时包含“适合初学者”和“包含数学推导”，一个向量可能丢掉其中的限制。可以先试关键词补充、改写查询或小候选集重排，再考虑多向量。多向量值得做的证据，是它找到了单向量漏掉的有用文档，而不只是用了更多查询预算。

比较时固定最后交给重排的候选数量，记录新增相关文档、重复比例、延迟和索引大小。准备挑具体模型，可以读 [Qwen3 Embedding 与 BGE-M3](embedding-models.md)；要组合几路结果，再看[混合检索与重排](hybrid-and-reranking.md)。
