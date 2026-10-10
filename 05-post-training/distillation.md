# 蒸馏：学生到底向老师学什么？

**中文** · [English](distillation.en.md)

> 最近审阅：2026-10 · 前置：[SFT](sft-and-its-ceiling.md)、[语言模型目标](../00-foundations/deep-dives/language-model-objective.md)

一个大模型把题做对了，你把答案交给小模型训练。这当然是一种蒸馏，但不是全部。你也可以让小模型学老师对不同答案的概率，或者让小模型先尝试，再请老师检查它走到的地方。

三种做法需要的接口、预算和训练数据都不同。先别急着问“用什么 KL”，先问：**老师实际能交给学生什么？**

第一次读，可以先看三种监督材料和 soft-target 算例。想实现训练，再看后面的 tokenizer 对齐和 on-policy 梯度：它们决定“看起来像蒸馏”的代码是否真的在学预期的目标。

## 同一个问题，可以有三种教学材料

假设一个虚构的客服分类任务只有 3 个标签：退款、物流、商品咨询。输入是“包裹还没到，能退吗？”，它确实有一点歧义。

| 老师提供什么 | 学生实际学什么 | 留下或丢掉了什么 |
| --- | --- | --- |
| 一条答案：“退款” | 用它做 hard-label CE / SFT | 简单，但没保留其他标签的概率 |
| `[0.60, 0.35, 0.05]` | 拟合老师的分布 | 保留了“物流也有可能”这个信息 |
| 对学生当前前缀给出分布 | 在学生走到的状态上学习 | 更贴近学生实际行为，但得反复调用老师 |

这些概率只是教学数值，不是置信度已校准的事实。分布更详细，不代表一定更正确。老师也可能误读问题，或者把自己的偏差教给学生。

## 先算一次 soft-target loss

令老师分布为 $p=[0.60,0.35,0.05]$，学生为 $q=[0.40,0.40,0.20]$。在同一个标签空间中，前向 KL 是：

$$
\begin{aligned}
D_{KL}(p\|q)&=\sum_c p_c\log\frac{p_c}{q_c}\\
&\approx0.1272.
\end{aligned}
$$

也可以训练 soft-target cross entropy：

$$
\begin{aligned}
H(p,q)&=-\sum_c p_c\log q_c\\
&\approx0.9509.
\end{aligned}
$$

两者相差老师自己的熵 $H(p)$。当老师固定时，这个差值不依赖学生参数，所以对学生有相同梯度；**损失数值不相同，不等于学习方向不同**。经典[知识蒸馏论文](https://arxiv.org/abs/1503.02531)讨论了使用软分布作为监督。

```python
import math

teacher = [0.60, 0.35, 0.05]
student = [0.40, 0.40, 0.20]
cross_entropy = -sum(prob * math.log(pred) for prob, pred in zip(teacher, student))
teacher_entropy = -sum(prob * math.log(prob) for prob in teacher)
forward_kl = sum(prob * math.log(prob / pred) for prob, pred in zip(teacher, student))
assert abs(cross_entropy - teacher_entropy - forward_kl) < 1e-12
print(round(cross_entropy, 4), round(forward_kl, 4))
```

与 softmax 相接时，soft CE 对学生 logit 的梯度是 $q-p=[-0.20,0.05,0.15]$。若只用老师的 argmax 作为 one-hot 标签，梯度则是 `[-0.60, 0.40, 0.20]`。第二种会更强地压低“物流”；第一种保留了老师认为它仍有可能的判断。

## 温度 T 怎样改变监督？ {#_2}

把 logits 除以温度 $T$ 后做 softmax：

$$
\begin{aligned}
p_T&=\operatorname{softmax}(z^{teacher}/T),\\
q_T&=\operatorname{softmax}(z^{student}/T),\\
\mathcal L_{KD}&=T^2D_{KL}(p_T\|q_T).
\end{aligned}
$$

正温度增大时，同一组 logits 的分布会变平，让非最大项有更多权重。$T^2$ 是经典蒸馏中用于补偿温度带来梯度尺度变化的做法；它不意味着不同温度下训练严格等价，也不能代替对 loss 权重的检查。

如果老师最初就把任务理解错了，升温不会把错误变成真相。它改变的是监督分配，不是老师的知识。

有真实标签时，也可以把两种监督合在一起：

$$
\begin{aligned}
\mathcal L&=(1-\lambda)\operatorname{CE}(y,q_1)\\
&\quad+\lambda T^2D_{KL}(p_T\|q_T),\\
&\qquad 0\leq\lambda\leq1.
\end{aligned}
$$

真实标签项通常用温度 1，蒸馏项让老师和学生使用同一个 $T$。$\lambda=0$ 是普通监督学习；$\lambda=1$ 只学老师。中间值不是必然更好：老师和人工标签不一致时，要先检查原因，再决定谁占多大权重。具体权重约定也可能不同。

## 放到语言模型里，需要对齐什么？

分类只有三个固定标签，语言模型则在每个前缀上预测下一 token。比较两份分布前，至少核对：

- **同一个前缀**：老师与学生是否看到相同的已生成内容？角色模板不同会不会改变语义？
- **同一个预测位置**：logits 与下一个 token 对齐，不能少移或多移一位。
- **同一个输出空间**：相同数字 ID 在两个 tokenizer 中可能代表不同文本。不同词表不能直接逐列 KL。
- **同一套有效位置**：padding、prompt、回答、截断后的尾部，哪些参与 loss？

老师是 VLM、学生是纯文本模型时，还要问学生在部署时能看到什么。如果教师标签依赖图里一项学生永远看不到的信息，小模型可能只学到统计关联，不能凭蒸馏凭空恢复证据。可以先把可验证的视觉证据提供给学生，但也要单独评估提取错误和输入缺失时的行为。

API 只给文本而不给完整 logits 时，response distillation 是更直接的起点，不要把它写成“完整分布 KL 蒸馏”。不同 tokenizer 的蒸馏也可以做，但需要明确的方法，例如在文本输出层对齐；不是直接比较同形状数组就够了。

## 不同 tokenizer：对上了文本，还没有对上整个分布

用一个人为分词例子：同一段 `bluebird`，老师切成一个 token，学生切成 `blue` 和 `bird`。老师对这段的 log-prob 是 −1.2，学生两步是 −0.7 和 −0.9，加起来为 −1.6。可以比较这一条固定文本路径的总 log-prob，但不能拿老师的一列 logits 对学生两列求逐 token KL。

| 做法 | 需要什么 | 少了什么或多了什么假设 |
| --- | --- | --- |
| 老师生成文本，学生做 SFT | 可用回答和各自的 tokenizer | 没有老师完整分布里的“其他选项” |
| 严格对齐位置 / 共同输出事件 | 前缀与候选含义都对齐 | 报告覆盖率与保留的概率质量（probability mass） |
| 对齐连续文本片段后聚合 | 字节或文本边界、一致的角色语义、片段 log-prob | 片段分数不是天然的逐 token 目标 |
| 专门的跨词表分布转换 | 明确的共同事件空间与概率映射 | 实现复杂，不能当作普通 KL 换个数组 |

例如把老师的 −1.2 平均分成 −0.6、−0.6，只保住了和，没有证明这就是老师对两步的条件概率。同理，按学生旧 log-prob 的比例分配，也是一种训练设计，不是概率链式法则自动推出的唯一答案。

“对齐成功”还要防空格清理、Unicode、特殊 token 和模板造成的假相等。应从结构化 messages 生成各自合法的模板，检查真实回答边界，不能直接复制另一种模型的角色标记。若对不上的位置被 mask 掉，同时报告被丢弃的样本类型；更高覆盖不一定是更可靠监督。

截至 2026-10-09，[跨模型家族 OPD](https://arxiv.org/abs/2606.09456) 已研究如何跨 tokenizer 传递训练信号；[2026-10 的后续分析](https://arxiv.org/abs/2610.08448)在其测试的模型组合中发现，扩大到错配片段的监督覆盖可能反而降低准确率。后者是特定实验的发现，不是“所有片段方法都不好”。本页不复现这些训练，也不把某个实验分支的配置当作 verl 的通用接口；先核对所用实现的版本、目标与梯度。

<details markdown="1">
<summary>算一下：同一片段的分数，怎样分给学生的两步？</summary>

仍用 `bluebird`。老师的片段分数是 −1.2，学生旧分数之和是 −1.6。按旧 log-prob 的比例分配时，缩放系数为 `−1.2 / −1.6 = 0.75`，得到两个目标：

| 学生 token | 旧 log-prob | 分配的目标 | 目标减去旧值 |
| --- | ---: | ---: | ---: |
| `blue` | −0.7 | −0.525 | +0.175 |
| `bird` | −0.9 | −0.675 | +0.225 |
| 合计 | −1.6 | −1.2 | +0.4 |

这保留了片段的总分，也让原本更意外的那一步得到更大的调整。[跨模型家族 OPD 的式 8–9](https://arxiv.org/html/2606.09456v1#S4.SS2)采用这种分配。**它是分配监督的规则，不是老师真的输出了 `blue`、`bird` 两步分布。** 旧分数之和接近 0 时，还需要单独处理数值问题。

更细的一点：token 路径的概率不一定等于文本概率。假设一个玩具 tokenizer 能通过 `[ab]` 或 `[a, b]` 两条完整路径生成 `ab`，含结束事件的路径概率分别是 0.2 和 0.3，那么文本概率是 0.5。只评分默认编码 `[ab]` 得到的是 0.2。片段对齐没有自动完成对所有编码路径的求和，因此不能只凭文本相同就宣称算出了严格的文本级 KL。

</details>

<details markdown="1">
<summary>代码：先找共同的字节边界，不急着造逐 token 标签</summary>

下面是字节片段的教学实现，不调用任何 tokenizer。它要求两侧拼起来是同一串非空字节，并返回学生 / 老师的半开 token 区间；真实 tokenizer 的特殊 token、解码清理和角色模板还要单独处理。

```python
def aligned_spans(student_pieces, teacher_pieces):
    for pieces in (student_pieces, teacher_pieces):
        if not pieces or any(not isinstance(piece, bytes) or not piece for piece in pieces):
            raise ValueError("expected nonempty byte pieces")
    if b"".join(student_pieces) != b"".join(teacher_pieces):
        raise ValueError("decoded bytes differ")

    teacher_ends = {}
    offset = 0
    for index, piece in enumerate(teacher_pieces, 1):
        offset += len(piece)
        teacher_ends[offset] = index

    spans = []
    offset = 0
    student_start = teacher_start = 0
    for student_end, piece in enumerate(student_pieces, 1):
        offset += len(piece)
        if offset in teacher_ends:
            teacher_end = teacher_ends[offset]
            spans.append((student_start, student_end, teacher_start, teacher_end))
            student_start, teacher_start = student_end, teacher_end
    return spans

assert aligned_spans([b"blue", b"bird", b"!"], [b"bluebird", b"!"]) == [
    (0, 2, 0, 1), (2, 3, 1, 2)
]
```

为什么用字节？一个 token 可能只覆盖某个 Unicode 字符的一部分 UTF-8 字节，逐 token 解码成字符串可能引入替换符。这里先检查整串字节相等，再比较边界。算法的时间是总字节数加 token 数的线性量级；它只解决对齐，不解决监督应该怎样分配。

</details>

## 只存 top-k，会丢掉什么？

继续用三个标签。老师 top-2 是 `[0.60,0.35]`，保留了 0.95 的概率质量。若重新归一化成 `[0.6316,0.3684]`，学生也只在这两项里归一化，那么测的是**条件分布**，不是原来的全分布。

| 学生 | top-2 内部归一化后 | 给第三项的概率 |
| --- | --- | ---: |
| A：`[0.60,0.35,0.05]` | `[0.6316,0.3684]` | 0.05 |
| B：`[0.30,0.175,0.525]` | `[0.6316,0.3684]` | 0.525 |

这种只看条件分布的 loss 会认为 A、B 同样好，却没发现 B 把大量概率放到了第三项。可选做法是保留剩余质量为一个 “other” 桶，或明确使用未经这种条件归一化的截断目标；但聚合尾部也看不到尾部内部如何分配。

这个反例针对**top-k 后两边重新归一化**的做法，不是说所有 top-k 蒸馏都有同样问题。写实验时，把 teacher normalization、student normalization、尾部处理和 token mask 一起记录。

## R1 的数据蒸馏属于哪一种？

[DeepSeek-R1 的公开说明](https://github.com/deepseek-ai/DeepSeek-R1)介绍了用 R1 整理的数据微调 Qwen / Llama 学生。这可以放在上面的“老师提供答案”一栏：学生用自己的 tokenizer 读文本，不要求逐 token 对齐老师的 logits。

不能仅凭词表不同，就反推出一个项目完整的训练方法。也别把 SFT 等同于全参数更新：用什么监督，与更新全部权重还是 LoRA，是两件事。要确认一个具体模型，还是看它的训练报告。

## On-policy 主要换的是前缀来源

假设题目是“3 盒，每盒 4 支笔，一共几支？”示范答案写的是“3 × 4 = 12”。学生却先写了“3 + 4 =”。如果训练永远沿着示范答案走，就不会在这个加号后面听到老师的意见。

On-policy distillation（在线蒸馏）让学生先生成，再把**学生实际生成的前缀**交给老师评分。老师不是另做一遍题再比较两个答案，而是回答：“在你已经写到这里的情况下，接下来各个 token 有多合适？”

<figure class="worked-update worked-update--pairs">
<figcaption>老师可以相同，训练时走过的前缀不同。</figcaption>
<ol>
<li><small>固定示范</small><strong>3 × 4 = → 下一步</strong><span>沿示范前缀获取监督。可预先整理数据，但不一定覆盖学生会犯的错。</span></li>
<li><small>学生采样</small><strong>3 + 4 = → 下一步</strong><span>老师对学生的前缀评分。更贴近学生当前行为，但需要重新生成和评分。</span></li>
</ol>
</figure>

这也说明了边界：老师若顺着加法预测 `7`，只是局部续写合理，整道题仍然错了。反馈更密不等于任务一定正确；还需要答案验证、回溯或其他训练设计。并且 RL 也可以有过程奖励，不是所有 RL 都只在最后打一次分。

### 前缀、监督、loss，是三项选择 {#opd-choices}

| 要决定什么 | 可选做法 | 不能由它单独推出什么 |
| --- | --- | --- |
| 前缀从哪来 | 固定数据、学生采样、混合 | 不能据此决定 KL 方向 |
| 老师返回什么 | 文本、全词表分布、top-k、采样 token 的 log-prob | 不能都当成完整分布监督 |
| 怎样更新 | 固定前缀上的分布拟合、采样动作的策略梯度 | 不能忽略采样与 stop-gradient 的位置 |

[GKD（ICLR 2024）](https://arxiv.org/html/2306.13649v3#S3.SS1)把前缀来源与 divergence 分开选择，可混合固定数据和学生生成的数据。其实现不对前缀采样求导，而是在这些前缀上拟合老师分布。因此，**on-policy 不等于 reverse KL，也不等于把 GRPO 的 reward 换掉。**

### Reverse KL 到底怎样影响一次更新？ {#opd-gradient}

只看一个固定前缀，假设下一步只有 A、B 两种选择。老师是 $p=[0.8,0.2]$，学生是 $q=[0.5,0.5]$。

| 采到的动作 | 老师 log-prob 减学生 log-prob | 直观含义 |
| --- | ---: | --- |
| A | $\log(0.8/0.5)\approx+0.4700$ | 学生给 A 的概率比老师低 |
| B | $\log(0.2/0.5)\approx-0.9163$ | 学生给 B 的概率比老师高 |

把这样的差值作为**冻结的**学习信号，正值鼓励采到的动作，负值压低它。这不是“答案正确率”：老师错了，这个信号也可能指错方向。

想理解公式，记 $q=q_\theta$，用 $f(a)=\log[q(a)/p(a)]$ 简写 log-ratio。在老师固定且两项概率都为正时：

$$
\begin{aligned}
J(\theta)&=\sum_a q(a)f(a),\\
\nabla J&=\mathbb E_{a\sim q}
\left[f(a)\nabla\log q(a)\right].
\end{aligned}
$$

第二行省掉的常数项，其期望为 $\sum_a\nabla q(a)=0$。所以从学生采样动作时，需要这个 score-function 梯度，而不只是对采样后的 `logq - logp` 直接求导。本例对两个学生 logits 的正确梯度约为 `[-0.3466, +0.3466]`；若把采样动作固定后，只微分那项 log-ratio，其期望梯度反而是 `[0, 0]`。

<details markdown="1">
<summary>用有限差分核对：loss 数值对，不代表梯度也对</summary>

下面把两个动作都枚举出来，没有随机误差。`score_gradient` 对应冻结采样权重与 log-ratio 的 score-function 更新；`naive_gradient` 对应只微分采到动作的 log-prob。代码检查的是一个固定前缀，不是完整 trainer。

```python
import math

teacher = [0.8, 0.2]
student = [0.5, 0.5]
log_ratio = [math.log(pred / target) for pred, target in zip(student, teacher)]
score_gradient = [
    sum(student[action] * log_ratio[action] * ((action == index) - student[index])
        for action in range(2))
    for index in range(2)
]
naive_gradient = [
    sum(student[action] * ((action == index) - student[index]) for action in range(2))
    for index in range(2)
]

def reverse_kl(logits):
    weights = [math.exp(value - max(logits)) for value in logits]
    probs = [value / sum(weights) for value in weights]
    return sum(prob * math.log(prob / target) for prob, target in zip(probs, teacher))

epsilon = 1e-5
finite_difference = (reverse_kl([epsilon, 0]) - reverse_kl([-epsilon, 0])) / (2 * epsilon)
assert abs(finite_difference - score_gradient[0]) < 1e-9
assert naive_gradient == [0.0, 0.0]
assert abs(score_gradient[0] + math.log(2) / 2) < 1e-12
```

</details>

也别把 forward KL 理解成“完全不管坏 token”。若老师是 `[1, 0]`，学生是 `[0.5, 0.5]`，soft CE 的 logit 梯度仍是 `[-0.5, +0.5]`：第二项会被压低，因为 softmax 各项共享归一化。

“Reverse KL 选中一个模式就够了”也不是无条件结论。老师为 `[0.5, 0.5]`，学生只保留第一项 `[1, 0]` 时，reverse KL 是 $\log2$，并非 0。没有模型容量等限制时，两种 KL 都在 $q=p$ 处达到最小值。模式偏好要结合可表达的分布与优化过程来谈。

### 从一个前缀走到整段回答 {#opd-sequence}

上面的推导故意固定前缀。整段生成还多了一层：前面的动作会影响后面能走到哪里。

- **固定采样前缀后做全词表 KL**：在每个已访问位置拟合分布，不对前缀采样求导。这是明确的训练约定，不是假装算了完整轨迹梯度。
- **优化完整序列的 reverse KL**：在共同 token / 结束空间上，序列 log-ratio 是逐步 log-ratio 的和；精确的策略梯度还要考虑动作对未来的影响，可写成 return-to-go 的形式。
- **把当前位置 log-prob 差直接当 advantage，再用 PPO clipping**：这是局部 surrogate。它可能实用，但不能只凭长得像第二行公式，就说与完整序列目标的梯度严格相同。

实际实现还要记录 behavior / current student 版本，确认 prompt、工具返回和 padding 不被误当成采样动作。只拿到 sampled-token log-prob 与拿到全词表 logits，也不是同一份信息。数据怎样进入训练，接着看 [rollout 与训练记录](post-training-infrastructure.md#rollout-record)。

### 多花的 teacher 成本值不值？ {#opd-budget}

先用固定 teacher 数据做基线，再比较学生采样。固定数据容易缓存；on-policy 会不断产生新前缀，缓存命中率和 teacher 调用成本需要实测。批量 teacher forcing 可以一次评分整段前缀，不意味着必须每生成一个 token 就发一次远程请求。

做对照时一起报学生训练量、采样 tokens、teacher 评分 tokens、总 GPU-hours / wall time，以及独立任务指标。相同 optimizer steps 不代表相同预算。若 teacher 的错误、输入缺失或 tokenizer 错配没有解决，多采样往往只会多收集一些不可靠监督。

## 怎样评估蒸馏后的学生？ {#_4}

| 对照 | 要排除的解释 |
| --- | --- |
| 原始学生 vs 等预算普通 SFT vs 蒸馏 | 提升是否只是多用了数据或训练步骤 |
| 与老师独立的人工标签或任务验证器 | 是否只是在复现老师的偏好与错误 |
| 熟悉任务、未见任务、证据不足的任务分别看 | 是否只会模仿固定题型、是否会编造 |
| 报质量、延迟、内存和教师数据成本 | 便宜的学生是否真的抵得过蒸馏成本 |

训练题目不重合还不够：若仍用同一个老师生成标签并判定输赢，老师的系统性偏差仍可能贯穿训练和评估。需要独立的验证依据。

学生也不一定永远差于老师：额外数据、不同训练目标或集成信号都可能改变结果。但“超过老师”同样需要独立证据，不能拿训练 KL 很小当作任务成功。

想继续看如何检查教师评分，读 [LLM-as-a-Judge](../07-evaluation/llm-as-a-judge/README.md)；想减少更新参数，读 [LoRA / QLoRA](lora-and-qlora.md)。
