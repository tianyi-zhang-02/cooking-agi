# 解码策略：temperature、top-k、top-p

**中文** · [English](decoding.en.md)

> 阅读时间：约 12 分钟 · 难度：必修 · 最近审阅：2026-10

下一步有三个候选，概率是 `0.6、0.3、0.1`。Greedy 总选第一个；top-k 取 k=2 时留下前两个；top-p 取 p=0.8 时，也要留下前两个才够 0.8 的概率质量。留下候选以后，还要重新归一化再采样。模型没有换，输出的选择方式已经不同了。

## 模型输出概率分布，解码策略负责选择 {#_1}

每一步前向之后，模型输出的是词表上的一个概率分布，比如：

```text
" the"   0.42
" a"     0.18
" my"    0.09
" one"   0.05
...      （剩下 5 万个 token 分掉剩下的 0.26）
```

权重固定时，解码器仍可以用不同规则从这个分布选 token。所以复现一次生成，除了模型版本，还要记录采样参数；只保存回答文本，通常无法解释行为为什么变了。

## Temperature：调整分布的陡峭程度 {#temperature}

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

**通常把 logits 除以 $T$，再做 softmax。** 给概率统一乘一个系数再归一化没有作用；给概率取 $1/T$ 次幂再归一化则与这个公式等价。

| $T$ | 效果 |
| --- | --- |
| $T \to 0^+$ | 若最大 logit 唯一，概率集中到该 token；并列最大时仍有多个候选 |
| $T < 1$ | 分布变陡，高概率 token 更占优势，输出更保守 |
| $T = 1$ | 模型原本的分布 |
| $T > 1$ | 分布变平，长尾 token 有机会，输出更发散 |

$T$ 不改变 token 的**排序**，只改变概率有多集中。低温不会让模型掌握新知识，也不能保证答案更正确。

> 实现上 $T=0$ 会除零，所以大多数 API 里的 `temperature=0` 其实是走 greedy 分支，不是真的做除法。

## Top-k：只留概率最高的 k 个 {#top-k-k}

把分布排序，保留前 $k$ 个，其余置零后重新归一化，再采样。

问题在于 $k$ 是固定的，而分布的形状一直在变：

- 「中华人民共和国万」后面几乎只有一个合理选择，分布极陡。此时 $k=50$ 会把 49 个几乎不可能的词也放进候选池。
- 「他觉得这部电影」后面有几百个合理的接续，分布很平。此时 $k=50$ 又切得太狠，砍掉了本来合理的选项。

**同一个 $k$ 不可能同时适配这两种情况。**

## Top-p（核采样）：按累积概率截断 {#top-p}

改成：按概率从高到低累加，取**刚好让累积概率 ≥ p 的最小集合**，其余丢弃后归一化。

$$\text{保留最小的集合 } V^{(p)} \text{ 使得} \sum_{i \in V^{(p)}} p_i \ge p$$

同样是 $p = 0.9$：

- 分布陡的时候，可能第一个 token 就有 0.93，候选池只有 **1 个**；
- 分布平的时候，可能要累加到第 80 个才够 0.9，候选池就是 **80 个**。

**候选池大小随分布变化**，不必事先指定留下几个 token。但 0.9 是模型分配的概率质量，不是“答案有九成把握正确”。Top-p 是否更合适，仍要看任务和实际输出。

两者可以叠加：先 top-k 兜住上限，再 top-p 动态收紧。很多实现默认就是这么串的。

## 常见的处理顺序 {#_2}

处理顺序要核对所用框架，不能当作通用规定。下面明确选择这条顺序来比较：

```text
Temperature → TopK → TopP
```

温度在**最前面**。这不是无关紧要的细节，因为它会改变 top-p 截出来的核有多大。

对 top-k 无所谓——温度是单调变换，不改变排序，而 top-k 只看排名。但 **top-p 看的是累积概率**，温度一改，累积速度就变了。

用概率 `0.40、0.25、0.15、0.08、0.05、0.03、0.025、0.015` 作例子，转成 logits，再设 `top_p=0.9`：

| temperature | 温度在前 → 核大小 | 截断在前 → 核大小 |
| --- | --- | --- |
| 0.7 | **4** | 5 |
| 1.0 | 5 | 5 |
| 1.5 | **6** | 5 |

顺序颠倒之后，同样的参数给出的候选池不一样。$T=1$ 时两者恰好相等，所以只在默认温度下测是发现不了这个差别的。

比较配置时，把 temperature 和 top-p 一起记录。固定顺序后再调参数，才能分清变化来自分布变平，还是候选被截掉。

## 频率惩罚与存在惩罚 {#_3}

另一类做法按生成历史**压低已经出现过的 token 的 logits**，也会改变最终分布：

$$z_i \leftarrow z_i - \alpha_{\text{presence}}\cdot\mathbb{1}[c_i > 0] - \alpha_{\text{frequency}}\cdot c_i$$

其中 $c_i$ 是 token $i$ 在已生成文本里出现的次数。两者的区别就在这个式子里：

- **存在惩罚**（presence）：只要出现过就扣一个**固定值**，出现 1 次和 10 次扣得一样多。作用是推动模型换新话题。
- **频率惩罚**（frequency）：按出现**次数累计**扣，出现越多压得越狠。作用是抑制重复。

这两种惩罚只识别 token 是否出现，不理解“话题”。压低旧 token 不保证出现新观点；换个词反复表达同一件事，也可能绕过去。

⚠️ 它们都很粗暴，因为**有些重复是应该的**。标点、代词、代码里的缩进和括号、以及文中反复提到的人名，本来就该高频出现。惩罚开大了，模型会开始回避这些必要的 token，输出会变得别扭甚至语法错误。代码生成尤其忌讳。

可以先保留不加惩罚的基线，再用相同 prompts 对照输出。参数范围和计数是否包含 prompt 要核对框架，不能给所有模型套一个固定步长或上限。

## Beam Search：为什么生成式 LLM 很少使用它 {#beam-search-llm}

Beam search 每一步扩展候选，再按分数保留 $k$ 条。最简单的分数是**累积对数概率**，实际还可能加入长度惩罚。最后选的是保留下来的候选里分数最高的那条，不保证全局最优。

一个两步例子：第一步 A 的概率是 0.6，B 是 0.4；A 后最好的接续概率是 0.5，B 后是 0.9。Greedy 先选 A，得到 0.30；宽度为 2 的 beam 能留下 B，找到 0.36。保留多个前缀有用，但已被剪掉的前缀通常找不回来。

翻译、语音识别等任务常用 beam search，并不是因为答案唯一。关键是模型分数、长度处理与任务质量是否一致。开放生成中，持续追求高似然可能带来重复，见 [Neural Text Degeneration](https://arxiv.org/abs/1904.09751)。

数学题也不只能 greedy。[Self-consistency](https://arxiv.org/abs/2203.11171) 会采样多条推理路径再汇总答案。选择策略时，要一起考虑候选预算、验证方式和最终指标。

## 实际参数怎样设置 {#_4}

| 场景 | temperature | top-p | 说明 |
| --- | --- | --- | --- |
| 单次抽取、分类 | greedy 基线 | — | 先测准确率与格式；需要时加入约束解码 |
| 代码、数学，多候选验证 | 对比低温与采样 | 按实验设置 | 同时报候选数与验证成本；pass@k 不是 pass@1 |
| 对话、写作 | 扫小范围 | 与温度一起调 | 看质量和重复，不只看多样性 |
| RL rollout | 由目标决定 | 由目标决定 | 对齐采样分布、log-prob 与训练目标 |
| 自洽性投票 | 使用多样采样 | 可以截断 | 推理聚合不直接套用策略梯度的要求 |

几条容易踩的：

**不要同时把 temperature 和 top-p 都调很低。** 两个都在收紧，叠起来常常直接退化成 greedy，多样性完全消失，但你以为自己在采样。

**RL 训练要区分目标策略和实际采样分布。** 温度和截断会改变分布，需要明确 log-prob 对应哪个分布、是否校正。截断还会让部分动作概率归零，不能只补一个重要性比就声称恢复了完整策略的无偏估计。

**要复现就固定随机种子并记下所有解码参数。** 只记模型版本不够——同一份权重配 `T=0.7` 和 `T=1.0` 是两种行为。

<details markdown="1">
<summary><b>进阶</b>：还有哪些截断方式</summary>

**min-p**：保留概率 ≥ `min_p × 最高概率` 的 token。它按峰值设相对门槛，不是累计概率门槛；是否改善高温输出，需要在目标任务上比较。

**repetition penalty 与 no-repeat n-gram 不是同一个操作**。前者调整已出现 token 的分数；后者禁止会组成重复 n-gram 的下一 token。例如历史是 `A B A`，禁止重复 bigram 时，下一个 `B` 会被屏蔽，但不是所有旧 token 都被禁止。实现可对照 [Transformers 4.57.1 的生成处理器](https://huggingface.co/docs/transformers/v4.57.1/en/internal/generation_utils#transformers.NoRepeatNGramLogitsProcessor)。

**typical sampling**：比较 token 的惊异度 $-\log p_i$ 与条件熵，而不是单按概率排名。它试图保留信息量接近当前期望的 token；依据和实验范围见 [Locally Typical Sampling](https://arxiv.org/abs/2202.00666)。

这些都是在同一个位置动手：**改 logits 或改候选集，然后采样**。理解了 temperature 和 top-p，其余的都是同一类操作的变体。

</details>

## 写一个能检查边界的 top-p {#top-p-example}

假设词表 ID 为 `0、1、2`，概率依次为 `0.1、0.6、0.3`，阈值为 0.8。按概率排序后先拿 ID 1，再拿 ID 2，累积质量从 0.6 到 0.9；**跨过阈值的那个 token 也要保留**。归一化后，两者的概率是 2/3 和 1/3。

下面只计算最终分布，不随机抽样，方便直接核对结果。它是有限 logits 的教学实现，不是 GPU 算子。

```python
import math

def nucleus_distribution(logits, threshold=0.9, temperature=1.0):
    if not logits or not 0 < threshold <= 1 or temperature <= 0:
        raise ValueError("Expected logits, 0 < threshold <= 1, and temperature > 0")
    if not math.isfinite(temperature) or not all(math.isfinite(value) for value in logits):
        raise ValueError("Expected finite inputs")
    maximum = max(logits)
    weights = [math.exp((value - maximum) / temperature) for value in logits]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    ordered_ids = sorted(range(len(logits)), key=lambda token_id: probabilities[token_id], reverse=True)
    selected_ids = []
    cumulative = 0.0
    for token_id in ordered_ids:
        selected_ids.append(token_id)
        cumulative += probabilities[token_id]
        if threshold < 1 and cumulative >= threshold:
            break
    return {token_id: probabilities[token_id] / cumulative for token_id in selected_ids}

example = nucleus_distribution([math.log(.1), math.log(.6), math.log(.3)], .8)
assert list(example) == [1, 2]
assert math.isclose(example[1], 2 / 3)
```

注意返回的是原词表 ID，不是排序后的名次。还可以试试 `threshold=1`（不截断）、只有一个候选、两个最大值相同，以及极大的正负 logits。不要只测最顺的一个例子。

## 面试常见问题 {#_5}

<details class="interview" markdown="1">
<summary>temperature 是作用在 logits 上还是概率上？为什么？</summary>

logits 上，在 softmax **之前**：$p_i = \text{softmax}(z_i/T)$。

若已有原始 softmax 概率，$p_i^{1/T}/\sum_j p_j^{1/T}$ 与 $\operatorname{softmax}(z/T)$ **数学上等价**，因为公共归一化因子会抵消。实现时通常保留 logits 来避免概率下溢。给所有概率乘同一个系数再归一化，则不会改变分布。

</details>

<details class="interview" markdown="1">
<summary>top-k 和 top-p 的区别？为什么 top-p 更常用？</summary>

top-k 固定保留 $k$ 个候选；top-p 保留累积概率刚好达到 $p$ 的最小集合。

区别在于**候选池是否随分布形状变化**。分布陡时，top-p 可以只留少数候选；分布平时则可能留很多。这个适应性不等于质量保证：模型也可能很自信地给错 token 高分。

</details>

<details class="interview" markdown="1">
<summary>为什么大模型开放生成不用 beam search？</summary>

模型似然不等于任务质量，开放生成中尤其要检查重复与多样性。Beam search 是有限宽度的近似搜索，不是全局最优算法；翻译和摘要也有多个合理答案。应与采样、重排或验证方案在同一预算下比较。

</details>

<details class="interview" markdown="1">
<summary>temperature=0 和 greedy 是一回事吗？</summary>

最大 logit 唯一时，$T \to 0^+$ 的概率集中到该 token；并列最大时，极限仍会在它们之间分配概率，而 greedy 通常用固定规则打破平局。

但实现上 $T=0$ 会除零，所以框架和 API 里的 `temperature=0` 通常是直接走 greedy 分支。另外注意 greedy 也未必完全确定——批处理时的浮点归约顺序、不同 kernel 实现，都可能让同一输入产生不同结果。

</details>

<details class="interview" markdown="1">
<summary>temperature、top-k、top-p 谁先作用？顺序会影响结果吗？</summary>

本文选择 temperature → top-k → top-p；具体框架的顺序需要核对。不能把任意两个步骤互换后仍当作同一配置。

**顺序对 top-k 没影响**：温度是单调变换，不改变排序，top-k 只看排名。

**对 top-p 有影响**：top-p 看的是累积概率，温度改变了各 token 的相对占比，累积到 $p$ 所需的 token 数就变了。实测同样 `top_p=0.9`，$T=0.7$ 时温度在前得到 4 个候选、在后得到 5 个；$T=1.5$ 时是 6 对 5。$T=1$ 时两者相等——所以只在默认温度下测试是发现不了的。

</details>

<details class="interview" markdown="1">
<summary>频率惩罚和存在惩罚有什么区别？</summary>

都作用在 logits 上：$z_i \leftarrow z_i - \alpha_p\mathbb{1}[c_i>0] - \alpha_f c_i$。

**存在惩罚**只看「出现过没有」，出现 1 次和 10 次扣的一样多，推动模型换话题。**频率惩罚**按次数累计，出现越多压得越狠，专治重复。

两者都会误伤本该重复的 token——标点、代词、代码缩进和括号、正文里反复出现的人名。调大了输出会变别扭，代码生成尤其容易被搞坏。

</details>

<details class="interview" markdown="1">
<summary>RL 训练做 rollout 时，采样参数该怎么设？</summary>

先确认实际 rollout 分布 $\mu$。未变换的旧策略采样是容易核对的基线；其他采样配置也可以是有意设计，但要明确目标策略、记录的概率和校正方式。

普通 PPO 的分母来自收集这批数据的旧策略，不是当前更新中的策略。若数据来自 $\mu$，却直接当作未变换的旧策略样本，估计量就改变了。还要检查支持集：$\mu$ 为零的动作无法从这批样本中恢复。

</details>

## 自检 {#_6}

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>temperature 作用在哪一步？直接给概率乘系数为什么没用？</li>
    <li>同样是 0.9，top-p 的候选池在陡峭分布和平坦分布上分别有多大？</li>
    <li>beam search 在什么任务上合适，什么任务上容易失败？为什么？</li>
    <li>Rollout 分布经过温度或截断后，log-prob 和训练目标需要检查什么？</li>
  </ol>
</div>

## 继续阅读 {#_7}

- [Decoder-only：自回归生成](decoder-only.md)：这些分布是怎么一步步产生的
- [RLHF 的三个阶段](../../05-post-training/rlhf/)：rollout 采样参数为什么会影响梯度

## 快速学习：模型概率与 decoding policy 是两层 {#decoding-policy}

<details class="interview" markdown="1">
<summary>Temperature、top-k、top-p 的主线与组合顺序</summary>

**快速记忆**：模型先输出 logits。Temperature 调整概率分布有多集中，top-k / top-p 决定哪些候选可以保留，最后才按这个分布随机采样。

**面试回答**

> Decoding 不改变模型参数，只把同一组 logits 转成不同的选择策略。通常先应用 penalties，再除以 temperature，随后做 top-k 或 top-p filtering，重新归一化后采样。Temperature 趋近 0 接近 argmax，但它本身不等于 greedy decoding。

<details markdown="1">
<summary><b>深挖</b>：为什么 top-p 比固定 top-k 更能适应上下文？</summary>

当分布很集中时，少数 token 已覆盖概率质量；当分布平坦时，需要更多候选。固定 $k$ 不会这样变化。具体采样开销还取决于排序、截断 kernel 和 batch，不能仅凭候选数变化就推断整套服务吞吐变差。

</details>
</details>
