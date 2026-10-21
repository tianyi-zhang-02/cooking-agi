# 解码策略：temperature、top-k、top-p

**中文** · [English](decoding.en.md)

> 阅读时间：约 12 分钟 · 最近审阅：2026-10

下一步有三个候选，概率是 `0.6、0.3、0.1`。Greedy 总选第一个；top-k 取 k=2 时留下前两个；top-p 取 p=0.8 时，也要留下前两个才够 0.8 的概率质量。留下候选以后，还要重新归一化再采样。模型没有换，输出的选择方式已经不同了。**模型只给分布；同一个分布换一种截断，候选池就不同。**

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
| $0 < T < 1$ | 分布变陡，高概率 token 更占优势 |
| $T = 1$ | 模型原本的分布 |
| $T > 1$ | 分布变平，长尾 token 有机会，输出更发散 |

正温度不改变 token 的**排序**，只改变概率有多集中。低温不会让模型掌握新知识，也不能保证答案更正确。

> 公式不能直接代入 $T=0$。有些接口把它当作 greedy 的快捷设置，另一些用单独的开关。例如 [Transformers 5.20.0](https://huggingface.co/docs/transformers/v5.20.0/en/main_classes/text_generation)（2026-10 核对）中，`do_sample=False, num_beams=1` 才是明确的 greedy 配置。

## Top-k：只留概率最高的 k 个 {#top-k-k}

把分布排序，保留前 $k$ 个，其余置零后重新归一化，再采样。

同样留下两个 token，保住的概率质量可能差很多。看两个假设分布：

| 下一 token 的概率 | top-k，k=2 | top-p，p=0.9 |
| --- | --- | --- |
| `0.98, 0.01, 0.01` | 留 2 个，原概率之和为 0.99 | 留 1 个，原概率之和为 0.98 |
| `0.34, 0.33, 0.33` | 留 2 个，原概率之和为 0.67 | 留 3 个，原概率之和为 1.00 |

**Top-k 固定数量，不固定留下的概率质量。** 它可以很好用，但同一个 k 在不同位置的截断力度不同。这也是下面 top-p 换一种标准的原因。表里并列的 token 按 ID 顺序打破平局。

## Top-p（核采样）：按累积概率截断 {#top-p}

改成：按概率从高到低累加，取**刚好让累积概率 ≥ p 的最小集合**，其余丢弃，剩余概率再归一化。

$$\text{保留最小的集合 } V^{(p)} \text{ 使得} \sum_{i \in V^{(p)}} p_i \ge p$$

同样是 $p = 0.9$：

- 分布陡的时候，可能第一个 token 就有 0.93，候选池只有 **1 个**；
- 分布平的时候，可能要累加到第 80 个才够 0.9，候选池就是 **80 个**。

**候选池大小随分布变化**，不必事先指定留下几个 token。但 0.9 是模型分配的概率质量，不是“答案有九成把握正确”。Top-p 是否更合适，仍要看任务和实际输出。

两者也可以叠加。例如先 top-k，再对剩余候选重新归一化后做 top-p；这时 top-p 的门槛针对剩余分布，不是原分布。是否同时启用、谁先作用，要看具体配置。

## 常见的处理顺序 {#_2}

处理顺序要核对所用框架，不能当作通用规定。下面明确选择这条顺序来比较：

```text
Temperature → TopK → TopP
```

温度在**最前面**。这不是无关紧要的细节，因为它会改变 top-p 截出来的核有多大。

对 top-k 的候选集合没影响——正温度不改变排序，而 top-k 只看排名（并列时保持相同的规则）。但 **top-p 看的是累积概率**，温度一改，累积速度就变了。

用概率 `0.40、0.25、0.15、0.08、0.05、0.03、0.025、0.015` 作例子，转成 logits，再设 `top_p=0.9`：

| temperature | 温度在前 → 核大小 | 截断在前 → 核大小 |
| --- | --- | --- |
| 0.7 | **4** | 5 |
| 1.0 | 5 | 5 |
| 1.5 | **6** | 5 |

顺序颠倒之后，同样的参数给出的候选池不一样。$T=1$ 时温度不改变分布，两者必然相等，所以只在默认温度下测是发现不了这个差别的。

比较配置时，把 temperature 和 top-p 一起记录。固定顺序后再调参数，才能分清变化来自分布变平，还是候选被截掉。

## 频率惩罚与存在惩罚 {#_3}

另一类做法按生成历史**压低已经出现过的 token 的 logits**，也会改变最终分布：

$$z_i \leftarrow z_i - \alpha_{\text{presence}}\cdot\mathbb{1}[c_i > 0] - \alpha_{\text{frequency}}\cdot c_i$$

其中 $c_i$ 是 token $i$ 在已生成文本里出现的次数。这里先设两个系数都非负；负系数会反过来鼓励出现过的 token。两者的区别就在这个式子里：

- **存在惩罚**（presence）：只要出现过就扣一个**固定值**，出现 1 次和 10 次扣得一样多，让没出现过的 token 相对更有优势。
- **频率惩罚**（frequency）：按出现**次数累计**扣分，出现越多，logit 降得越多，用来抑制重复。

这两种惩罚只识别 token 是否出现，不理解“话题”。压低旧 token 不保证出现新观点；换个词反复表达同一件事，也可能绕过去。

⚠️ 这两种规则都无法区分无意义重复和**必要的重复**。标点、代词、代码里的缩进和括号，以及文中反复提到的人名，本来就可能高频出现。惩罚过大时，模型会回避这些必要的 token，导致表达别扭甚至语法错误，代码生成尤其容易受影响。

可以先保留不加惩罚的基线，再用相同 prompts 对照输出。参数范围和计数是否包含 prompt 要核对框架，不能给所有模型套一个固定步长或上限。

## Beam Search：高概率不一定是好回答 {#beam-search-llm}

Beam search 每一步扩展候选，再按分数保留 $k$ 条。最简单的分数是**累积对数概率**，实际还可能加入长度惩罚。最后选的是保留下来的候选里分数最高的那条，不保证全局最优。

一个两步例子：第一步 A 的概率是 0.6，B 是 0.4；A 后最好的接续概率是 0.5，B 后是 0.9。Greedy 先选 A，得到 0.30；宽度为 2 的 beam 能留下 B，找到 0.36。保留多个前缀有用，但已被剪掉的前缀通常找不回来。

翻译、语音识别等任务常用 beam search，并不是因为答案唯一。关键是模型分数、长度处理与任务质量是否一致。开放生成中，持续追求高似然可能带来重复，见 [Neural Text Degeneration](https://arxiv.org/abs/1904.09751)。

数学题也不只能 greedy。[Self-consistency](https://arxiv.org/abs/2203.11171) 会采样多条推理路径再汇总答案；预算怎么分、用什么验证，见[并行采样与验证](../../11-frontier/test-time-compute/parallel-and-verify.md)。选择策略时，要一起考虑候选预算、验证方式和最终指标。

## 实际参数怎样设置 {#_4}

| 场景 | temperature | top-p | 说明 |
| --- | --- | --- | --- |
| 单次抽取、分类 | greedy 基线 | — | 先测准确率与格式；需要时加入约束解码 |
| 代码、数学，多候选验证 | 对比低温与采样 | 按实验设置 | 同时报候选数与验证成本；pass@k 不是 pass@1 |
| 对话、写作 | 扫小范围 | 与温度一起调 | 看质量和重复，不只看多样性 |
| RL rollout | 由目标决定 | 由目标决定 | 对齐采样分布、log-prob 与训练目标 |
| 自洽性投票 | 使用多样采样 | 可以截断 | 推理聚合不直接套用策略梯度的要求 |

几条容易踩的：

**低温和低 top-p 叠加，可能只剩一个候选。** 想要多样性时要检查实际候选数；如果本来就想要确定输出，这不一定是问题。但不要把“开启了采样”当作“结果一定多样”。

**RL 训练要区分目标策略和实际采样分布。** 温度和截断会改变分布，需要明确 log-prob 对应哪个分布、是否校正。截断还会让部分动作概率归零，不能只补一个重要性比就声称恢复了完整策略的无偏估计。

**要复现就固定随机种子并记下所有解码参数。** 只记模型版本不够——同一份权重配 `T=0.7` 和 `T=1.0` 是两种行为。

<details markdown="1">
<summary><b>进阶</b>：还有哪些截断方式</summary>

**min-p**：保留概率 ≥ `min_p × 最高概率` 的 token。它按峰值设相对门槛，不是累计概率门槛；是否改善高温输出，需要在目标任务上比较。

**repetition penalty 与 no-repeat n-gram 不是同一个操作**。前者调整已出现 token 的分数；后者禁止会组成重复 n-gram 的下一 token。例如历史是 `A B A`，禁止重复 bigram 时，下一个 `B` 会被屏蔽，但不是所有旧 token 都被禁止。实现可对照 [Transformers 5.20.0 的生成处理器](https://huggingface.co/docs/transformers/v5.20.0/en/internal/generation_utils#transformers.NoRepeatNGramLogitsProcessor)。

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
    if temperature >= 1:
        scaled_logits = [value / temperature for value in logits]
    else:
        maximum = max(logits)
        scaled_logits = [(value - maximum) / temperature for value in logits]
    maximum = max(scaled_logits)
    weights = [math.exp(value - maximum) for value in scaled_logits]
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

缩放顺序也有数值边界：`[1e308, -1e308]` 配 `T=1e308`，正确分布约为 `[0.881, 0.119]`；若先做两个 logits 的减法，差值已经溢出，后面再除温度也救不回来。代码在高温时先缩放；低温时先减最大值，避免共同的大偏置除以小温度后溢出。

## 面试常见问题 {#_5}

<details class="interview" markdown="1">
<summary>temperature 是作用在 logits 上还是概率上？为什么？</summary>

logits 上，在 softmax **之前**：$p_i = \text{softmax}(z_i/T)$。

若已有原始 softmax 概率，$p_i^{1/T}/\sum_j p_j^{1/T}$ 与 $\operatorname{softmax}(z/T)$ **数学上等价**，因为公共归一化因子会抵消。实现时通常保留 logits 来避免概率下溢。给所有概率乘同一个系数再归一化，则不会改变分布。

</details>

<details class="interview" markdown="1">
<summary>top-k 和 top-p 的区别？候选数怎样变化？</summary>

top-k 固定保留 $k$ 个候选；top-p 保留累积概率刚好达到 $p$ 的最小集合。

区别在于**候选池是否随分布形状变化**。分布陡时，top-p 可以只留少数候选；分布平时则可能留很多。这个适应性不等于质量保证：模型也可能很自信地给错 token 高分。

</details>

<details class="interview" markdown="1">
<summary>为什么 beam search 的分数更高，回答却未必更好？</summary>

模型似然不等于任务质量，开放生成中尤其要检查重复与多样性。Beam search 是有限宽度的近似搜索，不是全局最优算法；翻译和摘要也有多个合理答案。应与采样、重排或验证方案在同一预算下比较。

</details>

<details class="interview" markdown="1">
<summary>temperature=0 和 greedy 是一回事吗？</summary>

最大 logit 唯一时，$T \to 0^+$ 的概率集中到该 token；并列最大时，极限仍会在它们之间分配概率，而 greedy 通常用固定规则打破平局。

具体配置要看接口：有的用 `temperature=0` 表示 greedy，有的要求关闭采样，不能在公式里直接除零。Greedy 对固定 logits 和固定平局规则是确定的；不同 kernel 或批处理方式若改变了 logits，接近的候选仍可能换序。

</details>

<details class="interview" markdown="1">
<summary>temperature、top-k、top-p 谁先作用？顺序会影响结果吗？</summary>

本文选择 temperature → top-k → top-p；具体框架的顺序需要核对。不能把任意两个步骤互换后仍当作同一配置。

**正温度不改变 top-k 的候选集合**：它不改变排序；并列时仍需使用相同的取舍规则。

**对 top-p 有影响**：top-p 看的是累积概率，温度改变了各 token 的相对占比，累积到 $p$ 所需的 token 数就变了。实测同样 `top_p=0.9`，$T=0.7$ 时温度在前得到 4 个候选、在后得到 5 个；$T=1.5$ 时是 6 对 5。$T=1$ 时两者相等——所以只在默认温度下测试是发现不了的。

</details>

<details class="interview" markdown="1">
<summary>频率惩罚和存在惩罚有什么区别？</summary>

都作用在 logits 上：$z_i \leftarrow z_i - \alpha_p\mathbb{1}[c_i>0] - \alpha_f c_i$。

系数为正时，**存在惩罚**只看「出现过没有」，出现 1 次和 10 次扣得一样多；**频率惩罚**按次数累计，出现越多扣得越多。它们调整的是 token 的相对概率，不保证减少语义上的重复，也不保证换话题。

两者都可能压低本该重复的 token 的概率，包括标点、代词、代码缩进和括号、正文里反复出现的人名。惩罚过大时，表达会变得别扭，也尤其容易造成代码错误。

</details>

<details class="interview" markdown="1">
<summary>RL 训练做 rollout 时，采样参数该怎么设？</summary>

先确认实际 rollout 分布 $\mu$。未变换的旧策略采样是容易核对的基线；其他采样配置也可以是有意设计，但要明确目标策略、记录的概率和校正方式。

普通 PPO 的分母来自收集这批数据的旧策略，不是当前更新中的策略。若数据来自 $\mu$，却直接当作未变换的旧策略样本，估计量就改变了。还要检查支持集：$\mu$ 为零的动作无法从这批样本中恢复（离线估计里的同一个问题见 [OPE 估计量](../../05-post-training/deep-rl/ope-estimators.md)）。

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
- [旧回答还能拿来训练吗](../../05-post-training/rlhf/on-off-policy.md)：temperature、top-p 怎样改变 rollout 的行为分布，log-prob 该记录哪一个

## 快速学习：模型概率与 decoding policy 是两层 {#decoding-policy}

<details class="interview" markdown="1">
<summary>Temperature、top-k、top-p 的主线与组合顺序</summary>

**快速记忆**：模型先输出 logits。Temperature 调整概率分布有多集中，top-k / top-p 决定哪些候选可以保留，最后才按这个分布随机采样。

**面试回答**

> Decoding 不改变模型参数，只把同一组 logits 转成不同的选择策略。先明确处理顺序，例如 penalties → 正温度缩放 → top-k / top-p → 归一化 → 采样。Temperature 趋近 0 时，概率集中在最大 logits 上；若有并列，仍不同于固定规则选一个的 greedy。

<details markdown="1">
<summary><b>深挖</b>：为什么 top-p 比固定 top-k 更能适应上下文？</summary>

当分布很集中时，少数 token 已覆盖概率质量；当分布平坦时，需要更多候选。固定 $k$ 不会这样变化。具体采样开销还取决于排序、截断 kernel 和 batch，不能仅凭候选数变化就推断整套服务吞吐变差。

</details>
</details>
