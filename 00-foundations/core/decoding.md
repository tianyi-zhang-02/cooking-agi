# 解码策略：temperature、top-k、top-p

**中文** · [English](decoding.en.md)

> 阅读时间：约 8 分钟 · 难度：必修 · 最近审阅：2026-08

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>模型给出一个概率分布，怎么从里面挑一个 token</strong></div>
  <div><span>前置知识</span><strong>softmax · 自回归生成</strong></div>
  <div><span>核心机制</span><strong>先改分布形状（temperature），再截断（top-k / top-p），最后采样</strong></div>
  <div><span>常见错误</span><strong>以为 temperature 作用在概率上；开放生成里用 beam search</strong></div>
</div>

## 模型输出概率分布，解码策略负责选择

每一步前向之后，模型输出的是词表上的一个概率分布，比如：

```text
" the"   0.42
" a"     0.18
" my"    0.09
" one"   0.05
...      （剩下 5 万个 token 分掉剩下的 0.26）
```

**模型到这里就结束了。** 选哪个 token 是解码策略的事，和模型权重无关。同一个模型配不同的解码参数，行为可以差得像两个模型——这也是为什么复现别人的结果时，解码参数和权重一样重要。

## Temperature：调整分布的陡峭程度

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

**关键：$T$ 除的是 logits，在 softmax 之前，不是事后去缩放概率。** 这两者不等价——事后缩放再归一化只会得到原分布。

| $T$ | 效果 |
| --- | --- |
| $T \to 0$ | 分布趋近 one-hot，等价于 greedy |
| $T < 1$ | 分布变陡，高概率 token 更占优势，输出更保守 |
| $T = 1$ | 模型原本的分布 |
| $T > 1$ | 分布变平，长尾 token 有机会，输出更发散 |

直观地说，$T$ 控制的是「模型对自己有多确信」这件事的放大倍数。它不改变**排序**，只改变差距。

> 实现上 $T=0$ 会除零，所以大多数 API 里的 `temperature=0` 其实是走 greedy 分支，不是真的做除法。

## Top-k：只留概率最高的 k 个

把分布排序，保留前 $k$ 个，其余置零后重新归一化，再采样。

问题在于 $k$ 是固定的，而分布的形状一直在变：

- 「中华人民共和国万」后面几乎只有一个合理选择，分布极陡。此时 $k=50$ 会把 49 个几乎不可能的词也放进候选池。
- 「他觉得这部电影」后面有几百个合理的接续，分布很平。此时 $k=50$ 又切得太狠，砍掉了本来合理的选项。

**同一个 $k$ 不可能同时适配这两种情况。**

## Top-p（核采样）：按累积概率截断

改成：按概率从高到低累加，取**刚好让累积概率 ≥ p 的最小集合**，其余丢弃后归一化。

$$\text{保留最小的集合 } V^{(p)} \text{ 使得} \sum_{i \in V^{(p)}} p_i \ge p$$

同样是 $p = 0.9$：

- 分布陡的时候，可能第一个 token 就有 0.93，候选池只有 **1 个**；
- 分布平的时候，可能要累加到第 80 个才够 0.9，候选池就是 **80 个**。

**候选池大小随分布形状自动伸缩**，这正是 top-p 比 top-k 好用的原因。它调的不是「留几个」，而是「留多少把握」。

两者可以叠加：先 top-k 兜住上限，再 top-p 动态收紧。很多实现默认就是这么串的。

## 常见的处理顺序

网上不少材料写「先 top-k、再 top-p、最后 temperature」。**实际实现是反过来的。**

`transformers 4.44.1` 里 warper 的实际顺序是：

```text
Temperature → TopK → TopP → MinP
```

温度在**最前面**。这不是无关紧要的细节，因为它会改变 top-p 截出来的核有多大。

对 top-k 无所谓——温度是单调变换，不改变排序，而 top-k 只看排名。但 **top-p 看的是累积概率**，温度一改，累积速度就变了。

拿一个具体分布量一下（8 个 token，`top_p=0.9`）：

| temperature | 温度在前 → 核大小 | 截断在前 → 核大小 |
| --- | --- | --- |
| 0.7 | **4** | 5 |
| 1.0 | 5 | 5 |
| 1.5 | **6** | 5 |

顺序颠倒之后，同样的参数给出的候选池不一样。$T=1$ 时两者恰好相等，所以只在默认温度下测是发现不了这个差别的。

**结论：调参时把 temperature 和 top-p 当成一对来调**，因为温度会实际改变 top-p 的行为；反过来不会。

## 频率惩罚与存在惩罚

另一类做法不动分布形状，而是**直接压低已经出现过的 token 的 logits**：

$$z_i \leftarrow z_i - \alpha_{\text{presence}}\cdot\mathbb{1}[c_i > 0] - \alpha_{\text{frequency}}\cdot c_i$$

其中 $c_i$ 是 token $i$ 在已生成文本里出现的次数。两者的区别就在这个式子里：

- **存在惩罚**（presence）：只要出现过就扣一个**固定值**，出现 1 次和 10 次扣得一样多。作用是推动模型换新话题。
- **频率惩罚**（frequency）：按出现**次数累计**扣，出现越多压得越狠。作用是抑制重复。

「想让内容涉及更多话题」用存在惩罚，「想让同一个词别反复出现」用频率惩罚。

⚠️ 它们都很粗暴，因为**有些重复是应该的**。标点、代词、代码里的缩进和括号、以及文中反复提到的人名，本来就该高频出现。惩罚开大了，模型会开始回避这些必要的 token，输出会变得别扭甚至语法错误。代码生成尤其忌讳。

一般起点：两个都设 0，出现明显重复再往上加，一次加 0.1–0.3，很少需要超过 1.0。

## Beam Search：为什么生成式 LLM 很少使用它

Beam search 同时维护 $k$ 条候选序列，每步扩展后按**累积对数概率**保留最好的 $k$ 条，最后输出整体概率最高的那条。

它在机器翻译、摘要这类任务上是标准做法——因为那里**存在一个正确答案**，你要的就是概率最高的序列。

但开放式生成里它会失败，而且失败得很典型：**输出又干又重复**。

原因是「最可能的句子」和「好句子」根本不是一回事。人说话本身就带着不确定性，一个句句都挑最大概率的序列，读起来像一份平淡的模板。这现象叫 likelihood trap——有研究发现人写的文本，其 token 概率是**上下起伏**的，而 beam search 的输出概率曲线平滑地贴在高位。

所以：**有唯一正解的任务用 beam search，开放生成用采样。**

## 实际参数怎样设置

| 场景 | temperature | top-p | 说明 |
| --- | --- | --- | --- |
| 代码、数学、结构化抽取 | 0 | — | 要确定性和可复现；采样只会引入错误 |
| 事实问答、分类 | 0–0.3 | 1.0 | 略留一点余地，但不鼓励发散 |
| 一般对话 | 0.7 | 0.9 | 最常见的默认组合 |
| 创意写作、头脑风暴 | 0.9–1.1 | 0.95 | 明确要多样性 |
| RL rollout / 自洽性投票 | 1.0 | 1.0 | **必须**保留完整分布，否则采样偏了 |

几条容易踩的：

**不要同时把 temperature 和 top-p 都调很低。** 两个都在收紧，叠起来常常直接退化成 greedy，多样性完全消失，但你以为自己在采样。

**RL 训练时的采样参数会影响梯度。** rollout 用的分布如果被 top-p 截断过，它就不是策略 $\pi_\theta$ 本身了，重要性比算出来是错的。这是 RLHF 实现里一个安静的 bug 来源。

**要复现就固定随机种子并记下所有解码参数。** 只记模型版本不够——同一份权重配 `T=0.7` 和 `T=1.0` 是两种行为。

<details markdown="1">
<summary><b>进阶</b>：还有哪些截断方式</summary>

**min-p**：保留概率 ≥ `min_p × 最高概率` 的所有 token。相比 top-p 更直接地跟随分布的峰值高度，在高 temperature 下更稳。

**repetition penalty / no-repeat n-gram**：直接压低已出现过的 token 的 logits。有效但粗暴——它会连带惩罚那些本来就该重复的词（代词、标点、代码里的缩进）。

**typical sampling**：按 token 的信息量偏离条件熵的程度筛选，理由是人类语言倾向于保持信息密度稳定。

这些都是在同一个位置动手：**改 logits 或改候选集，然后采样**。理解了 temperature 和 top-p，其余的都是同一类操作的变体。

</details>

## 面试常见问题

<details class="interview" markdown="1">
<summary>temperature 是作用在 logits 上还是概率上？为什么？</summary>

logits 上，在 softmax **之前**：$p_i = \text{softmax}(z_i/T)$。

不能事后缩放概率再归一化——那样等于 $p_i^{1/T}$ 归一化，虽然也能改变陡峭程度，但不是同一个变换，而且数值上更不稳。更常见的错误是直接给概率乘一个系数再归一化，那个操作**什么也不改变**，归一化会把系数消掉。

</details>

<details class="interview" markdown="1">
<summary>top-k 和 top-p 的区别？为什么 top-p 更常用？</summary>

top-k 固定保留 $k$ 个候选；top-p 保留累积概率刚好达到 $p$ 的最小集合。

区别在于**候选池是否随分布形状变化**。分布很陡时（下一个词几乎唯一），top-k 会强行塞进一堆不可能的候选；分布很平时（很多合理接续），top-k 又切得太狠。top-p 两种情况下都自动给出合适的池子大小。

</details>

<details class="interview" markdown="1">
<summary>为什么大模型开放生成不用 beam search？</summary>

因为「概率最高的序列」不等于「好的回答」。beam search 找的是全局最大似然，结果是干瘪重复的文本——人类语言的 token 概率本来就是起伏的，一路挑最大概率会得到一份没人这么说话的模板。

机器翻译、摘要这类**有唯一正解**的任务仍然用它，那里要的确实就是最可能的序列。

</details>

<details class="interview" markdown="1">
<summary>temperature=0 和 greedy 是一回事吗？</summary>

数学上是极限等价：$T \to 0$ 时 softmax 趋近 one-hot，采样必然取到 argmax。

但实现上 $T=0$ 会除零，所以框架和 API 里的 `temperature=0` 通常是直接走 greedy 分支。另外注意 greedy 也未必完全确定——批处理时的浮点归约顺序、不同 kernel 实现，都可能让同一输入产生不同结果。

</details>

<details class="interview" markdown="1">
<summary>temperature、top-k、top-p 谁先作用？顺序会影响结果吗？</summary>

实际实现里 **temperature 在最前**（`transformers` 的顺序是 Temperature → TopK → TopP → MinP）。不少材料写成「最后做 temperature」，是反的。

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

一般设成 `temperature=1.0, top_p=1.0`，也就是不截断。

因为策略梯度里的重要性比 $\rho_t = \pi_\theta(a_t|s_t)/\pi_{\theta_\text{old}}(a_t|s_t)$ 假设动作是从 $\pi_\theta$ 采出来的。一旦用 top-p 截断，实际采样分布就不是 $\pi_\theta$ 了，而 log-prob 仍按完整分布计算，比值算出来是错的。

这是个不会报错、只会让训练慢慢跑偏的 bug。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>temperature 作用在哪一步？直接给概率乘系数为什么没用？</li>
    <li>同样是 0.9，top-p 的候选池在陡峭分布和平坦分布上分别有多大？</li>
    <li>beam search 在什么任务上合适，什么任务上容易失败？为什么？</li>
    <li>为什么 RL rollout 不能开 top-p 截断？</li>
  </ol>
</div>

## 继续阅读

- [Decoder-only：自回归生成](decoder-only.md)：这些分布是怎么一步步产生的
- [RLHF 的三个阶段](../../05-post-training/rlhf-pipeline.md)：rollout 采样参数为什么会影响梯度
