# MTP：多预测几个 token，究竟多学了什么？

**中文** · [English](multi-token-prediction.en.md)

想比较 MTP 与 DFlash，读完目标对齐后转到 [DFlash 的草拟、验证与成本](dflash.md)。那里也区分初版、DFlash 2 与新近的后缀修正思路，不把训练目标和推理算法混成一类。

> 最近审阅：2026-10 · 前置：[语言模型目标](language-model-objective.md)

普通语言模型在每个位置预测下一个 token。MTP（Multi-token Prediction）在训练时再加一些更远的预测任务，希望当前表示不只适合眼前一步。

它不是“模型从此不用自回归”，也不是只要加一个 head，生成速度就能翻倍。先把训练目标和推理算法分开。

## 同一段序列，标签怎样移动？

用 A、B、C、D 代表 4 个 token：

| 当前表示 | Next-token target | 再远一步的 target |
| --- | --- | --- |
| $h_A$ | B | C |
| $h_B$ | C | D |
| $h_C$ | D | 无 |
| $h_D$ | 无 | 无 |

后一列的有效位置更少。越过文档边界、padding 或序列末尾的标签不能硬补进去。MTP 看起来像多一个 loss，实际却多了一组 shift、mask 和分母约定。

下面仅构造**独立多步预测头**的标签对，假设 document IDs 对应连续的文档片段：

```python
def future_pairs(tokens, document_ids, horizon):
    if horizon < 1 or len(tokens) != len(document_ids):
        raise ValueError("Invalid horizon or document IDs")
    return [(tokens[position], tokens[position + horizon])
            for position in range(len(tokens) - horizon)
            if len(set(document_ids[position:position + horizon + 1])) == 1]

assert future_pairs(list("ABCD"), [0, 0, 0, 0], 1) == [
    ("A", "B"), ("B", "C"), ("C", "D")]
assert future_pairs(list("ABCD"), [0, 0, 0, 0], 2) == [
    ("A", "C"), ("B", "D")]
assert future_pairs(list("ABCD"), [0, 0, 1, 1], 2) == []
```

代码不包含模型前向；它先把最容易悄悄错掉的监督关系检查清楚。

## 独立 heads 与串行模块，不是同一种 MTP

[Gloeckle 等人的工作](https://arxiv.org/abs/2404.19737)讨论从共享表示预测多个未来 token 的多头训练。[DeepSeek-V3](https://arxiv.org/abs/2412.19437)使用串行附加模块，并保留逐步的因果条件关系。

可以用同一个位置区分：

```text
独立 heads：h_A ─→ B
               └→ C

串行示意： h_A ─→ B
           h_A + embedding(B) ─→ 附加模块 ─→ C
```

串行训练中使用真实 B，来预测后面的 C。这不等于偷看 C；但部署时 B 来自已经接受的或草拟的 token，不再保证等于训练数据中的真实 B。需要单独检查由此带来的分布差异。

附加模块读到 B，不能让**基础 next-token head**也提前读到 B，否则它预测 B 时就泄漏了答案。并行张量实现依然要遵守这一因果关系。

## 多个目标怎样一起训练？

一种便于解释的写法是：

$$
\mathcal L=\mathcal L_{\mathrm{NTP}}
+\frac{\lambda}{K}\sum_{k=1}^{K}\mathcal L_{\mathrm{aux},k}.
$$

每个 $\mathcal L_{\mathrm{aux},k}$ 只在该任务的有效位置上计算；独立 head 和串行模块的条件输入不同。这里采用各任务有效 token 均值，实际论文或实现可能采用不同分母，比较时要明确。

例如 NTP loss 为 2.0、两项辅助 loss 为 2.4 和 2.8，$\lambda=0.1$，则总 loss 为 $2.0+0.1(2.4+2.8)/2=2.26$。不能把这个总 loss 和只有 NTP 的 2.0 直接比较，说模型变差了。

共享 trunk 收到多个任务的梯度。它们可能鼓励更有用的表示，也可能互相冲突；“监督更密”不代表信息独立，也不保证任务效果更好。要调 $\lambda$，并测基础 head 与下游任务，而不只看辅助 loss。

## 训练用 MTP，推理可以不用

一种部署方式是去掉辅助模块，仍用基础模型逐 token 生成。此时可能留下训练获得的质量变化，但没有自动获得“每步多生成几个 token”的速度收益。

另一种是把附加模块用于 speculative decoding：先提议多个 tokens，再用目标模型验证。提议若经常被拒绝，草拟和验证可能反而增加成本。

严格保持目标采样分布需要正确的接受、拒绝和重采样算法，不能简单“把两个模型都同意的词留下”。[Speculative Decoding](https://arxiv.org/abs/2211.17192)讨论了这类验证机制。greedy 和随机采样的验证规则也应分别处理。

## 两个候选词，算一遍“接受”是什么意思

先只看同一前缀下的一个位置。目标模型分布为 $p$，草稿分布为 $q$。草稿抽到 token $x$ 后，用 $\min(1,p(x)/q(x))$ 的概率接受；拒绝时从归一化的 $[p-q]_+$ 重采样。这是[原算法](https://proceedings.mlr.press/v202/leviathan23a.html)的单步规则，不是比较两个 argmax 是否相同。

假设下一词只可能是“是”或“否”：

| Token | 目标 $p$ | 草稿 $q$ | 接受概率 | 直接接受的概率质量 |
| --- | --- | --- | --- | --- |
| 是 | 0.6 | 0.8 | 0.6 / 0.8 = 0.75 | 0.8 × 0.75 = 0.6 |
| 否 | 0.4 | 0.2 | 1 | 0.2 |

总接受概率是 0.8，剩下 0.2 被拒绝。目标里“否”还差 0.2，所以这次 residual distribution 全落在“否”。最终“是”为 0.6，“否”为 $0.2+0.2=0.4$，恰好回到目标分布。

只留下接受的样本再重新归一化，会得到 0.75 / 0.25，已经不是 0.6 / 0.4。这就是为什么拒绝后的处理不是小细节。

```python
import math

def speculative_one_step_distribution(target, draft):
    if not target or len(target) != len(draft):
        raise ValueError("Expected matching distributions")
    for distribution in (target, draft):
        if any(not math.isfinite(probability) or probability < 0 for probability in distribution):
            raise ValueError("Invalid probability")
        if not math.isclose(sum(distribution), 1.0, abs_tol=1e-12):
            raise ValueError("Probabilities must sum to one")
    accepted = [min(target_prob, draft_prob)
                for target_prob, draft_prob in zip(target, draft)]
    residual = [max(target_prob - draft_prob, 0.0)
                for target_prob, draft_prob in zip(target, draft)]
    rejection = sum(residual)
    corrected = [accepted_prob + remainder
                 for accepted_prob, remainder in zip(accepted, residual)]
    return corrected, rejection

corrected, rejection = speculative_one_step_distribution([0.6, 0.4], [0.8, 0.2])
assert all(math.isclose(actual, expected) for actual, expected in zip(corrected, [0.6, 0.4]))
assert math.isclose(rejection, 0.2)
```

代码算的是概率质量，不是生成器。这里 residual 的总质量就是拒绝概率，因此“拒绝概率 × 归一化 residual”又回到 residual 本身；若两分布相同，拒绝概率为 0，不必除以 0 去构造重采样分布。

扩展到一串草稿时，先用 causal target forward 计算各位置的条件分布，再从左到右检查。第一个拒绝发生后，后面的草稿不能照收，因为它们条件中的前缀已经不成立；缓存也要截回有效前缀。串行 MTP 模块的草拟还存在前后依赖，不能想当然地当作所有未来 token 完全并行产出。

## 算速度，要算时间，不是只数步骤

以下是**假设的计时**，不是某个模型的实测：普通 decode 每 token 10 ms；一轮草拟加验证共 18 ms，平均产出 3 个最终 tokens，则每 token 6 ms，约为 1.67 倍速度。如果平均只产出 1 个，则每 token 18 ms，反而更慢。

“平均产出”要包含接受的草稿以及算法在拒绝或全接受后实际输出的 token；计时也要包含采样、缓存处理和调度。不能把一次小模块前向和一次长 target 验证都当成相同的一步，然后直接承诺 $K/2$ 倍加速。

## 怎样证明它带来了你想要的收益？

| 问题 | 控制条件 | 看什么 |
| --- | --- | --- |
| 表示是否更好 | 相同数据、相近训练计算预算 | 基础 head 的验证 loss、下游任务 |
| 多花的训练是否划算 | 记录附加模块和输出头的计算 | 质量—训练成本曲线 |
| 推理是否更快 | 相同目标模型、输出条件、硬件和请求长度 | 接受率、每秒输出 tokens、延迟 |
| 实现是否正确 | 固定小序列与文档边界 | shift、mask、未来信息泄漏、有效分母 |

MTP 值得学的不是一个热门缩写，而是如何把额外训练信号放进表示里，同时不把训练收益、推理收益和更多计算混在一起。
