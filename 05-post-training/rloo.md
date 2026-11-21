# RLOO：用同题的其他回答做比较

**中文** · [English](rloo.en.md)

> 最近审阅：2026-11 · 前置：[策略梯度](deep-rl/policy-gradients.md)、[PPO、GRPO 与 DPO](after-ppo.md)

同一个问题，让模型回答三次，分数是 1、0、0.5。拿到 0.5 的回答算好吗？单看分数不好说；如果同题的另外两份回答平均也是 0.5，这份回答至少没有比它们更好。

**REINFORCE Leave-One-Out（RLOO）用其余回答的平均奖励作为基线（baseline）**，不另训一个价值模型。下面先算一次，再解释为什么要把自己排除。算例是教学设定，不是模型实验。

## 三份回答，算出三个基线 {#worked-group}

假设一个任务检查两个要求，每满足一项得 0.5 分。对同一输入独立采样三份回答，得到：

| 回答 | 奖励 $R_i$ | 其他回答的平均奖励 $b_i$ | 优势 $A_i=R_i-b_i$ |
| --- | ---: | ---: | ---: |
| A：满足两项 | 1 | $(0+0.5)/2=0.25$ | 0.75 |
| B：两项都不满足 | 0 | $(1+0.5)/2=0.75$ | −0.75 |
| C：满足一项 | 0.5 | $(1+0)/2=0.5$ | 0 |

正优势给出提高该回答概率的梯度信号，负优势给出相反信号。C 这次没有奖励项带来的直接梯度，但不能据此断言它的概率不会变：其他回答与它共享参数，正则项也可能参与更新。

对同一 prompt $x$，采样 $K\ge2$ 个回答：

$$
b_i=\frac{1}{K-1}\sum_{j\ne i}R_j,
\qquad A_i=R_i-b_i.
$$

这就是留一法（leave-one-out）：每个回答轮流当被比较的对象，其余回答提供参照。[Ahmadian 等的论文 §2.2–2.3](https://arxiv.org/html/2402.14740v1#S2.SS3)将这种估计用于语言模型偏好训练。多采样的留一估计并非该论文首次提出，论文把它归于 [Kool、van Hoof 与 Welling（2019）](https://openreview.net/forum?id=r1lgTGL5DE)。

## 为什么不把自己的分数也算进去？ {#why-leave-one-out}

先固定 prompt。设奖励函数不依赖正在更新的参数，$y_i$ 独立同分布地采自 $\pi_\theta(\cdot\mid x)$，且可以交换求导与期望。记 $g_i=\nabla_\theta\log\pi_\theta(y_i\mid x)$，则

$$
\mathbb E[g_i\mid x]
=\sum_y\pi_\theta(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)
=\nabla_\theta\sum_y\pi_\theta(y\mid x)=0.
$$

基线只用其他独立样本，因此在给定 $x$ 时与 $y_i$ 独立：

$$
\mathbb E[b_i g_i\mid x]
=\mathbb E[b_i\mid x]\mathbb E[g_i\mid x]=0.
$$

减去它，不改变期望中的奖励梯度。这里把采样结果、奖励和基线当作固定训练数据，不沿基线反向传播。减基线通常是为了降低方差，但无偏性本身不保证每个任务里的方差都更小。

如果用包含自己的组均值 $\bar R$，代数关系是：

$$
R_i-\bar R=\frac{K-1}{K}(R_i-b_i).
$$

上面的例子里，减组均值得到 `[0.5, −0.5, 0]`，恰好是留一优势的 $2/3$。在这些采样条件下，朴素的组均值估计会把期望梯度也缩小为 $(K-1)/K$ 倍。固定 $K$ 时，这只是统一的尺度变化；若不同 prompt 的 $K$ 不同，缩放也不同。接上裁剪、优化器、梯度裁剪或其他损失后，不能只说“调一下学习率就完全一样”。

独立采样不要求文本互不相同：两个独立样本可以恰巧一样。反过来，先生成很多份，再只留下最好的几份，已经改变了样本分布；不能直接沿用上面的无偏性证明。

## 整段回答的奖励，怎样更新每个 token？ {#sequence-gradient}

先只看一次 on-policy 更新，不加 KL。最小化的样本损失为：

$$
\widehat L=-\frac1K\sum_i\operatorname{stopgrad}(A_i)
\sum_{t=1}^{T_i}\log\pi_\theta(y_{i,t}\mid x,y_{i,<t}).
$$

内层是**求和，不是按回答长度取平均**：整段概率是条件概率的乘积，取对数后自然相加。若再除以 $T_i$，不同长度回答的权重就变了。具体的 mask、EOS 和分母检查见[从一批回答核对 RL loss](../practice/post-training/rl-batch-audit.md)。

同一个 $A_i$ 乘到这份回答的各个动作上。它没有识别出“第 7 个 token 是错误起点”，也没有给每个前缀训练单独的 value。这既省了 Critic 的训练，也放弃了这部分逐步估计。需要多步奖励归因时，可以回到 [Actor–Critic 与 GAE](deep-rl/actor-critic-gae.md)比较。

## 和 GRPO 比，哪些差别要分开看？ {#comparison}

| 选择 | 留一基线的 REINFORCE | 常见的组相对训练 |
| --- | --- | --- |
| 怎么减基线 | 减其他回答的均值 | 减包含自己的组均值 |
| 是否除标准差 | 基本估计不需要 | 原始 GRPO 使用组内标准差 |
| 怎样更新 | 本文先讲一次新鲜采样的一次梯度 | 常使用概率比和 PPO-style clipping |
| 怎样聚合 token | 序列 log-prob 求和，再平均样本 | 要检查具体的长度归一化方式 |

只减均值时，两种优势有上面的固定比例关系。再除以每组标准差后，就不只是全局缩放了：不同组会得到不同权重。标准差也由当前样本估计，因此不能直接套用留一基线的无偏性证明。[DeepSeekMath §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1)给出原始 GRPO 的目标，本站[组相对优势](after-ppo.md#grpo-advantage)有完整公式。[Dr. GRPO（Liu 等，2025）§3.1](https://arxiv.org/html/2503.20783v2#S3.SS1)把这种权重变化称为题目难度偏置：几乎全对或几乎全错的题，标准差小，除完以后反而权重更大。

基线也不一定来自其他采样。[ReMax（Li 等，2023）§4.2](https://arxiv.org/html/2310.10505v4#S4.SS2)用同一题贪心解码那份回答的奖励作基线。它同样不依赖被评价的样本，所以不改变期望梯度；代价是每题多生成一份贪心回答，而 RLOO 的参照来自本来就要训练的那几份。

**RLOO 不等于“任何实现都没有 clipping”。** 留一基线与更新规则是两个选择。例如 [TRL v0.22.2 的 RLOO 文档](https://huggingface.co/docs/trl/v0.22.2/en/rloo_trainer#computing-the-loss)就包含复用样本时的概率比和裁剪。这里引用固定版本，是为了说明一种具体实现，不把它当作所有框架的定义。

## 加 KL 后，奖励和梯度还要重新对齐 {#kl}

一种做法是在采样时构造调整后的分数：

$$
\widetilde R_i=R_i-\beta\left[
\log\pi_{\mathrm{old}}(y_i\mid x)
-\log\pi_{\mathrm{ref}}(y_i\mid x)\right].
$$

再对 $\widetilde R$ 算留一优势。旧策略（old policy）负责说明数据从哪里来；参考策略（reference policy）提供正则化参照，两者不是一个角色。

方括号是某个样本的 log-ratio，可以为负；在正确采样分布下对它取期望才得到相应的 KL。原始奖励全相同，也不代表加 KL 后的分数仍然相同。

前面的证明先假设奖励不含参数。对于精确 on-policy 的 $R-\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$，显式求导还会多出 $-\beta\mathbb E[g]$，这一项为 0，所以固定该分数的 score-function 写法仍能得到对应目标的梯度。换成旧样本、多次更新、裁剪或另一种 KL 估计后，需要重新检查目标，不能把这个结论直接搬过去。见 [KL 的几种估计](rlhf/kl-estimators.md)。

## 什么时候值得试，什么时候先别加样本？ {#checks}

如果同题分数有明显差异、终局奖励有用，而 Critic 的成本很高，RLOO 是一个容易检查的起点。但采样并不免费，$K$ 翻倍不一定比增加 prompt 覆盖更划算。

| 现象 | 先查什么 |
| --- | --- |
| 一组全 0 或全 1 | 奖励是否太稀疏，题目是否过难或过易；没有额外项时，留一优势全为 0 |
| 显存省了，墙钟时间却变长 | 多份回答的生成、评分与同步占了多少时间 |
| 换 batch 顺序后优势变了 | 是否把不同 prompt 的回答混到一组；不能只凭 `reshape` 猜分组 |
| 单次更新正常，多轮后发散 | old log-prob 是否固定，数据有多旧，采样端和训练端的概率是否对齐 |
| reward 涨了，实际回答变差 | 先复核奖励与独立评测，不急着更换算法 |

小算例和无偏性条件的数值检查在[批次检查程序](../practice/post-training/code/rl_batch_checks.py)。可以从仓库根目录运行 `python3 practice/post-training/code/rl_batch_checks.py`；无需模型或 GPU。它检查数学与数据对齐，不代表真实 RL 训练已经验证成功。
