# 手写 ML 模块：7 个值得练习的实现

**中文** · [English](hand-write-kit.en.md)

> 阅读时间：约 8 分钟 · 类型：速查与小实验 · 最近审阅：2026-10

<span id="_1"></span>

## 这七个实现练什么 {#_3}

这 7 个小实现适合练习从公式到代码：输入 shape 对不对、哪些数值会溢出、mask 到底挡住了什么。先写出能运行的版本，再用小例子检查边界，比只记住公式更有用。

参考实现在 [`code/interview_kit.py`](code/)，只依赖 NumPy，自带自检，包括**反向传播的数值梯度校验**。先跑一遍：

```bash
python3 00-foundations/code/interview_kit.py
```

## 七个实现，先检查这些地方 {#_4}

| 要写的 | 坑在哪 | 追问通常是 |
| --- | --- | --- |
| **softmax** | 不减 max 会溢出 | 「减了为什么还对？」——softmax 对输入的共同平移不变，分子分母中的共同因子 $e^c$ 会约掉，因此减 max 不改变数学结果 |
| **BCE from logits** | 先算 $p$ 再取 log，$p$ 下溢到 0 就 $-\infty$ | 「梯度是多少」——$p-y$，$\sigma'$ 被约掉了 |
| **LayerNorm** | `eps` 在根号里面；方差除以 $N$ 而不是 $N-1$；这份实现只沿最后一维 | 换成 `normalized_shape=(T,d)` 就会跨时间归一化，不再是逐 token 的配置 |
| **attention + causal mask** | 除 $\sqrt{d_k}$；掩码加在 softmax **之前** | 「softmax 之后置零行不行」——不行，破坏归一化（两个分数的例子见[多头注意力](core/multi-head-attention.md#_5)） |
| **KV cache 解码** | 单 token query，且 cache 只有合法的过去与当前位置时，可省 causal mask | padding、滑窗和多 token 解码仍可能需要 mask；不能一概省略 |
| **top-k / top-p** | top-p 要保留**跨过阈值的那一个**，且至少留 1 个 | 「最大概率就超过 p 怎么办」——写成 `cum <= p` 会把 token 全删光 |
| **MLP 前向 + 反向** | 先说明 loss 的 mean / sum，再沿链式法则传播 | 普通 ReLU 下 `z1 > 0` 和 `a1 > 0` 等价；其他激活函数不能照搬 |

采样代码还有个很容易漏掉的小细节：`[2, 2, 2, 1]` 取 top-k、k=2 时，到底留哪两个？本实现按 token ID 打破平局，保留 ID 0、1；如果只用“分数不低于第 k 名”来筛，就会留下前三个。两种约定都可以用，但不能一边写“固定 k 个”，一边在代码里保留所有并列项。已有 mask 不会被重新打开；`top_p=1` 表示不再截断。

## softmax：先把数字缩到能算的范围 {#softmax-numerics}

比如三个 logits 是 `[1000, 1001, 1002]`。直接算指数会溢出，但我们只关心它们之间的相对大小。每项都减去 1002，得到 `[-2, -1, 0]`，再算指数和归一化，概率约为 `[0.0900, 0.2447, 0.6652]`。

$$
p_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\qquad m=\max_j x_j.
$$

分子分母同时除以 $e^m$，数学结果没变。最大的指数变成 1，其余不超过 1，避免了指数溢出。但**减 max 不是对所有数值问题的保证**：

| 输入或场景 | 会发生什么 | 这份实现怎么处理 |
| --- | --- | --- |
| `[1000, 1001, 1002]` | 直接取指数会溢出 | 先减最大值 |
| FP16 的 65,536 个零 | 每个指数都是 1，但 FP16 求和装不下 65,536：NumPy 的成对求和会溢出成 `inf`，逐个累加则停在 2048，结果悄悄出错 | 至少用 FP32 计算，返回 FP32；不立即转回 FP16 |
| `[0, -inf]` | 第二项被 mask | 返回 `[1, 0]` |
| `[-inf, -inf]` | 没有可选位置，分布没有定义 | 报错；不擅自改成均匀分布 |
| 包含 `NaN` 或 `+inf` | 通常要检查上游计算 | 报错；不能把它们当作 mask |

这里说的是配套 NumPy 教学实现，不是所有框架的默认行为。整数输入还会检查转换是否精确：像 `[2**60, 2**60+1]`，直接转成 float64 会丢掉差值，因此报错，不默默给出均匀概率。更高精度也不能保住任意小的概率，或修复已经在上游溢出的 logits。

### 要的是 log-probability，就直接算它 {#log-softmax-example}

对 `[0, -1000]`，第二个概率在常见浮点格式里会下溢到 0。先 softmax 再取 log，得到 `-inf`；直接计算 log-softmax，却还能得到约 `-1000`。损失需要的是 log-probability，没必要先把它变成一个存不下的小概率。

```python
import numpy as np

logits = np.array([0.0, -1000.0], dtype=np.float64)
shifted = logits - logits.max()
weights = np.exp(shifted)
probabilities = weights / weights.sum()
with np.errstate(divide="ignore"):
    indirect = np.log(probabilities)
direct = shifted - np.log(weights.sum())
assert np.isneginf(indirect[1])
assert np.allclose(direct, [0.0, -1000.0])
```

[PyTorch 的 `log_softmax`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.log_softmax.html)（2.14，2026-10 核对）也明确区分直接计算与 `log(softmax(x))`。如果想接着理解怎样分块计算、为什么 attention 不必存整行概率，可以看 [online softmax 的例子](deep-dives/attention-kernels.md#online-output)。这三个问题分别关心数值范围、输出形式和中间量存储，不是同一件事。

## 把前向和反向连起来 {#_5}

前面的模块可以逐个测试；反向传播则要把它们连起来。建议从一个隐层 ReLU、一个输出 logit 的二分类 MLP 开始，先画出前向过程，再反过来检查每一步的 shape 和梯度。

设 $N$ 个样本的 BCE 取平均，$z_1$ 是隐层输入，$a_1=\max(z_1,0)$，$z_2$ 是输出 logit。关键几步是：

$$\frac{\partial \mathcal{L}}{\partial z_2} = \frac{\sigma(z_2)-y}{N} \quad\longrightarrow\quad \frac{\partial \mathcal{L}}{\partial W_2} = a_1^\top \frac{\partial \mathcal{L}}{\partial z_2} \quad\longrightarrow\quad \frac{\partial \mathcal{L}}{\partial z_1} = \left(\frac{\partial \mathcal{L}}{\partial z_2} W_2^\top\right)\odot \mathbb{1}[z_1 > 0]$$

**注意分母**：整个 $p-y$ 都要除以 $N$。如果换成 $\frac{1}{N}\sum_i(p_i-y_i)^2$，梯度是 $2(p-y)p(1-p)/N$；若 loss 还带 $1/2$，系数 2 才会消失。例子：$N=2,p=0.8,y=1$，单个 logit 的 mean-BCE 梯度是 -0.1。不要把单样本公式和 batch 平均混在一起，见[面试基础题](interview-basics.md)。

写完解析梯度，可以选一个参数，用中心差分对照：

$$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}$$

函数足够光滑时，中心差分的截断误差是 $O(\epsilon^2)$，前向差分是 $O(\epsilon)$。但 $\epsilon$ 太小会放大浮点舍入误差，跨过 ReLU 的不可导点也不适用这个误差结论。用 float64，检查几档步长，并避开激活值恰好为 0 的位置。

## 怎么练 {#_6}

可以先选一个模块，不必一次写完七个。比如先写 softmax，再检查每行概率和是否接近 1、被 mask 的位置是否为 0，以及加上同一个常数后结果是否在容差内一致。卡住时回来看对应的小节就好。

熟悉以后，试着先写测试再写实现。除了正常输入，也放进很大的 logits、部分 mask 和整行 mask。这样检查的不只是公式有没有写对，还有这段代码遇到不支持的输入时会怎么做。

## 继续阅读 {#_7}

- [面试基础题：大半在问同一件事](interview-basics.md)：概念那一半
- [多头注意力](core/multi-head-attention.md) · [归一化](core/normalization.md) · [解码策略](core/decoding.md)
- [参考实现](code/)：`interview_kit.py` 是这页的配套，`attention_numpy.py` 是完整的多头版本

## 快速学习：手写公式时先回答哪五件事 {#_2}

<details class="interview" markdown="1">
<summary>公式不是默写题：输入、目的、梯度、数值与条件</summary>

**快速记忆**

每写一个公式，都检查五件事：输入输出是什么、为什么这样定义、梯度是什么、哪些计算可能出现数值问题、结论在什么条件下成立。

**面试回答**

> 我不会只给公式。我会先定义变量和 shape，再从概率或优化目标解释公式；随后给关键梯度和稳定实现，最后明确假设。比如 CE 对 logits 的梯度是 $p-y$，实现时用 LogSumExp，CE 与 MLE 的等价依赖标签对应的条件似然。

<details markdown="1">
<summary><b>深挖</b>：为什么“条件”往往是区分度最高的一步？</summary>

比如线性回归里的 Gauss–Markov 定理，要检查设计矩阵满列秩、误差条件均值为零、同方差且不相关；不能只记一句“最优线性无偏”。再比如 softmax 对共同平移不变，但 LogSumExp 会随之增加同一个常数：稳定计算时减去的 max，最后要加回来。问清这些条件，比再背一遍公式更有帮助。

</details>

</details>
