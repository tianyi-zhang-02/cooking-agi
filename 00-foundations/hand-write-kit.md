# 手写 ML 模块：7 个值得练习的实现

**中文** · [English](hand-write-kit.en.md)

> 阅读时间：约 5 分钟 · 类型：速查 · 最近审阅：2026-10-09

<span id="_1"></span>

## 这七个公式分别在解决什么 {#_3}

这 7 个小实现适合练习从公式到代码：输入 shape 对不对、哪些数值会溢出、mask 到底挡住了什么。先写出能运行的版本，再用小例子检查边界，比只记住公式更有用。

参考实现在 [`code/interview_kit.py`](code/)，只依赖 NumPy，自带自检，包括**反向传播的数值梯度校验**。先跑一遍：

```bash
python3 00-foundations/code/interview_kit.py
```

## 七个，以及各自的坑 {#_4}

| 要写的 | 坑在哪 | 追问通常是 |
| --- | --- | --- |
| **softmax** | 不减 max 会溢出 | 「减了为什么还对？」——softmax 对输入平移不变，分子分母同乘 $e^c$ 约掉，所以减 max 是免费的 |
| **BCE from logits** | 先算 $p$ 再取 log，$p$ 下溢到 0 就 $-\infty$ | 「梯度是多少」——$p-y$，$\sigma'$ 被约掉了 |
| **LayerNorm** | `eps` 在根号**里面**；方差除以 $N$ 不是 $N-1$；只沿最后一维 | 「为什么不用 BatchNorm」 |
| **attention + causal mask** | 除 $\sqrt{d_k}$；掩码加在 softmax **之前** | 「softmax 之后置零行不行」——不行，破坏归一化 |
| **KV cache 解码** | 单 token query，且 cache 只有合法的过去与当前位置时，可省 causal mask | padding、滑窗和多 token 解码仍可能需要 mask；不能一概省略 |
| **top-k / top-p** | top-p 要保留**跨过阈值的那一个**，且至少留 1 个 | 「最大概率就超过 p 怎么办」——写成 `cum <= p` 会把 token 全删光 |
| **MLP 前向 + 反向** | 先说明 loss 的 mean / sum，再沿链式法则传播 | 普通 ReLU 下 `z1 > 0` 和 `a1 > 0` 等价；其他激活函数不能照搬 |

## 最后一条是分界线 {#_5}

前面的模块可以逐个测试；反向传播则要把它们连起来。建议从一个隐层 ReLU、一个输出 logit 的二分类 MLP 开始，先画出前向过程，再反过来检查每一步的 shape 和梯度。

设 $N$ 个样本的 BCE 取平均，$z_1$ 是隐层输入，$a_1=\max(z_1,0)$，$z_2$ 是输出 logit。关键几步是：

$$\frac{\partial \mathcal{L}}{\partial z_2} = \frac{\sigma(z_2)-y}{N} \quad\longrightarrow\quad \frac{\partial \mathcal{L}}{\partial W_2} = a_1^\top \frac{\partial \mathcal{L}}{\partial z_2} \quad\longrightarrow\quad \frac{\partial \mathcal{L}}{\partial z_1} = \left(\frac{\partial \mathcal{L}}{\partial z_2} W_2^\top\right)\odot \mathbb{1}[z_1 > 0]$$

**注意分母**：整个 $p-y$ 都要除以 $N$。如果换成 $\frac{1}{N}\sum_i(p_i-y_i)^2$，梯度是 $2(p-y)p(1-p)/N$；若 loss 还带 $1/2$，系数 2 才会消失。例子：$N=2,p=0.8,y=1$，单个 logit 的 mean-BCE 梯度是 -0.1。不要把单样本公式和 batch 平均混在一起，见[面试基础题](interview-basics.md)。

写完解析梯度，可以选一个参数，用中心差分对照：

$$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}$$

函数足够光滑时，中心差分的截断误差是 $O(\epsilon^2)$，前向差分是 $O(\epsilon)$。但 $\epsilon$ 太小会放大浮点舍入误差，跨过 ReLU 的不可导点也不适用这个误差结论。用 float64，检查几档步长，并避开激活值恰好为 0 的位置。

## 怎么练 {#_6}

**别读，写。** 关掉这一页，在空白文件里从头敲一遍，然后用 `interview_kit.py` 的自检对答案。卡住的地方就是你以为自己会、其实不会的地方。

一个更狠的练法：**先写自检，再写实现。** 你能说出「softmax 应该满足每行和为 1、且对输入平移不变」，说明你真的知道它是什么；写不出判据的那道题，你只是记住了公式的形状。

## 继续阅读 {#_7}

- [面试基础题：大半在问同一件事](interview-basics.md)：概念那一半
- [多头注意力](core/multi-head-attention.md) · [归一化](core/normalization.md) · [解码策略](core/decoding.md)
- [参考实现](code/)：`interview_kit.py` 是这页的配套，`attention_numpy.py` 是完整的多头版本

## 快速学习：手写公式时先回答哪五件事 {#_2}

<details class="interview" markdown="1">
<summary>公式不是默写题：输入、目的、梯度、数值与条件</summary>

**快速记忆**

每写一个公式，都顺着五问走：输入输出是什么、为什么这样定义、梯度是什么、数值上哪里会炸、结论在什么条件下成立。

**面试回答**

> 我不会只给公式。我会先定义变量和 shape，再从概率或优化目标解释公式；随后给关键梯度和稳定实现，最后明确假设。比如 CE 对 logits 的梯度是 $p-y$，实现时用 LogSumExp，CE 与 MLE 的等价依赖标签对应的条件似然。

<details markdown="1">
<summary><b>深挖</b>：为什么“条件”往往是区分度最高的一步？</summary>

BLUE 需要线性、无偏、同方差且误差不相关；L1 的稀疏性依赖不可导尖点与最优性条件；LogSumExp 的稳定写法依赖平移不变性。公式本身很多人能背，能说出何时失效才说明理解了 theorem 的边界。

</details>

</details>
