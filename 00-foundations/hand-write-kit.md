# 白板上会让你手写的那七个

**中文** · [English](hand-write-kit.en.md)

> 阅读时间：约 5 分钟 · 类型：速查 · 最近审阅：2026-08

## 快速学习：手写公式时先回答哪五件事

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

## 这七个公式分别在解决什么

手撕题考的从来不是「你记不记得公式」，是**你知不知道哪一行会炸**。每道题都有一个数值或语义上的坑，而追问必落在那一行上。

参考实现在 [`code/interview_kit.py`](code/)，纯 numpy、无依赖、自带 14 项自检——包括**反向传播对数值梯度的校验**。先跑一遍：

```bash
python3 00-foundations/code/interview_kit.py
```

## 七个，以及各自的坑

| 要写的 | 坑在哪 | 追问通常是 |
| --- | --- | --- |
| **softmax** | 不减 max 会溢出 | 「减了为什么还对？」——softmax 对输入平移不变，分子分母同乘 $e^c$ 约掉，所以减 max 是免费的 |
| **BCE from logits** | 先算 $p$ 再取 log，$p$ 下溢到 0 就 $-\infty$ | 「梯度是多少」——$p-y$，$\sigma'$ 被约掉了 |
| **LayerNorm** | `eps` 在根号**里面**；方差除以 $N$ 不是 $N-1$；只沿最后一维 | 「为什么不用 BatchNorm」 |
| **attention + causal mask** | 除 $\sqrt{d_k}$；掩码加在 softmax **之前** | 「softmax 之后置零行不行」——不行，破坏归一化 |
| **KV cache 解码** | **解码时不需要因果掩码** | 「为什么不用」——q 只有一个位置，cache 里全是过去，由构造保证 |
| **top-k / top-p** | top-p 要保留**跨过阈值的那一个**，且至少留 1 个 | 「最大概率就超过 p 怎么办」——写成 `cum <= p` 会把 token 全删光 |
| **MLP 前向 + 反向** | ReLU 的导数判据是 `z1 > 0` 不是 `a1 > 0` | 「你怎么知道反向写对了」——**数值梯度校验** |

## 最后一条是分界线

前六个背下来就能写。**手写反向传播不行**——它是「真的懂链式法则」和「会调框架 API」之间的那条线，也是唯一一个面试官能从你写代码的顺序看出你懂没懂的题。

关键的几步：

$$\frac{\partial \mathcal{L}}{\partial z_2} = \sigma(z_2) - y \;\Big/\, N \quad\longrightarrow\quad \frac{\partial \mathcal{L}}{\partial W_2} = a_1^\top \frac{\partial \mathcal{L}}{\partial z_2} \quad\longrightarrow\quad \frac{\partial \mathcal{L}}{\partial z_1} = \left(\frac{\partial \mathcal{L}}{\partial z_2} W_2^\top\right)\odot \mathbb{1}[z_1 > 0]$$

**注意第一项**：这里又是 BCE 那个约掉——如果你换成 MSE，这行就变成 $2(p-y)p(1-p)$，后面全跟着变。[面试基础题](interview-basics.md)那篇讲的就是这个约掉的后果。

**然后主动加一句**：「我一般会跑一次数值梯度校验」，并写出来——

$$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}$$

用中心差分不用前向差分（误差 $O(\epsilon^2)$ 而不是 $O(\epsilon)$）。这句话说出来，比你把梯度写对更能说明问题——**它说明你知道怎么验证自己**。

## 怎么练

**别读，写。** 关掉这一页，在空白文件里从头敲一遍，然后用 `interview_kit.py` 的自检对答案。卡住的地方就是你以为自己会、其实不会的地方。

一个更狠的练法：**先写自检，再写实现。** 你能说出「softmax 应该满足每行和为 1、且对输入平移不变」，说明你真的知道它是什么；写不出判据的那道题，你只是记住了公式的形状。

## 继续阅读

- [面试基础题：大半在问同一件事](interview-basics.md)：概念那一半
- [多头注意力](core/multi-head-attention.md) · [归一化](core/normalization.md) · [解码策略](core/decoding.md)
- [参考实现](code/)：`interview_kit.py` 是这页的配套，`attention_numpy.py` 是完整的多头版本
