# 概率：从三条公理出发

**中文** · [English](README.en.md)

> 阅读时间：约 6 分钟 · 难度：入门 · 最近审阅：2026-09

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>概率到底是什么，哪些是定义，哪些是推出来的</strong></div>
  <div><span>前置知识</span><strong>集合的交、并、补</strong></div>
  <div><span>核心机制</span><strong>样本空间 Ω、事件、概率函数 P，外加三条公理</strong></div>
  <div><span>常见错误</span><strong>把事件和随机变量混着写，比如写 P(X)</strong></div>
</div>

## 三样东西

概率论只从三样东西出发。拿「明天下不下雨」当例子：

- **样本空间 Ω**：所有可能结果的集合。这里 $\Omega = \{\text{下雨}, \text{不下雨}\}$。
- **事件**：Ω 的一个子集。$A = \{\text{下雨}\}$，它的补是 $A^c = \{\text{不下雨}\}$。
- **概率 P**：给每个事件配一个数的函数，要满足下面三条公理。

注意，「明天下雨」是一个**事件**，不是随机变量。这一点现在分清，后面能少走很多弯路。

## 三条公理

Kolmogorov 在 1933 年把概率定义成满足这三条的函数 P：

1. $P(A) \ge 0$，对任何事件 A；
2. $P(\Omega) = 1$；
3. 如果 $A_1, A_2, \dots$ 两两不相交，那么 $P(A_1 \cup A_2 \cup \cdots) = P(A_1) + P(A_2) + \cdots$。

**定义就这么多。** 概率论里其他的结论，要么能从这三条推出来，要么是另外加的定义，比如条件概率、独立和随机变量。

严格地说，完整的对象叫概率空间 $(\Omega, \mathcal{F}, P)$。$\mathcal{F}$ 是可以谈概率的事件的集合：离散情况下就是所有子集；连续情况下，是区间，以及由区间反复取可数并和补集得到的集合。面试里基本用不到它。

## 定义和算法是两回事

下面三种算法都很常用，但都不是定义：

- **古典概型**：Ω 有限、每个结果等可能时，$P(A) = |A| / |\Omega|$。它是由公理 3 推出来的。掷骰子、抽牌、抛硬币都用这个算。
- **连续情形**：$P(a \le X \le b) = \int_a^b f(x)\,dx$。密度 f 只是帮 P 满足公理的工具，它本身不是概率。
- **长期频率**：同一个试验重复 n 次，A 出现的比例会越来越接近 $P(A)$。这是大数定律，是一条定理，也说明了这三条公理为什么定得合理。

## 六个马上能推出来的结论

每条都不超过三行。建议先自己推一遍，再对照下面的写法：

1. $P(A^c) = 1 - P(A)$。A 和 $A^c$ 不相交，并起来是 Ω，所以 $P(A) + P(A^c) = P(\Omega) = 1$。比如 $P(\text{下雨}) = 0.8$，就有 $P(\text{不下雨}) = 0.2$。答案你本来就知道，这里只是把它从公理推了出来。
2. $P(\varnothing) = 0$。$\varnothing$ 是 Ω 的补，用第 1 条。
3. 若 $A \subseteq B$，则 $P(A) \le P(B)$。把 B 拆成 A 和 $B \setminus A$ 两块不相交的部分：$P(B) = P(A) + P(B \setminus A) \ge P(A)$。
4. $P(A) \le 1$。$A \subseteq \Omega$，用第 3 条。
5. $P(A \cup B) = P(A) + P(B) - P(A \cap B)$。把 $A \cup B$ 拆成 $A \setminus B$、$A \cap B$、$B \setminus A$ 三块不相交的部分，再分别写出 $P(A)$ 和 $P(B)$，交集那块被数了两次，减掉一次。
6. 并集上界 $P(A_1 \cup \cdots \cup A_n) \le \sum_i P(A_i)$。第 5 条给出 $P(A \cup B) \le P(A) + P(B)$，再归纳。算「至少有一个发生」的概率时经常用到。

## 事件不是随机变量

**随机变量是一个函数** $X: \Omega \to \mathbb{R}$，给每个结果配一个数。把事件变成随机变量，最常用的是指示变量（indicator）：发生记 1，不发生记 0。

$$X = \mathbf{1}_A = \begin{cases} 1 & \text{下雨} \\ 0 & \text{不下雨} \end{cases}$$

这时 $P(X = 1) = P(A) = 0.8$，$P(X = 0) = 0.2$，X 服从 Bernoulli(0.8)。很多概率面试题都靠下面这一行：

$$\mathbb{E}[\mathbf{1}_A] = 1 \cdot P(A) + 0 \cdot P(A^c) = P(A)$$

**指示变量的期望，就等于事件的概率。** 遇到「平均有几个……」这类题，通常可以把个数拆成若干个指示变量之和，再用期望的线性性一项一项算。

举个例子：$n$ 个人把帽子放在一起，再随机各拿回一顶，平均有几个人拿到自己的？设 $A_i$ 为第 $i$ 个人拿对，个数就是 $X = \sum_i \mathbf{1}_{A_i}$。每个人拿对的概率都是 $1/n$，所以 $\mathbb{E}[X] = n \cdot \frac{1}{n} = 1$，和 $n$ 无关。这些指示变量彼此并不独立，但期望的线性性不需要独立。

指示变量的方差也可以直接写出来：

$$\mathrm{Var}(\mathbf{1}_A) = P(A)\,\bigl(1 - P(A)\bigr) = 0.8 \times 0.2 = 0.16$$

## 什么记号配什么

| 对象 | 用什么 | 例子 |
| --- | --- | --- |
| 事件 | P | $P(A)$、$P(A \cap B)$、$P(A \mid B)$ |
| 随机变量 | E、Var，或者某个取值的 P | $\mathbb{E}[X]$、$\mathrm{Var}(X)$、$P(X = k)$ |

如果 X 是随机变量，却写成 $P(X)$，面试官会觉得概念没分清。这是小问题，现在改掉最省事。

下一篇把随机变量画出来：[把分布画出来](distributions.md)。

## 面试常见问题

<details class="interview" markdown="1">
<summary>掷两枚均匀硬币，A =「至少一个正面」。P(A) 是多少？用两种方法算。</summary>

$\Omega = \{HH, HT, TH, TT\}$，$A = \{HH, HT, TH\}$。直接数：$3/4$。走补集：$A^c = \{TT\}$，$1 - 1/4 = 3/4$。

</details>

<details class="interview" markdown="1">
<summary>接上题，指示变量 1_A 的期望和方差是多少？</summary>

$\mathbb{E}[\mathbf{1}_A] = P(A) = 3/4$，$\mathrm{Var}(\mathbf{1}_A) = 3/4 \times 1/4 = 3/16$。

</details>

<details class="interview" markdown="1">
<summary>只用三条公理，推出 P(A ∪ B) = P(A) + P(B) − P(A ∩ B)。</summary>

$A \cup B$ 拆成三块不相交的部分：$A \setminus B$、$A \cap B$、$B \setminus A$。由公理 3，$P(A \cup B)$ 是三块之和；同样 $P(A) = P(A \setminus B) + P(A \cap B)$，$P(B) = P(B \setminus A) + P(A \cap B)$。代进去，$A \cap B$ 被多数了一次，减掉即可。

</details>

<details class="interview" markdown="1">
<summary>期望的线性性要求随机变量相互独立吗？</summary>

不要求。$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ 对任意 X、Y 都成立。所以数个数的时候，即使那些指示变量互相影响（比如上面拿帽子的例子），也可以直接把每一项的期望加起来。需要独立的是方差：$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$ 只在协方差为 0 时成立。

</details>

<details class="interview" markdown="1">
<summary>「明天下雨」是随机变量吗？为什么不能写 P(X)？</summary>

不是，它是事件，Ω 的一个子集。随机变量是从 Ω 到实数的函数，比如指示变量 $\mathbf{1}_A$。事件配 P，随机变量配 E 和 Var，要谈某个取值就写 $P(X = k)$。

</details>
