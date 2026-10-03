# 事件证明：为什么可以这样拆

**中文** · [English](event-proofs.en.md)

> 阅读时间：约 7 分钟 · 前置：[概率公理](README.md) · 最近审阅：2026-10

“至少一个发生”的概率为什么不能直接相加？因为同一个结果可能被数了好几次。很多概率证明的第一步，不是找公式，而是把集合拆成互不重叠的部分。

## 1 · 从空集到补集

**只用三条公理。** 先把 $\Omega$ 写成 $\Omega,\varnothing,\varnothing,\ldots$ 的不交并。由可数可加性和非负性，$1=1+\sum_{n\ge1}P(\varnothing)$ 只能在 $P(\varnothing)=0$ 时成立。这也让有限个事件的可加性成为可数可加性的特例。

因为 $\Omega=A\mathbin{\dot\cup}A^c$，所以：

$$
P(A^c)=1-P(A).
$$

符号 $\dot\cup$ 特意提醒我们：这些部分互不相交。**可以直接加的原因是不交，不是独立。**

## 2 · 单调性与容斥

若 $A\subseteq B$，就有 $B=A\mathbin{\dot\cup}(B\setminus A)$。因此：

$$
P(B)=P(A)+P(B\setminus A)\ge P(A).
$$

两个任意事件则拆成三块：

$$
A\cup B=(A\setminus B)\mathbin{\dot\cup}(A\cap B)\mathbin{\dot\cup}(B\setminus A).
$$

把 $P(A)$ 与 $P(B)$ 分别展开，交集出现了两次，于是得到容斥（inclusion–exclusion）：

$$
P(A\cup B)=P(A)+P(B)-P(A\cap B).
$$

掷一颗公平骰子，$A=\{2,4,6\}$，$B=\{4,5,6\}$。直接相加得 1，可并集只有 $\{2,4,5,6\}$，概率为 $2/3$；多算的正是 $\{4,6\}$。

<details markdown="1">
<summary>换成 3 个事件，为什么最后要把三重交集加回来？</summary>

先加三个单事件，再减三个两两交集。一个同时属于三者的结果，计数从 3 变成了 0，需要补回 1：

$$
P(A\cup B\cup C)=P(A)+P(B)+P(C)-P(A\cap B)-P(A\cap C)-P(B\cap C)+P(A\cap B\cap C).
$$

这是逐个结果检查“被数了几次”，不是背正负号。

</details>

## 3 · 并集上界：不需要独立

**定理。** 对有限或可数个事件，

$$
P\left(\bigcup_{n\ge1}A_n\right)\le\sum_{n\ge1}P(A_n).
$$

把每个事件里“之前已经出现过”的结果去掉：

$$
B_1=A_1,\qquad B_n=A_n\setminus\bigcup_{k<n}A_k.
$$

现在 $B_n$ 两两不交，并集却没变。又因为 $B_n\subseteq A_n$，所以：

$$
P\left(\bigcup_n A_n\right)=\sum_nP(B_n)\le\sum_nP(A_n).
$$

比如做 20 次检查，每次误报概率不超过 $0.01$，那么至少一次误报的概率不超过 $0.2$，不管这些检查是否独立。这是**上界**，不是实际误报率。

若所有 $A_n$ 都是同一个事件，上界会非常松。松不等于错，它换来的是少做假设。

## 4 · 事件逐渐变大，概率为什么也能取极限

**定理：从下连续（continuity from below）。** 若 $A_1\subseteq A_2\subseteq\cdots$，令 $A=\bigcup_nA_n$，则 $P(A_n)\to P(A)$。

<details markdown="1">
<summary>展开证明：把每次新增的部分单独拿出来</summary>

设 $D_1=A_1$，$D_n=A_n\setminus A_{n-1}$。这些增量两两不交，且

$$
P(A_n)=\sum_{k=1}^{n}P(D_k),\qquad P(A)=\sum_{k=1}^{\infty}P(D_k).
$$

无穷级数就是部分和的极限，所以结论成立。这里真正用到的是可数可加性。

</details>

若事件逐渐变小，$A_n\downarrow A=\bigcap_nA_n$，对补集用刚才的结论，就得到 $P(A_n)\to P(A)$。概率总量有限，所以这一步没有“无穷减无穷”。

**应用：CDF 为什么右连续？** 固定 $x$，取 $x_n\downarrow x$。事件 $\{X\le x_n\}$ 递减到 $\{X\le x\}$，所以 $F(x_n)\to F(x)$。离散变量的 CDF 可以跳，但不能在右侧另留一个洞。

## 5 · 加餐：错误会不会无穷次出现

若 $\sum_nP(A_n)<\infty$，则 $A_n$ 无穷次发生的概率是 0。这是第一 Borel–Cantelli 引理，不要求独立。

<details markdown="1">
<summary>只用刚才的上界，推一次</summary>

“无穷次发生”意味着无论从哪一个 $N$ 开始，后面总还有一次。因此这个事件包含在每个 $\bigcup_{n\ge N}A_n$ 里：

$$
P(A_n\text{ infinitely often})\le P\left(\bigcup_{n\ge N}A_n\right)\le\sum_{n\ge N}P(A_n)\longrightarrow0.
$$

例如 $P(A_n)\le1/n^2$ 时适用。反过来，概率之和发散**不能单独推出**无穷次发生：若所有 $A_n=A$ 且 $P(A)=1/2$，总和发散，但无穷次发生的概率仍是 $1/2$。

</details>

## 接下来

加餐出处：[MIT 18.175：Borel–Cantelli 与强大数定律](https://ocw.mit.edu/courses/18-175-theory-of-probability-spring-2014/resources/mit18_175s14_lecture9/)。

这篇反复用的动作只有一个：**先改写集合，再算概率。** 下一篇把事件变成数字：[期望与方差的证明](expectation-proofs.md)。

基础参考：[MIT 6.041：概率模型与公理](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/resources/mit6_041f10_l01/)。Borel–Cantelli 是加餐，第一遍可跳过。
