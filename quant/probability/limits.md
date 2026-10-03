# 大数定律与 CLT：平均会稳定，误差长什么样

**中文** · [English](limits.en.md)

> 阅读时间：约 8 分钟 · 前置：[方差](expectation-proofs.md)、[Chebyshev](inequalities.md) · 最近审阅：2026-10

“样本多了就准了”和“样本多了就正态了”都说得太快。哪一个量在变化？要满足什么条件？先把这两个问题分开。

## 1 · 弱大数定律：离均值很远的概率变小

**这里的版本。** $X_1,X_2,\ldots$ 独立同分布（iid），均值为 $\mu$，方差为 $\sigma^2<\infty$。记 $\bar X_n=(X_1+\cdots+X_n)/n$。则对每个固定的 $\varepsilon>0$：

$$
P(|\bar X_n-\mu|\ge\varepsilon)\longrightarrow0.
$$

这叫依概率收敛（convergence in probability）。

**完整证明只有两步。** 线性性与独立给出 $\mathbb E[\bar X_n]=\mu$、$\operatorname{Var}(\bar X_n)=\sigma^2/n$。再用 Chebyshev：

$$
P(|\bar X_n-\mu|\ge\varepsilon)\le\frac{\sigma^2}{n\varepsilon^2}\longrightarrow0.
$$

有限方差是这个短证明的条件，不是所有大数定律版本的必要条件。它也没说每多采一个样本，误差就必定减小。

## 2 · 一个能算出样本量的例子

独立 Bernoulli($p$) 样本估计成功率，$\sigma^2=p(1-p)\le1/4$。希望绝对误差达到 $0.05$ 的概率不超过 $0.05$，Chebyshev 给出的充分条件是：

$$
\frac1{4n(0.05)^2}\le0.05
\quad\Longrightarrow\quad n\ge2000.
$$

这是一个保守保证，不是最少样本数，也不是“不管怎么采 2000 个就行”。换成相关样本，前面的方差计算就需要重做。

<details markdown="1">
<summary>反例：把同一个样本复制很多遍会怎样？</summary>

设所有 $X_i=Z$，其中 $Z$ 是一次公平的 0/1 结果。每个变量的均值都是 $1/2$，方差都是 $1/4$，但 $\bar X_n=Z$，并没有越来越稳定。

对 $\varepsilon=0.4$，$P(|\bar X_n-1/2|\ge0.4)=1$。同分布不是独立，多行数据也不等于更多独立信息。

</details>

## 3 · CLT：把缩小的波动放大再看

**经典 iid 版本。** 均值 $\mu$ 有限，方差 $0<\sigma^2<\infty$。中心极限定理（central limit theorem）说：

$$
Z_n=\frac{\sum_{i=1}^nX_i-n\mu}{\sigma\sqrt n}
=\frac{\sqrt n(\bar X_n-\mu)}{\sigma}
\xrightarrow{d}\mathcal N(0,1).
$$

这里的分布收敛，指 $P(Z_n\le z)\to\Phi(z)$。不是说原始 $X_i$ 越来越像正态，也不是有限 $n$ 时已经精确正态。

| 问题 | 看哪个量 | 结论 |
| --- | --- | --- |
| 平均是否靠近真均值 | $\bar X_n$ | 依概率靠近 $\mu$ |
| 误差通常是什么尺度 | $\bar X_n-\mu$ | 标准差为 $\sigma/\sqrt n$ |
| 放大后的误差是什么形状 | $\sqrt n(\bar X_n-\mu)/\sigma$ | 分布趋近标准正态 |

想把标准误减半，通常要把独立样本量增到 4 倍，而不是 2 倍。CLT 本身不提供统一的“超过 30 个就够了”。

## 4 · CLT 的证明思路：为什么会出现正态

<details markdown="1">
<summary>进阶：特征函数的证明骨架，不是完整的分析课程证明</summary>

令 $W_i=(X_i-\mu)/\sigma$。特征函数定义为 $\varphi_W(t)=\mathbb E[e^{itW}]$，总是存在。由零均值、单位方差，在 0 附近：

$$
\varphi_W(t)=1-\frac{t^2}{2}+o(t^2).
$$

这里的二阶展开需要用有限二阶矩控制余项，不能未经说明把随机变量当作有界常数展开。独立使和的特征函数相乘：

$$
\varphi_{Z_n}(t)=\left[\varphi_W\left(\frac{t}{\sqrt n}\right)\right]^n
=\left[1-\frac{t^2}{2n}+o(1/n)\right]^n
\longrightarrow e^{-t^2/2}.
$$

右边是标准正态的特征函数。最后调用 Lévy 连续性定理，把特征函数收敛转成分布收敛。这两个分析步骤——二阶展开的依据与 Lévy 定理——是本页引用的前置结果，没有在这里完整证明。

</details>

## 5 · 收敛不等于什么都能交换

**依概率收敛不自动保证期望收敛。** 设 $U$ 在 $(0,1)$ 上均匀，$X_n=n\mathbf1_{\{U\le1/n\}}$。对固定 $\varepsilon>0$，当 $n>\varepsilon$ 时：

$$
P(|X_n|>\varepsilon)=\frac1n\to0,\qquad \mathbb E[X_n]=1.
$$

越来越少出现的大值，仍然撑住了均值。不能只看“大部分时候接近 0”，就把极限搬进期望。支配收敛或一致可积等条件正是为了处理这种问题。

<details markdown="1">
<summary>强大数定律和这里有什么区别？</summary>

经典 iid、可积版本的强大数定律说：$\bar X_n\to\mu$ 几乎必然，也就是除了一个概率为 0 的例外集合，整条样本路径都收敛。它比依概率收敛更强。

本页只证明了有限方差版的弱大数定律，没有用 Chebyshev 的单次概率界冒充强大数定律的证明。

</details>

## 接下来

进阶证明参考：[MIT 18.175：特征函数与 CLT](https://ocw.mit.edu/courses/18-175-theory-of-probability-spring-2014/resources/mit18_175s14_lecture15/)。

[连续概率与微积分](continuous-calculus.md)。参考：[MIT 6.041：弱大数定律](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/resources/mit6_041f10_l19/)、[中心极限定理](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/resources/mit6_041f10_l20/)。特征函数证明是进阶选读，第一遍先掌握两个定理分别在说什么。
