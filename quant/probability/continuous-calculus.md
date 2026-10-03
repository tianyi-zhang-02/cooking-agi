# 连续概率与微积分：先画区域，再动积分

**中文** · [English](continuous-calculus.en.md)

> 阅读时间：约 8 分钟 · 前置：[PDF 与 CDF](distributions.md)、基本求导与积分 · 最近审阅：2026-10

连续题容易一上来就堆积分。其实先写取值范围、画事件所在区域，通常能少算很多。这篇练三件事：面积、变量变换，以及用对数把幂变简单。

## 1 · 面积什么时候才是概率

如果 $X,Y$ 独立且都在 $[0,T]$ 上均匀，联合密度是 $1/T^2$。因此区域 $D$ 的概率是：

$$
P((X,Y)\in D)=\iint_D\frac1{T^2}\,dx\,dy
=\frac{\operatorname{Area}(D)}{T^2}.
$$

**需要联合均匀，不是“二维就能算面积比例”。** 若到达时间集中在某一段，或两人商量好时间，密度可能不均匀或根本没有二维密度。

## 2 · 约见面：两个角落比中间那条带好算

两人独立地在 $[0,T]$ 内均匀到达，各自等 $w$，其中 $0\le w\le T$。能见面的条件是 $|X-Y|\le w$。

正方形中不能见面的区域是两个直角三角形，各有两条直角边 $T-w$。因此：

$$
P(\text{meet})=1-\frac{2\cdot\frac12(T-w)^2}{T^2}
=1-\left(1-\frac wT\right)^2.
$$

$T=60,w=15$ 时，答案是 $7/16$。检查端点：不等人时概率为 0，愿意等满 $T$ 时为 1。

<details markdown="1">
<summary>如果一个等 10 分钟，另一个等 20 分钟呢？</summary>

设 A 在 $X$ 到，等 $a$；B 在 $Y$ 到，等 $b$，且 $0\le a,b\le T$。见面的条件是 $-b\le Y-X\le a$。

两块错过区域的边长分别为 $T-a$、$T-b$：

$$
P(\text{meet})=1-\frac{(T-a)^2+(T-b)^2}{2T^2}.
$$

$T=60,a=10,b=20$ 得 $31/72$。不要用平均等待时间代替两个等待时间，平方会改变结果。

</details>

## 3 · 变量变换：不会换元时，先写 CDF

令 $X\sim\operatorname{Uniform}(0,1)$，$Y=X^2$。先写支持集 $0\le Y\le1$。对 $0\le y\le1$：

$$
F_Y(y)=P(X^2\le y)=P(X\le\sqrt y)=\sqrt y.
$$

区间外 CDF 分别为 0、1；区间内求导得到：

$$
f_Y(y)=\frac1{2\sqrt y},\qquad 0<y<1.
$$

密度在 0 附近很高，但积分为 1。用它算 $\mathbb E[Y]=\int_0^1y/(2\sqrt y)\,dy=1/3$，与直接算 $\mathbb E[X^2]$ 一致。

<details markdown="1">
<summary>如果 X 改成 Uniform(-1,1)，哪一步要小心？</summary>

现在 $X^2\le y$ 对应 $-\sqrt y\le X\le\sqrt y$，不能只保留正根。概率仍为 $(2\sqrt y)/2=\sqrt y$，所以这次恰好得到相同的 $Y$ 分布。

一般的非单调变换，要把每个逆像分支都算上。CDF 法让漏分支比较容易被发现。

</details>

## 4 · 幂的比较：取对数，再找单调性

比较 $e^\pi$ 和 $\pi^e$。因为 $\log$ 严格递增，只要比较 $\pi$ 与 $e\log\pi$，等价于比较 $\log e/e$ 和 $\log\pi/\pi$。

考虑 $g(x)=\log x/x$，$x>0$：

$$
g'(x)=\frac{1-\log x}{x^2}.
$$

它在 $(0,e)$ 递增，在 $(e,\infty)$ 递减。因为 $\pi>e$，所以 $g(e)>g(\pi)$，也就是 $e^\pi>\pi^e$。

同一个导数还回答了：$x^{1/x}$ 在哪里最大？由于 $\log(x^{1/x})=g(x)$，最大点为 $x=e$，最大值为 $e^{1/e}$。**求导后要看全区间的符号，不是找到驻点就结束。**

## 5 · 近似什么时候可信

常用 $\log(1+u)\approx u$，但“$u$ 很小”最好能落到误差上。Taylor 定理给出某个介于 0 和 $u$ 的 $\xi$：

$$
\log(1+u)=u-\frac{u^2}{2(1+\xi)^2}.
$$

若 $|u|\le1/2$，则 $|\log(1+u)-u|\le2u^2$。所以近似误差是二阶，而不是随口忽略。

<details markdown="1">
<summary>用这个界，解释为什么 (1+c/n)^n 趋近 exp(c)</summary>

固定实数 $c$，当 $n$ 足够大时 $|c/n|\le1/2$ 且底数为正。于是

$$
\left|n\log(1+c/n)-c\right|\le\frac{2c^2}{n}\to0.
$$

再用指数函数连续性，得到 $(1+c/n)^n\to e^c$。先控制误差，再把极限搬过连续函数。

</details>

## 接下来

[Markov 链与等待时间](markov-chains.md)。连续分布与变量变换可对照 [MIT 6.041 第 8–10 讲](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/pages/lecture-notes/)。上面的微积分例子只用导数符号、Taylor 余项和连续性。
