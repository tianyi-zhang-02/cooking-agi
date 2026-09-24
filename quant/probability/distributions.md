# 概率：把分布画出来

**中文** · [English](distributions.en.md)

> 阅读时间：约 6 分钟 · 难度：入门 · 最近审阅：2026-09

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>一个随机变量长什么样，怎么描述、怎么画</strong></div>
  <div><span>前置知识</span><strong>随机变量 · 指示变量</strong></div>
  <div><span>核心机制</span><strong>离散画柱子（PMF），连续看面积（PDF），CDF 对谁都成立</strong></div>
  <div><span>常见错误</span><strong>把密度 f(x) 当成概率；以为连续变量在某一点的概率是 f(x)</strong></div>
</div>

## 分布是什么

随机变量 X 的**分布**，说的是 X 落在各个范围里的概率分别是多少。一个分布通常一张图就能画清楚，画法取决于 X 属于哪一类。

## 离散：一根根柱子

如果 X 只取有限个或可数个值，就直接给每个值一个概率。这个函数叫 **PMF**（probability mass function，概率质量函数）：

$$p(k) = P(X = k), \qquad \sum_k p(k) = 1$$

画出来就是柱状图。比如抛 8 次硬币，每次正面朝上的概率是 p，正面的次数 X 服从 Binomial(8, p)，柱子落在 0 到 8 这 9 个整数上。期望就是按柱子高度加权的平均：$\mathbb{E}[X] = \sum_k k\,p(k) = 8p$。

## 连续：曲线下面的面积

如果 X 可以取一整段区间里的任何实数，就没法给每个点单独分概率了：点有无穷多个，每个点的概率只能是 0。这时要用**密度** f（PDF，probability density function，概率密度函数），概率等于曲线下面的面积：

$$P(a \le X \le b) = \int_a^b f(x)\,dx$$

这里有两个常见误解：

- **f(x) 不是概率，所以可以大于 1。** 比如 Uniform(0, 1/2) 的密度在区间上都是 2，总面积仍然是 1。
- **单个点的概率都是 0。** 所以对连续变量来说，$P(X \le b)$ 和 $P(X < b)$ 是一样的。

期望换成积分：$\mathbb{E}[X] = \int x\,f(x)\,dx$。

## CDF：对谁都成立的那张图

不管是离散、连续还是混合的变量，都能定义**累积分布函数**（CDF）：

$$F(x) = P(X \le x)$$

它从左往右把概率累加起来，所以一定不会下降，最左边趋于 0，最右边趋于 1，而且是右连续的：在跳跃点上，取的是跳上去之后的值。

- 离散变量的 CDF 是**台阶**：每一级的高度，正好是那一点的 PMF；
- 连续变量的 CDF 是**光滑的坡**，斜率就是密度，$F' = f$；
- 从 CDF 读概率，记住两条就够了：$P(a < X \le b) = F(b) - F(a)$；$P(X = x) = F(x) - F(x^-)$，也就是 CDF 在 x 处**跳了多高**。

拖动 x，看 F(x) 怎样把左边的概率一点点加进来：

<!-- widget:tx-prob-dist -->

## 混合分布：又有柱子，又有曲线

还有第三种，也是最有意思的一种。以保险赔付为例：有 π 的概率这一年没有出险，赔付正好是 0；如果出险，赔付金额大致服从正态分布，但最多赔 5.5。于是 0 和 5.5 这两个点**各自有一块实实在在的概率**，中间那一段则用密度描述。

- 只用 PMF 不够：中间那段每个点的概率都是 0；
- 只用 PDF 也不够：落在 0 和 5.5 这两个点上的概率，用密度表示不出来；
- CDF 可以：它是一段上升的曲线，在 0 和 5.5 各**跳**一下，跳多高，那一点的概率就是多少。

期望就是把点上的部分和连续的部分加在一起：

$$\mathbb{E}[X] = 0 \cdot P(X = 0) + 5.5 \cdot P(X = 5.5) + \int_0^{5.5} x\,f(x)\,dx$$

机器学习里也经常遇到这种分布。标准正态的 Z 经过 ReLU，$\mathrm{ReLU}(Z) = \max(Z, 0)$：有一半概率正好落在 0 上，另一半分布在正半轴上，画出来是一根竖线加半条钟形曲线。把奖励或梯度裁剪到 $[-c, c]$ 也是一样：超出范围的部分都被压到 $\pm c$ 这两个点上，CDF 会在那里跳一下。

条件概率和独立在下一篇：[条件概率与独立](conditional.md)。

## 面试常见问题

<details class="interview" markdown="1">
<summary>PDF 的值能大于 1 吗？</summary>

能。密度不是概率，面积才是。Uniform(0, 1/2) 的密度在整个区间上都是 2，积分还是 1。σ 很小的正态，峰值也远大于 1。

</details>

<details class="interview" markdown="1">
<summary>X 是连续随机变量，P(X = 3) 是多少？为什么不是 f(3)？</summary>

是 0。一个点的「面积」是 0。f(3) 是 3 附近单位长度上的概率密度：$P(3 \le X \le 3 + \varepsilon) \approx f(3)\,\varepsilon$，ε 趋于 0 时这个概率也趋于 0。

</details>

<details class="interview" markdown="1">
<summary>Z 服从标准正态，ReLU(Z) 的分布是什么？期望是多少？</summary>

混合分布：$P(\mathrm{ReLU}(Z) = 0) = P(Z \le 0) = 1/2$，是 0 处的一根竖线；x > 0 的部分密度就是 $\varphi(x)$。期望 $\mathbb{E}[\mathrm{ReLU}(Z)] = \int_0^\infty x\,\varphi(x)\,dx = \varphi(0) = 1/\sqrt{2\pi} \approx 0.399$。

</details>

<details class="interview" markdown="1">
<summary>X 非负，怎么只用 CDF 求 E[X]？几何分布的期望为什么是 1/p？</summary>

非负的 X 有一个很好用的公式：$\mathbb{E}[X] = \int_0^\infty P(X > x)\,dx = \int_0^\infty \bigl(1 - F(x)\bigr)\,dx$；取整数值时写成 $\mathbb{E}[X] = \sum_{k \ge 1} P(X \ge k)$。几何分布数的是「第一次成功在第几次」，前 $k-1$ 次都失败的概率是 $(1-p)^{k-1}$，所以 $\mathbb{E}[X] = \sum_{k \ge 1} (1-p)^{k-1} = 1/p$。

</details>

<details class="interview" markdown="1">
<summary>只给你 CDF，怎么求 P(2 &lt; X ≤ 5) 和 P(X = 3)？</summary>

$P(2 < X \le 5) = F(5) - F(2)$。$P(X = 3) = F(3) - F(3^-)$，也就是 CDF 在 3 处跳的高度；没有跳，就是 0。

</details>
