# 概率：把分布画出来

**中文** · [English](distributions.en.md)

> 阅读时间：约 5 分钟 · 难度：入门 · 最近审阅：2026-09

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>一个随机变量长什么样，怎么描述、怎么画</strong></div>
  <div><span>前置知识</span><strong>随机变量 · 指示变量</strong></div>
  <div><span>核心机制</span><strong>离散画柱子（PMF），连续看面积（PDF），CDF 对谁都成立</strong></div>
  <div><span>常见错误</span><strong>把密度 f(x) 当成概率；以为连续变量在某一点的概率是 f(x)</strong></div>
</div>

## 分布是什么

随机变量 X 的**分布**，就是「X 落在每个集合里的概率」。整个分布一张图就能画出来，只是画法要看 X 是哪一类。

## 离散：一根根柱子

X 只取可数个值的时候，直接给每个值配概率，叫 **PMF**（probability mass function）：

$$p(k) = P(X = k), \qquad \sum_k p(k) = 1$$

画出来就是柱状图。比如抛 8 次硬币、每次正面概率是 p，正面的个数 X 服从 Binomial(8, p)，柱子落在 0 到 8 这九个整数上。期望是柱子的加权平均：$\mathbb{E}[X] = \sum_k k\,p(k) = 8p$。

## 连续：曲线下面的面积

X 可以取一整段实数的时候，给单个点配概率就行不通了——点太多，每个点只能是 0。这时用**密度** f（PDF，probability density function），概率是曲线下面的面积：

$$P(a \le X \le b) = \int_a^b f(x)\,dx$$

两件事容易想错：

- **f(x) 不是概率，可以大于 1。** Uniform(0, 1/2) 的密度处处是 2，面积照样是 1。
- **任何一个点的概率都是 0。** 所以对连续变量，$P(X \le b)$ 和 $P(X < b)$ 没区别。

期望换成积分：$\mathbb{E}[X] = \int x\,f(x)\,dx$。

## CDF：对谁都成立的那张图

不管离散、连续还是别的，**累积分布函数**都有定义：

$$F(x) = P(X \le x)$$

它从左往右把概率一路攒起来，所以一定单调不减、左边趋于 0、右边趋于 1，而且是右连续的。

- 离散变量的 CDF 是**台阶**：每一级的高度，正好是那一点的 PMF；
- 连续变量的 CDF 是**光滑的坡**，斜率就是密度，$F' = f$；
- 读概率只要两条：$P(a < X \le b) = F(b) - F(a)$；$P(X = x) = F(x) - F(x^-)$，也就是在 x 处**跳了多高**。

拖一下 x，看 F(x) 怎么把左边的东西全收进来：

<!-- widget:tx-prob-dist -->

## 混合分布：又有柱子，又有曲线

最好玩的是第三种。拿保险赔付说：有 π 的概率这一年没出险，正好赔 0；出险了，赔付大致是正态的，但封顶 5.5。于是 0 和 5.5 这两个点**各自真有概率**，中间那段是密度。

- 只用 PMF 不行：中间那段每个点的概率都是 0；
- 只用 PDF 也不行：0 和 5.5 这两个点上的概率，密度写不出来；
- CDF 可以：一段坡，在 0 和 5.5 各**跳**一下，跳的高度就是那一点的概率。

期望把两部分加起来：

$$\mathbb{E}[X] = 0 \cdot P(X = 0) + 5.5 \cdot P(X = 5.5) + \int_0^{5.5} x\,f(x)\,dx$$

这在机器学习里其实到处都是。标准正态的 Z 过一个 ReLU，$\mathrm{ReLU}(Z) = \max(Z, 0)$：一半的概率正好落在 0 上，另一半铺在正半轴上——一根竖线加半条钟形曲线。把奖励或者梯度裁剪到 $[-c, c]$ 也一样，被裁掉的那部分全堆在 $\pm c$ 两个点上，CDF 在那里跳。

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
<summary>只给你 CDF，怎么求 P(2 &lt; X ≤ 5) 和 P(X = 3)？</summary>

$P(2 < X \le 5) = F(5) - F(2)$。$P(X = 3) = F(3) - F(3^-)$，也就是 CDF 在 3 处跳的高度；没有跳，就是 0。

</details>
