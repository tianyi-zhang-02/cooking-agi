# 归一化：BatchNorm 与 LayerNorm

**中文** · [English](normalization.en.md)

> 阅读时间：约 8 分钟 · 难度：必修 · 最近审阅：2026-08

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>把每层的输入拉回一个稳定的尺度</strong></div>
  <div><span>前置知识</span><strong>一批激活值 · 两个可学参数 γ 和 β</strong></div>
  <div><span>核心机制</span><strong>沿哪条轴求均值和方差</strong></div>
  <div><span>常见错误</span><strong>把 BatchNorm 用到变长序列和自回归生成上</strong></div>
</div>

## 核心区别：归一化轴不同

**BatchNorm 沿着「一批样本」求统计量，LayerNorm 沿着「一个样本自己的特征」求。**

剩下的差别，能不能处理变长、batch size 能不能是 1、能不能自回归生成、训练和推理是否一致，全都能从这句话推出来。

![三种归一化各自沿哪条轴](../assets/norm-axes.svg)

## 三个公式

**BatchNorm**（对每个特征 $j$，在批次维上统计）：

$$\mu_j = \frac{1}{N}\sum_{i=1}^{N} x_{ij}, \qquad \sigma_j^2 = \frac{1}{N}\sum_{i=1}^{N}(x_{ij}-\mu_j)^2$$

$$\hat{x}_{ij} = \frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}, \qquad y_{ij} = \gamma_j \hat x_{ij} + \beta_j$$

**LayerNorm**（对每个样本 $i$，在特征维上统计）：

$$\mu_i = \frac{1}{d}\sum_{j=1}^{d} x_{ij}, \qquad y_{ij} = \gamma_j\,\frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}} + \beta_j$$

**RMSNorm**（LayerNorm 去掉减均值，也去掉 $\beta$）：

$$y_{ij} = \gamma_j\,\frac{x_{ij}}{\sqrt{\frac{1}{d}\sum_{k} x_{ik}^2+\epsilon}}$$

三种写法里，$\gamma$ 和 $\beta$ 都是**按特征维**走的，长度都是 $d$。真正变的只有一件事：统计量沿哪条轴算。

## 为什么 BatchNorm 不适合语言模型

**1. 训练和推理是两套行为。** 训练时它用当前批次的统计量，推理时改用训练期间累积的滑动平均。同一份权重，两种前向，这在常见的层里几乎是独一份。微调、分布漂移、忘了调 `model.eval()`，都会在这里出事。

**2. 变长序列估不出统计量。** 一个批次里句子长短不一，位置 500 上可能只有 3 个样本有值。拿 3 个样本去估均值和方差，噪声大到不能用。

**3. 自回归生成时 batch size 可能是 1。** 一个样本算不出批统计量：方差是 0，$x-\mu$ 也是 0，输出里不剩任何信息。PyTorch 索性**直接报错**：

```
ValueError: Expected more than 1 value per channel when training,
got input size torch.Size([1, 8])
```

这不是可以将就的退化，是框架层面的硬拒绝。

**4. 它把样本之间耦合起来了。** 同一批次里换个别的句子，你这句的输出就变了。对图像分类这是无害的正则化；对逐 token 生成，这是不可接受的非确定性。

LayerNorm 对上面四条**全部免疫**，因为它只看一个 token 自己的那 $d$ 个数。

## 怎样选择归一化方法

| 场景 | 用什么 | 为什么 |
| --- | --- | --- |
| CNN 图像分类，batch 够大且固定 | **BatchNorm** | 批统计量稳定，顺带带来正则化效果，通常还更快收敛 |
| 任何 Transformer / 语言模型 | **LayerNorm / RMSNorm** | 变长、batch 可能为 1、生成必须确定 |
| RNN / LSTM | **LayerNorm** | 时间步之间批统计量不可比 |
| batch size 很小（检测、分割、大模型微调） | **GroupNorm / LayerNorm** | BatchNorm 在小批次下统计量噪声太大 |
| 强化学习、在线学习 | **LayerNorm** | 数据分布随策略变化，滑动平均会一直滞后 |
| GAN 判别器 | 常用 **InstanceNorm / LayerNorm** | 避免同批次样本互相泄漏信息 |

一句判据：**只要「同一个输入在不同批次里应该得到同一个输出」是硬要求，就不能用 BatchNorm。**

现代大模型进一步选 RMSNorm：少一次归约（不用算均值），显存和带宽都省一点，效果实测不掉。

<details markdown="1">
<summary><b>进阶</b>：归一化到底为什么有用</summary>

原论文的说法是减少 internal covariate shift。后来 [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) 用实验反驳了这个解释——他们在 BN 之后人为注入分布噪声，训练依然又快又稳。

目前更站得住的解释是**它让损失曲面更平滑**，梯度的 Lipschitz 常数变小，于是可以用更大的学习率而不发散。

还有一个常被忽略的角度：归一化把权重的**尺度**自由度消掉了。$\text{Norm}(\alpha W x) = \text{Norm}(Wx)$，所以权重整体放大不改变输出——只改变有效学习率。这也是为什么归一化层通常不做 weight decay。

</details>

## 动手验证

[`../code/norm_compare.py`](../code/norm_compare.py) 用同一批激活值分别跑三种归一化，打印各自沿哪条轴统计、以及把 batch size 降到 1 时 BatchNorm 怎么塌掉。

图由 [`../code/make_norm_figures.py`](../code/make_norm_figures.py) 生成。

## 面试常见问题

<details class="interview" markdown="1">
<summary>BatchNorm 和 LayerNorm 的区别是什么？</summary>

统计量沿的轴不同：BatchNorm 对每个特征在批次维上求均值方差，LayerNorm 对每个样本在特征维上求。

推论：BatchNorm 的输出依赖同批次的其他样本，LayerNorm 不依赖。所以 BatchNorm 需要维护滑动平均供推理使用，训练和推理行为不同；LayerNorm 训练推理完全一致。

</details>

<details class="interview" markdown="1">
<summary>为什么 Transformer 用 LayerNorm 不用 BatchNorm？</summary>

四条，任何一条都足以否决：序列变长导致尾部位置的批统计量样本太少；自回归生成时 batch 可能为 1，PyTorch 直接拒绝运行；同批次其他样本会改变本样本输出，生成不确定；训练/推理两套行为增加微调风险。

</details>

<details class="interview" markdown="1">
<summary>BatchNorm 在推理时怎么做？</summary>

用训练期间累积的滑动平均 $\hat\mu, \hat\sigma^2$，不再用当前批次。所以 `model.eval()` 必须调用——忘了调是最常见的线上事故之一，表现是「单条请求和批量请求结果不一样」。

另外滑动平均是在训练分布上估的，一旦推理分布漂移，它就一直是错的，而且不会报警。

</details>

<details class="interview" markdown="1">
<summary>RMSNorm 相比 LayerNorm 去掉了什么？为什么可以去掉？</summary>

去掉了减均值和偏置 $\beta$，只做 RMS 缩放。

可以去掉的原因是：重新居中带来的收益在实践中很小，真正起作用的是重新缩放。省掉一次归约在大模型里是实打实的带宽收益。注意平方和必须用 fp32 累加，bf16 会把方差算烂。

</details>

<details class="interview" markdown="1">
<summary>pre-norm 和 post-norm 有什么区别？</summary>

post-norm 是 $\text{Norm}(x + f(x))$，norm 压在残差通路上；pre-norm 是 $x + f(\text{Norm}(x))$，残差通路上没有 norm。

后果：post-norm 的梯度每层都要穿过一个 norm，深了会不稳，所以原版 Transformer 必须配 warmup。pre-norm 留了一条完全不经过 norm 的通路，可以用更简单的学习率调度。代价是最终输出的尺度会随深度增长，所以最后要补一个 final norm。

</details>

<details class="interview" markdown="1">
<summary>batch size 很小的时候怎么办？</summary>

换 GroupNorm（把通道分组，组内统计）或 LayerNorm。GroupNorm 的统计量和 batch 无关，在检测、分割这类 batch 只能开到 2–4 的任务上是标准做法。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>不看公式，说清 BatchNorm 和 LayerNorm 分别沿哪条轴求统计量？</li>
    <li>为什么 BatchNorm 需要滑动平均而 LayerNorm 不需要？</li>
    <li>batch size 为 1 时，两者分别会发生什么？</li>
    <li>$\gamma$ 和 $\beta$ 的长度是多少？它们和统计轴是同一条轴吗？</li>
  </ol>
</div>

## 继续阅读

归一化让每层输入的尺度可控，但深度真正可行还差另一半——[残差连接](residual-connections.md)。
