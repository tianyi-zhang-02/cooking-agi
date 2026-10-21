# 起步路线：从预测到训练

**中文** · [English](speedrun-start.en.md) · [快速复习](speedrun.md)

> 最近审阅：2026-10 · 用时指有效学习时间，可分次完成，不是学会整章的期限。

刚开始学 ML，容易一口气遇到很多新词：参数、loss、梯度、训练。我们用同一个小模型把它们串起来：先算一次预测，再算误差，最后更新一个参数。只需要 Python，暂时用不到 PyTorch 或 GPU。

下面也穿插了算法和读代码练习，方便衔接面试准备。熟悉的部分可以跳过；主攻通用 SWE 的读者，先选 Python、算法和读代码即可。

<nav class="study-route speedrun-route" aria-label="起步路线的学习段落">
<a href="#first-two"><small>先安排 2 小时</small><strong>函数、预测与 loss</strong><span>手算两个样本，再用 Python 核对</span></a>
<a href="#next-four"><small>再加 4 小时</small><strong>梯度与参数更新</strong><span>改一次参数，看看 loss 为什么会变</span></a>
<a href="#next-six"><small>再加 6 小时</small><strong>模型与系统</strong><span>继续看张量、神经网络和一次预测的完整流程</span></a>
</nav>

## 从哪里开始 {#starting-point}

- **函数、list、dict 还不熟**：从 [Python 常用写法](python.md)挑一小段跑，改输入，先看程序怎样走。
- **会 Python，不熟 ML**：直接看下面的预测例子；暂时不用装 PyTorch 或申请 GPU。
- **会训练模型，想练面试**：转到[快速复习](speedrun.md)，用实现和排错找需要补的部分。

## 前 2 小时：函数、预测与 loss {#first-two}

| 建议用时 | 做什么 | 这次先理解什么 |
| --- | --- | --- |
| 15 分钟 | 看岗位或学习目标，选一个起点 | 不熟的是 Python，还是模型概念 |
| 25 分钟 | 写一个数频次的函数，用 dict 保存计数 | 输入、循环、返回值；为什么空输入也要试 |
| 30 分钟 | 跟着[预测与 loss](#prediction)算一遍 | 参数不是输出；loss 是怎样算出来的 |
| 25 分钟 | 用 list / dict 做一道小算法练习 | 先写朴素方法，再解释能省掉哪一步 |
| 15 分钟 | 讲一个自己做过的小项目或课程作业 | 输入、做法、结果，哪些是自己写的 |
| 10 分钟 | 记下下一次从哪接着做 | 可以只是一处没想明白的循环 |

合计 120 分钟，可以分次完成。算法从[Hash Map](algorithms/hash-and-prefix.md)的计数例子开始，先看懂做法，再换输入练习。

### 两个样本，算一次预测 {#prediction}

假设有两个输入 `[1,2]`，目标是 `[2,4]`。模型只有一个参数 `weight`，预测规则是 `weight × 输入`。先令 `weight = 1`，预测就是 `[1,2]`。

<div class="compact-table" markdown="1">

| 输入 | 目标 | 预测 | 误差的平方 |
| --- | --- | --- | --- |
| 1 | 2 | 1 | 1 |
| 2 | 4 | 2 | 4 |

</div>

把平方误差平均，loss 是 `(1+4)/2 = 2.5`。这里的平方是我们的目标函数选择，不是所有任务都这样算；分类通常会换用别的目标，比如交叉熵。

```python
def predict(values, weight):
    return [weight * value for value in values]


def mean_squared_error(predictions, targets):
    if len(predictions) != len(targets) or not targets:
        raise ValueError("Need equally sized, nonempty inputs")
    errors = [(prediction - target) ** 2
              for prediction, target in zip(predictions, targets)]
    return sum(errors) / len(errors)


inputs = [1.0, 2.0]
targets = [2.0, 4.0]
print(mean_squared_error(predict(inputs, 1.0), targets))
```

把参数换成 1.5，先算再运行：预测 `[1.5,3]`，loss 是 `(0.25+1)/2 = 0.625`。在这个小例子里，调整参数后，模型更符合训练数据了。**这还没证明它对没见过的数据更好。**

运行后，试着在代码里指出输入、参数、预测、目标和 loss 分别在哪。它们各自是什么，后面的更新例子还会用到。

## 再加 4 小时：梯度与参数更新 {#next-four}

| 增加用时 | 做什么 | 这次先理解什么 |
| --- | --- | --- |
| 60 分钟 | 计数、二分或 BFS 选两项，各试一个小输入 | 方法依赖什么条件，不只记代码 |
| 60 分钟 | 从下面的梯度例子接到[线性模型](../00-foundations/from-linear-to-neural.md) | 梯度怎样告诉我们局部改动方向 |
| 45 分钟 | [张量与 shape](../00-foundations/pytorch/README.md)：手算一组维度再运行 | batch、特征和输出分别是哪一维 |
| 45 分钟 | 划分训练与验证，再改一组数据看 loss | 训练成绩不能直接当作新数据成绩 |
| 30 分钟 | 回头重做第一次卡住的函数 | 哪些已经理解，哪里还需要一个例子 |

加上前一段，累计六小时。如果第一次接触张量，先看懂二维数组的 shape 和运算，再接训练循环。

### 为什么把参数往这个方向改？ {#update}

把 `weight` 从 1 调到 1.01，loss 从 2.5 变成 2.45025：在这个位置，参数稍微增大，误差就会变小。梯度描述的就是这种局部变化；只有一个参数时，它是 loss 曲线的斜率。

还是刚才的样本。平方误差对 `weight` 的梯度，是每个样本 `2 × (预测−目标) × 输入` 的平均。`weight=1` 时，两项分别是 −2 和 −8，平均为 −5。

更新规则 `新参数 = 旧参数 − 学习率 × 梯度`。取学习率 0.1，得到 `1 − 0.1 × (−5) = 1.5`，恰好是刚才试过的值，loss 降到了 0.625。

```python
weight = 1.0
gradient = sum(2 * (weight * value - target) * value
               for value, target in zip(inputs, targets)) / len(inputs)
weight -= 0.1 * gradient
print(weight, mean_squared_error(predict(inputs, weight), targets))
```

这不是“梯度更新总能降 loss”的保证。把学习率改成 1，会跳到 `weight=6`，loss 变成 40，反而更差。先理解这一个变化，再读学习率、优化器和更复杂的网络，就有东西可以对照了。

## 再加 6 小时：模型与系统 {#next-six}

| 增加用时 | 做什么 | 这次练习的重点 |
| --- | --- | --- |
| 90 分钟 | 选一个小 MLP 的 forward，或先做矩阵乘法 | 说清每层输入输出的 shape；backward 可下一次再学 |
| 60 分钟 | 看 [Tokenization](../00-foundations/core/tokenization.md)与 [Attention 图解](../00-foundations/transformer-lab.md)的一小段 | 能说明文字怎样变成输入、权重怎样混合信息；不要求手写整网 |
| 60 分钟 | [读代码](code-comprehension.md#cache)，一次追两个调用 | 发现一个可复现的 bug，并说明修复理由 |
| 60 分钟 | 再练两道相关算法或一个旧题变体 | 换条件后重新判断，不照抄模板 |
| 45 分钟 | [完整 ML 系统](../practice/ml-system/README.md)：只看一次预测经过哪里 | 区分数据、训练、上线和评估，不必记住所有组件 |
| 45 分钟 | 选一个熟悉内容，给朋友讲或自己录音练习 | 找到一处说不顺的地方，留给下一次 |

累计十二小时。这些练习只是起点，还不足以覆盖一轮 ML 面试。接下来可以沿某个例子深入，也可以到[快速复习](speedrun.md#core)检查还缺哪些基础。

## 下次接着做什么 {#keep-going}

比如这次算过了一步梯度更新，下次就试着循环更新十次，打印每次的参数和 loss。先猜结果，再运行，遇到和预期不同的数值就停下来查。这样比直接换一个更大的模型更容易看清问题。

如果马上有面试，就按已确认的轮次选练习，先把相关部分做熟；没有实践过的内容，面试时如实说明。
