# 激活与初始化：梯度为什么传不动？

**中文** · [English](activation-and-initialization.en.md)

> 阅读时间：约 14 分钟 · 难度：基础到进阶 · 最近审阅：2026-10

“Sigmoid 容易梯度消失，所以不要用。”这句话少了位置和任务：放在深层网络中间，和放在二分类输出端，效果并不一样。初始化也是如此，不是记住 Xavier / He 两个名字就够了。

我们先看一个错误但很自信的分类结果，再顺着梯度往前走，最后解释初始化为什么要随输入宽度变化。

## 隐藏层与输出层，需要的不是同一种函数

| 函数 | 计算 / 值域 | 适合理解的用途 | 需要留意 |
| --- | --- | --- | --- |
| Sigmoid | $\sigma(z)=1/(1+e^{-z})$；`(0,1)` | 二分类概率、多标签独立概率、门值 | 大幅度输入会饱和；输出有界不代表已校准 |
| Tanh | $\tanh(z)$；`(-1,1)` | 有正有负的候选状态 | 两端也会饱和；函数对称不保证实际输出均值为 0 |
| ReLU | $\max(0,z)$ | 简单隐藏层 | 负半轴局部梯度为 0；正半轴不饱和不代表整网不会梯度消失 |
| Leaky ReLU / PReLU | $\max(0,z)+a\min(0,z)$ | 保留负半轴梯度 | 前者固定 $a$，后者学习 $a$；不是必胜 ReLU |
| GELU / SiLU | $z\Phi(z)$ / $z\sigma(z)$ | 平滑隐藏层变换 | 不是概率输出；与门控 FFN 是相关但不同的概念 |
| Softmax | $p_i=e^{z_i}/\sum_j e^{z_j}$ | 互斥类别 / attention 权重 | 是整组数的变换，不是独立地处理每个数 |

只有线性层相接，仍是一个仿射变换。激活函数改变了可表达的函数族；为什么能拟合 XOR，见[非线性的小实验](../from-linear-to-neural.md)。GELU 与 SiLU 的使用可继续接到 [FFN / SwiGLU](../core/ffn-and-gates.md)。不要因为新模型用了某个函数，就把输出层的概率约束也一起换掉。

## 一个分错的答案，梯度真的会消失吗？

二分类里，$p=\sigma(z)$，标签 $y\in\{0,1\}$。交叉熵对 logit 的导数是

$$\frac{\partial L_{BCE}}{\partial z}=p-y.$$

若 $z=10,y=0$，模型很自信但错了，梯度接近 1，并没有因为 sigmoid 饱和就接近 0。若对概率使用 $\frac12(p-y)^2$，导数变成

$$\frac{\partial L_{MSE}}{\partial z}=(p-y)p(1-p),$$

才会多一个很小的因子。训练中通常直接使用稳定的 logits loss，不先算概率再取 log，避免舍入到 0 或 1。[BCEWithLogitsLoss](https://docs.pytorch.org/docs/main/generated/torch.nn.BCEWithLogitsLoss.html)将这两步合在一起。

多分类也一样，softmax + cross-entropy 对 logits 的梯度是 $p-\operatorname{onehot}(y)$。`[1000, 10, 5]` 若标签是第 2 类，近似梯度是 `[1, -1, 0]`；不能说“softmax 很尖，所以模型完全更新不了”。作为 attention 内部模块时，后接的计算与梯度又不同，不能直接套输出层结论。

```python
import math

def categorical_loss_gradient(logits, target):
    if not logits or not all(math.isfinite(value) for value in logits):
        raise ValueError("finite nonempty logits required")
    if type(target) is not int or not 0 <= target < len(logits):
        raise ValueError("invalid target")
    maximum = max(logits)
    shifted = [value - maximum for value in logits]
    total = sum(math.exp(value) for value in shifted)
    probabilities = [math.exp(value) / total for value in shifted]
    loss = math.log(total) - shifted[target]
    gradient = [value - (index == target) for index, value in enumerate(probabilities)]
    return loss, gradient
```

把所有 logits 加同一个数，结果不变。因此“输入数值很小就变均匀”也不准确；决定分布的是**相对差值和温度**。这段代码对有限、正常范围的教学输入做稳定化，不是支持任意极端浮点范围的生产实现。PyTorch 的 [CrossEntropyLoss](https://docs.pytorch.org/docs/main/generated/torch.nn.CrossEntropyLoss.html)同样直接接收 logits。

## 深处的梯度看的是整条链

对 $h_\ell=\phi(W_\ell h_{\ell-1}+b_\ell)$，局部 Jacobian 为

$$J_\ell=\operatorname{diag}(\phi'(a_\ell))W_\ell.$$

总梯度穿过的是一串 $J_\ell$，不是只连乘激活函数的导数。若每层算子范数都至多 $c<1$，乘积范数至多为 $c^L$，这是梯度衰减的一个充分条件。反过来，某一层最大奇异值大于 1，并不保证整条链爆炸：方向可能在后续层被压缩。

ReLU 负输入一次为 0，也不等于这个神经元永久死亡。更值得担心的是：它对训练中几乎所有样本都落在负半轴，且没有其他更新路径把它带回来。小权重本身不会让对称、零均值输入“几乎都变负”。

先看每层 pre-activation、激活、梯度的分布，再决定是学习率、初始化、饱和、loss scale 还是计算图出了问题。裁剪梯度能限制爆炸，不能恢复已消失的信息。

## Xavier：宽度变了，单个权重也要跟着变

假设零均值权重独立，方差为 $s^2$，并与输入独立。一个输出为 $z=\sum_{j=1}^{n}w_jh_j$。若输入二阶矩相同为 $q=\mathbb E[h_j^2]$，交叉项消失后

$$\mathbb E[z^2]=n s^2q.$$

若希望在线性近似下保持前向尺度，取 $s^2\approx1/n_{in}$。反向传播给出另一个与 $n_{out}$ 有关的要求；Xavier 的常见折中是

$$\operatorname{Var}(w)=\frac{2}{n_{in}+n_{out}}.$$

对应零均值正态分布，标准差是这个数的平方根；均匀分布的边界为 $\pm\sqrt{6/(n_{in}+n_{out})}$。这是初始化时的近似尺度分析，不是全程梯度稳定定理。[Glorot 与 Bengio 的原论文](https://proceedings.mlr.press/v9/glorot10a.html)讨论了激活、梯度和层 Jacobian，而不是证明所有优化方法都归于一个“Glorot 条件”。

## He：减半的是二阶矩，不一定是方差

对关于 0 对称的 $z$，令 $r=\max(0,z)$，有

$$\mathbb E[r^2]=\tfrac12\mathbb E[z^2].$$

但 $r$ 的均值通常为正，因此 $\operatorname{Var}(r)=\mathbb E[r^2]-\mathbb E[r]^2$，不是简单地把方差除以 2。例如 $z$ 等概率取 `-1` 或 `1`，原方差为 1；ReLU 后取 `0` 或 `1`，二阶矩为 0.5，**方差为 0.25**。

把这个二阶矩关系接到下一层的零均值独立权重上，得到 He 的常见 fan-in 设置 $\operatorname{Var}(w)=2/n_{in}$。对于负半轴斜率固定为 $a$ 的 Leaky ReLU，对应为

$$\operatorname{Var}(w)=\frac{2}{(1+a^2)n_{in}}.$$

当输入宽度为 128，ReLU 设置的权重标准差是 0.125。这里的独立、对称、近似同分布假设不能丢；残差结构、门控和训练后的相关性都会改变实际尺度。[He 等人的推导](https://arxiv.org/html/1502.01852v1#S2.SS2)明确使用了非线性后的二阶矩。

```python
import math

def initialization_std(fan_in, fan_out, kind, negative_slope=0.0):
    if any(type(width) is not int or width <= 0 for width in (fan_in, fan_out)):
        raise ValueError("positive integer widths required")
    if not math.isfinite(negative_slope) or negative_slope < 0:
        raise ValueError("nonnegative finite slope required")
    if kind == "xavier":
        return math.sqrt(2.0 / (fan_in + fan_out))
    if kind == "he":
        return math.sqrt(2.0 / ((1.0 + negative_slope ** 2) * fan_in))
    raise ValueError("unknown initialization")
```

## 真正写代码时，最后检查哪几件事

- `Linear` 权重通常是 `[out_features, in_features]`。手写成另一种方向时，要核对初始化函数如何计算 fan-in / fan-out。
- 0 初始化隐藏层权重会留下对称性；偏置设 0 通常没有同样的问题。不是“所有参数都必须随机”。
- 载入预训练模型后，不要顺手对整个模型重新初始化；通常只初始化新加的模块。
- 框架默认初始化不等于你推导出的某个 ReLU 配方，查看实际 [PyTorch init 文档](https://docs.pytorch.org/docs/main/nn.init.html)和模块实现。
- 一次 forward 后记录尺度，一次 backward 后确认有限且非零的梯度，再做小 batch 过拟合测试。初始化好，只代表起点合理，不代表训练已经成功。
