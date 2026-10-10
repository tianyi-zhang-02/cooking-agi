# 交叉熵：从一个预测到训练里的 loss

**中文** · [English](cross-entropy.en.md)

> 阅读时间：约 14 分钟 · 运行环境：PyTorch / CPU · 最近审阅：2026-10

模型给正确答案 40% 的概率，和给它 1% 的概率，都可能被算成“预测错了”。但训练时，我们希望分清这两种错误：后者更需要改。交叉熵（cross-entropy，CE）就能表达这种差别。

先看一个三分类的小例子，再把它接到语言模型。第一次读懂前 3 节就够了；要写训练代码，再看后面的 mask、平均方式和梯度。

## 正确类别拿到了多少概率？ {#one-prediction}

假设任务是判断一句话在说“退款”“物流”还是“其他”。模型给出概率 `[0.2, 0.3, 0.5]`，这条数据的标签是“其他”。单条样本的 loss 是：

$$
\ell=-\log 0.5\approx0.6931.
$$

这里用自然对数，单位是 nat。正确类别的概率变成 0.9 时，loss 约为 0.105；降到 0.01 时，loss 约为 4.605。**正确答案拿到的概率越低，惩罚越大。**

<figure class="worked-update">
<ol>
<li><small>01 / 模型输出</small><strong>先得到 logits</strong><span>[log 2, log 3, log 5]<br>它们是分数，不是概率。</span></li>
<li><small>02 / 换算概率</small><strong>Softmax → [0.2, 0.3, 0.5]</strong><span>三个类别共享总计为 1 的概率。</span></li>
<li><small>03 / 对照标签</small><strong>标签是“其他”</strong><span>取第 3 类：−log 0.5 ≈ 0.6931。</span></li>
</ol>
<figcaption>这是方便手算的独立例子。标签告诉 loss 该检查哪一类，不是提前把答案交给模型前向。</figcaption>
</figure>

为什么不只数对错？当正确类别还排第 2 名时，它的概率从 0.1 涨到 0.4，准确率可能完全不变，CE 却能反映进步。反过来，CE 下降也不保证生成质量、校准或每个群体的准确率都变好。

## 从 logits 直接算，为什么更稳？ {#stable-ce}

令 $z_c$ 是类别分数，$y$ 是正确类别。把 Softmax 代入负对数：

$$
\ell=-\log p_y
=\log\sum_c e^{z_c}-z_y.
$$

直接算 `exp(1000)` 会溢出。我们先减去这一行最大的分数 $m$：

$$
\ell=\log\sum_c e^{z_c-m}-(z_y-m).
$$

所有分数减去同一个数，不会改变 Softmax。最大的指数现在是 $e^0=1$，其他指数不超过 1。下面手写一个单 batch、类别在最后一维的版本，再和库函数核对：

```python
import math
import torch
import torch.nn.functional as functional

def ce_from_logits(logits, targets):
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_partition = shifted.exp().sum(dim=-1).log()
    selected = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_partition - selected).mean()

logits = torch.tensor([[2., 3., 5.]], dtype=torch.float64).log()
targets = torch.tensor([2])
loss = ce_from_logits(logits, targets)
assert math.isclose(loss.item(), math.log(2))
torch.testing.assert_close(loss, functional.cross_entropy(logits, targets))
large_logits = torch.tensor([[1000., 999., -1000.]], dtype=torch.float64)
torch.testing.assert_close(
    ce_from_logits(large_logits, targets),
    functional.cross_entropy(large_logits, targets),
)
```

这里没有先生成概率再取 `log`。后者可能先把很小的概率舍入成 0，再得到无穷大的 loss。实际训练优先用现成的稳定实现，手写代码是为了看清它在算什么。

## 硬标签和软标签，差在哪里？ {#soft-targets}

硬标签（hard target）只指定一个类别，相当于 one-hot 分布 `[0, 0, 1]`。软标签（soft target）可以是 `[0.1, 0.2, 0.7]`：目标不是把全部概率压到第 3 类，而是接近这一整个分布。

$$
\ell=-\sum_c q_c\log p_c.
$$

沿用刚才的预测 `[0.2, 0.3, 0.5]`，软标签 CE 约为 0.8869。比硬标签的 0.6931 大，并不说明模型突然变差了：**目标分布变了，不能直接比两个数字。**

```python
soft_targets = torch.tensor([[.1, .2, .7]], dtype=torch.float64)
log_probabilities = functional.log_softmax(logits, dim=-1)
soft_loss = -(soft_targets * log_probabilities).sum(-1).mean()
torch.testing.assert_close(
    soft_loss, functional.cross_entropy(logits, soft_targets)
)
assert math.isclose(soft_loss.item(), 0.8869413785, abs_tol=1e-9)
```

蒸馏可以用 teacher 的分布作目标；label smoothing 则把 one-hot 和一个平滑分布混合。这些目标都要检查非负、归一化及类别顺序。若目标 $q$ 固定，最小化 CE 与最小化 $D_{KL}(q\|p)$ 的梯度相同，因为两者只差与模型无关的 $H(q)$；数值不必相同。完整算例见[蒸馏](../../05-post-training/distillation.md)。

### 梯度为什么是“预测减目标”？

对单条、未加权、归一化目标的 Softmax CE：

$$
\frac{\partial\ell}{\partial z_c}=p_c-q_c.
$$

硬标签例子的梯度是 `[0.2, 0.3, −0.5]`。梯度下降会降低前两类的分数，提高正确类别的分数。换成软标签，梯度变成 `[0.1, 0.1, −0.2]`，更新方向仍然合理，但没那么急着把第 3 类推到 100%。

<details markdown="1">
<summary>展开推导：把 log Softmax 拆开就够了</summary>

利用 $\sum_c q_c=1$：

$$
\ell=\log\sum_j e^{z_j}-\sum_c q_c z_c.
$$

第一项对 $z_c$ 的导数是 $p_c$，第二项是 $-q_c$。批次取平均后再除以样本数；类别加权或未归一化目标会改变这个表达式。

注意 $-\log p_y$ 对“概率”求导会出现 $-1/p_y$，但对 logits 求导还要经过 Softmax。不能看见前者就断言 CE 的 logit 梯度会在低概率处爆炸。

</details>

## 二分类：别靠 clamp 补救 log(0) {#binary-ce}

二分类可以只输出一个 logit $z$，通过 sigmoid 得到正类概率。标签 $y\in\{0,1\}$ 时：

$$
\begin{aligned}
\ell={}&-y\log\sigma(z)\\
&-(1-y)\log(1-\sigma(z)).
\end{aligned}
$$

它也能直接在 logits 上稳定计算。先算 sigmoid、再把概率夹到 `[1e-7, 1−1e-7]`，虽然可能避免 `log(0)`，却改了目标：落在截断区间外时，clamp 的梯度是 0。

例如正确标签为 1，模型却给出 logit −100。稳定 BCE 的 loss 约为 100，梯度约为 −1，应该明显提高分数。夹住概率后，loss 只有约 16.12，而且传回 logit 的梯度为 0，恰好不再纠正这个严重错误。

```python
wrong_logit = torch.tensor([-100.], dtype=torch.float64, requires_grad=True)
binary_target = torch.ones_like(wrong_logit)
stable_loss = functional.binary_cross_entropy_with_logits(wrong_logit, binary_target)
stable_loss.backward()
assert math.isclose(stable_loss.item(), 100.)
assert math.isclose(wrong_logit.grad.item(), -1.)
wrong_logit.grad = None
clamped_loss = -wrong_logit.sigmoid().clamp(1e-7, 1 - 1e-7).log().mean()
clamped_loss.backward()
assert wrong_logit.grad.item() == 0.
```

[BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)把 sigmoid 与 BCE 合并计算，不需要你提前做 sigmoid。多标签任务通常给每个标签一个这样的 logit；“退款”和“物流”若可以同时成立，就不该强行要求它们的概率之和为 1。

## 到了语言模型：哪些位置算分？ {#mask-and-reduction}

语言模型每个位置输出词表上的分类分数，常见 shape 是 `[batch, sequence, vocabulary]`。不过不能直接把整张张量平均：

- **先对齐目标**：causal LM 的当前位置预测下一个 token；只 shift 一次。
- **再选位置**：padding、未监督的用户文本等不计入 loss，但仍需正确的 attention mask。
- **最后定分母**：这里按有效目标 token 数取平均，不按 padding 后的长度。

一个小 batch 的有效 loss 分别是 `[2]` 和 `[1, 1, 1]`。按 token 平均得到 $5/4=1.25$；先按样本平均再平均得到 $(2+1)/2=1.5$。前者每个 token 等权，后者每条样本等权。两者都能定义成目标，但别把它们当成同一件事。

同理，gradient accumulation 不能在有效 token 数不同时，未经调整就把每个 microbatch 的 mean 再平均。要先确定训练目标的分母；分布式时还要对齐跨 rank 的计数和梯度归约。实作见[训练循环](training-loop.md)，完整对话例子见 [SFT](../../05-post-training/sft-and-its-ceiling.md#mask-example)。

<details markdown="1">
<summary>工程检查：为什么同样叫 CE，换一种标签写法结果就不同？</summary>

[PyTorch CE 接口](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html)区分整数类别与概率目标。整数标签可用 `ignore_index`；概率目标需要自己选择有效位置。带类别权重时，二者的默认 `mean` 分母也不同。

例如两个样本的 logits 都是 `[0, 0]`，标签分别为 0、1，类别权重为 `[1, 3]`。整数标签的加权 mean 是 $\log2$；把标签换成 float one-hot，默认 mean 是 $2\log2$。标签表达了同一个类别，**归约约定却不一样**。

```python
weighted_logits = torch.zeros(2, 2, dtype=torch.float64)
hard_targets = torch.tensor([0, 1])
class_weights = torch.tensor([1., 3.], dtype=torch.float64)
one_hot_targets = functional.one_hot(hard_targets, 2).double()
hard_mean = functional.cross_entropy(weighted_logits, hard_targets, weight=class_weights)
soft_mean = functional.cross_entropy(weighted_logits, one_hot_targets, weight=class_weights)
assert math.isclose(hard_mean.item(), math.log(2))
assert math.isclose(soft_mean.item(), 2 * math.log(2))
```

另外，PyTorch 的多维 CE 把类别轴放在第 2 维，不是永远放在最后一维。对 `[B,T,V]`，先展成 `[B×T,V]`，或明确换轴。不要依靠恰好相等的维度混过去。

全部位置被 mask 时，没有可平均的目标。应明确报错或按训练协议处理空批次，不用 `nan_to_num` 悄悄把它变成一次“成功更新”。

</details>

## 怎样确认自己真的写对了？ {#checks}

先拿小张量检查，不急着跑大模型。本文代码能按顺序在 CPU 执行；更完整的[教学实现](../code/loss_contracts.py)还检查输入、空 mask、soft target 和一次 shift，配套[回归测试](../../site/tests/test_loss_contracts.py)包含梯度和极端数值对照。

| 检查 | 预期 |
| --- | --- |
| 所有 logits 加同一个常数 | CE 不变 |
| 正确类别概率提高，目标不变 | 单样本硬标签 CE 下降 |
| 用 one-hot 替代整数标签，且不加权 | loss 和梯度一致 |
| 增加不参与监督的 padding | 有效 token mean 不变 |
| `z = −100, y = 1` 的 BCE | 有限 loss，仍有纠错梯度 |
| 所有目标都被忽略 | 明确处理，不静默吞掉 NaN |

这些检查能发现实现错误，不等于证明模型效果好。跑通之后，还得回到数据覆盖、验证集和任务本身。
