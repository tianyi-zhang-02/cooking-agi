# RoPE 与长上下文：能塞进去，就能读懂吗？

**中文** · [English](position-and-context.en.md)

> 最近审阅：2026-10 · 前置：[多头注意力](../core/multi-head-attention.md)

把配置里的 context length 从 8K 改成 128K，模型不报错，只能说明程序容纳了这些位置。它有没有学过这么远的关系，能不能找回中间的证据，还要另测。

先看 RoPE 到底改变了哪个计算。它不是给 token 贴一个编号后就结束了，而是在 attention 的点积里引入位置关系。

## 从二维旋转开始

对二维向量，旋转矩阵为：

$$
R(\phi)=
\begin{bmatrix}
\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi
\end{bmatrix}.
$$

位置 $p$ 的 query 变为 $R(p\theta)q$，位置 $r$ 的 key 变为 $R(r\theta)k$。于是

$$
(R(p\theta)q)^\top R(r\theta)k
=q^\top R((r-p)\theta)k.
$$

这是相对距离怎样进入点积的关键。对固定的未旋转 $q,k$，两者位置同时平移，点积不变；但完整模型的 hidden states 会随上下文变化，不能把这个恒等式说成模型对所有平移都完全不变。

用 $\theta=\pi/2$ 做示意，取 $q=k=[1,0]$。位置 0 与 1 的点积为 0；换成位置 3 与 4，仍为 0。距离变成 2 时点积为 -1。旋转没有改变各自长度，只改变了方向关系。这不是实际模型的频率配置。

## 为什么不只用一个频率？

把 head 的维度拆成多个二维对，每对使用一个频率。常见基础写法是

$$
\theta_j=b^{-2j/d},\quad j=0,\ldots,d/2-1.
$$

$d$ 是这里旋转的维度，$b$ 是基数。假设 $d=4,b=10000$，两对的频率是 1 和 0.01；相隔 10 个位置时，旋转角差分别是 10 和 0.1 弧度。不同频率让模型能利用不同尺度的位置变化。

[RoFormer](https://arxiv.org/abs/2104.09864)给出了旋转式位置编码。实现时还要确认二维配对方式、是否只旋转部分维度、频率约定以及 cache 中的位置偏移。配对顺序与 checkpoint 不一致，shape 一样也会算错。

```python
import math

def rotate_pair(vector, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return [cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1]]

def dot(left, right):
    return sum(first * second for first, second in zip(left, right))

query, key = [1.0, 2.0], [3.0, -1.0]
frequency = 0.2
original = dot(rotate_pair(query, 2 * frequency), rotate_pair(key, 5 * frequency))
shifted = dot(rotate_pair(query, 9 * frequency), rotate_pair(key, 12 * frequency))
relative = dot(query, rotate_pair(key, 3 * frequency))
assert math.isclose(original, shifted, abs_tol=1e-12)
assert math.isclose(original, relative, abs_tol=1e-12)
```

## 外推为什么可能出问题？

训练时只见过短距离，并不保证所有频率在更远距离上都好用。旋转是周期性的，不同距离可能在一些维度上接近；多个频率和学习到的表示共同决定能否区分。

因此不能简单说“RoPE 只依赖相对位置，所以任意长度都成立”，也不能反过来说“某个频率转满一圈就彻底失效”。公式能计算，与训练分布下学到的行为能否推广，是两件事。

检查长上下文时，除了长度，还要变证据位置、干扰内容和所需信息的跨度。只在最后放一个明显关键词，通常不足以证明模型会整合整篇文档。

## 插值、改频率、继续训练，各在做什么？

简单位置插值把位置 $p$ 换成 $p/s$，例如希望长度扩成 4 倍时使用 $s=4$。原来的相邻位置也只差 0.25，缓解远距离超出原范围的问题，同时改变了局部位置分辨率。

| 方法 | 改变什么 | 不能保证什么 |
| --- | --- | --- |
| 不缩放，直接使用更长位置 | 输入范围 | 不保证模型在新距离上稳定 |
| 位置插值 | 把位置压回较短范围 | 不保证短距离关系完全不受影响 |
| 频率相关的缩放，如 YaRN | 不同频段采用不同调整，并配合相应 attention scaling | 不是随便改一个 base 就等价 |
| 长上下文继续训练 | 让模型适应新长度与位置设置 | 要花数据和计算，也需要验证短文本能力 |

[Position Interpolation](https://arxiv.org/abs/2306.15595)与[YaRN](https://arxiv.org/abs/2309.00071)提供了具体方案。这里比较的是设计目的，不复用论文的提升数字作为我们模型的承诺。

## YaRN：哪些频率该缩，哪些先保留？

想把上下文扩成 4 倍，最直接是所有频率都除以 4。但高频分量本来就能区分相邻位置，一起缩小也会改变这些局部关系。YaRN 的分段处理让不同频率承担不同调整，而不是给每个 token 学一套新频率。

一个直观参照是：这对维度在原训练长度 $L$ 内转了多少圈？

$$
r_j=\frac{L\theta_j}{2\pi},\qquad
\theta'_j=\left(\frac{1-\gamma_j}{s}+\gamma_j\right)\theta_j.
$$

$s$ 是长度扩展倍数；$\gamma_j$ 是随 $r_j$ 从 0 过渡到 1 的权重。低频端更多采用插值，高频端更多保留原频率，中间平滑过渡。**高低频说的是向量的不同二维分量，不是文本靠前或靠后的位置**：同一个 token 同时有这些分量。

用一组方便算的阈值演示：$s=4$，在 1 圈以下取 $\gamma=0$，4 圈以上取 1，中间线性变化。这是教学配置，不是某个 checkpoint 的参数。

| 原训练窗口内的圈数 $r$ | $\gamma$ | 新频率 / 原频率 | 直观变化 |
| --- | --- | --- | --- |
| 0.5 | 0 | 0.25 | 用插值，转得更慢 |
| 2.5 | 0.5 | 0.625 | 两端折中 |
| 8 | 1 | 1 | 保留原频率 |

```python
def yarn_frequency_multiplier(rotations, scale=4.0, low=1.0, high=4.0):
    if rotations < 0 or scale < 1 or not 0 <= low < high:
        raise ValueError("Invalid scaling configuration")
    blend = min(1.0, max(0.0, (rotations - low) / (high - low)))
    return (1 - blend) / scale + blend

assert yarn_frequency_multiplier(0.5) == 0.25
assert yarn_frequency_multiplier(2.5) == 0.625
assert yarn_frequency_multiplier(8.0) == 1.0
assert yarn_frequency_multiplier(2.5, scale=1.0) == 1.0
```

这只是频率分段的数学示意。实际实现还可能把过渡写在维度索引上，涉及取整、原训练长度和 checkpoint 配置，不能直接拿这段函数替换模型代码。

YaRN 还调整 attention 的尺度。若两边 Q、K 都乘 $a$，点积就乘 $a^2$；对同一组分数，这会改变 softmax 的尖锐程度。比如分数 `[0,1]` 的概率约为 `[0.269,0.731]`，乘 2 后约为 `[0.119,0.881]`。这和生成时调 token sampling temperature 不是同一个位置的操作。

因此迁移长上下文配置时，要一起检查频率、attention scaling、训练长度和缓存行为。只看到模型里有个 `rope_theta`，并不等于其他改动都可以省略。

### Attention scaling 和 cache 的两个实现细节

[YaRN 原论文 §3.4](https://arxiv.org/html/2309.00071v2#S3.SS4)在 Llama 系列的实验中采用 $a=1+0.1\ln s$，将 Q、K 的幅度各乘 $a$，对应 logits 乘 $a^2$。$s=4$ 时约为 1.2965 倍，$s=1$ 时回到 1；这是一组经验设置，不是所有新模型都必须用的常数。

推导很短：$(aQ)(aK)^\top=a^2QK^\top$。如果只对部分 rotary 维度乘 $a$，就不能再把**整个点积**写成乘 $a^2$。例如 $Q=K=[2,3]$，原点积 13；两维都乘 2 得到 52，只把第二维乘 2 得到 40。迁移到 partial RoPE / MLA 时必须检查缩放放在哪里。

[作者实现](https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py)把过渡区落实到维度索引和取整边界。论文的连续说明、教学函数和某个 checkpoint 的实现，不应不加核对地互换。

另一个问题是动态 scale：如果历史 key 已经按旧频率旋转，新的 query 却按新频率旋转，它们不再使用同一套位置规则。要么保持一个请求内约定不变，要么保留能重新构造 / 旋转历史 key 的状态；不能只改 query。测试时跨过 scale 变化的长度边界，比较完整 prefill 与逐 token decode，而不只测边界之前的短输入。

## DCA：跨过 chunk 边界，距离怎么算？

[Dual Chunk Attention](https://arxiv.org/abs/2402.17463) 不直接把所有位置按相同比例压缩，而是区分同 chunk、前一个 chunk 和更远的 chunk。下面讲的是该论文的位置处理，不代表所有 2026 长上下文模型都采用它。

设原窗口长度为 $c$，chunk 长度为 $s<c$，query 在位置 $i$、key 在 $j\le i$。key 使用 $j\bmod s$；query 的位置随两者关系变化：

| 关系 | query 的局部位置 | 想保留什么 |
| --- | --- | --- |
| 同一个 chunk | $i\bmod s$ | chunk 内的相对距离 |
| 前一个 chunk | $\min(s+i\bmod s,c-1)$ | 边界附近的连续关系 |
| 更早的 chunk | $c-1$ | 避免很大的、未训练过的位置差 |

最终 RoPE 距离是该 query 位置减去 key 局部位置。后一种会牺牲远距离的精细区分；它不是恢复了无限精确的位置。

取 $c=8,s=5$。位置 5 看位置 4，本来相隔 1；如果两边直接取模，会变成 $0-4=-4$。相邻 chunk 的处理给出 $5-4=1$。再往后，位置 8 看位置 4，得到 $7-4=3$，不再等于真实距离 4：这是限制位置范围的代价。

```python
def dca_distance(query_position, key_position, window=8, chunk=5):
    values = (query_position, key_position, window, chunk)
    if any(type(value) is not int for value in values):
        raise ValueError("Expected integer positions and sizes")
    if not 0 <= key_position <= query_position or not 0 < chunk < window:
        raise ValueError("Expected causal positions and chunk < window")
    gap = query_position // chunk - key_position // chunk
    if gap == 0:
        query_local = query_position % chunk
    elif gap == 1:
        query_local = min(chunk + query_position % chunk, window - 1)
    else:
        query_local = window - 1
    return query_local - key_position % chunk

assert dca_distance(4, 3) == 1
assert dca_distance(5, 4) == 1
assert dca_distance(6, 4) == 2
assert dca_distance(8, 4) == 3
assert dca_distance(10, 0) == 7
```

这是单个 query–key 对的位置算例，不是 DCA kernel。完整实现还要对不同区域的结果做**全局归一化**，不能平均各 chunk 已归一化的 attention 输出。例如一个 chunk 有 1 个 value 0，另一个有 3 个 value 4，score 全为 0，正确输出是 3，不是两个局部输出的平均值 2。

DCA 不等于只看几个 chunk：它仍可读取全部历史，也没有自动消除 dense attention 的二次计算。验证时除长文任务，还应专门移动证据跨越 chunk 边界，检查边界、padding、cached / uncached logits。训练之外能运行，不等于换到任何 checkpoint 都不掉质量。[作者实现](https://github.com/HKUNLP/ChunkLlama) 提供了具体接法。

## 位置问题解决后，还要付哪些成本？

固定模型与 batch，从 8K 到 128K，长度是 16 倍：普通 KV 存储约为 16 倍，dense prefill 的 attention 配对数约为 256 倍。整体运行时间不必正好是这个倍数，因为 FFN、kernel、并行与内存访问也参与其中。

“能访问长文”还不等于“必须一次塞进全部长文”。检索后读证据更省，但可能漏掉需要联合理解的信息；摘要更短，但可能丢细节。选择取决于任务，不是 context window 越大就一定越好。

## 怎么做一次像样的长文本检查？

固定模型版本、RoPE 配置、tokenizer 与解码设置。把同一关键证据放到开头、中间、结尾，增加相似干扰项；再加入必须联合两处证据才能回答的问题，不能只做单点查找。

同时报告准确率、引用是否真支持答案、TTFT、峰值显存和短文本回归。cache 开关前后的 logits 也要在容差内对齐，尤其检查 padding、位置偏移和多 token decode 的 mask。

这时才有资格说清楚：增加的是输入容量、可用的检索距离，还是跨长文的任务能力。

如果接着好奇“那为什么有些模型不加 RoPE？”，读 [NoPE 与顺序信息](nope-and-order.md)。从一个换序算例开始，再看 causal mask、循环状态和 Kimi K3 的混合设计。
