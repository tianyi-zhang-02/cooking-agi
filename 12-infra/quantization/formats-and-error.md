# 量化：数字压进几个格子时，误差从哪来

**中文** · [English](formats-and-error.en.md)

> 阅读时间：约 11 分钟 · 草稿，待审核

同一组 8 个权重，整组共用一个 scale 压成 INT4，均方误差（MSE）是 0.0042；把其中一个 1.40 换成 14.0，其余 7 个数全被压成 0。**量化误差由格子的步长决定，步长又由共用同一个 scale 的那组数里最大的那个决定。格式决定格子怎么排（整数等距，浮点按比例），粒度决定谁和谁共用一个 scale；离群值（outlier）的伤害，就是它把整组的步长撑大了。**

> **待审核草稿。** 这一篇还没有经过站主审阅，事实、取舍和措辞都可能修改；引用前请回到文中链接的一手资料核对。

FP16、BF16 的位数和舍入见[精度与显存](../../00-foundations/deep-dives/precision-and-memory.md)，各代 Tensor Core 支持的格式表和一个 NVFP4 小例子见 [GPU 架构](../gpu-architecture.md)，这里不重复。本文的数字除注明出处的以外，都是可以手算的教学例子。

## 一个实数怎样存成整数：scale 和 zero-point

把实数 $x$ 存成 $b$ 位整数 $q$，最常用的是仿射映射：

$$
q=\operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{x}{s}\right)+z,\ q_{\min},\ q_{\max}\right),\qquad \hat x=s\,(q-z).
$$

$s$ 是 scale，也就是相邻两个格子之间的步长；$z$ 是 zero-point，即实数 0 对应的那个整数；$\hat x$ 是读回来的近似值。[Jacob 等（Google，2017-12）](https://arxiv.org/abs/1712.05877)这样定义 $z$，理由是让 0 能被精确表示，因为很多算子要在边界补零。两种常见取法：

- **对称**：$z=0$，INT4 取 $-7$ 到 $7$（为了正负对称，不用 $-8$），张量不全为零时 $s=\max|x|/7$。
- **非对称**：用 $0$ 到 $15$ 的无符号 4 位码值。先把校准区间扩到包含 0：$a=\min(0,\min x)$，$b=\max(0,\max x)$。再取 $s=(b-a)/15$，$z=\operatorname{clamp}(\operatorname{round}(-a/s),0,15)$。这样 zero-point 落在合法码值范围里，实数 0 才能精确还原。

全零张量可以统一取 $s=1$、$z=0$，码值全为 0，避免除零。所有值都相同但不为零时，扩过的非对称区间也能正常算。比如校准数据是 $[1,2]$，应该用 $[0,2]$，得到 $s=2/15$、$z=0$。若直接用 $[1,2]$，就会算出 $z=-15$；实数 0 的码值被截到 0 后，读回来反而是 **1**。

只要 $x$ 在范围内，舍入误差不超过 $s/2$；如果误差在 $[-s/2,s/2]$ 上均匀分布，均方误差约为 $s^2/12$。**步长减半，误差平方降到四分之一。**

手算一次。6 个非负数 $[0,\ 0.2,\ 0.5,\ 0.9,\ 1.3,\ 2.0]$（比如某个激活之后的值）。对称取法 $s=2.0/7\approx0.286$，码值只用到 0 到 7，负的那一半格子白白空着，MSE 约 0.00514。非对称取法 $s=2.0/15\approx0.133$、$z=0$，码值是 $[0,2,4,7,10,15]$，MSE 约 0.00130，约为前者的四分之一，接近步长之比的平方 $(0.286/0.133)^2\approx4.6$（只有 6 个数，$s^2/12$ 只是近似）。

## 手算 8 个权重：分组能省下多少误差

教学权重 $w=[0.12,\ -0.53,\ 0.88,\ -0.07,\ 0.31,\ 1.40,\ -0.95,\ 0.02]$，对称 INT4。

- **per-tensor**：8 个数共用一个 scale，$s=1.40/7=0.2$。
- **per-group，每组 4 个**：前 4 个的最大绝对值是 0.88，$s=0.88/7\approx0.1257$；后 4 个仍是 $s=0.2$。

| 原值 | per-tensor 码值 | 读回 | per-group 码值 | 读回 |
| ---: | ---: | ---: | ---: | ---: |
| 0.12 | 1 | 0.2 | 1 | 0.1257 |
| −0.53 | −3 | −0.6 | −4 | −0.5029 |
| 0.88 | 4 | 0.8 | 7 | 0.88 |
| −0.07 | 0 | 0 | −1 | −0.1257 |
| 0.31 | 2 | 0.4 | 2 | 0.4 |
| 1.40 | 7 | 1.4 | 7 | 1.4 |
| −0.95 | −5 | −1.0 | −5 | −1.0 |
| 0.02 | 0 | 0 | 0 | 0 |

per-tensor 的 MSE 是 0.0042，和 $s^2/12=0.0033$ 同一量级；分组以后降到 0.00186。代价是多存一个 scale：真实做法里组长常取 128，每组一个 16 位 scale，每个权重平均多 $16/128=0.125$ 位。

### 换进一个离群值

把 1.40 换成 14.0。per-tensor 的 scale 变成 $14/7=2$，其余 7 个数除以 2 以后的绝对值都小于 0.5，全部舍成 0。前后都只看**同样的 7 个未改动的数**，MSE 从 0.0048 涨到约 0.29623，约 61.7 倍。如果都按 8 个数算，则是 0.0042 → 0.2592，倍数相同，因为最大值在两种情况下都能精确还原。分组时，前一组不受影响，后一组的另外 3 个数被压成 0，那 7 个未改动的数的 MSE 约 0.143。**分组不能消除离群值，只能把它的伤害关在自己那一组里。**

[SmoothQuant（2022-11）](https://arxiv.org/abs/2211.10438)把这件事写成一个数：8 位 per-tensor 量化时，最大值为 $m_i$ 的通道只能用到约 $2^8\cdot m_i/m$ 个格子，$m$ 是整个矩阵的最大值。他们在 OPT 等模型的激活里看到离群值比多数值大约 100 倍，非离群通道只剩两三个格子可用。离群值也不是随机出现的：

- [LLM.int8()（2022-08）](https://arxiv.org/abs/2208.07339)报告，模型到 6.7B 左右，所有层、约 75% 的序列位置都出现大幅值特征，却集中在整个模型的 6 个特征维度上（约占输入特征的 0.1%），幅值可达其他维度的 20 倍；把这些维度置零，验证集困惑度变差 600%–1000%。
- SmoothQuant 观察到离群值固定在少数通道里，每个 token 都在同样的通道上很大；权重的分布则平坦得多，好量化得多。

所以同一个模型，权重往往压到 4 位还行，激活压到 8 位就可能出问题。下一篇的 SmoothQuant 和 AWQ 都利用了“离群值固定在少数通道”这一点。

## 粒度：谁和谁共用一个 scale

一个线性层 $Y=XW$，$X$ 是 $T\times C_i$（$T$ 个 token，$C_i$ 个输入通道），$W$ 是 $C_i\times C_o$。

| 粒度 | 共用一个 scale 的是 | 每个值多出的位数（16 位 scale） | 能不能在整数 GEMM 之后再乘回去 | 常见用途 |
| --- | --- | --- | --- | --- |
| per-tensor | 整个矩阵 | ≈ 0 | 能 | 静态 FP8 激活 |
| per-channel（权重的输出通道） | $W$ 的一列 | $16/C_i$，$C_i=4096$ 时约 0.004 | 能 | INT8、FP8 权重 |
| per-token（激活的一行） | 一个 token 的所有通道 | ≈ 0 | 能 | 动态激活 scale |
| per-group / per-block（沿 $C_i$） | 连续 16、32 或 128 个值 | g128 约 0.125 | 不能，要在 kernel 里先反量化，或用支持块缩放的 MMA | INT4 权重、MX 格式、NVFP4 |
| per-channel（激活的输入通道） | 一个输入通道的所有 token | — | 不能 | SmoothQuant 改用等价变换绕开 |

关键在“能不能乘回去”。SmoothQuant 的推导是，只有外侧两个维度（token 维 $T$ 和输出通道 $C_o$）的 scale 能放到矩阵乘结束以后：

$$
Y=\operatorname{diag}(\Delta_X)\,\big(\bar X_{\mathrm{INT8}}\,\bar W_{\mathrm{INT8}}\big)\,\operatorname{diag}(\Delta_W).
$$

$\bar X$、$\bar W$ 是整数码值，$\Delta_X$ 是每个 token 一个的 scale，$\Delta_W$ 是每个输出通道一个的 scale。沿内侧维度 $C_i$ 分组，scale 在点积求和的中途就变了，Tensor Core 的整数累加里插不进去：要么像 [GPTQ](https://arxiv.org/abs/2210.17323) 的 kernel 那样，读权重时就地反量化（dequantize）成 FP16 再乘，要么由硬件在 MMA 指令里直接乘块 scale。后一种在 Blackwell 上有了：[PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 里，数据中心 Blackwell（sm_100a）的 `tcgen05.mma`（§9.7.18.10.7）和 RTX Blackwell（sm_120a）的 `mma.sync`（§9.7.16.3）都有块缩放版本，先算 $(A\cdot \text{scale}_A)(B\cdot\text{scale}_B)$ 再加到累加值上；MX 格式用 UE8M0 scale，NVFP4 用 UE4M3 scale。

## 浮点格式：格子按比例排

INT 的格子等距，FP 的格子在每个 2 的幂区间里等分，越靠近 0 越密。8 位浮点有两种（[Micikevicius 等，2022-09](https://arxiv.org/abs/2209.05433)；数值取自 [OCP MX 规范 v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 表 2，该表转载自 OCP FP8 规范）：

| 格式 | 符号 / 指数 / 尾数位 | 指数偏置 | 最大值 | 最小正规数 | 相邻两数的相对间隔 |
| --- | --- | ---: | ---: | ---: | ---: |
| E4M3 | 1 / 4 / 3 | 7 | 448 | $2^{-6}$ | $2^{-3}$，即 12.5% |
| E5M2 | 1 / 5 / 2 | 15 | 57,344 | $2^{-14}$ | $2^{-2}$，即 25% |
| INT8（对称，对照） | — | — | $127s$ | — | 处处都是 $s$ |

E4M3 不表示无穷大，NaN 只占一个尾数编码，省下的编码用来把最大值推到 448；E5M2 沿用 IEEE 754 的特殊值。E4M3 在 8 到 16 之间的间隔是 1，在 32 到 64 之间是 4。**缩放后的数落在正规数范围（normal range）内，且没有溢出或截断时**，就近舍入的相对误差上界约为 6.25%，绝对误差则随数值变大。

次正规数（subnormal）和下溢不受这个界约束。scale 为 1 时，E4M3 把 $0.001$ 舍入成 $2^{-9}=0.001953125$，相对误差是 95.3125%；更小的非零数可能直接舍成 0，相对误差达到 100%。哪些数会落到这里，和 scale 的选择有关。

手算对比。4 个数 $[0.02,\ -0.3,\ 1.7,\ 12.0]$，都用 per-tensor scale 把最大值对到格式的顶端：E4M3 取 $s=12/448\approx0.0268$，INT8 取 $s=12/127\approx0.0945$。

| 原值 | E4M3：$x/s$ → 舍入 → 读回 | 相对误差 | INT8：码值 → 读回 | 相对误差 |
| ---: | --- | ---: | --- | ---: |
| 0.02 | 0.747 → 0.75 → 0.0201 | 0.45% | 0 → 0 | 100% |
| −0.3 | −11.2 → −11 → −0.2946 | 1.8% | −3 → −0.2835 | 5.5% |
| 1.7 | 63.47 → 64 → 1.714 | 0.84% | 18 → 1.701 | 0.05% |
| 12.0 | 448 → 448 → 12.0 | 0 | 127 → 12.0 | 0 |

靠近最大值的地方 INT8 更准，远小于最大值的地方 FP8 更准。Qualcomm AI Research 的 [van Baalen 等（2023-03）](https://arxiv.org/abs/2303.17951)总结为“all about the outliers”：分布里有明显离群值时 FP8 误差更小，分布规整时 INT8 更准；他们还估算专用硬件里 FP8 乘加单元比 INT8 低效 50%–180%，因此认为端侧推理不值得为 FP8 专门做一套实现。数据中心 GPU 上怎么选，还要看 Tensor Core 支持什么，见下文的硬件一节。

同叫 FP8，位模式也不一定一样。AMD 的 [HIP 7.16 文档](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/fp8_numbers.html)写明，CDNA 3（MI300 系列）硬件支持 FNUZ 变体，CDNA 4（MI350 系列）支持 OCP 变体。按 [ONNX float8 说明](https://onnx.ai/onnx/technical/float8.html)的定义，E4M3FNUZ 的指数偏置是 8（OCP 是 7），没有无穷大和负零，NaN 占用负零的位模式，所以全 1 的 `S.1111.111` 也是普通数。由这个定义推出最大值是 $1.875\times2^{15-8}=240$（ONNX 表里只给位模式），OCP 的 E4M3 是 448。同一个位模式按 FNUZ 读，数值正好减半，所以 OCP 格式的 checkpoint 要把 scale 乘 2、另行处理负零，才能当 FNUZ 用（这是按位定义的推算）。

## 只剩 4 位时为什么要块缩放：MXFP4 与 NVFP4

元素只剩 4 位时，E2M1 只能表示 0、0.5、1、1.5、2、3、4、6 及其负数，一个张量一个 scale 是不够的，必须小块各自缩放（block scaling）。

| 格式 | 元素 | 块大小 | 块 scale | 每个值平均位数 |
| --- | --- | ---: | --- | ---: |
| MXFP8 | E4M3 或 E5M2 | 32 | E8M0（只能是 2 的幂） | 8.25 |
| MXFP4 | E2M1 | 32 | E8M0 | 4.25 |
| NVFP4 | E2M1 | 16 | E4M3，再乘一个张量级 FP32 scale | 4.5 |
| INT4 g128（对照） | INT4 | 128 | FP16 | 4.125 |

MX 两行出自 OCP MX 规范 v1.0 的表 1（同一张表还有 MXFP6 和 MXINT8）；NVFP4 出自 [NVIDIA 的 NVFP4 博客（2025-06-24）](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)。MX 规范 §6.3 给出实现至少应支持的一种转换：块 scale $X$ 取“不超过块内最大绝对值的最大 2 的幂”，除以“元素格式能表示的最大 2 的幂”（E2M1 是 4）；元素取 $V_i/X$ 舍入，超过元素最大值的截到最大值。

手算一个块 $[0.3,\ -1.2,\ 2.4,\ 5.0]$（教学例子，只取 4 个值，忽略 NVFP4 的张量级 scale）：

- **MXFP4**：不超过 5.0 的最大 2 的幂是 4，$X=4/4=1$。元素就是原值：0.3 → 0.5，−1.2 → −1，2.4 → 2，5.0 正好在 4 和 6 中间，按就近取偶得 4。读回 $[0.5,-1,2,4]$，MSE 0.31。最大值丢了 20%，最高的格子 6 根本没用上。
- **NVFP4**：scale 取 $5.0/6\approx0.833$，存成 E4M3 最近的是 0.8125。元素 $[0.369,-1.477,2.954,6.154]$ 舍入为 $[0.5,-1.5,3,6]$，读回 $[0.406,-1.219,2.438,4.875]$，MSE 约 0.0072，比 MXFP4 低约 43 倍，代价是每个值多 0.25 位。

2 的幂 scale 只能在 0.5、1、2 之间跳，块内最大值落在两个 2 的幂之间时，要么浪费顶端的格子，要么被截断。把块最大值换成 3.5 试试：$X=2/4=0.5$，$3.5/0.5=7$ 超过 6，被截成 6，读回 3.0。规范的取法相当于把 scale 向下取整，所以最大的那个元素可能被截。NVIDIA 的 [MXFP8 训练配方（2025-05）](https://arxiv.org/abs/2506.08027)改为把 scale 的指数向上取整，理由正是保证缩放后没有值超出元素的最大值。

## 哪代硬件能直接算哪种格式

一种格式省显存，不等于它算得快：Tensor Core 不支持，就只能先反量化成 16 位再算。截至 2026 年 10 月，[CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)的 Tensor Core 输入类型表（页面 2026-09-10 更新）里，8 位及以下的部分是这样的；GPU 和 compute capability 的对应取自 [CUDA GPU 列表](https://developer.nvidia.com/cuda-gpus)：

| Compute capability | 例子 | 8 位及以下的 Tensor Core 输入 |
| --- | --- | --- |
| 7.5、8.0、8.6 | T4、A100、A10 | INT8、INT4 |
| 8.9 | L4、L40S、RTX 4090 | FP8、INT8、INT4 |
| 9.0 | H100、H200 | FP8、INT8（没有 INT4） |
| 10.0、10.3、12.x | B200、B300、RTX PRO 6000 Blackwell、RTX 5090 | FP8、FP6、FP4、INT8 |

9.0 一行和 [H100 白皮书 V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)一致：它列出的 MMA 数据类型是 FP8、FP16、BF16、TF32、FP64 和 INT8，没有 INT4。

表里有，也不代表够快。截至 2026 年 10 月，NVIDIA 的 [HGX 页面](https://www.nvidia.com/en-us/data-center/hgx/)（8 卡整机，标的是稀疏口径，dense 减半）上，HGX B300 的 INT8 是 3 POPS，FP8 是 72 PFLOPS；HGX B200 的 INT8 和 FP8 都是 72。同一台 B300 上 INT8 只有 FP8 的 1/24，所以在 B300 上 INT8 W8A8 已经不是加速路线，8 位计算应该走 FP8。各代吞吐的对比见 [NVIDIA 各代 GPU](../nvidia-generations.md)。AMD 这边，ROCm 10.1 文档的[精度支持页](https://rocm.docs.amd.com/en/latest/reference/precision-support.html)（截至 2026 年 10 月）列出：CDNA 3（MI300 系列）的矩阵核心支持 INT8 和 FP8，FP8 只有 FNUZ 变体；CDNA 4（MI350 系列）支持 INT8 和 OCP 的 FP8、FP6、FP4。[MI350 架构页](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html)还列出 MXFP8、MXFP6、MXFP4，峰值表里 INT8 和 FP8 同速。硬件不支持某种格式时，引擎往往退回只压权重，第 3 篇有 vLLM 的例子。

## 自己算一遍

```python
def quantize_sym(values, bits=4):
    qmax = 2 ** (bits - 1) - 1
    maximum = max(abs(value) for value in values)
    scale = maximum / qmax if maximum else 1.0
    codes = [max(-qmax, min(qmax, round(value / scale))) for value in values]
    return scale, codes, [code * scale for code in codes]

def quantize_asym(values, bits=4):
    qmax = 2 ** bits - 1
    lower = min(0.0, min(values))
    upper = max(0.0, max(values))
    scale = (upper - lower) / qmax if upper > lower else 1.0
    zero_point = max(0, min(qmax, round(-lower / scale)))
    codes = [max(0, min(qmax, round(value / scale) + zero_point)) for value in values]
    return scale, zero_point, codes, [scale * (code - zero_point) for code in codes]

def mse(expected, restored):
    return sum((value - estimate) ** 2 for value, estimate in zip(expected, restored)) / len(expected)

weights = [0.12, -0.53, 0.88, -0.07, 0.31, 1.40, -0.95, 0.02]
scale, codes, restored = quantize_sym(weights)
assert codes == [1, -3, 4, 0, 2, 7, -5, 0]
assert abs(mse(weights, restored) - 0.0042) < 1e-9

grouped = quantize_sym(weights[:4])[2] + quantize_sym(weights[4:])[2]
assert abs(mse(weights, grouped) - 0.0018592) < 1e-6

outlier_weights = weights[:5] + [14.0] + weights[6:]
_, codes_out, restored_out = quantize_sym(outlier_weights)
assert codes_out == [0, 0, 0, 0, 0, 7, 0, 0]
assert abs(mse(outlier_weights, restored_out) - 0.2592) < 1e-9

unchanged_weights = weights[:5] + weights[6:]
before = mse(unchanged_weights, restored[:5] + restored[6:])
after = mse(unchanged_weights, restored_out[:5] + restored_out[6:])
assert abs(before - 0.0048) < 1e-9
assert abs(after - 0.2962285714) < 1e-9
assert abs(after / before - 61.7142857143) < 1e-9

scale, zero_point, codes, restored = quantize_asym([1.0, 2.0])
assert abs(scale - 2.0 / 15) < 1e-12
assert zero_point == 0
assert scale * (zero_point - zero_point) == 0.0
assert quantize_asym([0.0, 0.0]) == (1.0, 0, [0, 0], [0.0, 0.0])
```

把 `bits` 改成 8，或者把组长改成 2，看 MSE 怎样变。Python 的 `round` 是就近取偶，和多数硬件的默认舍入一致。

## 这些格子说明不了什么

| 先试什么 | 应该看到什么 | 不能据此推出什么 |
| --- | --- | --- |
| 8 个权重从 per-tensor 改成每 4 个一组 | MSE 从 0.0042 降到 0.00186 | 模型准确率也按同样比例变好 |
| 同一组数分别用 E4M3 和 INT8 | 小数值 FP8 准，大数值 INT8 准 | 哪种格式对整个模型更好；要看每层的分布 |
| 同一个块从 MXFP4 换成 NVFP4 | 本例 MSE 低约 43 倍 | 端到端质量也差 43 倍；NVFP4 每个值多花 0.25 位 |
| 给离群值所在的组单独一个 scale | 只有这一组受损 | 离群值可以随便截掉：LLM.int8() 显示置零这些维度会让困惑度大幅变差 |

单个张量的 MSE 只是起点。它没告诉你误差怎样穿过几十层累积，同一个输出里的误差是相互抵消还是同向叠加（下一篇 GPTQ 的例子里，两个同向误差让输出误差翻倍），也没说明累加用了多少位精度（第 3 篇讲 DeepSeek-V3 在 FP8 GEMM 里遇到的问题），更没说明哪些层最敏感。下一篇：[只压权重，与训练后量化](weight-only-and-ptq.md)。
