# 量化：FP8 推理与 KV cache 量化

**中文** · [English](fp8-and-kv-cache.en.md)

> 阅读时间：约 11 分钟 · 草稿，待审核

本系列的 8B 教学模型在 128K 上下文时，一条序列的 BF16 KV 就有 16 GiB，比 BF16 权重本身（16 GB，约 14.9 GiB）还大；一张 80 GB 的 H100 上，只放得下 3 条这样的序列。KV 改存 FP8，就能放 6 条；仍是 3 条时，每步要读的字节少 38%。**长上下文时，KV cache 的字节比权重还多，压 KV 比压权重更直接地决定能同时服务几条请求、每一步读多少字节。FP8 是目前最常用的 8 位起点，权重、激活、KV 都能用，稳不稳主要看 scale 怎么定；KV 再压到 4 位、2 位，key 要按通道、value 要按 token 分开量化，而且只能拿长上下文任务来验收。**

> **待审核草稿。** 这一篇还没有经过站主审阅，事实、取舍和措辞都可能修改；引用前请回到文中链接的一手资料核对。

E4M3、E5M2 的格子和块缩放见[格式与误差](formats-and-error.md)，KV 显存的基本公式见 [KV cache](../../00-foundations/deep-dives/kv-cache-and-inference.md)。

## FP8 W8A8：权重和激活都是 8 位

FP8 W8A8 同时换来两样东西：权重字节减半，decode 跟着快（上一篇的表里 batch 1 是 2.54 对 4.77 ms）；激活也是 FP8，矩阵乘走 FP8 Tensor Core，H100 SXM5 的 dense 峰值从 BF16 的 989.4 TFLOPS 翻到 1,978.9 TFLOPS（[白皮书 V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)），prefill 和大 batch 也受益。前提是硬件有 FP8 Tensor Core，即 compute capability 8.9 以上；[vLLM v0.31.0 的 FP8 文档](https://docs.vllm.ai/en/v0.31.0/features/quantization/llm_compressor/fp8/)写明，Turing、Ampere 上同一个 FP8 模型只能用 Marlin kernel 跑 W8A16，权重省了，算力没有涨。

scale 的常见定法，就是这份文档推荐的 `FP8_DYNAMIC` 方案：权重按输出通道各一个静态 scale；激活按 token 动态算 scale，不需要校准数据。另一种在线模式 `fp8_per_tensor` 给权重一个 per-tensor scale，激活的最大最小值每次前向都现算，文档说这种模式下延迟改善有限。文档自报 FP8 让模型显存减半、吞吐最多提高 1.6 倍。

质量上，[Kurtic 等（2024-11，ACL 2025）](https://arxiv.org/abs/2411.02355)的 FP8 W8A8 只用 RTN：权重 per-output-channel 对称量化，激活 per-token 动态量化，不用校准数据。在 Llama-3.1 8B、70B、405B 上，Open LLM Leaderboard v2 的平均分恢复率（量化后得分 ÷ BF16 得分）分别是 101.2%、100.0%、99.9%，论文称之为近似无损；同一篇里调好的 INT8 W8A8 掉 1–3%。

### 动态 scale，还是静态 scale

E4M3 最大是 448。两个 token 的激活（教学数字）：A 是 $[0.01,\ -0.8,\ 3.2]$，B 是 $[2.0,\ -45,\ 180]$。

- **按 token 动态**：A 的 scale 是 $3.2/448\approx0.00714$，B 的是 $180/448\approx0.402$。每个 token 的最大值都正好落在 448 上，三个数都读回得很准，A 里的 0.01 读回 0.0098。
- **静态 per-tensor**：假设校准时见过的最大值是 120，$s=120/448\approx0.268$。A 没问题，0.01 读回 0.0105，误差 4.6%，FP8 的指数位让小数值仍有相对精度。B 的 180 除以 $s$ 是 672，超过 448，按饱和模式截到 448，读回 120，少了三分之一。

动态 scale 的代价是每次前向都要对每个 token 求一次最大值；静态 scale 运行时不花钱，但遇到比校准集更极端的输入就会截断。截断的行为也要看配置：OCP 的 FP8 转换要求同时支持饱和与溢出两种模式，E4M3 在溢出模式下超出范围会变成 NaN（[OCP MX 规范 v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 表 3 转载）。[SmoothQuant](https://arxiv.org/abs/2211.10438) 的测量里，静态 per-tensor 的 O3 比动态的 O1 快，所以只要精度允许，他们建议用更粗的粒度。

## 训练也用 FP8 或更低：多出来的几个问题

推理只需要前向，训练还要让梯度和权重更新在低精度下不出错，问题多了几样。这里只列出各家的做法；BF16 下小更新被舍掉的例子见[精度与显存](../../00-foundations/deep-dives/precision-and-memory.md#rounded-away-update)，原理相同。

| 做法 | 出处 | scale 怎么定 | 哪些不用低精度 | 自报结果 |
| --- | --- | --- | --- | --- |
| FP8 delayed scaling | [Transformer Engine 2.18 文档](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/examples/fp8_primer.html) | 用前几次迭代的最大绝对值（amax history）定本次 scale；HYBRID 格式前向 E4M3、反向 E5M2 | 由框架决定 | — |
| 细粒度 FP8 | [DeepSeek-V3（2024-12）](https://arxiv.org/abs/2412.19437) §3.3 | 激活每 1×128、权重每 128×128 一个 scale，在线算当前最大值；全部用 E4M3 | embedding、输出头、MoE gating、归一化、attention；主权重、梯度用 FP32，AdamW 的矩估计用 BF16 | 在约 16B 和 230B 的验证模型上，相对 BF16 的 loss 相对误差一直低于 0.25%；之后用于 671B MoE |
| MXFP8 | [Mishra 等（NVIDIA，2025-05）](https://arxiv.org/abs/2506.08027) | 每 32 个值一个 E8M0 scale，指数向上取整；权重、激活、激活梯度都用 E4M3 | attention 的矩阵乘和 softmax、embedding、输出层 | 最大 8B、15T token（在 Hopper 上模拟 MXFP8），验证困惑度与 BF16 相差不到 0.5% |
| NVFP4 | [NVIDIA（2025-09）](https://arxiv.org/abs/2509.25149)、TE 2.18 | 每 16 个值一个 E4M3 scale 加张量级 FP32 scale；权重用 16×16 的二维块；梯度随机舍入（stochastic rounding）；随机 Hadamard 变换压离群值，只用在算权重梯度的那一步 | 约 16% 的线性层（前 2 个和最后 8 个 block）保持 BF16；TE 建议最后几层用 MXFP8，但不自动做 | 12B（Mamba 与 Transformer 混合）、10T token，loss 和下游准确率与 FP8 基线相当 |

两个细节值得单独说。第一，累加精度。DeepSeek-V3 发现 H800 上 FP8 GEMM 的累加只保留约 14 位，在 $K=4096$ 的随机矩阵测试里最大相对误差接近 2%；他们每累加 128 个元素（相当于 4 条 WGMMA 指令），就把部分和搬到 CUDA core 上用 FP32 累加。第二，转置。TE 文档说明，Blackwell 的 Tensor Core 要求 MXFP8 数据沿归约维度连续，而反向传播要用转置后的矩阵，转置 MXFP8 数据等于重新量化，所以 TE 从原始高精度张量分别量化出正常和转置两份。NVFP4 的权重用 16×16 二维块，也是为了让一个张量和它的转置共用同一组 scale。

## KV cache：长上下文时真正的大头

教学配置：32 层、8 个 KV head、head dim 128。每个 token 的 K 和 V 合起来有 $2\times32\times8\times128=65{,}536$ 个值：

| KV 格式 | 每个值平均位数 | 每个 token | 128K token 一条序列 |
| --- | ---: | ---: | ---: |
| BF16 | 16 | 128 KiB | 16 GiB |
| FP8，per-tensor scale | 8 | 64 KiB | 8 GiB |
| INT4，每 32 个一组，FP16 scale 和 zero-point | 5 | 40 KiB | 5 GiB |
| INT2，同样分组 | 3 | 24 KiB | 3 GiB |

INT4 和 INT2 两行按 [KIVI](https://arxiv.org/abs/2402.02750) 的分组方式（组长 32）估算，每组多存一个 FP16 scale 和一个 FP16 zero-point，每个值平均多 $32/32=1$ 位；KIVI 另外把最近 128 个 token 保持全精度，这部分 BF16 只有 16 MiB，表里没算。

放进 H100 SXM5 的 80 GB，给运行时留 6 GB（教学假设），看能放几条 128K 的序列：

| 权重 | BF16 KV | FP8 KV | INT4 KV |
| --- | ---: | ---: | ---: |
| BF16（16 GB） | 3 条 | 6 条 | 10 条 |
| FP8（8.5 GB，输出层 BF16） | 3 条 | 7 条 | 12 条 |

batch 3、每条 128K 时，每一步 decode 要读权重加 3 份 KV。按 3,352 GB/s 算下限：

| 权重 + KV | 每步读的字节 | 每步至少 | KV 占读取量 |
| --- | ---: | ---: | ---: |
| BF16 + BF16 | 67.5 GB | 20.2 ms | 76% |
| FP8 + BF16 | 60.0 GB | 17.9 ms | 86% |
| BF16 + FP8 | 41.8 GB | 12.5 ms | 62% |
| FP8 + FP8 | 34.3 GB | 10.2 ms | 75% |
| BF16 + INT4 | 32.1 GB | 9.6 ms | 50% |

在这个长度上，只压权重从 20.2 ms 到 17.9 ms，只压 KV 到 12.5 ms：**谁占字节的大头，就先压谁。** decode 时 attention 部分的算术强度和 batch 无关，只有每字节几次运算，见[性能](../performance.md#decode-batching)，所以少读的字节几乎都会变成时间。短上下文时情况相反：2K token 时一条序列的 BF16 KV 只有 0.27 GB，权重才是大头。

### key 和 value 为什么要分开量化

KIVI（2024-02，ICML 2024）统计了 KV cache 的分布：key 里有少数几个固定通道的幅值特别大，value 没有明显的离群模式；而 attention 的输出主要是少数几个重要 token 的 value 的加权和，所以 value 按 token 量化，误差只留在各自的 token 里。于是他们的方案是 **key 按通道（per-channel）、value 按 token（per-token）**。

**手算一次。** 4 个 token、4 个通道的 key，第 3 个通道是离群通道（教学数字）；query 取 $q=[1,\ 1,\ 0.1,\ 1]$；非对称 2 bit，每组 4 个格子。

$$
K=\begin{bmatrix}0.4&-0.2&9.0&0.1\\-0.3&0.5&11.0&-0.4\\0.2&-0.6&10.0&0.3\\0.5&0.1&12.0&-0.2\end{bmatrix},\qquad Kq=\begin{bmatrix}1.2\\0.9\\0.9\\1.6\end{bmatrix}.
$$

- **按 token**：第 1 行最小 −0.2、最大 9.0，步长 $9.2/3\approx3.07$，格子是 $-0.2,\ 2.87,\ 5.93,\ 9.0$，三个普通通道全部变成 −0.2。四个分数变成 $[0.3,\ -0.1,\ -0.8,\ 0.6]$，原来分数相同的第 2、3 个 token 被拉开，最大误差 1.7。
- **按通道**：第 1 列 $[0.4,-0.3,0.2,0.5]$ 自己一组，步长 0.267；离群通道也自己一组。分数变成 $[1.23,\ 0.90,\ 0.93,\ 1.67]$，最大误差 0.067。

两种方式下 $K$ 的相对误差（Frobenius 范数）是 8.6% 对 0.6%，看起来都不算大，但按 token 时分数几乎全错：离群通道撑大了范数，query 却主要看其他通道。**量 KV 量化的误差，要看 attention 分数和输出，不能只看 K 本身的相对误差。**

KIVI 的实测（Llama-2-7B，表 3）：16 位时 GSM8K 13.50；用模拟量化把所有 token 都压到 2 bit 时，key 按通道、value 按 token 是 5.76，两者都按 token 只剩 0.83；完整的 KIVI-2（组长 32，最近 128 个 token 保持全精度）是 12.74。LongBench 平均分 16 位 44.52、KIVI-2 44.27（表 4）。论文自报连同权重的峰值显存少 2.6 倍，batch 最多大 4 倍，吞吐提高 2.35–3.47 倍。[KVQuant（2024-01，NeurIPS 2024）](https://arxiv.org/abs/2401.18079)也按通道量化 key，并在 RoPE 之前量化 key 以减小位置编码对量化的影响，再加上按层的非均匀格子和逐向量的离群值分离；自报 3 bit 时 WikiText-2 和 C4 困惑度损失小于 0.1，LLaMA-7B 在一张 A100-80GB 上能跑 100 万 token 上下文、8 卡上 1,000 万。

### 在引擎里怎么开

截至 2026 年 10 月，vLLM v0.31.0 的 [Quantized KV Cache 文档](https://docs.vllm.ai/en/v0.31.0/features/quantization/quantized_kvcache.html)列出 `kv_cache_dtype` 可取 `auto`、`fp8`、`fp8_e4m3`（CUDA 11.8 以上和 ROCm）、`fp8_e5m2`（CUDA 11.8 以上），没有列出 INT4、INT2（API 参考里能看到一个 INT4 KV 的 attention 算子，但功能文档没有给用法）。几个要点：

- 不做校准时，**所有 scale 都设成 1.0**；文档推荐用 llm-compressor 在数据上校准。
- scale 可以每个张量一个（Q、K、V 各一个），也可以每个 attention head 一个，后者只在 Flash Attention backend 上可用，并且要校准。
- 用 Flash Attention 3 backend 时，attention 本身也在 FP8 里算，query 也会被量化成 FP8。
- 文档提醒 sliding-window 等层对 KV 量化更敏感，可以用 `--kv-cache-dtype-skip-layers` 让这些层保留原精度。

scale 为 1.0 意味着什么，可以按 E4M3 推一下（这是本文的推算，不是文档的说法）：绝对值超过 448 的 K、V 会被截断；最小的次正规数（subnormal）是 $2^{-9}\approx0.00195$，0.001 会被舍成它，误差近一倍，再小一点就变成 0。所以未校准的 FP8 KV 安不安全，取决于这个模型的 K、V 落在什么范围，要先统计一下再决定。

## 和其他省 KV 的办法比

| 办法 | 省的是什么 | 要不要改模型 | 代价 |
| --- | --- | --- | --- |
| KV 量化 | 每个值的位数 | 不用，训练后就能做 | 长上下文的误差；要 kernel 支持 |
| GQA / MQA | KV head 的个数 | 要（从头训练或继续训练） | 表示方式变了，不能直接删 head |
| MLA | 把 KV 压成更小的潜在向量 | 要 | 实现更复杂，见 [MLA 与稀疏注意力](../../00-foundations/deep-dives/latent-and-sparse-attention.md) |
| 滑动窗口 | 只留最近的若干位置 | 由架构决定 | 窗口外的信息看不到 |
| 分页、前缀共享、offload | 碎片和重复存储，或把 KV 挪出显存 | 不用 | 管理和搬运的开销，见 [vLLM 与 SGLang](../../06-systems/inference-frameworks.md) |

这些办法可以相乘：KV head 从 32 个减到 8 个，再存成 FP8，KV 只剩 MHA 加 BF16 的 $\tfrac{8}{32}\times\tfrac12=\tfrac18$。GQA 的算例见 [KV cache](../../00-foundations/deep-dives/kv-cache-and-inference.md)。

## 这些数字的边界

- 引擎文档没有给 FP8 KV 的精度数字；KIVI、KVQuant 的结果来自 Llama-2 那一代模型，是作者自报。换模型、换上下文长度都要重测，而且要用长上下文任务测，见[怎样测出丢了多少](checking-quality.md)。
- 上面的时间是“字节 ÷ 峰值带宽”的下限，没算 attention 的计算、kernel 效率和分页碎片。
- 训练配方一节只列了各家公开的做法和自报结果，不同配方之间没有在同一模型上的对照。
