# 量化：少存几位，换来什么、丢掉什么

**中文** · [English](README.en.md)

> 阅读时间：约 7 分钟 · 最近审阅：2026-10

一个 70B 参数的 dense 模型，仅 BF16 权重就占 140 GB，一张 80 GB 的 GPU 放不下。即使显存够用，在 batch size 为 1、逐个生成 token 的情况下，每一步读取权重也要花不少时间。

量化（quantization）用更少的位数存储数值。压缩权重可以减少显存占用和读取量；同时量化激活，还可能用上低精度 Tensor Core；长上下文下，量化 KV cache 也能节省大量显存。选哪一种，取决于当前瓶颈和硬件支持。

节省多少空间可以先算，质量变化则要测。后面会用同一批题比较量化前后的输出，既看总分，也看哪些任务退步了。

## 先记一笔账：70B 模型和一张 80 GB 的卡

下面是**教学假设**：一个 70B dense 模型，有 80 层、8 个 KV head，head dim 为 128。因此，每个 token 的 BF16 KV 是 $2\times80\times8\times128\times2=327{,}680$ 字节（320 KiB）。GPU 取 H100 SXM5 的 80 GB 显存和 3,352 GB/s 带宽（[H100 架构白皮书 V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)），给运行时预留 4 GB。

计算时假设所有权重都按同一格式存储，不单独计算 embedding、输出层和 scale 以外的元数据。

| 权重格式 | 每个参数 | 权重 | 剩给 KV | KV 能放多少 token（BF16 / FP8 KV） | batch 1 每个 token 至少 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 16 bit | 140 GB | 放不下 | — | 单卡不成立 |
| FP8 | 8 bit | 70 GB | 6 GB | 约 1.8 万 / 3.7 万 | 20.9 ms |
| INT4，每 128 个权重一组 | 4.125 bit | 36.1 GB | 39.9 GB | 约 12.2 万 / 24.4 万 | 10.8 ms |

4.125 bit 包括 4 位整数，以及每 128 个权重共用的一个 16 位 scale（$16/128=0.125$）。最后一列用“权重字节 ÷ 带宽”估算读取时间，假设带宽利用率达到 100%，因此只是理论下限；BF16 分到两张卡时，下限也是 20.9 ms，见 [NVIDIA 各代 GPU](../nvidia-generations.md)。这些数能帮助估算容量和速度，但还不能判断量化后的回答质量。后半部分会专门讨论怎样评估。

## 三样东西可以压，换来的不一样

名字的读法：W4A16 是权重 4 bit、激活 16 bit；W8A8 是两者都 8 bit；g128 是每 128 个值共用一个 scale；KV8 或 FP8 KV 指 KV cache 存成 8 位。

| 压什么 | 常见写法 | 换来什么 | 需要什么 | 通常先坏在哪里 |
| --- | --- | --- | --- | --- |
| 只压权重 | W4A16、W8A16（GPTQ、AWQ） | 权重显存；decode 每步读的字节 | 边读边反量化（dequantize）的 kernel；GPTQ、AWQ 还要一小份校准数据 | batch 一大优势就没了；3 bit 及以下误差急剧变大 |
| 权重和激活 | W8A8（INT8、FP8），W4A4（NVFP4、MXFP4） | 上面两样，加上低精度 Tensor Core 的算力 | 硬件支持这种格式；激活的 scale 怎么定 | 激活里的离群值（outlier）；INT8 往往要 SmoothQuant 这类预处理 |
| KV cache | FP8 KV，INT4 / INT2 KV | 长上下文时 KV 的显存和每步读的字节 | attention kernel 支持；K 和 V 各自的 scale | 长上下文任务；key 里固定的离群通道 |

选哪一种，取决于你落在 roofline 的哪一边（[性能](../performance.md#decode-batching)）。batch 1 的 decode 每读 1 字节只做约 1 次运算，远在 H100 BF16 的 ridge point（约 295 FLOP/字节）左边，少读字节就直接变快；batch 大到算力受限以后，W4A16 仍用 BF16 Tensor Core 算，优势消失，而 W8A8 的 FP8 能把 dense 峰值从 989.4 提到 1,978.9 TFLOPS（同一份白皮书）。

## 本系列的教学例子

后面几篇用同一个**教学模型**：8B dense，32 层、32 个 query head、8 个 KV head、head dim 128，与 [KV cache](../../00-foundations/deep-dives/kv-cache-and-inference.md) 和[性能](../performance.md)两篇的假设配置相同；GPU 仍按 H100 SXM5 的公开规格手算。每篇讲什么，见文末的阅读顺序。

## 时间线

| 日期 | 工作 | 一句话 |
| --- | --- | --- |
| 2017-12 | [Jacob 等](https://arxiv.org/abs/1712.05877)（Google） | 整数推理的 scale 加 zero-point 映射，zero-point 保证实数 0 能被精确表示 |
| 2021-06 | [Nagel 等的量化白皮书](https://arxiv.org/abs/2106.08295) | 训练后量化（PTQ）和量化感知训练（QAT）的流程；多数情况下 PTQ 就能做到接近浮点的 8 bit |
| 2022-08 | [LLM.int8()](https://arxiv.org/abs/2208.07339)（NeurIPS 2022） | 6.7B 左右开始出现系统性的离群特征；约 0.1% 的维度改用 16 位乘 |
| 2022-09 | [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) | 提出 E4M3 和 E5M2 两种 FP8 编码 |
| 2022-10 | [GPTQ](https://arxiv.org/abs/2210.17323)（ICLR 2023） | 用二阶信息把量化误差推给还没量化的权重；175B 模型约 4 个 GPU 小时压到 3–4 bit |
| 2022-11 | [SmoothQuant](https://arxiv.org/abs/2211.10438)（ICML 2023） | 把激活的离群值等价地“挪”到权重上，做 W8A8 |
| 2023-06 | [AWQ](https://arxiv.org/abs/2306.00978)（MLSys 2024 最佳论文） | 按激活大小找出重要通道，放大以后再量化 |
| 2023-09 | [OCP MX 规范 v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) | AMD、Arm、Intel、Meta、Microsoft、NVIDIA、Qualcomm 的作者共同定义：每 32 个值共享一个 2 的幂 scale |
| 2024-02 | [KIVI](https://arxiv.org/abs/2402.02750)（ICML 2024） | key 按通道、value 按 token 量化，KV cache 压到 2 bit |
| 2024-03 | [QuaRot](https://arxiv.org/abs/2404.00456) | 先旋转再量化，权重、激活、KV cache 都用 4 bit |
| 2024-07 | [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141)（NeurIPS 2024） | 准确率几乎不变，逐题看却有不少题对错翻转 |
| 2024-08 | [Marlin](https://arxiv.org/abs/2408.11743) | 4 bit 权重乘 FP16 激活的 kernel，A10 上的大矩阵在 batch 16–32 以内接近理想的 4 倍 |
| 2024-11 | [“Give Me BF16 or Give Me Death”?](https://arxiv.org/abs/2411.02355)（ACL 2025） | Llama-3.1 上 50 多万次评测：FP8 近似无损，INT8 掉 1–3%，W4A16 与 8 bit 相当 |
| 2024-12 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 671B 参数的 MoE 用细粒度 FP8 训练 |
| 2025-05 | [MXFP8 训练配方](https://arxiv.org/abs/2506.08027)（NVIDIA） | 最大 8B 模型、15T token，在 Hopper 上模拟 MXFP8 计算，自报验证困惑度与 BF16 相差不到 0.5% |
| 2025-05 | [长上下文任务上的量化](https://arxiv.org/abs/2505.20276)（EMNLP 2025） | 8 bit 平均最多掉约 0.8 个点；4 bit 在长输入任务上最多掉 59 个点 |
| 2025-06 | [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) | 每 16 个值一个 E4M3 scale，再乘一个张量级 FP32 scale |
| 2025-09 | [NVFP4 预训练](https://arxiv.org/abs/2509.25149) | 12B 模型、10T token，自报 loss 和下游准确率与 FP8 基线相当 |

## 哪些已经比较确定，哪些还在变

每格都对应一条带日期的公开记录；第三列写的是什么样的结果会改变判断，不是预测。

| 比较确定 | 还在变 | 接下来看什么信号 |
| --- | --- | --- |
| 8 bit 权重在大模型上基本无损：Kurtic 等（2024-11）在 Llama-3.1 8B 到 405B 上测 FP8 W8A8 近似无损；Dutta 等在 MMLU 上测 GPTQ W8A16，只有 0.26%–1.40% 的题翻转 | 4 bit 及以下怎么做：GPTQ、AWQ 之外，有旋转（QuaRot），也有 NVFP4、MXFP4 这类硬件块格式 | 在长上下文、多语言和推理任务上逐题对照的 4 bit 结果 |
| 小 batch 时 W4A16 主要换来 decode 速度，batch 一大优势就消失：Marlin（2024-08）、Kurtic 等的部署建议 | W4A4 的推理质量：NVIDIA 自报 DeepSeek-R1-0528 从 FP8 转 NVFP4 后，所列 7 项评测最多掉 1 个百分点（[NVFP4 博客](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)） | 第三方按切片报告的 W4A4 评测 |
| 平均准确率看不出逐题变化（Dutta 等，2024-07）；长上下文、非英语和数学这类难任务最先坏（Mekala 等 2025-05，Marchisio 等 2024-07） | 低精度训练：FP8 已用到 671B（DeepSeek-V3），MXFP8 到 8B / 15T（模拟量化），NVFP4 到 12B / 10T | 更大规模、公开完整配方的 FP4 训练 |
| 硬件决定能不能真的加速：FP8 Tensor Core 需要 compute capability 8.9 以上，9.0（Hopper）没有 INT4 Tensor Core（[CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)） | 新硬件的取舍：截至 2026 年 10 月的 [HGX 页面](https://www.nvidia.com/en-us/data-center/hgx/)（8 卡，稀疏口径，dense 减半），HGX B300 的 INT8 只有 3 POPS、FP8 是 72 PFLOPS，HGX B200 两者都是 72；AMD 的 [MI350 系列](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html)加入 MX 格式 | 下一代数据表里 INT8、FP8、FP4 各自的吞吐 |

## 这一组怎么读

先读这些会更顺：[精度与显存](../../00-foundations/deep-dives/precision-and-memory.md)（FP16 和 BF16 的取舍），[GPU 架构](../gpu-architecture.md)（各代 Tensor Core 支持的数据类型，NVFP4 的手算例子），[性能](../performance.md)（roofline、decode 的 batch 账），[KV cache](../../00-foundations/deep-dives/kv-cache-and-inference.md)（KV 显存预算）。相邻的几篇：[NVIDIA 各代 GPU](../nvidia-generations.md)讲 70B 模型在哪代卡上要几张；[vLLM 与 SGLang](../../06-systems/inference-frameworks.md)讲量化在生成引擎里处于哪一层；[LoRA 与 QLoRA](../../05-post-training/lora-and-qlora.md)讲 NF4 和在 4 bit 基座上训练；[分数涨了，究竟能说明什么](../../07-evaluation/metric-robustness.md)讲配对比较的统计。

1. [格式与误差](formats-and-error.md)：scale 和 zero-point，per-tensor、per-channel、per-group，INT 与 FP 的格子，块缩放的 MXFP4 与 NVFP4，离群值为什么伤害大，哪代硬件能算哪种格式
2. [只压权重，与训练后量化](weight-only-and-ptq.md)：W4A16 为什么让 decode 变快、batch 多大时失效，GPTQ 的误差补偿，AWQ 的通道放大，SmoothQuant 的 W8A8
3. [FP8 推理与 KV cache 量化](fp8-and-kv-cache.md)：FP8 的 scale 怎么定，FP8 / MXFP8 / NVFP4 训练要处理什么，128K 上下文时 KV 量化省下多少，KIVI 为什么选择按通道量化 key
4. [怎样测出丢了多少](checking-quality.md)：困惑度（perplexity）、KL 散度、任务准确率和逐题翻转，配对比较，长上下文和多语言切片，每次量化要记下什么，常见故障
5. [复习题](review.md)：面试题和自检
