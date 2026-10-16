# 量化：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 3 分钟 · 草稿，待审核

这一页把本系列的要点压成面试题和自检题；答案里的数字和结论都能在前四篇找到出处。

> **待审核草稿。** 这一篇还没有经过站主审阅，事实、取舍和措辞都可能修改；引用前请回到文中链接的一手资料核对。

## 面试常见问题

<details class="interview" markdown="1">
<summary>W4A16、W8A8 和 FP8 KV cache 各换来什么？什么时候选哪一个？</summary>

W4A16 只压权重：省显存，decode 每步少读字节，batch 小时几乎按字节数提速，但反量化后仍用 16 位 Tensor Core 算，算力受限时没有收益。W8A8（INT8 或 FP8）连激活一起压：除了字节减半，还能用 8 位 Tensor Core，H100 SXM5 的 dense 峰值从 989.4 翻到 1,978.9 TFLOPS，prefill 和大 batch 也快。FP8 KV 压的是 KV cache：长上下文时它的字节常常比权重还多，教学模型 128K 时一条序列 16 GiB，存成 FP8 后 80 GB 卡上能放的序列从 3 条变成 6 条。先算出瓶颈在权重字节、算力还是 KV，再选。

</details>

<details class="interview" markdown="1">
<summary>一个离群值为什么能毁掉 per-tensor INT4？per-channel、per-group 各解决了多少？激活为什么不能按输入通道设 scale？</summary>

步长由共用 scale 的那组数里的最大值决定。8 个权重的例子里最大值是 1.40 时 $s=0.2$，MSE 0.0042；换成 14.0，$s=2$，其余 7 个数全舍成 0，MSE 约 0.296。分组把伤害关在一组里，不能消除离群值。权重的输出通道、激活的 token 是 GEMM 的外侧维度，scale 可以在整数矩阵乘之后乘回去；激活的输入通道是求和的维度，scale 在点积中途变化，Tensor Core 的整数累加里插不进去。LLM 激活的离群值恰好固定在少数输入通道上，这就是 SmoothQuant 要做等价变换的原因。

</details>

<details class="interview" markdown="1">
<summary>FP8（E4M3）和 INT8 的误差有什么不同？MXFP4 和 NVFP4 又差在哪里？</summary>

INT8 格子等距，绝对误差处处一样；E4M3 在每个 2 的幂区间里有 8 个格子，相对误差大致不变，最坏约 6.25%。对 $[0.02,-0.3,1.7,12]$ 做 per-tensor 量化，0.02 在 INT8 下变成 0，在 E4M3 下误差 0.45%；1.7 则是 INT8 更准。Qualcomm 的结论是看离群值：分布有明显离群值时 FP8 好，规整时 INT8 好。4 位时都要块缩放：MXFP4 每 32 个值共享一个 2 的幂 scale（E8M0），平均 4.25 位；NVFP4 每 16 个值一个 E4M3 scale，再乘张量级 FP32，平均 4.5 位。块 $[0.3,-1.2,2.4,5.0]$：MX 的 scale 只能取 1，5.0 读回 4，MSE 0.31；NVFP4 的 scale 取 0.8125，MSE 约 0.0072。

</details>

<details class="interview" markdown="1">
<summary>为什么 batch 1 时 W4A16 能快三倍多，batch 大了优势就没了？交点怎么估？</summary>

batch 1 的 decode 每读 1 字节只做约 1 次运算，时间约等于权重字节除以带宽。教学 8B 模型 BF16 是 16 GB、4.77 ms；W4A16 g128（输出层保留 BF16）是 4.87 GB、1.45 ms，约 3.29 倍，不到 4 倍是因为没压的输出层和 scale。W4A16 用 BF16 算，batch $B$ 时运算时间是 $B\times2N/989.4$ TFLOPS，令它等于 1.45 ms，得 $B\approx90$；之后就和 BF16 一样被算力卡住，FP8 W8A8 的屋顶则高一倍。实测更早掉队：Marlin 自报 batch 16–32 以内接近 4 倍；Kurtic 等建议同步低延迟场景用 W4A16，高吞吐连续组 batch 用 W8A8。

</details>

<details class="interview" markdown="1">
<summary>GPTQ 怎样补偿量化误差？AWQ 的思路有什么不同？</summary>

GPTQ 逐层最小化 $\|WX-\hat WX\|^2$，Hessian 是 $H=2XX^\top$；量化一个权重后，按 $\delta_F=-\frac{w_q-\mathrm{quant}(w_q)}{[H_F^{-1}]_{qq}}(H_F^{-1})_{:,q}$ 更新还没量化的权重。两个高度相关的输入、$w=[0.62,0.37]$、步长 0.25：RTN 得 $[0.5,0.25]$，输出误差平方和 0.353；GPTQ 先把 $w_1$ 量成 0.5，把 0.12 的误差按 $H_{12}/H_{22}\approx1.01$ 推给 $w_2$，得 $[0.5,0.5]$，误差 0.00176。工程上靠固定顺序共用 $H^{-1}$、128 列一块的懒更新和 Cholesky 分解扩展到 175B。AWQ 不改舍入方向，而是按激活大小找出重要通道，把它们的权重放大 $s$ 倍、输入缩小 $s$ 倍，$s=s_X^{\alpha}$ 网格搜索；需要的校准数据更少，跨领域时更稳（OPT-6.7B 上困惑度升高 0.5–0.6，对 GPTQ 的 2.3–4.9）。

</details>

<details class="interview" markdown="1">
<summary>SmoothQuant 怎样让 W8A8 可行？$\alpha$ 起什么作用？</summary>

激活的离群通道让 per-tensor 或 per-token 量化下的普通通道只剩两三个格子，而按输入通道给激活设 scale 又进不了 INT8 GEMM。SmoothQuant 做等价变换 $Y=(X\operatorname{diag}(s)^{-1})(\operatorname{diag}(s)W)$，$s_j=\max|X_j|^{\alpha}/\max|W_j|^{1-\alpha}$，把难度从激活挪给权重，$\operatorname{diag}(s)^{-1}$ 离线融进前一层。$\alpha=0.5$ 时两边每个通道的最大值都变成 $\sqrt{\max|X_j|\max|W_j|}$；教学例子里第 1 个通道的可用格子从约 3 个变成约 40 个，输出相对误差从 1.95% 降到 0.15%。$\alpha$ 太小激活难量化，太大权重难量化，OPT-175B 上 0.4–0.6 最好。

</details>

<details class="interview" markdown="1">
<summary>KV cache 量化时，为什么 key 要按通道、value 要按 token？</summary>

KIVI 观察到 key 里有少数固定通道幅值特别大，value 没有明显离群模式，而 attention 输出主要来自少数重要 token 的 value。key 按 token 量化时，一行的步长被离群通道撑大：4×4 的例子里 2 bit 下普通通道全变成同一个值，分数从 $[1.2,0.9,0.9,1.6]$ 变成 $[0.3,-0.1,-0.8,0.6]$；按通道量化时最大误差只有 0.067。$K$ 本身的相对误差 8.6% 对 0.6%，低估了分数被破坏的程度。KIVI 在 Llama-2-7B 上，2 bit 两者都按 token 时 GSM8K 只剩 0.83，KIVI-2 是 12.74（16 位 13.50）。部署时还要注意引擎的默认值：vLLM 不校准时 KV 的 scale 全是 1.0。

</details>

<details class="interview" markdown="1">
<summary>怎样验收一个量化模型？为什么只看准确率不够？</summary>

准确率几乎不变时，逐题可能有很多对错互换：Dutta 等测到 Llama2-70B-chat 的 GPTQ W4A16 在 MMLU 上只少 0.45 个点，翻转却有 4.26%。要做三件事：固定文本上的平均 KL 和 top-1 一致率（接近平局的位置一翻转，后文就分叉）；同一批题的配对比较，1,000 题、翻转 56 题、净变化 −0.4 时 95% 区间约 $[-1.87,+1.07]$ 个点；事先定好的切片，长上下文（4 bit 方法最多掉 59 个点）、非英语（日语自动指标 1.7% 对人工 16.0%）、数学推理。还要记下校准数据是否和评测重叠、各部分的格式与 scale、引擎和 kernel，硬件不支持时会退回只压权重。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>给一个 70B 模型和一张 80 GB 的卡，BF16、FP8、INT4 g128 三种权重各占多少，剩下多少给 KV，batch 1 每个 token 的下限是多少？</li>
    <li>为什么权重的输出通道 scale 能在 GEMM 之后乘回去，沿输入维度分组的 scale 却要在 kernel 里反量化，或者靠 Blackwell 的块缩放 MMA？</li>
    <li>MX 规范把块 scale 取成不超过最大值的 2 的幂，块最大值是 3.5 时 FP4 元素会发生什么？MXFP8 训练配方为什么改成向上取整？</li>
    <li>128K 上下文、batch 3 时，只压权重和只压 KV，每步读的字节各降多少？为什么短上下文时结论反过来？</li>
    <li>一次量化之后，你会记录哪些字段、报告哪几种指标和切片，才能让别人复现并相信“没掉多少”？</li>
  </ol>
</div>

## 参考论文

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)：scale 与 zero-point 的仿射映射
- [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)：PTQ 与 QAT 的标准流程
- [LLM.int8()](https://arxiv.org/abs/2208.07339)：大模型激活里的系统性离群特征，混合精度分解
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)：E4M3 与 E5M2
- [FP8 versus INT8 for efficient deep learning inference](https://arxiv.org/abs/2303.17951)：两种格式的误差与硬件代价
- [OCP Microscaling Formats (MX) v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 与 [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537)：MXFP8、MXFP6、MXFP4、MXINT8
- [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)：16 个值一组、E4M3 scale 的 4 位格式
- [GPTQ](https://arxiv.org/abs/2210.17323)：二阶误差补偿的训练后量化
- [AWQ](https://arxiv.org/abs/2306.00978)：按激活找重要通道并缩放
- [SmoothQuant](https://arxiv.org/abs/2211.10438)：把激活的量化难度挪给权重
- [QuaRot](https://arxiv.org/abs/2404.00456)：旋转后的 4 位权重、激活与 KV
- [MARLIN](https://arxiv.org/abs/2408.11743)：4 位权重乘 FP16 激活的 kernel 与 batch 的关系
- [KIVI](https://arxiv.org/abs/2402.02750)、[KVQuant](https://arxiv.org/abs/2401.18079)：KV cache 的按通道与按 token 量化
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：细粒度 FP8 训练与累加精度
- [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027)、[Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)：块缩放格式的训练配方
- [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141)：翻转与 KL 散度
- [“Give Me BF16 or Give Me Death”?](https://arxiv.org/abs/2411.02355)：Llama-3.1 上的大规模量化评测与部署建议
- [Does quantization affect models' performance on long-context tasks?](https://arxiv.org/abs/2505.20276)、[How Does Quantization Affect Multilingual LLMs?](https://arxiv.org/abs/2407.03211)、[Quantization Hurts Reasoning?](https://arxiv.org/abs/2504.04823)：长上下文、多语言与推理切片
