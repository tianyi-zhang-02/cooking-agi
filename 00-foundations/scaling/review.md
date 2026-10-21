# Scaling：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 5 分钟 · 最近审阅：2026-10

这一页把本系列的要点压成面试题和自检题；答案里的数字都能在前三篇找到出处，手算部分是教学数字。

## 面试常见问题

<details class="interview" markdown="1">
<summary>$C\approx6ND$ 是怎么来的？64 张 H100 跑 4.6 天，按 Chinchilla 该训多大的模型？</summary>

dense Transformer 每个训练 token 前向约 $2N$ FLOPs（每个权重一次乘加），反向约是前向的两倍，合计 $6N$，乘以 token 数 $D$。它略去了随上下文增长的 attention 项；Kaplan 的 $N$ 不含 embedding，Chinchilla 的含。预算等于 GPU 数 × 峰值 × 模型浮点利用率（MFU）× 时间：$64\times989.4$ TFLOPS（BF16 dense）$\times0.4\times3.95\times10^{5}$ 秒 $\approx10^{22}$ FLOPs。按每参数 20 个 token，$N=\sqrt{C/120}\approx9.1\times10^{9}$，$D\approx1.8\times10^{11}$。用 sparse 峰值算会把预算高估一倍。

</details>

<details class="interview" markdown="1">
<summary>Kaplan 和 Chinchilla 的最优分配为什么不一样？</summary>

Kaplan（2020）得到 $N_{\rm opt}\propto C^{0.73}$，Chinchilla（2022）得到 0.46–0.50。Chinchilla 指出 Kaplan 所有 run 用同一个固定长度的 cosine，中途截取时学习率（LR）还没退下来，loss 偏高，低估了少数据的效果，而且模型偏小；Pearce 和 Song 指出只数非 embedding 参数、在小规模上拟合会把指数推高；Porian 等在 OpenWebText2 上从 0.864 出发，依次修正最后一层的计算、warmup、cosine 和随规模调的超参数（含 AdamW $\beta_2$），降到 0.518，并发现不做学习率衰减也能得到 Chinchilla 的结果。

</details>

<details class="interview" markdown="1">
<summary>“每个参数 20 个 token”在什么条件下成立？偏离它代价多大？</summary>

来自 Chinchilla 方法 1 的表 3（如 10 亿参数配 202 亿 token）。条件：MassiveText，数据大体只看一遍；$N$ 含 embedding，token 按他们的 tokenizer 计；cosine 衰减 10 倍并对准训练长度。方法 3 的原始拟合偏向更小的模型，Besiroglu 等重拟后也回到约 20；DeepSeek LLM 发现数据质量会把指数从 0.450 移到 0.578。最优点附近很平：用重拟曲面算，$10^{22}$ FLOPs 下模型做成最优的一半或两倍，loss 只高约 0.01，多花 15%–26% 算力即可追平。

</details>

<details class="interview" markdown="1">
<summary>怎样用几个小 run 拟合 scaling law？外推能信多少？</summary>

IsoFLOP：3–5 个小预算，每个训 5–10 个大小，schedule 对准训练长度，对 loss–$\log N$ 拟抛物线取谷底，再在 log–log 坐标里对谷底拟直线。教学数字 $(10^{18},9\times10^7)$ 到 $(10^{21},2.9\times10^9)$ 的斜率约 0.50，外推到 $10^{22}$ 得约 92 亿参数。指数差 ±0.04，外推 10 倍只差约 10%，外推 1000 倍就从 700 亿变到 1210 亿。风险：前沿会弯（Chinchilla 附录 E），定律在某处必然失效（Kaplan §6.3），种子噪声约 0.02，loss 平滑不代表下游指标平滑。GPT-4、DeepSeek LLM、Llama 3 的成功预测都是自报。

</details>

<details class="interview" markdown="1">
<summary>为什么 Llama 3 8B 要训 15T token？推理量怎样改变最优点？</summary>

Chinchilla 只最小化训练算力。Sardana 等固定目标 loss，最小化 $6ND_{\rm tr}+2ND_{\rm inf}$：推理越多，模型越该小、数据越该多。用重拟曲面、以 $10^{22}$ FLOPs 最优模型的 loss 为目标，$D_{\rm inf}=10^{12}$ 时最优约 46 亿参数、4470 亿 token，训练多花 24%，总账省 23%。Sardana 等也发现用常规比例拟合会高估极端比例下的数据收益，所以推理量很大时的节省是乐观上限。Llama 3 报告自报较小的模型训练得远超算力最优，在同样推理预算下更好；405B 则接近它自己拟合的算力最优（402B、16.55T）。

</details>

<details class="interview" markdown="1">
<summary>数据不够时，重复训练值多少？</summary>

Muennighoff 等把重复折算成有效数据：$D'=U_D+U_DR_D^*(1-e^{-R_D/R_D^*})$，$R_D^*\approx15.4$。1000 亿不重复 token 训 4 遍，有效约 3726 亿（93%）；16 遍约 1.06T（66%）；上限约 16.4 倍。论文结论是固定算力下重复约 4 遍和新数据几乎一样（8.7B 模型 4 个 epoch 的验证 loss 只高约 0.5%），之后收益迅速变小。多余参数的衰减常数 $R_N^*\approx5.3$ 更小，所以数据受限时多跑几遍比加大模型划算。

</details>

<details class="interview" markdown="1">
<summary>什么是临界 batch size？batch 和学习率一起变时该怎么换算？</summary>

McCandlish 等：$S=S_{\min}(1+B_{\rm crit}/B)$，$E=E_{\min}(1+B/B_{\rm crit})$，$B_{\rm crit}=E_{\min}/S_{\min}$，约等于梯度噪声尺度。教学数字 $B_{\rm crit}=2$M：batch 从 4M 加到 16M 只省 25% 的步数，数据量却变成原来的 3 倍。它随 loss 下降变大（Kaplan：loss 降约 13% 翻一倍），近期工作发现主要随数据量增长（Zhang 等：固定数据只放大模型，指数只有 0.087）。换算：SGD 用线性规则；Adam 用平方根规则（学习率乘 $\sqrt\kappa$，$1-\beta$ 乘 $\kappa$），只适用于噪声主导的小 batch 区间。也可以用 DeepSeek LLM 的拟合当起点：$10^{22}$ FLOPs 时约 $5.5\times10^{-4}$、4.6M token。

</details>

<details class="interview" markdown="1">
<summary>最大更新参数化（µP）解决什么问题？为什么隐藏层学习率要随宽度缩小？哪些东西不能迁移？</summary>

标准参数化（SP）下最优学习率随宽度移动（TP5 的 MLP 上宽度 256 到 8192 移动约一个数量级）。在[超参数一节](hyperparameters.md)的 batch=1、零初始化矩估计、不计 weight decay 且非零梯度远大于 $\epsilon$ 的条件下，Adam 第一步更新约为 $-\eta\,{\rm sign}(g)$，和输入 $x$ 对齐。对非零 $\delta_i$，变化幅度为 $|(\Delta Wx)_i|\approx\eta\,n\,\mathbb E|x|$：$x=\pm1$、$\eta=0.01$ 时宽 512 变化 5.12，宽 4096 变化 40.96。这里说的是幅度，方向还取决于 $\operatorname{sign}(\delta_i)$。µP 让隐藏层和输出层的 Adam 学习率 $\propto1/\text{fan\_in}$，输出层初始化方差 $1/\text{fan\_in}^2$，attention 用 $1/d$，于是宽 4096 的隐藏层学习率取 0.00125。不能迁移的：dropout、weight decay 等正则化；深度、batch、训练长度只有经验验证，Transformer 的多层块在深度上有根本限制（TP6）。Everett 等发现设对逐层学习率的 SP 也能迁移。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>给你 GPU 数、峰值、MFU 和天数，怎样一步算到 Chinchilla 意义下的 $N$ 和 $D$？哪一步最容易把预算算错一倍？</li>
    <li>为什么 Kaplan 固定 cosine 长度会让“少数据”的模型看起来更差，并把指数推高？</li>
    <li>拟合出的指数差 0.04，外推 10 倍和 1000 倍时的影响分别有多大？为什么最优点附近的平坦让这件事没那么可怕？</li>
    <li>推理 token 从 0 增到 $10^{13}$，最优模型怎样变？为什么表里最右边的节省是偏乐观的？</li>
    <li>宽度乘 8 时，SP 下 Adam 第一步对激活的影响乘多少？µP 怎样把它拉回来，又不能保证什么？</li>
  </ol>
</div>

## 参考论文

- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)：梯度噪声尺度与临界 batch size
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)：幂律、$C\approx6ND$、$N\propto C^{0.73}$
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)：Chinchilla 的三种方法与约 20 tokens/parameter
- [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102)：方法 3 的重拟与置信区间
- [Reconciling Kaplan and Chinchilla Scaling Laws](https://arxiv.org/abs/2406.12907)、[Resolving Discrepancies in Compute-Optimal Scaling](https://arxiv.org/abs/2406.19146)：两条定律的分歧从哪来
- [DeepSeek LLM](https://arxiv.org/abs/2401.02954)：学习率和 batch size 的算力拟合；数据质量改变分配
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)：自己的 IsoFLOP 拟合、下游预测和训练配方
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)：可预测的 scaling（自报）
- [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)：指标选择与“涌现”
- [LLaMA](https://arxiv.org/abs/2302.13971)、[Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)：按推理成本选模型
- [Language models scale reliably with over-training and on downstream tasks](https://arxiv.org/abs/2403.08540)：过度训练和下游误差的外推
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)：重复数据的有效数据量
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[OLMo 2](https://arxiv.org/abs/2501.00656)、[SmolLM2](https://arxiv.org/abs/2502.02737)、[Gemma 3](https://arxiv.org/abs/2503.19786)、[Qwen3](https://arxiv.org/abs/2505.09388)、[Kimi K2](https://arxiv.org/abs/2507.20534)、[DeepSeek-V4 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)与[报告](https://arxiv.org/abs/2606.19348)、[Llama 3 模型卡](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)：开源模型的训练量
- [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)、[On the SDEs and Scaling Rules for Adaptive Gradient Algorithms](https://arxiv.org/abs/2205.10287)：线性与平方根换算规则
- [How Does Critical Batch Size Scale in Pre-training?](https://arxiv.org/abs/2410.21676)、[Power Lines](https://arxiv.org/abs/2505.13738)：batch size 随数据量的规律
- [SGDR](https://arxiv.org/abs/1608.03983)、[MiniCPM](https://arxiv.org/abs/2404.06395)、[Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/abs/2405.18392)、[Understanding Warmup-Stable-Decay Learning Rates](https://arxiv.org/abs/2410.05192)：cosine 与 warmup-stable-decay（WSD）
- [Tensor Programs V](https://arxiv.org/abs/2203.03466)、[Tensor Programs VI](https://arxiv.org/abs/2310.02244)、[Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872)：µP、深度迁移与其他参数化
