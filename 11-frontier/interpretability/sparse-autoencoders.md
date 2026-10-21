# 可解释性：稀疏自编码器（SAE）

**中文** · [English](sparse-autoencoders.en.md)

> 阅读时间：约 12 分钟 · 最近审阅：2026-10

“从 Claude 3 Sonnet 里拆出 3400 万个特征”“在 GPT-4 上训练 1600 万个 latent”，这些数字说的是什么？为什么 Google DeepMind 的可解释性团队在 2025 年 3 月宣布暂缓基础 SAE 研究？

稀疏自编码器（sparse autoencoder，SAE）把一个激活写成少数几个字典方向的加权和。它不用标签就能找出大量可以赋予语义解释、还能用来引导模型的方向，适合探索和调试。不过，重建准确、标签通顺，都不保证找到的是模型真正在用的特征。在有强基线的下游任务上，SAE 常常不如线性探针、均值差和提示词。

## 目标：重建加稀疏

[上一篇](features-and-superposition.md)说，如果特征是叠加在一起的方向，就要把它们找出来。SAE 是字典学习的一种近似：一个很宽的单隐层自编码器，要求还原输入，同时隐层大部分是 0。Anthropic 在 [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) 里用的写法：

$$
f(x)=\mathrm{ReLU}(W_{\rm enc}x+b_{\rm enc}),\qquad \hat x=W_{\rm dec}f(x)+b_{\rm dec}
$$

$$
L=\mathbb E_x\Big[\|x-\hat x\|_2^2+\lambda\sum_i f_i(x)\,\|W_{\rm dec}^{(i)}\|_2\Big]
$$

$x\in\mathbb R^d$ 是模型某处的激活，$f(x)\in\mathbb R^F$ 是稀疏编码，$F\gg d$（过完备）。$W_{\rm dec}$ 的第 $i$ 列 $W_{\rm dec}^{(i)}$ 就是第 $i$ 个特征的方向。第一项要重建准，第二项是 L1 惩罚，$\lambda$ 决定两者怎么权衡；乘上解码列的范数，是为了防止模型把 $f_i$ 缩小、把解码向量放大来逃避惩罚。

| 变体 | 怎么控制稀疏 | 解决了什么 | 代价或问题 | 用在哪 |
| --- | --- | --- | --- | --- |
| ReLU + L1 | 罚激活之和，调 $\lambda$ | 简单 | 收缩（shrinkage）：非零激活被系统性低估 | [Bricken 等 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html)、Templeton 等 2024 |
| TopK | 每个 token 只保留最大的 $k$ 个，损失只剩重建项 | 直接设定稀疏度，没有 L1 带来的收缩 | 换随机种子时学到的特征更不稳定 | [Gao 等 2024](https://arxiv.org/abs/2406.04093)（OpenAI） |
| JumpReLU + L0 | 每个 latent 学一个阈值，直接罚非零个数 | 同样稀疏度下重建更准 | 阈值处不可导，要用直通估计器训练 | [Rajamanoharan 等 2024](https://arxiv.org/abs/2407.14435)、Gemma Scope |
| Transcoder | 用 MLP 的输入预测 MLP 的输出 | 可以直接替换 MLP，便于画电路 | 只是对 MLP 的近似，机制可能不同 | [Dunefsky 等 2024](https://arxiv.org/abs/2406.11944)，见[第 3 篇](circuits.md) |

## 手算：一个 2 维激活，三个字典方向（教学例子）

取 $d=2$，字典里有三个单位方向：$d_1=(1,0)$，$d_2=(0,1)$，$d_3=(0.707,0.707)$。激活 $x=(1,1)$。为了看清目标函数本身，这里直接求最优的非负系数，而不经过编码器：

$$
\min_{f\ge 0}\ \|x-Df\|_2^2+\lambda\,(f_1+f_2+f_3)
$$

$\lambda=0$ 时，解有无穷多个：$(1,1,0)$、$(0,0,\sqrt2)$，以及两者之间的混合，都能完全还原。只要重建，目标函数挑不出哪一种“才是”特征。

加上 L1 就不一样了。完全还原时，只用 $d_3$ 的 L1 代价是 $\sqrt2\approx1.414$，用 $d_1+d_2$ 是 2。比较这两种方案在两个 $\lambda$ 下的最优系数：只用 $d_3$ 时 $a=\sqrt2-\lambda/2$；只用 $d_1,d_2$ 时 $b=1-\lambda/2$（把损失对系数求导、令其为 0 即得）。

| $\lambda$ | 方案 | 系数 | 重建 | 平方误差 | $\lambda\times$ L1 | 总损失 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 只用 $d_3$ | $a=1.364$ | (0.965, 0.965) | 0.0025 | 0.136 | **0.139** |
| 0.1 | $d_1+d_2$ | $b=0.95$ | (0.95, 0.95) | 0.005 | 0.190 | 0.195 |
| 1.0 | 只用 $d_3$ | $a=0.914$ | (0.646, 0.646) | 0.25 | 0.914 | **1.164** |
| 1.0 | $d_1+d_2$ | $b=0.5$ | (0.5, 0.5) | 0.5 | 1.0 | 1.5 |

两个 $\lambda$ 下都是只用 $d_3$ 更好（用 KKT 条件可以验证它是全局最优）。从表里能读出三件事：

- **L1 偏爱少而“正对着”的方向。** 如果数据里其实是两个独立特征经常同时出现，SAE 可能学出一个合并的方向，而不是两个原子特征。
- **收缩。** 不加惩罚时 $d_3$ 的系数应是 $\sqrt2\approx1.414$。$\lambda=0.1$ 时少了 0.05；$\lambda=1.0$ 时只剩 0.914，平方重建误差为 0.25，相对 $\|x\|^2=2$ 是 12.5%，所以 $1-\|x-\hat x\|^2/\|x\|^2=87.5\%$。这个比例描述相对平方误差的补数，不是重建向量保留了 87.5% 的能量。$\lambda\ge2\sqrt2\approx2.83$ 时，最优解是全 0，什么都不激活。
- **TopK 换了一种取舍。** 取 $k=1$，选最能解释 $x$ 的一个方向 $d_3$，系数等于投影 $\sqrt2$，完全还原、没有收缩；代价是每个 token 激活几个 latent 是事先定死的。

真实 SAE 不会为每个输入单独求最优系数，而是用一层编码器直接算出系数。因此，在同一目标下，编码器给出的结果最多达到最优解，也可能更差。

<!-- paper-figure:jumprelu-toy -->

## 死 latent、特征分裂与吸收

**死 latent。** 上面的例子里，$x$ 落在对角线上时 $d_2$ 永远不会被选中。如果整个训练集都不让某个 ReLU latent 激活，它从重建项拿不到梯度，就再也回不来。Anthropic 把在 $10^7$ 个 token 上都没激活过的特征算作死特征：1M、4M、34M 三个 SAE 分别约有 2%、35%、65% 是死的，34M 那个只有约 1200 万个活着（[Templeton 等，2024-05](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)）。OpenAI 报告，不加缓解措施时死 latent 可达 90%；用解码矩阵的转置初始化编码器，再加一个让死 latent 去拟合重建残差的辅助损失（AuxK），1600 万宽的 SAE 只有 7% 是死的（[Gao 等，2024-06](https://arxiv.org/abs/2406.04093)）。

**特征分裂。** 字典变大，一个特征会分成几个更细的。Bricken 等在单层模型里看到一个 base64 特征分成三个；Templeton 等看到 1M SAE 里的一个 “San Francisco” 特征，在 4M 里分成 2 个，在 34M 里分成 11 个。不同大小的字典相当于不同的分辨率。

**吸收。** [Chanin 等（2024-09）](https://arxiv.org/abs/2409.14507)发现，当特征有层级关系时，看起来单义的父特征会在该激活的地方不激活，它的作用被子特征“吸收”了；他们认为这是优化稀疏度造成的，改字典大小或稀疏度解决不了。

**换种子就变。** [Paulo & Belrose（2025-01）](https://arxiv.org/abs/2501.16615)在 Llama 3 8B 的一个 MLP 上训练 131K 宽的 SAE，只换随机种子，只有约 30% 的特征是共同的；TopK SAE 比 L1 的更依赖种子。作者的结论是，SAE 找到的特征应看作“a pragmatically useful decomposition of activation space, rather than an exhaustive and universal list …”，即一种实用的分解，不是模型“真正在用”的特征的完整清单。

## 怎么给特征贴标签：自动解释和它的边界

几百万个特征没法人工看，于是用模型给模型贴标签。流程是：收集某个特征激活最强的上下文，让一个 LLM 写一句解释，再给解释打分。[Bills 等（OpenAI，2023-05）](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)用 GPT-4 解释 GPT-2 XL 的神经元，打分方式是模拟：另一个模型只看解释去预测激活，再算预测和真实激活的 Pearson 相关。[Paulo 等（2024-10）](https://arxiv.org/abs/2410.13928)补了几种更便宜的打分，并加了干预打分。

| 打分方式 | 问的是 | 看不到什么 |
| --- | --- | --- |
| 模拟 | 只凭解释能否预测激活值 | 这个特征对输出起什么作用 |
| 检测、fuzzing | 只凭解释能否分出激活和不激活的上下文（或 token） | 好几个不同特征是不是被同一句话描述 |
| 干预打分 | 只凭解释能否预测改动这个特征后输出怎么变 | 自然输入上模型是否这样用它 |

边界有三层。第一，标签说的是“什么时候激活”，不是“它做了什么”：[Huang 等（2023-09）](https://arxiv.org/abs/2309.10312)检查 Bills 等的解释，发现即使最有把握的那些，错误率也很高，因果作用几乎为零。第二，要有随机基线：[Heap 等（2025-01）](https://arxiv.org/abs/2501.17727)发现，在随机初始化的 Pythia 上训练的 SAE，自动解释分数和重建指标在很多设置下与训练过的模型差不多。第三，标签会撞车：一篇单作者预印本（[McCann，2026-05](https://arxiv.org/abs/2605.12874)）重新分析了 722 个人工标注的特征（Gemma 2 2B 和 Pythia 70M），平均每个标签被 3.07 个特征共用，82.1% 的特征至少和另一个特征同名，“plural nouns” 一个标签就对应 101 个特征；作者证明检测类打分对这种撞车完全不敏感。

所以 Anthropic 在 2024 年的工作里，除了看激活样例，还会把特征钳高钳低，看行为是否符合标签（上一篇的金门大桥就是一例）。

## 规模化：几组公开数字

| 工作 | 模型与位置 | 字典大小 | 训练数据 | 报告的数字 |
| --- | --- | --- | --- | --- |
| Towards Monosemanticity（Anthropic，2023-10） | 单层 Transformer 的 MLP，512 个神经元 | 512 到 131,072（1×–256×） | 80 亿个激活样本 | 人工和自动评估都认为大多数特征可解释 |
| Scaling Monosemanticity（Anthropic，2024-05） | Claude 3 Sonnet 中间层残差流 | 1M、4M、34M | 未公开 | 每个 token 平均激活不到 300 个特征；至少解释 65% 的方差；死特征约 2%、35%、65% |
| TopK SAE（OpenAI，2024-06） | GPT-4 残差流，约 5/6 深度处 | 16M | 400 亿 token | 死 latent 7%；把 SAE 换进 GPT-4 后，语言建模损失相当于只用 GPT-4 预训练算力 10% 训出的模型 |
| [Gemma Scope](https://arxiv.org/abs/2408.05147)（Google DeepMind，2024-08） | Gemma 2 2B、9B 的全部层和子层，27B 的部分层 | 400 多个 SAE，共 3000 万以上特征 | 每个 40 亿到 160 亿 token | 用了超过 GPT-3 训练算力 20% 的计算；存了约 20 PiB 激活 |
| [Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)（Google DeepMind，2025-12-19） | Gemma 3 的 270M 到 27B，每一层 | SAE、transcoder、skip transcoder、跨层 transcoder | 未在博客中给出 | 自报：存储约 110 PB 数据，训练的总参数超过 1 万亿 |

怎么读这些数字：“解释了多少方差”和“换进模型后损失多少”是两回事。OpenAI 指出，以“激活全置零”为基线算损失恢复比例，很差的 SAE 也能得高分（他们的 16M SAE 按这种算法是 98.2%），所以改用“相当于多少预训练算力”；10% 比较的是换入 SAE 后的语言建模损失与相应预训练算力水平，不是参数量比例，也不能单凭它确定一个等效模型有多小。Anthropic 自己也写道，他们很可能离“找全所有特征”差几个数量级，而真要找全，算力可能比训练模型本身还多。

## 2025–2026 的批评：在下游任务上

| 工作 | 任务 | 比较的基线 | 结果 |
| --- | --- | --- | --- |
| [Kantamneni 等](https://arxiv.org/abs/2502.16681)（2025-02） | 稀疏探测：数据少、类别不平衡、标签噪声、分布偏移 | 线性探针等 | SAE 偶尔在单个数据集上更好，但把它加进组合也不能稳定胜过只用基线的组合 |
| [Google DeepMind 团队](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)（2025-03-26） | 识别用户有害意图，测分布外泛化（Gemma 2 9B IT 第 20 层） | 残差流上的线性探针，分布外 AUROC 0.999 | k 个 latent 的 SAE 探针分布外明显更差；团队暂缓基础 SAE 研究 |
| [AxBench](https://arxiv.org/abs/2501.17148)（2025-01） | 引导与概念检测（Gemma 2 2B、9B） | 提示词、微调、均值差 | 引导：提示词最好，其次微调；检测：均值差最好；两项上 SAE 都没有竞争力 |
| [SAEBench](https://arxiv.org/abs/2503.09532)（2025-03） | 8 项指标，200 多个 SAE | 8 种架构互相比较 | 代理指标的提升不能可靠地变成实际表现 |
| [Anthropic](https://transformer-circuits.pub/2024/features-as-classifiers/index.html)（2024-10） | 生物武器相关提示的分类 | 原始激活上的线性分类器 | 特征分类器有时相当或略好；原始激活是强基线；特征的好处是能看出训练数据里的伪相关 |

SAE 也有派上用场的时候，多半是“不知道要找什么”的场合。Anthropic 的审计游戏里，一支允许用 SAE 的队伍，一个人 90 分钟就通过一个“奖励模型偏差”特征找到了隐藏目标；但论文也写明，这个解法用更简单的 embedding 语义搜索也能复现（[Marks 等，2025-03](https://arxiv.org/abs/2503.10965)）。OpenAI 用 SAE 比较微调前后的模型，找到一个 “toxic persona” 特征，它最强地控制着“涌现的不对齐”，还能预测模型会不会出现这种行为（[Wang 等，2025-06](https://arxiv.org/abs/2506.19823)）。Google DeepMind 团队也说，SAE 对探索性调试和找数据集里的伪相关有用，值得留在工具箱里。

2026 年两种声音都有：一篇预印本称，用有监督的流程挑选并标注特征后，SAE 在 AxBench 上能接近 LoRA 参考水平（[Jørgensen & Hansen，2026-05](https://arxiv.org/abs/2605.31183)）；Google DeepMind 团队 2025-12 的[务实路线](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability)则主张先用提示、引导、探针，不够再加复杂度。实用的结论是：已经知道要找哪个概念、手里有标签时，先试线性探针或均值差；不知道要找什么、需要无监督地冒出假设时，SAE 才有优势。

## 往哪走：transcoder、跨层字典与自然语言读出

SAE 只重建同一处的激活，不说明计算怎么从一层流到下一层。[Transcoder](https://arxiv.org/abs/2406.11944) 改为用 MLP 的输入预测它的输出，作者报告在 120M、410M、1.4B 的模型上，稀疏度、忠实度和可解释性至少不输 SAE；Anthropic 的 [crosscoder](https://transformer-circuits.pub/2024/crosscoders/index.html)（2024-10）让一个特征跨多层读写，也用来比较两个模型。跨层 transcoder 是[第 3 篇](circuits.md)里归因图的基础。

<!-- paper-figure:transcoder-vs-sae -->

另一条路是放弃固定字典，直接让模型把激活翻译成文字。Anthropic 的 [NLA](https://transformer-circuits.pub/2026/nla/index.html)（2026-05-07）用两个从目标模型初始化的模块，一个把激活写成解释，一个从解释还原激活，用 RL 联合训练以还原残差流。作者报告它在 Claude Opus 4.6 的部署前审计里有用，同时列出局限：会编造上下文里没有的细节，说不清解释依据的是激活的哪一部分，两个模块表达能力太强，训练和使用都贵。

## 这条线接下来看什么

1. **在强基线面前能不能赢？** 截至 2026-10，Kantamneni 等、Google DeepMind 和 AxBench 的结果都是 SAE 不占优。能改变判断的结果：新的字典方法在事先固定的代理任务上，稳定胜过线性探针和均值差。
2. **标签能不能信？** Heap 等（2025-01）的随机模型基线、McCann（2026-05）的撞车问题都说明现有打分会高估。能改变判断的结果：主要的 SAE 发布开始默认报告随机模型基线，并用能惩罚撞车的指标。
3. **特征稳不稳？** 只换种子、约 30% 共同特征（2025-01）。能改变判断的结果：跨种子、跨架构重合度明显更高的训练方法，并说明这些共同特征在模型里被用到。
4. **字典还是文字？** NLA 论文报告，在它自己的自动审计基准上，配 NLA 的 agent 胜过配 SAE 的基线（2026-05，Anthropic 自报）。能改变判断的结果：第三方在其他模型和基准上复现这个比较，并量化编造的比例。

下一篇：[电路、归因与因果干预](circuits.md)。
