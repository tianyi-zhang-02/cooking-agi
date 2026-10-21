# DeepSeek 精读：把架构、系统与推理训练放在一起

**中文** · [English](deepseek.en.md)

> 阅读时间：约 13 分钟 · 类型：模型家族精读 · 最近审阅：2026-10

<div class="lesson-recipe advanced">
  <div><span>核心问题</span><strong>怎样在大容量模型里控制 attention、FFN 与 reasoning 的成本？</strong></div>
  <div><span>重点组件</span><strong>MLA · fine-grained MoE · shared expert · MTP</strong></div>
  <div><span>训练主线</span><strong>V3-Base → R1 多阶段训练；小模型蒸馏是另一条分支</strong></div>
  <div><span>读完要会</span><strong>不要把 V3 的架构创新和 R1 的 post-training 混成一个故事</strong></div>
</div>

## 一句话定位

读 DeepSeek，值得留意的是架构、训练方法和系统实现怎样配合，也就是 **co-design**：MLA 减少 KV cache，MoE 增大 FFN 的总容量，训练与通信的优化让这套架构能高效运行；R1 则进一步研究推理能力的后训练。只记住“用了 MoE”，会漏掉很多关键设计。

## 先把两条线分开

<figure class="worked-update" id="deepseek-architecture" lang="zh-CN">
  <figcaption>先看 V3 怎么计算和训练</figcaption>
  <ol>
    <li><strong>MLA</strong><span>缩小每个 token 的 KV 缓存；不是把整段历史合成一条记录。</span></li>
    <li><strong>DeepSeekMoE</strong><span>用较少的激活专家提供 FFN 计算，同时保留更大的总容量。</span></li>
    <li><strong>MTP 与训练系统</strong><span>增加后续位置的预测目标；并行与通信实现决定这些计算怎样运行。</span></li>
  </ol>
</figure>

<figure class="worked-update worked-update--pairs" id="deepseek-training-routes" lang="zh-CN">
  <figcaption>再看从 V3-Base 出发的两条后训练路线</figcaption>
  <ol>
    <li><strong>V3 对话模型</strong><span>从 V3-Base 做 SFT 和 RL，改善回答行为。</span></li>
    <li><strong>R1 与小模型蒸馏</strong><span>从 V3-Base 做多阶段后训练。另用 R1 生成、筛选文本，给 Qwen / Llama 小模型做 SFT。</span></li>
  </ol>
</figure>

V3 的对话后训练与 R1 是从 V3-Base 出发的不同路线，不是“先训出 V3 对话模型，再接一个 R1 模块”。下面先讲 V2 / V3 / R1；V4 的新注意力结构不能直接套进这张图。

这张图区分的是权重起点，不代表两条路线没有数据联系：V3 报告明确说，部分推理 SFT 数据借助了内部 R1 模型。[V3 §5.1](https://arxiv.org/html/2412.19437v2#S5.SS1)

V3 和 V4 之间还有两次更新，本文不展开。2025-08-21 的 [V3.1](https://api-docs.deepseek.com/news/news250821) 把 thinking 与 non-thinking 合进同一个模型，API 里 `deepseek-reasoner` 走 thinking、`deepseek-chat` 走 non-thinking。2025-12 的 [V3.2 报告](https://arxiv.org/abs/2512.02556)在 MLA 旁边加了 DeepSeek Sparse Attention（DSA）：先用便宜的 indexer 挑出 top-k 位置，主 attention 只读这些位置。DSA 的复杂度和错选代价，见 [MLA 与稀疏注意力](../deep-dives/latent-and-sparse-attention.md)的 DSA 一节。

**版本范围：**本文按 V2 报告 v5、V3 报告 v2 和原始 R1 报告 v1 核对；不把结论自动推广到 Lite、后续小版本或带日期的 checkpoint。以下官方 `config.json` 读取于 2026-10-10，`main` 不是不可变版本。R1 报告后来发表于 [Nature 645, 633–638](https://doi.org/10.1038/s41586-025-09422-z)（2025），arXiv 也在 2026-01-04 更新到 [v2](https://arxiv.org/abs/2501.12948v2)：R1-Zero 在 §2，R1 的两轮 RL 在 §3，数据配方移到附录 B.3。下面的四阶段顺序在 v2 里不变；“第三步重新微调 V3-Base”出自 v1 §2.3.3，v2 的正文和附录 B.3 没有再写明这一步的起点。

| 配置 | Attention | MoE 层中的 routed / 每 token 选中 / shared | 不能混用的细节 |
| --- | --- | --- | --- |
| [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json) | MLA | 160 / 6 / 2 | softmax affinity；首层是 dense FFN |
| [原始 DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) | MLA | 256 / 8 / 1 | sigmoid affinity；前三层是 dense FFN |
| [原始 DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/config.json) | V3 型 MLA | 256 / 8 / 1 | 这是大模型配置，不是 R1-Distill 的架构 |

## 真正值得抓住的三件事

1. **MLA 少存的是每个 token 的 KV，不是把整段历史合成一个向量。** 每来一个 token，仍要追加低维 latent 和解耦的位置 key。历史从 1,000 个 token 变成 2,000 个，缓存条目也随之增长。部分投影可在推理时吸收，不一定先展开完整 K/V。[V2 §2.1](https://arxiv.org/html/2405.04434v5#S2.SS1)
2. **MoE 分开了总容量与每 token 的计算。** 多个 routed experts 提供容量，token 只调用少数；shared expert 的设计目的是承接可复用的模式，不保证自然学出清晰的领域分工。还要付路由、通信和存储成本。
3. **R1 研究的是后训练能改变什么。** R1-Zero 从已经预训练的 V3-Base 开始做 RL，不是从随机权重出发。它不能证明预训练不重要，也不能把奖励之外的所有能力一并归功于 RL。

“细粒度”是把宽专家切成更多窄专家、每次多选几个：总 expert 参数和每 token 的主要乘加量不变，变的是可选组合，以及路由和通信开销。8 选 2 与 32 选 8 的对照算例，以及额外加 shared expert 为什么会增加预算，见[细粒度专家与共享专家](../moe/fine-grained-and-shared.md)。

### MLA 的缓存账：一份 latent，不是两份 {#mla-cache-ledger}

设 batch 为 $B$、缓存 token 数为 $T$、层数为 $L$、KV latent 宽度为 $d_c$、解耦 RoPE key 宽度为 $d_R$，两部分都用每元素 $s$ 字节。紧凑 MLA 缓存的逻辑载荷是：

$$
\text{bytes}=BTL(d_c+d_R)s.
$$

这里没有额外乘 2：K、V 共用同一份 latent；位置 key 也跨 query heads 共享。令 $q$ 为内容 query、$c$ 为缓存 latent、$U_K$ 为内容 key 的上投影，则 $q^T U_Kc=(U_K^Tq)^Tc$，所以可以变换 query 后直接读 latent。RoPE 走解耦分支，不能随意把带位置旋转的矩阵也吸收进去。[V2 §2.1](https://arxiv.org/html/2405.04434v5#S2.SS1)

```python
def mla_cache_bytes(batch, tokens, layers, latent_width, rope_width, bytes_per_element):
    dimensions = (batch, layers, latent_width, rope_width, bytes_per_element)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("Expected positive integer dimensions and element bytes")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("Expected a nonnegative integer token count")
    return batch * tokens * layers * (latent_width + rope_width) * bytes_per_element

assert mla_cache_bytes(1, 8192, 61, 512, 64, 2) == 549 * 2**20
```

用 V3 配置的 `61` 层、`512 + 64` 维，假设缓存用 BF16（2 bytes/element），单请求 8,192 token 得到 **549 MiB**（1 MiB = $2^{20}$ bytes）。这是推导的载荷，不是实测显存；不含权重、MTP、分页空位、量化元数据、临时张量或并行副本，也不代表所有运行时都采用紧凑存法。MLA 本身仍会关注可见历史，并没有因缓存变窄就变成稀疏选 token。

## V3 的负载均衡：选谁和给多少权重，不是同一个分数

V3 用原始 affinity 加负载 bias 来选 routed expert，但混合输出仍使用选中 expert 的原始 affinity，再归一化。过载会降低 bias，欠载会提高它。论文还保留很小的 sequence-wise balance loss，所以“auxiliary-loss-free”不能读成全模型没有任何均衡辅助项。[V3 §2.1.2](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS2)

用一个自拟的 3-expert 例子算一遍。affinity 为 `[0.6, 0.3, 0.1]`，选择 top-2；为了给第三个 expert 更多机会，bias 为 `[0, 0, 0.4]`。

| 步骤 | 数值 | 结果 |
| --- | --- | --- |
| 选择分数 | `[0.6, 0.3, 0.5]` | 选第 1、3 个 expert |
| 混合权重 | 原始的 `0.6` 和 `0.1` 归一化 | `6/7` 和 `1/7` |
| 错误做法 | 把选择分数直接当混合权重 | 变成 `6/11` 和 `5/11` |

如果两个选中 expert 的输出分别是 2 和 10，正确混合为 `22/7`，错误混合为 `62/11`。一次看似无关紧要的“复用分数”，改变了模型的前向结果。

```python
import math

def route_with_bias(affinities, biases, selected_count):
    if len(affinities) != len(biases) or not affinities:
        raise ValueError("affinities and biases must have matching nonzero lengths")
    if type(selected_count) is not int or not 1 <= selected_count <= len(affinities):
        raise ValueError("invalid selected_count")
    if not all(math.isfinite(value) for value in [*affinities, *biases]):
        raise ValueError("scores must be finite")
    if any(value <= 0 for value in affinities):
        raise ValueError("this example expects positive affinities")
    selected = sorted(
        range(len(affinities)), key=lambda index: (-(affinities[index] + biases[index]), index)
    )[:selected_count]
    denominator = sum(affinities[index] for index in selected)
    return [(index, affinities[index] / denominator) for index in selected]

routed = route_with_bias([0.6, 0.3, 0.1], [0.0, 0.0, 0.4], 2)
assert [index for index, _ in routed] == [0, 2]
assert math.isclose(sum(weight * [2, 0, 10][index] for index, weight in routed), 22 / 7)
```

代码只演示 routed 分支，不含 shared expert、节点限制、完整缩放和通信。选择偏置在上一步负载统计之后调整，不是给当前 token 添加一个“更相关”的证据。看性能时还要检查每卡接收多少 token：平均 expert 负载接近，并不保证跨节点通信均匀。

另一组加上 bias 后改变专家选择的算例，以及为什么要分别统计专家负载和卡间负载，见[负载均衡](../moe/load-balancing.md#bias-example)。

### MTP：多一个目标，不是跳过验证 {#mtp-causal-chain}

V3 的 MTP 深度为 1：主干状态 $h_i$ 预测 $t_{i+1}$；额外模块把 $h_i$ 与真实的 $t_{i+1}$ 的 embedding 结合，再预测 $t_{i+2}$。模块按因果顺序连接，embedding 和输出 head 与主干共享，不是几个互不相干的预测头。[V3 §2.2、§4.2](https://arxiv.org/html/2412.19437v2#S2.SS2)

对自拟序列 `A B C D`，主干在 `A` 处预测 `B`；MTP 接收 `B` 后预测 `C`。`B` 是它的条件，不是它的答案；把目标 `C` 也喂进去才是泄漏。下面只列单条、无 padding 序列的标签对齐，不实现模型或 loss；packed 数据还必须屏蔽跨文档目标：

```python
def mtp_training_rows(tokens):
    return [
        (tuple(tokens[:position + 1]), tokens[position + 1], tokens[position + 2])
        for position in range(len(tokens) - 2)
    ]

assert mtp_training_rows(["A", "B", "C", "D"]) == [
    (("A",), "B", "C"), (("A", "B"), "C", "D")
]
```

训练时额外目标加入主干 next-token loss；推理时可以丢弃 MTP，仍按普通自回归方式运行，也可将它用于 speculative decoding。后者必须由主模型验证草稿，不能直接把多个预测都当作已接受输出。训练目标、草稿接受率和端到端延迟是三件事，不存在“预测两步就必然快两倍”的推论。

## R1 的四个阶段，不要把蒸馏算进去

按 [R1 原始报告 §2.3–2.4](https://arxiv.org/html/2501.12948v1#S2.SS3)读，会清楚很多：

| 阶段 | 主要作用 |
| --- | --- |
| Cold-start SFT | 先给 V3-Base 少量可读的长推理示范 |
| Reasoning-oriented RL | 用可检验的任务奖励继续训练推理行为 |
| Rejection sampling + SFT | 从 RL checkpoint 采样、筛选，再混入通用任务数据；用这些数据重新微调 V3-Base |
| All-scenarios RL | 进一步兼顾推理、帮助性与安全等目标 |

第三步尤其容易画错：**生成数据的 checkpoint 和接收 SFT 的起点不是同一个角色**。这是一条既传模型权重、也传训练数据的流程，不能全部画成一条连续更新的箭头。

小模型蒸馏则另起一条线：拿筛选后的数据训练 Qwen / Llama。原报告里的这些 distilled models 使用 SFT，不是把 V3 的 MoE 压缩成小 MoE，也不是每个小模型都再跑一遍 R1 的 RL。

### 奖励能检查结果，不一定能检查理由

R1-Zero 的规则奖励检查答案准确性和格式；R1 后续不同阶段还涉及语言一致性与通用偏好目标。不能把最初阶段的“规则奖励”概括为整个 R1 从头到尾只看数学答案。[R1 §2.2–2.3](https://arxiv.org/html/2501.12948v1#S2.SS2) 只奖励答案时，回答为什么会自己变长，见[长思维链是怎样训练出来的](../../11-frontier/test-time-compute/long-chains.md)的 R1-Zero 一节。

自己造个更小的例子：题目问两个整数相加，输出恰好是正确的 12，但解释写成“5 + 8”。只核对最后数字的 checker 会给分，逐步检查却能发现错误。相反，一个正确算法如果输出了不合约定的单位，脆弱的 parser 也可能判错。于是要分别检查 **verifier 是否可靠、奖励是否覆盖需求、训练是否真的利用了这些信号**。verifier 的误判率怎样限制 best-of-N 的收益，见[多采样、投票与验证器](../../11-frontier/test-time-compute/parallel-and-verify.md)。

做本地验证时，至少准备四类测试样例：正确答案、格式变化但语义正确、结果正确但理由错误、结果错误但语气流畅。不要只用训练时的同一个 checker 来做评估；还需要独立的检查方法。

再看 rejection sampling：假设每题采样 4 次，其中一道题 4 次全对，另一道只有 1 次对。如果把 5 条全部作为独立 SFT 样本，第一题的监督权重就高 4 倍；如果每题只留 1 条，两题权重才相同。这个例子不是报告的数据统计，它说明“只留正确答案”仍然留下采样与权重选择。数据筛选也属于训练目标的一部分。

## 接着读 V4：压缩的对象变了

[V4 专篇](deepseek-v4.md)继续拆 CSA / HCA 混合注意力、mHC 与后训练。几个词不能混用：序列压缩不同于 MLA 的通道压缩；mHC 的残差映射约束不同于 Muon 的更新矩阵正交化；局部窗口大小固定，也不代表整个模型的历史缓存恒定。

报告中的实现与性能还需要分开看。压缩比不是无损保证，正交化不是收敛保证，某个长度下的 FLOPs 比例也不是所有请求的延迟比例。V4 之后的 V4.1-Flash 换了结构，部分旧 API 名称也已转到新权重，但不能一概而论：截至 2026-10-21，官方模型表与 9 月公告对 `deepseek-v4-pro` 的对应版本仍有不同说法。具体见 V4 专篇开头的说明，以及[官方模型表](https://api-docs.deepseek.com/quick_start/pricing/)与[9 月公告](https://api-docs.deepseek.com/news/news260910)。

## 取舍表

| 选择 | 得到什么 | 付出什么 |
| --- | --- | --- |
| MLA | 更小的逻辑 KV 载荷；紧凑实现可减少缓存读写 | 实现更复杂，实际延迟还取决于 kernel 与并行方式 |
| Fine-grained MoE | 总容量大、每 token 激活较少 | all-to-all、路由和负载均衡困难 |
| Multi-token prediction | 更密的训练信号与潜在 decode 收益 | 训练目标和实现更复杂 |
| Reasoning RL | 用奖励优化任务表现；长链不等于正确 | rollout 成本、reward 边界与行为退化风险 |

## 我会怎样使用这个家族

读 DeepSeek 时，我会特别注意不要把不同来源的提升混为一谈。分析任何结果时，我会先问：这是 cache / compute 的系统收益，expert capacity 的架构收益，预训练数据收益，还是 reasoning post-training 的行为收益？如果消融实验还不足以区分，就先不急着下结论。

## 自检

想把上面的组件算一遍，可以接着读 [MLA 与稀疏注意力](../deep-dives/latent-and-sparse-attention.md)、[MTP 的标签与因果关系](../deep-dives/multi-token-prediction.md)，以及 [FFN 与 SwiGLU](../core/ffn-and-gates.md)。这些讲机制，不能代替对具体模型版本的核对。

- MLA 为什么主要改善 KV cache，而不是直接证明 reasoning 更强？
- MoE 省了哪些计算，又把成本转移到了哪里？
- R1-Zero 与 R1 的训练差别说明了什么？
- 为什么蒸馏后的 dense model 不能被简单称为“小号 R1 架构”？

## 原始资料

- [DeepSeek-V3 Technical Report，v2](https://arxiv.org/abs/2412.19437v2)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning，v1](https://arxiv.org/abs/2501.12948v1)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model，v5](https://arxiv.org/abs/2405.04434v5)
