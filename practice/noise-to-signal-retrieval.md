# 从噪声反馈到可服务的现代检索系统

**中文** · [English](noise-to-signal-retrieval.en.md)

> 这是一份公开且刻意抽象化的设计笔记，只讨论通用问题与取舍，不对应任何公司的数据规模、字段、模型配置、内部术语或生产系统。脱敏边界见[板块说明](README.md)。

## 先用一句话讲清楚

现代检索系统真正困难的地方，不只是换一个更大的模型，而是：

> **怎样从极少、受旧策略影响的正反馈中学习一个可扩展的 proposal function，在海量未观测内容里找到更可能产生正反馈的候选，同时不把“模型高分”误当成新的真实标签。**

## Problem Framing：从稀疏正样本到候选假设

对用户 $u$，设 $\mathcal{C}$ 是全部可检索内容，$\mathcal{E}_u\subset\mathcal{C}$ 是旧策略真正曝光过的内容，而 $\mathcal{P}_u\subset\mathcal{E}_u$ 是其中获得可靠正反馈的极小子集。系统面对的不是普通分类问题，因为 $\mathcal{C}\setminus\mathcal{E}_u$ 中绝大多数内容**没有标签，而不是负样本**。

Retriever 学习一个打分函数 $f_\theta(u,c)$，再从未观测空间提出候选：

\[
\mathcal{H}_u
=\operatorname{TopK}_{c\in\mathcal{C}\setminus\mathcal{E}_u} f_\theta(u,c).
\]

$\mathcal{H}_u$ 是 **positive hypotheses**，不是新的 positives。它们只有经过 downstream ranker、真实曝光和行为验证后，才可能形成新的训练信号。因此整套系统可以抽象为四个职责：

1. **Signal construction**：判断历史日志中哪些反馈足以成为监督；
2. **Retriever**：在巨大未观测空间中低成本提出高潜力候选；
3. **Ranker / policy**：结合请求级上下文决定哪些候选真正获得曝光；
4. **Evaluation / experiment**：验证候选是否带来用户价值，并将新证据写回下一轮训练数据。

所以更准确的名字不是“自动寻找 positive labels”，而是 **candidate discovery from sparse, selectively observed feedback**。Retriever 是 hypothesis generator；线上行为才是新的 evidence generator。

```mermaid
flowchart LR
    A["稀疏且有偏的行为反馈"] --> B["Signal Construction<br/>置信度 · 曝光上下文 · 负样本类型"]
    C["帖子文本 + 图片"] --> D["轻量 Image Router"]
    D -->|"信息量低"| E["跳过视觉增强"]
    D -->|"信息量高"| F["离线预训练 VLM<br/>提取有依据的视觉信息"]
    E --> G["Candidate Representation"]
    F --> G
    B --> H["Embedding Retriever<br/>Supervised Contrastive Post-Training"]
    G --> H
    I["Profile + History"] --> H
    H --> J["ANN Candidate Hypotheses"]
    J --> K["Existing Downstream Ranker"]
    K --> L["Controlled Exposure"]
    L --> M["Behavioral Validation"]
    M -.->|"new evidence, not automatic truth"| A
```

## 1. 先把 Noise 变成 Signal

日志中的点击、停留和跳过不是天然的偏好标签。用户只会对旧系统展示过的内容产生反馈；没有互动，也可能只是没有看到。

训练样本至少要区分：

- **可靠正样本**：有较强行为证据支持；
- **曝光负样本**：确实获得展示机会，但出现明确负向行为；
- **Hard negative**：语义接近或模型高分，但与当前用户上下文不匹配；
- **未观测候选**：没有足够证据判断相关或不相关，不能全部标成负样本。

这一步比先争论 SFT、DPO 还是 RL 更重要。训练算法只能放大收到的监督；如果标签语义不清楚，更复杂的 post-training 只会更稳定地学习错误目标。

## 2. 选择性理解图片，而不是全量运行大模型

Text-only embedding 会漏掉截图、图表、海报和普通照片里的关键信息，但对所有图片运行大 VLM 又会浪费计算。

先用轻量 router 判断处理需求：

```text
image
→ natural photo / screenshot / slide-like / chart / poster / decorative
→ 根据类型和置信度分配计算
```

Router 不需要完整理解图片，只需要回答：**这张图是否包含额外信息，以及应该走哪条处理路径？**

对于高信息量图片，再使用预训练 VLM 提取：

- 可见文字与关键实体；
- 图片、截图或图表表达的主要内容；
- 原文没有明确写出的视觉证据；
- 图文是否一致，以及判断置信度。

第一版通常不需要微调 VLM。先用固定 schema 做结构化抽取；只有出现稳定、可复现且影响下游检索的领域错误时，才考虑 LoRA 或 SFT。

## 3. Post-Training：究竟训练什么

VLM 不是在线检索模型，而是一个离线、可替换的 **visual teacher**。它在内容创建或更新时读取图片，并输出带来源和置信度的结构化证据：

```json
{
  "visual_type": "photo | screenshot | chart | poster | decorative",
  "visual_evidence": ["visible entities", "readable text", "visual claim"],
  "text_image_relation": "support | complement | conflict | irrelevant",
  "confidence": "calibrated score",
  "provenance": "which region supports each claim"
}
```

原文与这份证据再交给轻量 candidate encoder，生成可提前计算并写入 ANN index 的向量。这样，VLM 提供更完整的监督和内容理解，但不会进入每次请求的在线路径。

### 模块怎样连接

```mermaid
flowchart TB
    subgraph C["Content understanding · offline"]
        A["Text + image"] --> B["Image router"]
        B --> C1["Frozen or lightly adapted VLM teacher"]
        C1 --> D["Grounded evidence contract"]
        A --> E["Candidate tower<br/>Qwen3-Embedding-0.6B"]
        D --> E
    end
    subgraph U["用户理解 · 准实时"]
        F["用户画像 view<br/>Qwen3-Embedding-0.6B"] --> H["多个用户 view"]
        G["行为历史 view<br/>Qwen3-Embedding-0.6B"] --> H
        H --> I["Learnable routing"]
    end
    E --> J["Candidate vector"]
    I --> K["Contrastive retrieval score"]
    J --> K
    K --> L["ANN retrieval"]
    L --> M["Existing ranker"]
```

### 模型选择：为什么是 Qwen3-Embedding-0.6B

这里的模型选择不是“找 benchmark 最大的模型”，而是为每个模块选择刚好足够的能力：

| 模块 | 起始选择 | 选择标准 | 为什么不直接用更大的模型 |
| --- | --- | --- | --- |
| Image router | 小型视觉分类器或 SigLIP-like encoder | 内容类型、信息量与置信度校准 | Router 只决定是否调用 teacher，不负责完整理解 |
| Visual teacher | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) 作为快速基线，4B 作为容量对照 | grounding、OCR、图文冲突、schema adherence 与置信度 | 离线选择性调用已经足够；先避免领域 SFT 和线上生成成本 |
| Candidate/用户塔 encoder | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 检索质量、吞吐、长文本、向量大小与训练成本 | 4B/8B 可以作为容量上界，但会显著增加训练、准实时 编码与迭代成本 |
| Downstream ranker | 复用已有 ranker | 请求级 relevance、quality、freshness | First-stage retriever 不需要重复承担精排职责 |

Qwen3-Embedding-0.6B 是一个合理的起点，因为官方模型提供 **0.6B 参数、32K context、最高 1024 维 embedding、MRL 可变维度和 instruction-aware encoding**。这些特性分别对应实际约束：

- **0.6B**：足以从通用 embedding 初始化，同时允许频繁重训；相较更大模型，也更适合 准实时 刷新 用户 representation；
- **32K context**：能容纳结构化 profile 和较长 history，但不代表每次都应塞满；输入仍要按 evidence value 截断；
- **Instruction-aware**：用户/query 侧可明确任务，例如“表示长期兴趣”或“表示近期行为”，candidate 侧保持稳定的内容语义；
- **MRL / 32–1024 维**：模型容量与 ANN index 大小可以分开调节。先保留 1024 维质量基线，再在固定候选预算下比较 512/256 维，而不是凭感觉选择向量宽度。

通用 benchmark 只能说明它适合作为初始化，不能证明它适合某个具体反馈分布。真正的选择仍由后面的 lifecycle、breadth、complementarity、latency 和 freshness gates 决定。[Qwen 的技术介绍](https://qwenlm.github.io/blog/qwen3-embedding/)也将该系列定位为 dual-encoder embedding 与 cross-encoder reranking 两类模型；这里选择前者，是因为 first-stage retrieval 必须预计算 candidate vectors 并使用 ANN。

VLM 的选择遵循另一套标准。`Qwen3-VL-2B-Instruct` 适合作为第一版 teacher，不是因为它“会聊天”，而是因为 2B 规模允许批量离线处理，同时可以用固定 prompt 输出 OCR、实体、视觉主张、图文关系和 evidence provenance。4B 或更大模型只在同一套 held-out grounding/conflict set 上作为容量对照；如果 2B 已满足 schema 与校准门槛，就没有必要把更大模型加入默认路径。

也可以评估直接使用 [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) 产生 multimodal candidate vectors，但它解决的是另一种取舍：端到端表示更直接，却会把视觉理解、向量空间和索引刷新绑在同一个模型版本上。默认的 **VLM teacher → grounded evidence → Qwen3-Embedding-0.6B** 两阶段设计更容易审计、缓存、降级和独立升级，也允许纯文本内容复用同一个 ANN index。

### 参数怎样共享

这是一个**非对称 dual-encoder**：用户 与 candidate 最终位于同一个向量空间，但输入分布和更新频率不同。

```text
Candidate tower: post text + grounded visual evidence → z_c
用户 views:    instructed profile / history / latent intent → z_u,r
Similarity:      cosine(z_u,r, z_c), after identical pooling + L2 normalization
```

一个实用起点是：所有塔从同一个 Qwen3-Embedding-0.6B checkpoint 初始化，candidate 与 用户 两侧使用独立参数或 adapter；profile/history 在 用户 侧共享主干，再使用不同 instruction、adapter 或 projection head。这样既保留共同语义空间，也允许不同输入学习不同的压缩方式。

必须固定 train/serve parity：同一 tokenizer、instruction template、last-token pooling、L2 normalization、截断规则和视觉 evidence schema。否则离线训练的相似度不再等于线上 ANN 使用的相似度。

最容易被忽略的一处 parity 破坏不在这个列表里：**candidate index 相对 encoder 是会过期的**。重训 candidate tower 之后，索引里每一个向量都要重新编码，而在灰度期间索引是新旧向量混合的——此时向量空间不一致，相似度不可比。索引重建策略和 evidence table 的版本号，必须和模型版本绑在同一个 manifest 里。

用户侧不必把所有信息平均成一个点。Profile 与 history 可以保留为不同 view，也可以继续展开成多个 latent intent vectors。Router 根据上下文决定各 view 的权重；它的价值不在于“塔更多”，而在于不同 view 是否提供**互补的相关候选**：

- profile 提供稳定但通常更弱的长期证据；
- history 提供更强但可能更窄的近期证据；
- latent views 防止少数兴趣被一个平均向量稀释；
- routing 只有在不同 view 带来 unique relevant candidates 时才值得增加复杂度。

Serving 时不能先把这些向量平均掉，否则又回到了单点表示。更直接的做法是在一个固定总预算 $B$ 下分配每个 view 的查询额度：

\[
K_r = \operatorname{round}\!\left(B\cdot \operatorname{softmax}(g(u))_r\right),
\qquad
\mathcal{C}(u)=\bigcup_r \operatorname{ANN}(\mathbf{z}_{u,r},K_r).
\]

各 view 分别查询同一个 candidate index，随后 union、去重，并把来源 view、相似度和 router weight 交给 downstream ranker。这样多意图增加的是候选互补性，而不是无界增加在线候选量。

### 训练分成三层

**第一层：构造监督。** 从可靠正样本、曝光负样本、语义 hard negatives 和未观测候选中构造带置信度的训练批次，避免把“没看见”误当成“不喜欢”。

**第二层：监督式对比 post-training。** 用 用户–candidate 配对训练 embedding retriever；in-batch negatives 提供规模，曝光负样本保留策略上下文，hard negatives 教模型区分“语义相近”和“当前真正相关”。

**第三层：防止模块坍缩。** 单独约束 router 和模态使用：

- 对视觉信息冗余的样本，完整输入和 text-only 输入应保持相近；
- 对视觉信息关键的样本，遮掉图片后分数应该发生可解释的变化；
- 对图文冲突样本，模型应降低置信度或保留冲突标记，而不是静默地把两者拼在一起；
- 对多 view 用户 representation，监控路由熵、view 使用率与 unique-target contribution，避免所有样本都退化到同一个塔。

一个抽象目标可以写成：

\[
\mathcal{L}
=\mathcal{L}_{\text{retrieval}}
+\lambda_{r}\mathcal{L}_{\text{routing}}
+\lambda_{m}\mathcal{L}_{\text{modality}}
+\lambda_{c}\mathcal{L}_{\text{calibration}}.
\]

这里的核心仍是 **supervised contrastive fine-tuning**，而不是直接对固定候选检索使用生成式 GRPO。只有当任务变成跨多轮、需要优化长期 slate reward 或 exploration policy 时，RL 才解决了一个不同且合理的问题。

### 一次完整的训练与发布怎样运行

1. **冻结时间点**：按事件时间生成 用户快照、content snapshot 和 exposure context，防止使用未来信息；
2. **生成视觉证据**：router 决定哪些图片调用 teacher，结果写入带版本号的 evidence table；
3. **编译训练样本**：将行为标签、负样本类型、用户 views、原文和视觉证据绑定到同一 manifest；
4. **训练 retriever**：先冻结 teacher，只更新 用户塔 encoder、candidate encoder 与 router，分别记录每项 loss 和各 view 使用率；
5. **导出与回放**：批量生成 candidate vectors，建立隔离的 ANN index，在固定 ranker 上执行离线 replay 和 ablation matrix；
6. **逐级发布**：通过 representation、retrieval、slice 和 systems gates 后，再进入 shadow traffic 与受控线上验证。

如果某张图片解析失败，流水线必须回退到原文表示；如果某个 用户 view 缺失，router 必须对剩余 view 重新归一化。**Fallback 是训练和 Serving contract 的一部分，而不是上线后的补丁。**

Retriever 负责扩大高质量候选空间；已有 downstream ranker 继续负责请求级别的精细 relevance、quality、freshness 与最终排序。

## 4. 把复杂计算留在 Offline

```text
Offline:  image routing → selective VLM → candidate embedding → ANN index
准实时: profile/history update → cached 用户 representation
Online:   cache lookup → ANN retrieval → existing ranker → final slate
```

这样，多模态能力增加的是候选理解，而不是每次请求的生成式推理成本。视觉处理失败时应逐级回退到可见文字，最终回退到纯文本 embedding，不能阻止新内容进入索引。

## 5. Evaluation：怎样证明每个模块真的有用

> 本节描述的是实验协议和发布门槛，不包含任何观察到的结果。

聚合分数可能掩盖用户群体退化，也可能把“结果更相关、但语义越来越窄”误写成全面提升。评估因此要回答五个独立问题。

### 5.1 模型真的使用了视觉信息吗

只比较 text-only 和 multimodal 两个总分不够，因为模型可能只是利用文本先验。需要构造配对的 counterfactual tests：

| 输入条件 | 目的 | 应观察什么 |
| --- | --- | --- |
| 完整图文 | 正常路径 | 基准排序与置信度 |
| 遮掉图片 | modality ablation | 视觉关键样本的排序是否有针对性地变化 |
| 遮掉文字 | image-only probe | 图片能否独立提供最低限度的语义证据 |
| 随机交换图片 | prior control | 无关图片是否会错误影响排序 |
| 保留图片但删除 VLM evidence | module ablation | 收益究竟来自 router、teacher 还是 encoder |

最关键的集合不是随机样本，而是 **visual-essential subset**：文本本身不足以区分候选，而图片包含任务所需信息。只有在这里发生方向正确、可归因的变化，才能说明模型真的使用了视觉信号。

### 5.2 图文冲突时会发生什么

构造语义匹配但事实冲突的 pair，例如正文描述一个对象，而图片展示另一个对象；同时加入无冲突的 matched controls。评估三个层面：

1. teacher 是否输出 `conflict`，并指出冲突来自哪里；
2. candidate encoder 是否避免把矛盾证据压成一个过度自信的向量；
3. 最终 retrieval/ranking 是否降低不可靠候选的置信度，而不是仅仅提高一个中间 conflict-classification 分数。

### 5.3 新模态是否改善最终任务

所有实验固定候选池规模、ANN 预算、下游 ranker 和评估样本，只替换被测模块。建议按下面的阶梯逐层比较：

```text
T0  text-only candidate representation
T1  + image router
T2  + selective VLM evidence
T3  + modality-aware post-training
T4  + multi-view 用户 routing
```

每一级都同时记录：

- **最终任务**：Recall、NDCG、有效候选命中与 downstream ranker 可选择的 relevant set；
- **breadth**：topic coverage、within-list similarity、长尾候选覆盖；
- **complementarity**：新增模块贡献的 unique relevant candidates；
- **中间诊断**：router accuracy、grounding、conflict detection，只用于解释最终任务变化；
- **systems**：VLM 调用率、单内容处理成本、索引 freshness、缓存命中与在线 latency。

中间指标变好但最终候选质量不变，不足以支持上线；最终任务改善但 freshness 或成本越界，同样不能通过。

### 5.4 不同内容和用户是否都受益

同一个实验需要在预先定义的 slices 上重复，而不是在看到结果后挑群体：

- **内容**：text-rich、visual-essential、screenshot/chart、natural image、长尾主题与语言；
- **用户**：lifecycle、历史长度、反馈强度、兴趣集中度与冷启动程度；
- **交叉切片**：例如 sparse-history × visual-essential，检查新增模态是否只帮助数据本来就丰富的群体。

报告每个 slice 的 relevance、breadth、coverage、calibration 与失败率，并附样本量和置信区间。这里的“公平受益”不是要求所有群体获得完全相同的数值，而是确保总体平均值不会遮住稳定、可解释的 subgroup harm。

### 5.5 一套可执行的发布门槛

```mermaid
flowchart LR
    A["Representation checks<br/>grounding · conflict · ablation"] --> B["Retrieval checks<br/>relevance · breadth · complementarity"]
    B --> C["Slice checks<br/>content × lifecycle × signal"]
    C --> D["System checks<br/>cost · freshness · latency"]
    D --> E["Shadow / replay"]
    E --> F["Controlled online validation"]
```

每次实验都应绑定同一份 manifest：数据快照、标签定义、teacher 与 encoder 版本、negative sampler、ANN index、ranker 版本、slice 定义和随机种子。这样观察到的变化才能归因到具体模块，而不是数据或评估配方漂移。

User simulation 可以用于 repeated exposure、topic fatigue 和 exploration 的压力测试，但它不能替代真实用户。它更适合作为 online test 之前的 failure discovery layer，而不是产生最终性能结论。

## 这套设计真正优化什么

```text
更可靠的训练信号
→ 更完整的用户与内容表征
→ 更多互补的相关候选
→ downstream ranker 拥有更好的选择空间
→ 用多维评估和线上实验验证真实价值
```

目标不是堆叠热门模块，而是让每一层都为最终决策增加新的、可验证的信息。
