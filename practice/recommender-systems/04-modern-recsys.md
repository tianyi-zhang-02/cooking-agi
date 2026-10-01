# 04 · 今天的推荐：变强的是哪一层？

**中文** · [English](04-modern-recsys.en.md)

> 核验日期：2026-09-30。讨论公开代码与论文，不声称知道任何平台完整的线上配置。

## 先别把“新”理解成“全部推倒”

前面读的是 2023 快照。较新的 [X 公开仓库](https://github.com/xai-org/x-algorithm/blob/77d431aabf409ca1c1eed9bec7e2183f7c914e23/README.md)列出 Thunder、Phoenix 和 SimClusters 等来源，区分排序与可见性判断；这不能用旧仓库的模块表直接替代。

[Phoenix 文档](https://github.com/xai-org/x-algorithm/blob/77d431aabf409ca1c1eed9bec7e2183f7c914e23/phoenix/README.md)仍区分双塔检索与更丰富的 Transformer 排序，并说明这一版提供训练和 serving 实现、以及合成数据的本地运行路径。代码公开，不意味着拿到了真实训练数据或能复现生产收益。

## 读一个新模型，先问它替换哪里

读到一个新名字，先写下三件事：**输入是什么、输出是什么、替换旧架构的哪一段**。这比先判断“是不是用了 LLM”更有用。迷路时回到[架构导读](00-architecture.md)，把模型放进对应的框。

```mermaid
flowchart LR
 A["历史行为与内容"] --> B["表示 / 序列建模"]
 B --> C["候选获取"]
 C --> D["多目标打分"]
 D --> E["列表选择与资格检查"]
 E --> F["曝光与反馈"]
 F -. "更新数据与目标" .-> A
```

同样叫 Transformer，可能用于用户历史编码、候选打分，也可能自回归生成 item ID。只看到名字，还不能判断它替换了这条链上的哪一段。

| 方向 | 改了什么 | 读资料时追问 |
| --- | --- | --- |
| 长序列用户建模 | 怎样表示过去的行为与顺序 | 历史是否可用？重计算放在哪里？ |
| 更强的候选交互 | 用户上下文怎样影响每条候选的预测 | 候选能否互相看到？换 batch 会不会变分？ |
| 多模态内容表示 | 图片、文字等提供什么可用信息 | 信息是否增量？模型是否真的使用了该模态？ |
| 生成式检索 | 直接生成内容标识，而非只做向量近邻 | ID 合法性、新内容、解码成本怎样处理？ |

## 三份资料，三个不同问题

1. **X Phoenix：预测会不会被同批候选干扰？** 文档强调 candidate isolation：候选读取用户上下文，而不是互相 attend。我的理解是，这让单候选打分与列表多样性更容易分开分析；不等于跨请求、跨用户的预测都可无条件缓存。[架构说明](https://github.com/xai-org/x-algorithm/blob/77d431aabf409ca1c1eed9bec7e2183f7c914e23/phoenix/README.md)
2. **Meta 多阶段序列模型：重计算放在哪儿？** 2026 年公开文章把较重的用户建模与轻量在线排序分开。这里值得带走的是计算复用与新鲜度的取舍，而不是照搬它的规模或效果数字。[官方工程文章](https://engineering.fb.com/2026/08/05/ml-applications/from-user-sequences-to-scaling-laws-a-multi-stage-architecture-for-metas-ads-ranking/)
3. **TIGER：检索能否变成生成标识？** 论文用 Semantic ID 与序列模型生成下一物品标识。这是一个研究路线，不代表所有平台都已替换 ANN，也不是“让聊天机器人写推荐理由”。[论文](https://arxiv.org/abs/2305.05065v3)

## 什么并没有自动消失

反馈仍受曝光策略影响，新内容仍然缺互动，模型仍有延迟与成本约束。更大的模型可以改善表示能力，但不会凭空告诉你未展示内容的真实偏好。

所以先写一句可检验的假设：“我认为某类信息在当前表示中丢失了”，再比较新旧方案。别只写“换成生成式，因为这是趋势”。

## 自测

<details><summary>用了 Transformer，就算生成式推荐吗？</summary><p>不算。Encoder 可以只产向量，ranker 可以只预测行为概率。要看训练目标、输出是什么以及是否在生成候选标识。</p></details>

<details><summary>新版能用合成数据跑通，为什么不能报生产效果？</summary><p>跑通验证接口、数值与执行路径；真实效果还依赖数据分布、候选池、配置、实验和用户行为，合成数据不能替代。</p></details>

下一期：[评估小实验](05-evaluation-lab.md)。
