# Production ML：从模型到线上结果

**中文** · [English](README.en.md)

一个推荐模型离线分数不错，上线后却可能遇到别的问题：候选里没有用户想看的内容，特征比训练时晚了一小时，或者请求还没算完，用户已经走了。这一组从推荐 Feed 出发，讲模型、数据、服务和实验怎样配合。

这里的 Production ML 指的是**让机器学习系统持续为真实请求提供结果，并能检查和改进这些结果**。不只包括训练，也包括数据更新、发布、监控与故障处理。例子都是虚构的小系统，不对应任何公司的内部实现。

## 从哪里开始 {#route}

| 你现在想弄清什么 | 建议顺序 | 读完做一件事 |
| --- | --- | --- |
| 知道模型，但不熟悉完整产品 | [一次推荐请求](ranking-system.md) → [数据与实验](data-and-experiments.md) | 画出请求和反馈两条线，标清每一步交出什么 |
| 已经做过模型，想查容易漏的环节 | [时间一致性](data-and-experiments.md#time) → [发布版本](ranking-system.md#release) → [实验切片](data-and-experiments.md#experiment) | 找一个“离线变好、线上变差”的原因，设计能区分它的检查 |
| 想动手或准备系统设计 | [请求预算](ranking-system.md#latency) → [最小实验](data-and-experiments.md#exercise) → [容量与故障](../15-ai-infrastructure-and-distributed-systems/queues-and-failures.md) | 给方案加上超时、降级和回滚，而不只画正常路径 |

向量相似度不熟，先看[双塔检索](../04-search/dual-encoder.md)；概率、loss 和训练还没接起来，先读[从线性模型到神经网络](../00-foundations/from-linear-to-neural.md)。不需要先学完大模型架构。

## 第一轮要讲清的三件事 {#core}

| 问题 | 一个常见误判 | 这里怎样展开 |
| --- | --- | --- |
| 召回与排序怎样配合？ | 排序模型更强，就能找回没召回的内容 | 用同一个候选漏斗，分别算每层丢了什么 |
| 哪个指标说明产品变好了？ | 离线 AUC 或总点击率涨了，就可以上线 | 分开模型质量、线上行为和服务质量，再看人群切片 |
| 训练和线上看到的是同一份信息吗？ | 都从同一张表取特征，就不会不一致 | 对照事件时间、数据可用时间、转换版本和实际服务日志 |

例如，一条行为发生在 09:58，却到 10:04 才写进在线特征。拿它训练 10:00 的预测，会让模型在训练时多看见一条当时还拿不到的信息。这个问题不会因为换了更好的排序模型就消失。[数据篇](data-and-experiments.md#time)会把这笔时间账展开。

## 已有笔记怎样接着读 {#deeper}

这里负责把系统串起来，单个模块的推导与实现仍放在原来的文章里。

| 模块 | 深入阅读 | 带着什么问题去读 |
| --- | --- | --- |
| 候选生成 | [召回](../practice/recommender-systems/02-candidate-retrieval.md) · [为什么双塔](../practice/recommender-systems/06-why-two-towers.md) | 哪些东西能提前计算？候选数量怎样影响成本和覆盖？ |
| 排序与重排 | [排序与多样性](../practice/recommender-systems/03-ranking-and-diversity.md) | 分数预测什么？排序后还要满足哪些约束？ |
| 特征与训练 | [数据到模型](../practice/ml-system/data-to-model.md) · [数据在教什么](../01-data-and-feedback/data-as-supervision.md) | 每个标签和特征在什么时候才确定？ |
| 评估 | [评估小实验](../practice/recommender-systems/05-evaluation-lab.md) · [对照与切片](../07-evaluation/ablation-and-slices.md) | 比较时变了哪一个条件？哪些用户没有被总体指标代表？ |
| 发布与反馈 | [服务生命周期](../practice/recommender-systems/08-serving-lifecycle.md) · [上线与反馈](../practice/ml-system/serving-and-feedback.md) | 模型回滚时，特征、索引和配置是否也能一起退回？ |

## 谁负责哪一段 {#ownership}

数据、模型、平台和产品往往由不同的人负责。学全链路不是要求一个人做完所有工作，而是要能说清边界：输入缺了怎么办，版本不匹配由谁拦下，指标异常先找谁，以及修复后如何证明问题确实消失。

下一条[AI Infra 路线](../15-ai-infrastructure-and-distributed-systems/README.md)从同一个产品继续往下拆：请求为什么排队，副本为什么没及时接住流量，训练作业和在线服务又该怎样共享资源。五个方向之间的关系见[系统学习路线](../practice/learning-routes.md)。
