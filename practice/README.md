# 工程实践

**中文** · [English](README.en.md)

这里从具体项目出发，讲数据、模型、服务和评估怎样组成一个系统，以及为什么这样设计。

- **第一次做完整项目：**从[本站与阅读清单](web-lab/README.md)开始，了解 Markdown 怎样发布，再连接网页、API 和数据库。不需要 GPU，也不用先学前端框架。
- **想了解 ML 系统：**读推荐系统、RAG 或后训练实验，顺着数据看各个模块如何配合。推荐系统结合公开代码，其余项目从小规模教学实现开始。
- **准备系统设计面试：**直接看[设计练习](#system-design)，先自己提出方案，再对照项目里的实现。

更多建议顺序见[项目路线](../learn/choose-your-path.md#projects)。工程实践既包括设计，也包括实现和排错；可以先读讲解，有空再动手。

## 选一个问题开始

| 你想了解什么 | 阅读入口 | 重点 |
| --- | --- | --- |
| 学过模型，但不知道完整产品有哪些环节 | [完整 ML 系统](ml-system/README.md) | 三条时间线、数据与服务接口、模块验收和团队分工 |
| 想弄清 GPU、多卡训练和 RL 框架怎么接起来 | [ML / LLM 运行链路](../06-systems/mlsys-llmsys-map.md#learning-route) | 从 CPU / GPU 执行到 Megatron、集群与异步 RL，不需要先租 GPU |
| 想把原理对应到开源库，知道慢了或训坏了怎么查 | [源码拆解](runtime-reading/README.md) | Prefill / decode、vLLM / SGLang 调度、NeMo RL 更新与训练稳定性 |
| 写过代码，但还没把网页、服务和自动检查接起来 | [小项目入门](web-lab/README.md) | 静态站、CI/CD、请求、数据保存与失败提示 |
| 拿到需求，不知道从哪一步设计 | [系统设计练习](../learn/system-design/README.md) | 先明确目标和约束，再比较组件、成本与失败处理 |
| 好内容没有被推荐出来 | [推荐系统拆解](recommender-systems/README.md) | 是没召回、被过滤，还是排序靠后 |
| 资料搜到了，回答还是错的 | [RAG 知识库](rag/README.md) | 资料是否有效，模型是否看到、用对了它 |
| loss 降了，实际表现却没变好 | [后训练实验](post-training/README.md) | 训练目标是否合适，评估是否漏掉了问题 |

推荐、RAG 和后训练这三类项目都要处理数据、成本和评估，但检查方法并不相同。例如，推荐中没有点击不一定是不喜欢；知识库中找不到依据，就不能肯定地给出答案。读的时候可以对照这些差别。

## 系统设计：先自己做一遍选择 {#system-design}

想先有一张全貌图，可以读[从数据到用户](ml-system/README.md)：内容更新、模型训练和在线请求是三条不同的线。再跟着[数据到模型](ml-system/data-to-model.md)看样本、框架、checkpoint 与索引，最后读[上线与反馈](ml-system/serving-and-feedback.md)。不用一个人把每层都做完，但要知道每层交接什么。

可以选 [Feed](../learn/system-design/feed.md)、[RAG](../learn/system-design/rag.md) 或[长期记忆](../learn/system-design/memory.md)。先写需求、数据量和延迟预算，画一个最小方案，再改变一个条件：内容更新更频繁了，权限变了，或者某个服务超时了。这个方案哪里要跟着改？

然后读下面的项目，看看实现里是怎么处理这些问题的。设计题不是标准答案；公开项目也有特定场景，不能把它的所有选择原样搬过来。

## 推荐系统：沿着一次刷新往下看

先看 [00 · 架构全貌](recommender-systems/00-architecture.md)，把请求、内容更新、训练三条时间线分开。然后顺着 [01 · 请求](recommender-systems/01-feed-pipeline.md) → [02 · 召回](recommender-systems/02-candidate-retrieval.md) → [03 · 排序](recommender-systems/03-ranking-and-diversity.md) → [05 · 评估](recommender-systems/05-evaluation-lab.md) 走一遍。

理解流程以后，再问更难一点的问题：为什么用[双塔](recommender-systems/06-why-two-towers.md)，[组件怎么选](recommender-systems/07-component-choices.md)，[线上故障怎么处理](recommender-systems/08-serving-lifecycle.md)？最后看[较新的公开架构](recommender-systems/04-modern-recsys.md)，判断新模型到底改了哪一层。

这一组会具体算：两路召回怎么分配名额，缺失标签怎么处理，以及为什么各组表现没变，总分却能上涨。

## RAG：从找到资料，到答对问题

活动报名原本周五截止，后来改成了周三。助手怎么才能用上新规则？[数据与检索](rag/data-and-retrieval.md)先处理版本和权限，再讲如何合并检索结果，以及上下文长度不够时该保留哪些段落。

[证据与评估](rag/evidence-and-evaluation.md)接着检查回答。引用了真的文档，也可能读错里面的日期。我们会修改日期、删掉支持句、加入冲突资料，逐一说明该怎样测试。

## 后训练：先检查它在学什么

沿用这个助手：资料已经找对，模型却经常漏掉引用，怎么办？[数据与目标](post-training/data-and-objectives.md)从准备示范讲起，算清哪些 token 参与 loss、长回答占多大权重，再检查相似样本有没有混进测试集。

接下来是训练中的具体选择：[SFT、LoRA 与框架](post-training/training-plan.md)、[数据与 batch 构建](post-training/data-pipeline.md)、[多卡训练](post-training/distributed-training.md)和[中断恢复](post-training/checkpoint-and-resume.md)。每一部分都会比较适用条件、成本，以及实现时需要检查的问题。

[实验与发布](post-training/experiments-and-release.md)用 6 道题比较新旧模型。总体多答对一道，可能同时多编了一次没有依据的答案。看清具体变化后，再决定继续改、扩大实验，还是保留旧版。

## 先跑几个小例子

准备部署模型，可以先读[一次 LLM 请求怎样跑完](../06-systems/llm-serving.md)：请求排队、长输入分块、取消请求后的清理，以及增加副本或拆分模型的条件。理解这些环节之后，再按预期请求量和延迟要求做压测、选框架。

在仓库根目录运行，不需要 GPU 或 API key：

```bash
python3 practice/rag/code/evidence_pipeline.py
python3 practice/post-training/code/experiment_checks.py
python3 practice/post-training/code/training_contracts.py
```

这些程序验证小例子的规则和数值，不代表训练或部署过完整模型。需要补原理，回[基础与原理](../learn/README.md)；想练单个函数、ML 问答或算法题，去[面试准备](../interview/README.md)。

## 加入我们！

调试时卡在哪里、试过哪些办法、最后怎么找到原因，都可以写下来。实验没得到预期结果也没关系，讲清楚过程同样有帮助。参与方式见[贡献指南](../CONTRIBUTING.md)。

这里只使用公开代码、论文和合成数据。公开项目的行为、本站教学设计、真正测量过的结果会分开说明，不放公司内部项目或私有数据。
