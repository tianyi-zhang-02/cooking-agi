# 工程实践

**中文** · [English](README.en.md)

会调用模型以后，接下来往往会遇到这些问题：数据从哪里来？哪些计算可以提前做？结果不对，是模型的问题，还是前面的某一步出了错？

这里选了推荐系统、RAG 知识库和后训练实验 3 类任务。推荐系统结合公开代码来读；另外两类从小项目入手，跟着数据走一遍，再试着改一个条件，看哪里会出问题。先看讲解也可以，想动手时再打开代码。

**工程实践不只等于系统设计。** 这里有两种读法：从需求开始，自己推一个方案；或者沿着已有项目，看数据、组件、训练、服务和评估怎样接起来。前者练判断，后者用实现检查判断是否站得住。

## 选一个问题开始

| 你遇到了什么问题 | 从这里开始 | 重点看什么 |
| --- | --- | --- |
| 拿到需求，不知道从哪一步设计 | [系统设计练习](../learn/system-design/README.md) | 先明确目标和约束，再比较组件、成本与失败处理 |
| 好内容没有被推荐出来 | [推荐系统拆解](recommender-systems/README.md) | 是没召回、被过滤，还是排序靠后 |
| 资料搜到了，回答还是错的 | [RAG 知识库](rag/README.md) | 资料是否有效，模型是否看到、用对了它 |
| loss 降了，实际表现却没变好 | [后训练实验](post-training/README.md) | 训练目标是否合适，评估是否漏掉了问题 |

三类任务都要处理数据、成本和评估，但检查方法并不相同。例如，推荐中没有点击不一定是不喜欢；知识库中找不到依据，就不能肯定地给出答案。读的时候可以对照这些差别。

## 系统设计：先自己做一遍选择 {#system-design}

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

这条路线现在也讲中间的工程决定：[SFT、LoRA 与框架怎么选](post-training/training-plan.md)，[数据怎么变成 batch](post-training/data-pipeline.md)，以及[多卡](post-training/distributed-training.md)和[中断恢复](post-training/checkpoint-and-resume.md)。不是列一串工具名，而是说清什么时候需要、付出什么成本、怎么检查没改错目标。

[实验与发布](post-training/experiments-and-release.md)用 6 道题比较新旧模型。总体多答对一道，可能同时多编了一次没有依据的答案。看清具体变化后，再决定继续改、扩大实验，还是保留旧版。

## 先跑几个小例子

模型训练完，下一步往往是让别人稳定地用起来。[一次 LLM 请求怎样跑完](../06-systems/llm-serving.md)连接推理原理与服务设计：请求怎么排队、长输入怎么分块、取消怎么清理，以及什么时候该增加副本或拆分模型。先读这篇，再带着目标负载去做压测，比直接挑一个“最快框架”更有用。

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
