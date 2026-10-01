# 系统设计：阅读与取舍

**中文** · [English](system-design.en.md)

> 阅读时间：约 2 分钟 · 最近审阅：2026-09

> **先读这个**：面试的形式和重点变得很快，这里的内容有时效性；这里只写公开的资源和我自己的理解，不写任何一家公司的面试题或流程。

我也还在补系统设计。这页保留我读过的公开资源和一些思考；想动手练习，可以先去[系统设计练习](../learn/system-design/README.md)，从信息流、RAG 或带记忆的助手选一个。

## 没有固定公式

我的体会是，系统设计没有一套背下来就能用的模板。真正被考察的是两件事：**你知不知道每个选择的 trade-off**，以及**你能不能把整个系统串起来**。

所以从最基本的出发就行：

1. 要解决的是什么问题？成功长什么样？
2. 手里有什么数据？没有什么数据？
3. 约束是什么——规模、延迟、成本、freshness？
4. 每一步选这个方案，放弃了什么？

## 一个例子：推荐系统

比如要做推荐。先看手上有什么：

- **如果有关注、好友或互动关系**，可以先用这些关系找候选。图也不只指社交图：[PinSage](https://arxiv.org/abs/1806.01973) 用的是 Pinterest 的物品与收藏板关系，别把几种不同的数据结构混为一谈。
- **如果没有关系数据呢？** 那就退回到行为本身：item–item 的共现、双塔召回、内容特征（标题、类目、多模态 embedding）。冷启动阶段，内容特征和热门兜底往往比复杂模型更管用。

再往下每一层都是取舍：召回要多快、排序能用多大的模型、特征要多新鲜、离线指标涨了线上会不会涨、复杂度值不值得维护。把这些说清楚，比背一套架构图有用得多。

## 公开资源

通用系统设计：

- [System Design Primer](https://github.com/donnemartin/system-design-primer)：免费，覆盖面广，适合搭骨架
- [ByteByteGo](https://bytebytego.com)：Alex Xu 的《System Design Interview》系列，图多、好读
- [Designing Data-Intensive Applications](https://www.dataintensive.net)：想真正理解底层取舍，这本最值得花时间

ML / 推荐方向：

- [Designing Machine Learning Systems](https://huyenchip.com/books/)（Chip Huyen）
- [Evidently 的 ML system design 案例合集](https://www.evidentlyai.com/ml-system-design)：几百个真实系统的公开分享
- [Eugene Yan: system design for discovery](https://eugeneyan.com/writing/system-design-for-discovery/)：搜索与推荐系统的分层设计
- [Twitter 开源的推荐算法](https://github.com/twitter/the-algorithm)：推荐链路中的公开组件，不是完整可部署的生产系统

## 站内相关

- [Search：模型怎样找到它现在不知道的东西](../04-search/)
- [现代 AI 系统：模型只是其中一部分](../06-systems/)
- [Agents：不同场景怎么用](../10-agents/scenarios.md)
- [Evaluation：我们凭什么说系统变好了](../07-evaluation/)
