# MoE：训练和推理的系统代价

**中文** · [English](systems.en.md)

> 阅读时间：约 3 分钟 · 难度：进阶 · 最近审阅：2026-10-09

## Expert parallelism 与两次 all-to-all

模型一大，expert 就得分散在不同的卡上。一层 MoE 的前向于是变成三步（GShard 的做法）：

1. **dispatch**：一次 all-to-all，把每个 token 发到它选中的 expert 所在的卡；
2. 各卡算自己的 expert；
3. **combine**：再一次 all-to-all，把结果送回 token 原来所在的卡。

这是 expert parallelism 的典型逻辑流程，不代表任何部署都必须发两次跨卡集合通信；专家都在本地时不需要这一步。跨卡流量还取决于选中的专家落在哪里、是否合并同节点发送。排查慢 step 时，分别看派发、expert 计算、回收及重叠程度。

## 限制一个 token 跨几台机器

跨机器的带宽比机器内部低得多。DeepSeek-V3 的做法是 node-limited routing：一个 token 最多只发往 4 个节点，节点按「这个节点上最高的几个 expert 分数之和」来挑。top-8 的选择保住了，跨节点的流量也压住了。

## 显存：装得下才谈得上稀疏

全 GPU 常驻时，要为全部权重安排存储，而不是只按激活参数分配。DeepSeek-V3 的主模型为 671B，包含 MTP 模块的发布权重还多约 14B；gpt-oss 的 expert 权重使用 MXFP4，约 4.25 bit/参数，其他张量不全是这个精度。Offload 可以把部分权重放到 CPU，但带宽和缓存命中就会进入延迟账。具体型号来源见[模型报告列表](review.md)。

## 解码时，batch 大小会改变账

小 batch 解码常受到内存带宽限制，但长上下文的 KV 读取、跨卡通信和较大的矩阵计算，也可能成为瓶颈，最好用 profiler 确认。

- **batch 小的时候**，一步可能只用少数 expert，读取权重较少；小矩阵效率、派发开销仍可能抵消收益。
- **batch 大的时候**，如果路由分散，更多 expert 会被用到，读权重可能接近总量，但可以摊给更多 token；偏斜路由则可能继续只用少数 expert。

所以「每个 token 只算一小部分」这个好处，在不同 batch 下兑现的方式并不一样；再加上 all-to-all，真实吞吐还得看具体怎么部署。

## 数值精度

router 的 softmax 对数值很敏感。Switch Transformer 让 router 单独用 fp32 算，其余部分照旧低精度；[负载均衡那篇](load-balancing.md)讲的 z-loss，针对的也是这个问题。

## 小结

卡和显存都够、想要更多参数、又想把每个 token 的计算压住——这种场景适合 MoE，典型是大规模在线服务。单卡、显存紧、追求最低延迟的小 batch 部署，dense 模型往往更省心。
