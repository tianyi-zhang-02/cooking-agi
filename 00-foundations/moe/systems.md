# MoE：训练和推理的系统代价

**中文** · [English](systems.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

## Expert parallelism 与两次 all-to-all

模型大了，expert 就分放在不同的卡上。一层 MoE 的前向于是变成三步（GShard 的做法）：

1. **dispatch**：一次 all-to-all，把每个 token 发到它选中的 expert 所在的卡；
2. 各卡算自己的 expert；
3. **combine**：再一次 all-to-all，把结果送回 token 原来所在的卡。

每一层都要来回两次，$k$ 越大、token 越多，通信越重。MoE 训练和推理的效率，很大程度上取决于这两次通信能不能和计算重叠起来。

## 限制一个 token 跨几台机器

跨机器的带宽比机器内部低得多。DeepSeek-V3 用了 node-limited routing：每个 token 最多只能发往 4 个节点，节点按「该节点上最高的几个 expert 分数之和」来挑。这样既保留了 top-8 的选择，又把跨节点的流量控制住。

## 显存：装得下才谈得上稀疏

显存要装下**全部**参数。DeepSeek-V3 有 671B 参数（Hugging Face 上显示的 685B 还包括 14B 的 multi-token prediction 模块），不管激活参数只有 37B，权重本身就需要多张卡。gpt-oss 把 MoE 部分的权重存成 MXFP4，每个参数约 4.25 bit，就是为了把总参数压进更少的显存。

## 解码时，batch 大小会改变账

解码通常是 memory-bound：每一步都要把用到的权重从显存读一遍。

- **batch 很小时**，一步里只有少数几个 expert 被用到，读的权重大致和激活参数成正比，这正是 MoE 快的地方；
- **batch 很大时**，不同 token 选了不同的 expert，一步里几乎所有 expert 都会被用到，读的权重接近总参数，只是摊到了更多 token 上。

所以 MoE「每个 token 只算一小部分」的好处，在不同的 batch 大小下兑现的方式不一样；再加上 all-to-all，实际吞吐要看具体部署。

## 数值精度

router 的 softmax 对数值很敏感。Switch Transformer 让 router 用 fp32 计算，其余部分仍用低精度；ST-MoE 的 z-loss（上一篇）也是为了同样的问题。

## 小结

MoE 适合「显存和卡够、想要更多参数、又想控制每个 token 计算」的场景，比如大规模的在线服务。单卡、显存紧、追求最低延迟的小 batch 部署，dense 模型往往更省心。
