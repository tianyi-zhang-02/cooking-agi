# MoE：训练和推理的系统代价

**中文** · [English](systems.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

## Expert parallelism 与两次 all-to-all

模型一大，expert 就得分散在不同的卡上。一层 MoE 的前向于是变成三步（GShard 的做法）：

1. **dispatch**：一次 all-to-all，把每个 token 发到它选中的 expert 所在的卡；
2. 各卡算自己的 expert；
3. **combine**：再一次 all-to-all，把结果送回 token 原来所在的卡。

每层都要这么来回两趟，$k$ 越大、token 越多，通信越重。MoE 训练和推理快不快，很大程度上就看这两趟通信能不能和计算重叠。

## 限制一个 token 跨几台机器

跨机器的带宽比机器内部低得多。DeepSeek-V3 的做法是 node-limited routing：一个 token 最多只发往 4 个节点，节点按「这个节点上最高的几个 expert 分数之和」来挑。top-8 的选择保住了，跨节点的流量也压住了。

## 显存：装得下才谈得上稀疏

显存要装下**全部**参数。DeepSeek-V3 是 671B（Hugging Face 上写的 685B 还含 14B 的 multi-token prediction 模块），哪怕激活参数只有 37B，光权重就得摊在多张卡上。gpt-oss 把 MoE 权重存成 MXFP4、每个参数约 4.25 bit，就是为了把总参数塞进更少的显存。

## 解码时，batch 大小会改变账

解码基本是 memory-bound：每走一步，都要把用到的权重从显存里读一遍。

- **batch 小的时候**，一步里只用到少数几个 expert，读进来的权重大致和激活参数成正比——MoE 快就快在这里；
- **batch 大的时候**，不同 token 挑了不同的 expert，一步下来几乎每个 expert 都被碰到，读进来的权重接近总参数，只是摊给了更多 token。

所以「每个 token 只算一小部分」这个好处，在不同 batch 下兑现的方式并不一样；再加上 all-to-all，真实吞吐还得看具体怎么部署。

## 数值精度

router 的 softmax 对数值很敏感。Switch Transformer 让 router 单独用 fp32 算，其余部分照旧低精度；[负载均衡那篇](load-balancing.md)讲的 z-loss，针对的也是这个问题。

## 小结

卡和显存都够、想要更多参数、又想把每个 token 的计算压住——这种场景适合 MoE，典型是大规模在线服务。单卡、显存紧、追求最低延迟的小 batch 部署，dense 模型往往更省心。
