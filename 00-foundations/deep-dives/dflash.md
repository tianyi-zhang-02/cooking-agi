# DFlash：草拟、验证与加速

**中文** · [English](dflash.en.md)

> 核对：2026-10-09 · 前置：[生成与 KV cache](kv-cache-and-inference.md) · 第一遍读流程和算例；准备实现时再看采样、缓存与实验表。

一次生成多个 token，听起来都像“并行预测”。但你得先问：**谁在猜，谁说了算，猜错的计算怎么办？** MTP 主要讨论训练时的预测目标和模块；DFlash 讨论怎样用一个小 drafter 帮目标模型（target model）更快生成。两个名字并不在同一个分类层级。

## 先把 3 件事分开

| 你想改什么 | 对应问题 | 不能顺带假定什么 |
| --- | --- | --- |
| 训练目标 | 除了下一个 token，要不要预测更远的位置？ | 加了 MTP loss 就一定推理加速 |
| 草稿生成 | 小模型或辅助模块怎么一次提出多个候选？ | 候选都可以直接输出 |
| 验证与调度 | 哪些候选能接受，怎样管理缓存和请求？ | 验证一整块永远与验证 1 token 一样便宜 |

[早期 MTP 工作](https://arxiv.org/abs/2404.19737)在共享主干上用多个预测头；[DeepSeek-V3](https://arxiv.org/abs/2412.19437)采用串行的 MTP 模块。不要把所有 MTP 都画成同一种结构。它们可以为 speculative decoding 提供草稿，但要另有验证流程才能把“多预测”变成可靠输出。

## DFlash 的一轮生成

[DFlash 原论文](https://arxiv.org/abs/2602.06036)保持目标自回归模型冻结，训练轻量 block-diffusion drafter：目标模型的上下文特征注入 draft 各层的 K/V；一个已确认的 token 作锚点，其余位置是 mask，一次前向并行提出候选。不是把目标模型整体换成 diffusion LLM。

```text
已确认的前缀 → target 特征 + 已确认的锚点 token
                          ↓
              [锚点 | MASK | MASK | MASK]
                          ↓  一次 draft 前向
              [锚点 | 候选1 | 候选2 | 候选3]
                          ↓  target 因果验证
              接受的连续前缀 + target 补出的 token
                          ↓
                修正缓存，进入下一轮
```

训练也要遵守这个信息边界：训练样本可以有完整答案，但 drafter 预测某块时，不能读取该块未来答案对应的 target 特征。原文用随机锚点和专门的 attention mask 对齐训练与推理，并提高靠前位置的 loss 权重。[项目说明](https://z-lab.ai/projects/dflash/)补充了特征注入与共享 embedding / LM head 的结构。

为什么重视靠前的位置？下面这个例子不用跑大模型就能看出来。

## 为什么后面猜对了也不能直接留下？

用 `A B X D` 表示草稿的 4 个 token，不对应特定 tokenizer。先看 greedy decoding：target 选概率最高的 token。

| 位置 | 草稿 | target 在草稿前缀下的选择 | 本轮处理 |
| --- | --- | --- | --- |
| 1 | A | A | 接受 |
| 2 | B | B | 接受 |
| 3 | X | C | 首次不同，输出 C |
| 4 | D | D | 不能接受；它是在含 X 的前缀下算的 |

这一轮可以输出 `A B C`，但不能宣布 `A B C D` 都正确。把第 3 个位置从 X 改成 C 后，第 4 个位置的条件分布可能变化。要保留的是**连续接受前缀（accepted prefix）**，不是整块里所有碰巧相同的位置。

下面只写 greedy 验证的控制逻辑。`target_choices[position]` 必须来自相应草稿前缀，最后一个元素是整块接受时的额外 token。实际实现还需要 logits、attention mask 和缓存；这不是完整 DFlash 推理器。

```python
def verify_greedy_block(draft, target_choices, eos_token=None):
    if len(target_choices) != len(draft) + 1:
        raise ValueError("expected one target choice per draft token plus a bonus")
    output = []
    accepted = 0
    for position, candidate in enumerate(draft):
        expected = target_choices[position]
        if candidate != expected:
            output.append(expected)
            return output, accepted
        output.append(candidate)
        accepted += 1
        if eos_token is not None and candidate == eos_token:
            return output, accepted
    output.append(target_choices[-1])
    return output, accepted


emitted, accepted = verify_greedy_block([10, 20, 99, 40], [10, 20, 30, 40, 50])
assert emitted == [10, 20, 30] and accepted == 2
```

这也是缓存回滚的原因。验证时可能给错误的 X 和后缀计算了 K/V，这些状态不能留在正式前缀里。只保留有效前缀对应的状态；新补出的 C 若还没做前向，需要在后续步骤计算它的 K/V。不能把“已输出 token 数”直接当成“当前缓存长度”。Draft 与 target 的缓存还要各自对齐。

## 有温度的采样，不能只比较 argmax

Greedy 的一致性比较容易理解。随机采样要保持的是 target 的**输出分布**，不是同一个 seed 下逐字相同的句子。[Speculative sampling](https://arxiv.org/abs/2302.01318)使用接受 / 拒绝校正，而不是“看着差不多就接受”。

设当前位置的目标分布为 $p$，真实 proposal 分布为 $q$，候选 $y\sim q$，接受概率为

$$\min\left(1,\frac{p(y)}{q(y)}\right).$$

拒绝时从归一化的 $[p-q]_+$ 补采。具体的 2-token 精确求和见 [MTP 篇的采样算例](multi-token-prediction.md)。

关键在“真实 proposal”：如果先做 top-k、temperature，或者用 selector 改过候选分布，验证器需要对应的 $q$，不能继续拿原始 softmax。并行猜出的多个位置也不能被当成已经按 target 条件分布顺序采好了。该证明保证分布，不保证错误缓存、错误 mask 或错位概率还能无损。

## 省不省时间，算完整的一轮

设基线每输出一个 token 用 $T_{\mathrm{AR}}$；一次 draft、verify 和其他开销分别为 $T_d,T_v,T_o$；该轮平均实际输出 $E[N]$ 个新 token，包括接受的草稿和补出的 token，不重复计已知锚点。那么粗略地：

$$\text{speedup}\approx
\frac{T_{\mathrm{AR}}E[N]}{T_d+T_v+T_o}.$$

这只是固定负载下的摊销账，不是完整排队模型。下面数字是**教学假设，不是 DFlash benchmark**：基线 10 ms/token，draft 3 ms，verify 12 ms，其他 1 ms。

| 一轮实际输出 | 平均每 token 耗时 | 相对基线 |
| --- | --- | --- |
| 1 | 16 ms | 0.625×，更慢 |
| 2 | 8 ms | 1.25× |
| 4 | 4 ms | 2.5× |

写成代码，可以自己换开销：

```python
import math


def cycle_speedup(baseline_ms, draft_ms, verify_ms, overhead_ms, emitted_tokens):
    values = (baseline_ms, draft_ms, verify_ms, overhead_ms, emitted_tokens)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("inputs must be finite")
    if baseline_ms <= 0 or emitted_tokens <= 0:
        raise ValueError("baseline and emitted tokens must be positive")
    if min(draft_ms, verify_ms, overhead_ms) < 0:
        raise ValueError("costs must be nonnegative")
    cycle_ms = draft_ms + verify_ms + overhead_ms
    if cycle_ms <= 0:
        raise ValueError("cycle cost must be positive")
    return baseline_ms * emitted_tokens / cycle_ms


assert cycle_speedup(10, 3, 12, 1, 4) == 2.5
assert cycle_speedup(10, 3, 12, 1, 1) == 0.625
```

Block 更长不一定更好：后段可能更容易被拒绝，验证与缓存成本也会变。并行不是免费；高并发下，draft 占掉的算力和显存也可能影响其他请求。不能从 batch=1 的低延迟结果直接推导整台服务器的吞吐收益。

## 2026 年的新进展：并行猜，怎样猜得更连贯？

初版 DFlash 并行提出各位置候选，但后一个位置没有看到前一个位置刚选中的 token。这给局部连贯性留下了改进空间。

[DFlash 2 的官方介绍](https://inco.ai/blog/dflash2/)发表于 2026-08-18：它保留并行主干，用轻量相邻候选打分选择路径，并加入短卷积传递块内局部信息。选路径仍有顺序操作；“并行”不等于系统里没有任何依赖。

可以用一个自编的小例子理解动机：两列候选分别是 `{New, Los}` 和 `{York, Angeles}`。各列独立 top-1 可能拼成 `New Angeles`；给相邻词的搭配打分，就有机会选到 `New York`。这只解释“选得更连贯”，不代表 selector 知道正确答案；最终仍由 target 验证。

另一条路线是 2026-10-05 的 [D-Loop](https://arxiv.org/abs/2610.06011)：复用同一 drafter 再跑一遍，让后缀读取选定的前缀。它花额外前向换草稿质量。本页只核对其摘要与机制概述，尚未复现实验，不能据此给各种硬件排速度名次。

| 方案 | 怎样处理多个未来位置 | 主要要付什么代价 |
| --- | --- | --- |
| MTP | 额外训练目标 / 预测模块；具体结构不同 | 训练和模块开销；加速另需验证 |
| DFlash 初版 | 条件于 target 特征的单次并行草拟 | 草稿质量、验证成本与额外状态 |
| DFlash 2 | 并行草拟，加局部混合和路径选择 | 局部模块与选择开销 |
| D-Loop | 草拟后复用主干修正后缀 | 第二次 draft 前向 |

版本也影响可运行性。[当前官方仓库](https://github.com/z-lab/dflash)已列 DFlash 2 与不同 backend 的支持范围；旧项目页还写着 vLLM 集成进行中。运行前核对具体 commit、target/draft 权重和 backend，不直接复制某个旧网页的安装命令。

## 真正部署前，要交出什么证据？

| 检查项 | 最小对照 | 要防什么 |
| --- | --- | --- |
| 输出正确性 | Greedy 序列对齐；采样做分布检查 | 把相同 seed 当成采样正确性标准 |
| 提前结束 | 首位拒绝、整块接受、EOS、长度边界 | 多输出、漏输出或保留失效缓存 |
| 稳定加速 | 固定输入集、长度、温度、并发和硬件 | 用最长接受块代替平均每轮输出 |
| 用户体验 | TTFT、流式 token 间隔、尾延迟、总吞吐 | 只报平均 tokens/s，忽略突发输出 |
| 接入成本 | 两套模型 / 缓存、特征提取和引擎版本 | 只算 draft 权重大小 |

先在可控环境里跑对，再看哪个负载值得开。这里的代码验证的是局部逻辑和算例；没有运行完整 drafter 训练或 GPU serving 性能复现。想继续看目标模型为何能并行评分，回到 [causal mask 与 decode](../core/decoder-only.md)；想看训练时多个目标怎样对齐，读 [MTP](multi-token-prediction.md)。
