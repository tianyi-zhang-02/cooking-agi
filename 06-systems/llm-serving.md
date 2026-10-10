# 一次 LLM 请求，怎样变成屏幕上的回答？

**中文** · [English](llm-serving.en.md)

> 最近审阅：2026-10 · 前置：[KV cache](../00-foundations/deep-dives/kv-cache-and-inference.md)

同一个模型，本地单独问一句很快，接到网站上却让人等很久。慢的未必是模型本身：请求可能还在排队，CPU 正在处理输入，或者新来的长文档占用了这一轮计算。

这一篇沿着请求走一遍。先看各部分负责什么，再用一个小调度表解释为什么“吞吐更高”和“回答更流畅”不总是一回事。这里只讨论普通自回归生成；推测解码、混合状态模型和跨机器缓存有额外路径。

## 先看谁在做什么 {#request-map}

<figure class="worked-update">
<ol>
<li><small>01 · API server</small><strong>接收与整理输入</strong><span>检查请求，套用对话模板，把文本和图片等输入整理成模型能处理的形式。</span></li>
<li><small>02 · Engine / scheduler</small><strong>决定这一轮算谁</strong><span>维护队列，检查缓存和 token 预算，安排各请求本轮要计算的位置。</span></li>
<li><small>03 · Worker → output</small><strong>计算，再返回结果</strong><span>模型计算与采样得到 token；输出端逐步拼成文本，处理停止与清理。</span></li>
</ol>
<figcaption>还没生成完的请求会回到下一轮调度。多个请求交错前进，不是一个回答写完了，才轮到下一个。</figcaption>
</figure>

vLLM V1 将 API、engine core 和 GPU worker 的职责分开；engine 管调度和 KV，worker 执行模型。具体进程数取决于部署方式，图里的 3 类工作不是“固定只开 3 个进程”。[官方架构说明](https://docs.vllm.ai/en/latest/design/arch_overview/)

## 跟着一个短请求走 {#request-trace}

假设整理好的输入只有 `[A, B, C]`，生成结果是 `[D, E]`。字母只是 token 的占位符。

| 到哪一步了 | 发生什么 | 此时还不能认为 |
| --- | --- | --- |
| 接收请求 | 校验参数和长度；chat 模板加入角色、分隔符等 | 用户看到的文字长度就是实际 token 数 |
| 登记与排队 | 分配 request ID，保存采样参数，等待资源 | HTTP 已接收就代表 GPU 已开始算 |
| 查缓存、安排计算 | 找到可复用的前缀，给剩余位置安排 KV 槽和本轮预算 | 命中缓存就不用排队或计算了 |
| Prefill | 处理 A、B、C；C 位置的输出用于采样 D | D 的 K/V 已经存在 |
| 下一轮 decode | 将 D 送入模型，生成 D 的 K/V，再采样 E | 只取出了一个事先存好的答案 |
| 输出与结束 | 输出端更新文本，检查停止条件，清理请求状态 | 一次网络消息必然等于一个 token |

第一个生成 token 也经过采样。不要把“prefill、decode、sampling”画成只各执行一次的 3 段：每当模型要选下一个 token，都要按相应规则处理 logits。分块 prefill 的中间块一般还不能输出回答，要等该请求到达可生成的位置。

输出也不是把 token ID 直接转成一个字符就发走。一个字符可能跨多个 token，stop string 也可能跨几次输出。输出端需要保留必要的缓冲，区分“算出了 token”和“已有可发送文本”。可以沿固定版本的 [`AsyncLLM.generate`](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/async_llm.py) 和 [output processor](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/output_processor.py) 看两端如何衔接。

## 长请求来了，正在输出的人怎么办？ {#token-budget}

有 2 个请求正在 decode，各需要本轮处理 1 个 token。又来了一个 10-token 的 prompt。为了看清分块的作用，假设每轮最多处理 6 个输入位置，先安排 decode，再用余量处理 prompt。

| 轮次 | 2 个老请求 | 新 prompt 本轮处理 | 新 prompt 还剩 |
| --- | ---: | ---: | ---: |
| 1 | 2 tokens | 4 tokens | 6 |
| 2 | 2 tokens | 4 tokens | 2 |
| 3 | 2 tokens | 2 tokens | 0 |

新请求第 3 轮完成 prefill 后才有机会给出首 token。老请求则每轮都得到计算机会。把预算改成 12，新 prompt 可以一轮处理完，但这一轮也可能更长，正在等后续 token 的用户会感到停顿。

这是**轮数与预算的算例，不是延迟预测**。Prefill 的一个位置和 decode 的一个位置，计算成本并不相同；实际还受 KV 容量、batch 形状、优先级和 kernel 影响。即使按表安排，6 个位置也不代表固定多少毫秒。

<details markdown="1">
<summary>用一个小函数核对分块数</summary>

```python
def prefill_chunks(prompt_tokens, decode_tokens, token_budget):
    if any(type(value) is not int for value in
           (prompt_tokens, decode_tokens, token_budget)):
        raise ValueError("Token counts must be integers")
    if prompt_tokens < 0 or not 0 <= decode_tokens <= token_budget or token_budget < 1:
        raise ValueError("Invalid token budget")
    remaining_budget = token_budget - decode_tokens
    if prompt_tokens and remaining_budget <= 0:
        raise ValueError("No capacity left for prefill in this toy schedule")
    chunks = []
    while prompt_tokens:
        scheduled = min(prompt_tokens, remaining_budget)
        chunks.append(scheduled)
        prompt_tokens -= scheduled
    return chunks

assert prefill_chunks(10, 2, 6) == [4, 4, 2]
assert prefill_chunks(10, 2, 12) == [10]
```

它只验证固定 decode 占用下的预算分配，不模拟 vLLM 的调度器，也不模拟请求完成、抢占或抢占后的重算。

</details>

[vLLM 的调优文档](https://docs.vllm.ai/en/latest/configuration/optimization/)讨论了 chunked prefill 的 TTFT / ITL 取舍。但当前 [V1 scheduler](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py) 并不是维护两套完全独立的“prefill 阶段 / decode 阶段”：它会追踪请求有多少位置已经计算，再分配本轮计算量。上面的表是理解预算的简化模型，不是源码逐步复刻。

## 点了停止，后台也要真正停下来 {#request-lifecycle}

假设用户只想生成 50 个 token，却在第 5 个就关了页面。如果服务还把剩下的算完，用户看不到，但 GPU 和 KV 容量依然被占着。

清理要覆盖正常结束和取消：停止接收该请求的后续输出，通知 engine，不再安排新计算，释放它持有的缓存引用。已经发出的 GPU 工作不一定能在任意位置立即打断，因此“取消成功”还要有清楚的完成语义。固定版本的 [`generate` 取消处理](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/async_llm.py#L763)会向 engine 发起 abort；[engine 的单步循环](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/core.py#L647)也处理执行期间收到的取消。

释放引用不等于把显存还给操作系统。共享前缀可能仍被别的请求使用；没人使用的缓存也可能留在预分配池里，等待复用或被覆盖。想看这部分，接着读[缓存生命周期](../00-foundations/deep-dives/attention-kernels.md#cache-lifecycle)。

## 先量清楚，再改参数 {#latency-diagnosis}

首 token 的等待可以先拆成：输入处理、排队、首次所需模型计算、输出与传输。这是方便埋点的分法；流水线可能重叠，不要把几个各自测得的均值直接相加，当成端到端 P95。

| 你观察到什么 | 先查哪里 | 一个容易选错的优化 |
| --- | --- | --- |
| 并发一高，TTFT 就涨 | 到达率、队列、KV 容量、抢占 | 没查排队就换 tokenizer |
| 首 token 不慢，但输出一顿一顿 | ITL 分布、长 prefill 混入、同步和网络缓冲 | 只报整体 tokens/s |
| 相似 prompt 还是慢 | 实际 token 前缀、adapter、缓存副本与命中量 | 把文字相似度当 cache hit |
| GPU 利用率低，CPU 很忙 | 输入处理、调度、序列化和多模态预处理 | 继续增加 GPU，却不看 CPU 队列 |
| 取消后容量迟迟不恢复 | request ID 的终止路径与缓存引用 | 只关闭前端连接 |

还要分清负载怎么来的。固定并发压测里，一个请求完成才补下一个；模型越慢，新请求进入得也越慢。固定到达率则可能持续积压。两种测法都能用，但不能把它们的尾延迟直接横比。至少记录输入 / 输出长度、到达方式、冷 / 热缓存、硬件与版本，并单列失败和超时。

## 有了这张图，再选部署方式 {#serving-choices}

模型能装进一张卡，不代表这张卡能满足目标并发。[完整显存账](../00-foundations/deep-dives/kv-cache-and-inference.md#inference-budget)要把权重、KV 和运行时余量一起算。需要扩容时，先问自己缺的是哪一种资源。

| 选择 | 可能解决什么 | 要付出什么 |
| --- | --- | --- |
| 多开独立副本 | 分担请求，提高总服务容量 | 重复权重；请求换副本可能失去热缓存 |
| TP：层内拆分 | 分摊单个模型的权重和部分状态 | 各层需要通信，对互联敏感 |
| PP：按层拆分 | 跨设备放下模型 | 中间激活传输，pipeline 空泡与调度复杂度 |
| Prefill / decode 分离 | 两阶段分别选硬件与调度 | KV 传输、队列协调和故障处理 |

这里不是框架排行榜。先选一种能跑通的简单部署，在代表性请求上测清质量、成本和延迟，再决定多加一层系统复杂度是否值得。[并行原理](distributed-training.md)可以解释拆分方式；推理的容量和服务指标仍要单独验证。

## 源码从哪里读？ {#source-route}

本篇核对 vLLM `10cc2f6`，链接固定到完整 commit，不把会变化的 main 当永久说明。先读 [API 的 completion 入口](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/entrypoints/openai/completion/serving.py)，再跟 `AsyncLLM.generate` → `EngineCore.step` → `Scheduler.schedule` → worker 执行 / 采样 → output processor。Chat、异步调度和推测解码会有分支，不要求所有请求严格走同一条函数栈。

本站只运行预算算例和文档检查，没有启动这一版本的完整 serving 服务，也没有测出某个框架更快。要复现性能，还需要固定模型、硬件、引擎配置和请求回放。
