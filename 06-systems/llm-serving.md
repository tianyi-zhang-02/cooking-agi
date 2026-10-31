# 一次 LLM 请求，怎样变成屏幕上的回答？

**中文** · [English](llm-serving.en.md)

> 最近审阅：2026-10 · 前置：[键值缓存（KV cache）](../00-foundations/deep-dives/kv-cache-and-inference.md)

同一个模型，本地单独问一句很快，接到网站上却让人等很久。慢的未必是模型本身：请求可能还在排队，CPU 正在处理输入，或者新来的长文档占用了这一轮计算。

这一篇沿着请求走一遍：谁接收输入，谁决定何时计算，谁把结果送到屏幕上。先用一个短请求看清词元（token）怎样产生，再用调度表和排队算例解释为什么“吞吐更高”不一定意味着“更早看到回答”或“输出更流畅”。这里只讨论普通自回归生成；推测解码、混合状态模型和跨机器缓存有额外路径。

## 先看谁在做什么 {#request-map}

<figure class="worked-update">
<ol>
<li><small>01 · API server</small><strong>接收与整理输入</strong><span>检查请求，套用对话模板，把文本和图片等输入整理成模型能处理的形式。</span></li>
<li><small>02 · Engine / scheduler</small><strong>决定这一轮算谁</strong><span>维护队列，检查缓存和 token 预算，安排各请求本轮要计算的位置。</span></li>
<li><small>03 · Worker → output</small><strong>计算，再返回结果</strong><span>模型计算与采样得到 token；输出端逐步拼成文本，处理停止与清理。</span></li>
</ol>
<figcaption>还没生成完的请求会回到下一轮调度。多个请求交错前进，不是一个回答写完了，才轮到下一个。</figcaption>
</figure>

vLLM V1 将 API 服务、引擎核心和 GPU worker 的职责分开：引擎管调度和 KV，worker 执行模型。具体进程数取决于部署方式，图里的 3 类工作不是“固定只开 3 个进程”。可对照[官方架构说明](https://docs.vllm.ai/en/latest/design/arch_overview/)与 [v0.31.0 的 V1 进程架构章节](https://docs.vllm.ai/en/v0.31.0/design/arch_overview/#v1-process-architecture)；这里引用版本化文档来辨认职责，不改变文末固定的源码版本。

## 跟着一个短请求走 {#request-trace}

假设整理好的输入只有 `[A, B, C]`，生成结果是 `[D, E]`。字母只是 token 的占位符。先分清两件事：模型已经处理了哪些位置，以及刚刚选出了哪个 token；选出一个 token，不等于它已经被送回模型计算。

| 到哪一步了 | 发生什么 | 此时还不能认为 |
| --- | --- | --- |
| 接收请求 | 校验参数和长度；对话模板加入角色、分隔符等 | 用户看到的文字长度就是实际 token 数 |
| 登记与排队 | 分配请求标识，保存采样参数，等待资源 | HTTP 已接收就代表 GPU 已开始算 |
| 查缓存、安排计算 | 找到可复用的前缀，给剩余位置安排 KV 槽和本轮预算 | 命中缓存就不用排队或计算了 |
| 预填充（prefill） | 处理 A、B、C；C 位置的输出用于采样 D | D 的 K/V 已经存在 |
| 下一轮解码（decode） | 将 D 送入模型，生成 D 的 K/V，再采样 E | 只取出了一个事先存好的答案 |
| 输出与结束 | 输出端更新文本，检查停止条件，清理请求状态 | 一次网络消息必然等于一个 token |

第一个生成 token 也经过采样。不要把“预填充、解码、采样”画成只各执行一次的 3 段：每当模型要选下一个 token，都要按相应规则处理模型输出的 logits。分块预填充的中间块一般还不能输出回答，要等该请求到达可生成的位置。

输出也不是把 token ID 直接转成一个字符就发走。一个字符可能跨多个 token，停止字符串也可能跨几次输出。输出端需要保留必要的缓冲，区分“算出了 token”和“已有可发送文本”。可以沿固定版本的 [`AsyncLLM.generate`](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/async_llm.py) 和[输出处理器](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/output_processor.py)看两端如何衔接。

## 长请求来了，正在输出的人怎么办？ {#token-budget}

现在把一个请求放回多人共用的服务里。有 2 个请求正在解码，各需要本轮处理 1 个 token。又来了一个包含 10 个 token 的输入。为了看清分块的作用，假设每轮最多处理 6 个输入位置，先安排解码，再用余量处理新输入。

| 轮次 | 2 个老请求 | 新输入本轮处理 | 新输入还剩 |
| --- | ---: | ---: | ---: |
| 1 | 2 tokens | 4 tokens | 6 |
| 2 | 2 tokens | 4 tokens | 2 |
| 3 | 2 tokens | 2 tokens | 0 |

新请求第 3 轮完成预填充后才有机会给出首个 token。老请求则每轮都得到计算机会。把预算改成 12，新输入可以一轮处理完，但这一轮也可能更长，正在等后续 token 的用户会感到停顿。这里的取舍是：让新请求少等几轮，还是让已经开始的回答保持节奏？

这是**轮数与预算的算例，不是延迟预测**。预填充的一个位置和解码的一个位置，计算成本并不相同；实际还受 KV 容量、批次（batch）形状、优先级和计算内核影响。即使按表安排，6 个位置也不代表固定多少毫秒。为什么解码批次变大后会从带宽受限走向算力受限，见[性能：解码的批处理](../12-infra/performance.md#decode-batching)。

<!-- widget:step-chunked-prefill -->

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

它只验证固定解码占用下的预算分配，不模拟 vLLM 的调度器，也不模拟请求完成、抢占或抢占后的重算。

</details>

[vLLM 的调优文档](https://docs.vllm.ai/en/latest/configuration/optimization/)讨论了分块预填充对首 token 延迟（TTFT）和相邻 token 输出间隔（ITL）的影响。但本文固定版本的 [V1 调度器](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py)并不是维护两套完全独立的“预填充阶段 / 解码阶段”：它会追踪请求有多少位置已经计算，再分配本轮计算量。上面的表是理解预算的简化模型，不是源码逐步复刻。

## 点了停止，后台也要真正停下来 {#request-lifecycle}

调度不只要决定谁能进入，也要知道谁已经离开。假设用户只想生成 50 个 token，却在第 5 个就关了页面。如果服务还把剩下的算完，用户看不到，但 GPU 和 KV 容量依然被占着。

清理要覆盖正常结束和取消：停止接收该请求的后续输出，通知引擎，不再安排新计算，释放它持有的缓存引用。已经发出的 GPU 工作不一定能在任意位置立即打断，因此“取消成功”还要明确指的是哪一步已经完成。固定版本的 [`generate` 取消处理](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/async_llm.py#L763)会向引擎发起中止请求；[引擎的单步循环](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/core.py#L647)也处理执行期间收到的取消。

释放引用不等于把显存还给操作系统。共享前缀可能仍被别的请求使用；没人使用的缓存也可能留在预分配池里，等待复用或被覆盖。想看这部分，接着读[缓存生命周期](../00-foundations/deep-dives/attention-kernels.md#cache-lifecycle)。

## 先量清楚，再改参数 {#latency-diagnosis}

沿着前面的请求路径，首个 token 的等待可以先拆成：输入处理、排队、首次所需模型计算、输出与传输。这是方便埋点的分法；流水线可能重叠，不要把几个各自测得的均值直接相加，当成端到端 P95。

| 你观察到什么 | 先查哪里 | 一个容易选错的优化 |
| --- | --- | --- |
| 并发一高，首 token 就等得更久 | 到达率、队列、KV 容量、抢占 | 没查排队就换分词器 |
| 首个 token 不慢，但输出一顿一顿 | 输出间隔的分布、长输入的预填充混入、同步和网络缓冲 | 只报整体 tokens/s |
| 相似输入还是慢 | 实际 token 前缀、适配器、缓存副本与命中量 | 把文字相似度当缓存命中 |
| GPU 利用率低，CPU 很忙 | 输入处理、调度、序列化和多模态预处理 | 继续增加 GPU，却不看 CPU 队列 |
| 取消后容量迟迟不恢复 | 按请求标识追踪终止路径与缓存引用 | 只关闭前端连接 |

还要分清负载怎么来的。固定并发压测里，一个请求完成才补下一个；模型越慢，新请求进入得也越慢。固定到达率则可能持续积压。两种测法都能用，但不能把它们的尾延迟直接横比。至少记录输入 / 输出长度、到达方式、冷 / 热缓存、硬件与版本，并单列失败和超时。

### 小算例：多等一会儿凑批，谁受益？ {#batching-wait}

先看一个**简化的排队例子，数字不是实测结果**。4 个请求分别在时刻 0、1、2、3 到达同一个空闲执行器。假设单独处理一个请求耗时 4，合成一批处理 4 个请求耗时 6；时间均为任意单位，不计其他开销，批次之间不能重叠。为单独观察排队，这里把每个请求视为一整段计算，不模拟逐 token 输出。

| 安排方式 | 计算时段 | 4 个请求各自的完成时刻 | 第一个请求从到达到完成 |
| --- | --- | --- | --- |
| 按到达顺序，空闲就处理一个 | 0–4、4–8、8–12、12–16 | 4、8、12、16 | 4 |
| 等 4 个都到齐，再一起处理 | 3–9 | 9、9、9、9 | 9，其中排队 3 |

从时刻 0 到全部完成，第一种安排的平均完成速率是 `4 / 16 = 0.25`，第二种是 `4 / 9 ≈ 0.44` 个请求 / 时间单位。第二种更早处理完全部请求，却让第一个请求更晚完成。这里算的是**有限窗口内的完成速率，不是稳态吞吐，也不是首 token 延迟**；更不能据此要求真实服务总要等满一批。连续批处理（continuous batching）可以在迭代边界调整成员，不必等待整批回答全部结束后才接纳新请求，相关课程段落见文末。

## 有了这张图，再选部署方式 {#serving-choices}

模型能装进一张卡，不代表这张卡能满足目标并发。[完整显存账](../00-foundations/deep-dives/kv-cache-and-inference.md#inference-budget)要把权重、键值缓存和运行时余量一起算。需要扩容时，先问自己缺的是哪一种资源。多开独立副本、张量并行（TP）、流水线并行（PP）、预填充 / 解码分离各解决什么、要付出什么，以及同样 8 张卡怎样算权重账，见 [vLLM 与 SGLang：多张 GPU 的拼法](inference-frameworks.md#deployment-map)；副本在集群里怎样调度和扩容，见 [Kubernetes 与 GPU 集群](../12-infra/kubernetes.md)。

这里不是框架排行榜。先选一种能跑通的简单部署，在代表性请求上测清质量、成本和延迟，再决定多加一层系统复杂度是否值得。[并行原理](distributed-training.md)可以解释拆分方式；推理的容量和服务指标仍要单独验证。

## 源码从哪里读？ {#source-route}

可以先带着两个问题读 [Stanford CS336 第 10 讲](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)：

- `throughput_and_latency()`：看批次大小与延迟的取舍；其带宽模型忽略开销并假设理想重叠，不是线上延迟预测。
- `continuous_batching()`：看请求怎样在迭代边界加入批次；对照上面的预算表，但不要把课程示意当成具体版本的调度代码。

再进入实现。本篇核对 vLLM `10cc2f6`，链接固定到完整提交，避免 `main` 更新后与说明不符。先读 [API 的补全请求入口](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/entrypoints/openai/completion/serving.py)，再跟 `AsyncLLM.generate` → `EngineCore.step` → `Scheduler.schedule` → 执行进程计算 / 采样 → 输出处理器。对话和推测解码会有分支，并非所有请求都经过完全相同的函数调用链。

异步调度在这一提交的兼容配置下默认开启（`AsyncScheduler` 继承 `Scheduler`）。CPU 排下一轮时，GPU 还在跑上一轮，因此执行轨迹里的两步可以重叠。某些池化模型、执行器或推测解码配置会关闭异步调度，阅读源码时还要检查这些条件，不能只看默认开关。

本站只运行预算算例和文档检查，没有启动这一版本的完整推理服务，也没有测出某个框架更快。要复现性能，还需要固定模型、硬件、引擎配置和请求回放。
