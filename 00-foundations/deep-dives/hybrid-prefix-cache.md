# 混合模型的前缀缓存：同一个 prompt 的 8 个样本，为什么各算了一遍？

**中文** · [English](hybrid-prefix-cache.en.md)

> 最近审阅：2026-11 · 前置：[Gated DeltaNet](gated-deltanet.md)、[KV cache](kv-cache-and-inference.md#paged-budget)

GRPO 的一步里，每个 prompt 要采样好几个回答，比如 8 个。按理说，引擎算一次 prompt 并缓存下来，另外 7 个样本就能直接复用，这就是前缀缓存（prefix caching）。

实测一步并不是这样。测试条件是 SGLang 跑 Qwen3.5-9B，用 verl 的默认 rollout 设置。Qwen3.5 是把 Gated DeltaNet 和 attention 层叠在一起的混合模型。结果：

- 72% 的样本没用上缓存的 prompt，各自重算了超过 512 个 token；
- 加起来，算掉的 prompt token 是去重后的 2.6–3.0 倍。

原因有两层：

1. 同一个 prompt 的样本挨着进 batch，谁都还没把 prompt 存进缓存。这一层只和调度顺序有关，和 Gated DeltaNet 层无关。
2. 先把 prompt 算一次，再发样本，就绕开了第一层。但混合模型的状态池很小，状态往往等不到样本就被挤掉了。这一层才是混合模型特有的。

## 复用前缀，混合模型要多存一样东西

Attention 层给每个 token 存一份 K 和 V。想复用前 $L$ 个 token，取出前 $L$ 份就行，前缀长短都能切。

Gated DeltaNet 层只有一块固定大小的状态，每读一个 token 就改写一次（见 [Gated DeltaNet](gated-deltanet.md)）。读完 $L+100$ 个 token 后的状态，退不回第 $L$ 个 token 时的样子。所以要复用一段前缀，得有一份停在那里的状态快照。

SGLang 0.5.23 默认用 `extra_buffer` 策略，在两处存快照：

- prefill 每算完一块，向下对齐到 64 个 token 的边界（[schedule_batch.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/schedule_batch.py#L3170-L3186)）；
- decode 每 256 个 token。

因为要对齐，命中缓存的样本通常还要重算最后几十个 token。

快照放在一个启动时就定好大小的池子里，满了按最近最少使用（LRU）淘汰。一份快照包含所有 Gated DeltaNet 层的状态，个头不小。下面的数都是实测，条件见后面的[实测](#measured)：

- 单张 A100 80 GB、`--mem-fraction-static 0.5` 时，SGLang 拿出 10.2 GB 做这个池子，只放得下 212 份快照，每份约 48 MB。
- 池子大小还决定并发上限：SGLang 按每个请求 5 个槽位来算，把上限定成“池子大小 ÷ 5”（[kv_cache_configurator.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/kv_cache_configurator.py#L3062-L3086)）。所以 verl 要的 256 个并发请求被压到了 42 个。实际跑起来，每个请求平均占 3 个多槽位。
- 换成 verl 默认的两卡张量并行（TP 2），每张卡只存一半的权重和一半的状态，池子变成 592 份，并发上限 118，仍然低于 256。

## 第一种丢法：8 个样本同时到，谁也用不上谁的

先看一个能手算的小例子：

- prompt 长 2,000 token，8 个样本同时到达；
- 每个 prefill batch 最多放 8,192 个 token。

调度器每组一次 batch，都要先查缓存，看已经有多长的前缀可以复用。

1. **第 1 个 batch**放进样本 1–4，共 8,000 个 token。组 batch 时，缓存里还没有这个 prompt，4 个样本只能各算一遍。prompt 的状态要等这个 batch 算完，才会写进缓存。
2. **第 2 个 batch**放进样本 5–8。SGLang 用的是重叠调度（overlap scheduling）：第 1 个 batch 还在 GPU 上跑，CPU 已经在组第 2 个 batch 了（[scheduler.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/scheduler.py#L2061)）。所以样本 5–8 查缓存时也查不到。

结果，8 个样本各算了一遍：一共 16,000 个 token，其实算 2,000 个就够了。

```python
def concurrent_misses(prompt_len, samples, batch_tokens, lag=1):
    """Samples that recompute the prompt when all of them arrive at once.

    A batch takes as many whole prompts as fit in batch_tokens. A sample reuses
    the prompt only if its batch is formed after the first batch's state has
    been written to the cache. With overlap scheduling, the next `lag` batches
    are formed before that, so they miss too.
    """
    per_batch = max(1, batch_tokens // prompt_len)
    return min(samples, per_batch * (1 + lag))
```

只改一个条件，看看结果怎么变：

```python
assert concurrent_misses(2000, 8, 8192) == 8          # 两个 batch，全部重算
assert concurrent_misses(2000, 8, 4096) == 4          # batch 变小，只有前两个 batch 重算
assert concurrent_misses(2000, 8, 8192, lag=0) == 4   # 没有重叠调度，只有第一个 batch 重算
```

这只是讲调度顺序的教学模型。真实的 batch 里还会混着其他 prompt 和正在 decode 的请求，长 prompt 也会被切成几块来算。

最后一行看上去像是说“关掉重叠调度就好了”，但那样 CPU 和 GPU 就不能并行了，这里没有测。

真实情况可以直接查。SGLang 给每个请求记了两个时间：

- `forward_entry_time`：进入第一个计算 batch 的时刻，紧跟在查缓存之后；
- `prefill_finished_time`：prefill 结束、状态马上要写进缓存的时刻。

在 TP 2 下跑了 3 次，结果是：

- 每个没用上缓存的样本，都是在同组第一个 prefill 结束之前就进了 batch。之后进去的，没有一个没用上。
- 每次还有 388–455 个这样的样本，复用了 prompt 的前一段。很可能是兄弟样本的 prefill 还没跑完，但已经算好了几块。

所以这些样本重算，不是因为状态被淘汰了，而是因为那时状态还不存在。这一层只和调度顺序有关；纯 attention 模型在同样的调度下应该也会这样，不过这里没有测。

## 先 prefill 一次：不再重复算，状态却可能被挤掉

直接的改法是：先单独发一个只做 prefill 的请求，算完再发 8 个样本。SGLang 自己处理 `n > 1` 的请求，就是这么做的：先发一个 `max_new_tokens=0` 的请求，再发 n 个样本（[tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/tokenizer_manager.py#L2130-L2151)）。

这样，样本到达时状态已经在缓存里了，但它们还得排队。一步里有约 1,000 个样本在排，前面的请求不断把新快照写进池子，这个 prompt 的状态在 LRU 顺序里越来越旧，可能等不到自己的样本就被挤掉了。

再算一个小例子：

- 4 个 prompt A–D，池子里只空着 3 个位置，每个 prompt 2 个样本；
- 先把 A–D 都 prefill 一遍，再按顺序跑样本。

1. prefill D 时，池子满了，最久没用的 A 被挤掉。
2. 轮到 A 的样本：第一个查不到，只好重算，重算出的状态又挤掉了 B；第二个样本能复用。
3. B、C、D 依次重复这个过程。

最后每个 prompt 的第一个样本都重算了，8 个里有 4 个。

实测也是这个样子。在单卡上先 prefill 一次、不加别的保护，128 个 prompt 里有 121–123 个的第一个样本照样重算。这一步反而比直接发样本还慢了 10.6%–11.4%：每个 prompt 多了一个请求，状态还是没留住。

## 保住状态，还要限制在途的组数

开了 `--enable-session-radix-cache` 之后，SGLang 可以用 session 保护一个 prompt 的状态，直到客户端调用 `/close_session`。这里的做法是在同组最后一个样本开始出 token 时关掉 session。

但这种保护是软的，不是钉死的（[PR #29173](https://github.com/sgl-project/sglang/pull/29173)）：池子里全是受保护的状态时，新状态进来，仍会挤掉最旧的那个。

所以光开 session 还不够，还要限制同时在等样本的 prompt 组数，让受保护的状态放得进池子。

还是上面的例子：

```python
def queued_misses(prompts, free_slots, policy, samples=2, cap=None):
    """Samples that miss when each prompt is prefilled once before its samples.

    The pool holds free_slots prompt states and evicts the least recently used.
    "warm" prefills every prompt first, then runs the samples prompt by prompt.
    "hold" also protects each state until its samples start; the protection is
    soft, so a pool full of protected states still evicts the oldest one.
    "hold_cap" is "hold" with at most `cap` prompts prefilled and waiting.
    """
    pool, held, misses = [], set(), 0      # pool: least recently used first

    def put(prompt):
        if prompt in pool:
            pool.remove(prompt)
        elif len(pool) == free_slots:
            unheld = [p for p in pool if p not in held]
            pool.remove(unheld[0] if unheld else pool[0])
        pool.append(prompt)

    window = cap if policy == "hold_cap" else len(prompts)
    queue, waiting = list(prompts), []
    while queue or waiting:
        while queue and len(waiting) < window:
            prompt = queue.pop(0)
            put(prompt)
            waiting.append(prompt)
            if policy != "warm":
                held.add(prompt)
        prompt = waiting.pop(0)
        held.discard(prompt)               # released when its samples start
        for _ in range(samples):
            misses += prompt not in pool
            put(prompt)
    return misses
```

```python
four = list("ABCD")
assert queued_misses(four, 3, "warm") == 4                # 每个 prompt 的第一个样本都重算
assert queued_misses(four, 3, "hold") == 2                # 4 个受保护的状态挤 3 个位置
assert queued_misses(four, 3, "hold_cap", cap=3) == 0     # 最多 3 组在等，放得下
assert queued_misses(four, 3, "hold_cap", cap=4) == 2     # 上限超过空位，又回到 hold
```

这也是教学模型：只数 prompt 的状态，不算正在跑的请求占的槽位、decode 快照，也不管时间先后。

上限怎么定？拿真实的数算一遍：

- **状态池：** 单卡时有 212 个槽位。42 个请求同时跑时，实测占了 132–143 个，剩下约 70–80 个：放得下 32 组受保护的状态，放不下 128 组。
- **KV 池：** 也要放得下。seed 0 的 prompt 平均 5.9k token，32 个就需要约 188k token 的 KV；默认 KV 池有 373,823 个 token，够用。
- **反例：** 用 `--mamba-full-memory-ratio 4` 把显存更多地分给状态池时，KV 池缩到了 142,739 个 token，这时同样的改法反而慢了 4.3%。原因很可能是 KV 放不下，但没有单独验证。

## 实测：verl 默认设置下差多少 {#measured}

测试条件：

- **硬件、软件和模型：** A100 SXM 80 GB，SGLang 0.5.23，Qwen3.5-9B，bf16。
- **服务器设置：** `--mem-fraction-static 0.5 --max-running-requests 256`，是 verl 的 `gpu_memory_utilization` 和 `max_num_seqs` 的默认值；另外加了 `--context-length 36864 --enable-session-radix-cache --enable-metrics`，其余用 SGLang 的默认值。
- **工作负载：** 模拟 GRPO 的一步。
  - 128 或 256 个由随机 token 组成的 prompt，长度在 1k–16k 之间按对数均匀分布，各 seed 平均 4.8k–5.9k；
  - 每个 prompt 8 个样本，每个样本生成 512 个 token；
  - 所有请求按 prompt 顺序一次发出，和 verl 的 agent loop 一样。

注意，verl 默认的 `prompt_length` 只有 512，这里的 prompt 长得多。

“改后”指三件事一起做：每个 prompt 先用 session 做一次 prefill，再发它的 8 个样本，同时最多 32 组在等样本开始。

两卡 A100，TP 2（verl 的默认设置）：

| Prompt 数, seed | 现状 | 算掉的 prompt token | 改后 | 算掉的 prompt token |
| --- | --- | --- | --- | --- |
| 128, 0 | 169.5 s | 去重后的 2.6 倍 | 130.7 s（−22.9%） | 1.07 倍 |
| 128, 1 | 159.6 s | 3.0 倍 | 128.7 s（−19.4%） | 1.11 倍 |
| 256, 0 | 332.3 s | 2.6 倍 | 243.5 s（−26.7%） | 1.08 倍 |

单卡，TP 1，3 个 seed。三样拆开单独用，效果都不大：

| 做法 | step 时间，相对现状 |
| --- | --- |
| 先 prefill + session + 限制组数 | −21.8% 到 −22.8% |
| 先 prefill + session | −0.9% 到 −6.3% |
| 先 prefill + 限制组数 | −1.8% 到 −3.1% |
| 只先 prefill（2 个 seed） | +10.6% 到 +11.4% |

单卡 seed 0 的“现状”和“只先 prefill”各少了 3 个样本（1,024 个里完成 1,021 个），起因是客户端的文件句柄上限，所以这两个时间略偏快。

原始数据、脚本和计时拆分都在 [hybrid-tree-state](https://github.com/tianyi-zhang-02/hybrid-tree-state/tree/main/runs)，可以从 `runs/verl-tp2` 和 `runs/verl-check3` 开始看。

## 几种做法比一比

| 做法 | 省了什么 | 代价 | 实测 / 在哪里失效 |
| --- | --- | --- | --- |
| 加大状态池（TP 2，或调 `--mamba-full-memory-ratio`） | 能存更多快照 | 并发上限跟着涨；KV 池可能变小 | 同时到的兄弟样本根本没有状态可用：TP 2 仍有 72–73% 没用上缓存 |
| `--schedule-policy lpm` | 共享前缀的请求先放一个进 batch | 每次调度要做前缀匹配 | 等待请求超过 128 个时整体退回 fcfs（[schedule_policy.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/schedule_policy.py#L304-L314)）：1 个 seed 下重算的样本少 16%，step 快 4.7% |
| 先 prefill 一次 | 不再并发重算 | 每个 prompt 多一个请求 | 状态排队时被挤掉：step 变慢 |
| 再加 session | 状态不被普通 LRU 挤掉 | 受保护的状态占着槽位 | 受保护的状态多过空位，照样被挤 |
| 再限制在途组数 | 受保护的状态放得下 | 要按状态池和 KV 池定上限，还要改客户端 | 三者一起：TP 2 下快 19%–27% |

## 没测的，和以后要重新核对的

- **没测：**
  - 纯 attention 模型的对照：第一种丢法应该也会出现；
  - 并发上限不变、只把状态池加大；
  - 关掉重叠调度（`--disable-overlap-schedule`）；
  - 真实 prompt，以及更长的回答（decode 占比变大，收益应该会变小）；
  - 样本数不是 8，或上限不是 32；
  - 多轮 agent、其他混合模型、多个推理副本；
  - 在 verl 内部跑。verl 在进程内调用引擎，这里用的是 HTTP 服务。
- **会随版本变的行为：** 下面这些都是 SGLang 0.5.23 的实现，换版本要重新查：
  - 并发上限按“池子大小 ÷ 5”来定；
  - `extra_buffer` 的快照位置和 64 个 token 的对齐；
  - lpm 在等待请求超过 128 个时退回 fcfs。

核对日期：2026-11-25。依据：

- 上面各处链接的 SGLang v0.5.23 源码；
- session radix cache 的 [PR #29173](https://github.com/sgl-project/sglang/pull/29173)；
- 上面链接的测量仓库。

小例子由 [hybrid_prefix_cache.py](../code/hybrid_prefix_cache.py) 计算。

接着读 [KV cache 的共享前缀怎么算](kv-cache-and-inference.md#paged-budget)，或者回到[后训练基础设施](../../05-post-training/post-training-infrastructure.md#rollout-record)，看一条 rollout 记录还要留下什么。
