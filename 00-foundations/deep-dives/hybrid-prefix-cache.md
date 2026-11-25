# 混合模型的前缀缓存：8 个样本为什么各算了一遍 prompt？

**中文** · [English](hybrid-prefix-cache.en.md)

> 最近审阅：2026-11 · 前置：[Gated DeltaNet](gated-deltanet.md)、[KV cache](kv-cache-and-inference.md#paged-budget)

GRPO 每一步要给同一个 prompt 采样好几个回答，比如 8 个。这 8 个样本的开头完全一样，引擎只要算一次 prompt、存进缓存，另外 7 个就能直接接着用。这就是前缀缓存（prefix caching）。

实际测一步，情况差得远。用 SGLang 跑 Qwen3.5-9B，设置照搬 verl 的默认 rollout 配置。Qwen3.5 是把 Gated DeltaNet 和 attention 层叠在一起的混合模型。结果 72% 的样本没用上缓存，各自又把 prompt 重算了 512 个 token 以上。整步算掉的 prompt token，是去重后的 2.6–3.0 倍。

原因分两层：

1. 第一层和模型结构无关：8 个样本几乎同时进了 batch，那时谁都还没把 prompt 存好。
2. 第二层才和混合模型有关：就算先把 prompt 算好再发样本，混合模型存状态的池子很小，状态常常等不到样本就被挤掉了。

下面先说混合模型的缓存多了什么，再分别看这两层，最后给出实测和几种改法的对比。

## 复用前缀时，混合模型多存了什么

Attention 层给每个 token 存一份 K 和 V。要复用前 $L$ 个 token，取前 $L$ 份就行，前缀想切多长都可以。

Gated DeltaNet 层只有一块固定大小的状态，每读一个 token 就改写一次（见 [Gated DeltaNet](gated-deltanet.md)）。读完 $L+100$ 个 token 的状态，退不回第 $L$ 个 token 时的样子。所以要复用一段前缀，就得事先在那个位置存一份状态快照。

SGLang 0.5.23 默认的 `extra_buffer` 策略在两种时候存快照：prefill 每算完一块，对齐到最近的 64 个 token 边界存一份（[schedule_batch.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/schedule_batch.py#L3170-L3186)）；decode 每走 256 个 token 再存一份。因为要对齐，即使命中缓存，样本通常也还要重算最后几十个 token。

这些快照放在一个启动时就定好大小的池子里，满了就按最近最少使用（LRU）淘汰。一份快照要装下所有 Gated DeltaNet 层的状态，个头不小。用 `--mem-fraction-static 0.5` 实测：

| 配置 | 状态池 | 每份快照 | 并发上限 |
| --- | --- | --- | --- |
| 单张 A100 80 GB，TP 1 | 212 份，共 10.2 GB | 约 48 MB | 42 |
| 两张 A100，TP 2（verl 默认） | 592 份 | 每张卡存一半 | 118 |

最后一列容易被忽略。SGLang 按每个请求 5 个槽位来估算，把并发上限直接定成“池子大小 ÷ 5”（[kv_cache_configurator.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/kv_cache_configurator.py#L3062-L3086)）。所以 verl 要求的 256 个并发请求，被压到了 42 或 118 个。实际跑起来，每个请求平均只占 3 个多槽位。

## 第一层：8 个样本同时到，谁都还没存好

先看一个能手算的例子：prompt 长 2,000 个 token，8 个样本同时到达，每个 prefill batch 最多装 8,192 个 token。调度器每组一个 batch，都会先查缓存，看每个请求有多长的前缀可以复用。

第 1 个 batch 装下样本 1–4，共 8,000 个 token。组 batch 时缓存里还没有这个 prompt，4 个样本只能各算一遍；prompt 的状态要等这个 batch 算完才会写进缓存。

第 2 个 batch 装样本 5–8。SGLang 用了重叠调度（overlap scheduling）：GPU 还在算第 1 个 batch 的时候，CPU 已经在组第 2 个 batch 了（[scheduler.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/scheduler.py#L2061)）。所以样本 5–8 查缓存时，也什么都查不到：

```text
GPU │ batch 1：样本 1–4 各算一遍 prompt │ batch 2：样本 5–8 各算一遍 prompt │
CPU │ 组 batch 2，查缓存：还没有        │ 把 batch 1 的状态写进缓存         │
```

8 个样本各算了一遍，一共 16,000 个 token，其实 2,000 个就够了。

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

每次只改一个条件：

```python
assert concurrent_misses(2000, 8, 8192) == 8          # 两个 batch，全部重算
assert concurrent_misses(2000, 8, 4096) == 4          # batch 变小，只有前两个 batch 重算
assert concurrent_misses(2000, 8, 8192, lag=0) == 4   # 没有重叠调度，只有第一个 batch 重算
```

这只是讲清调度顺序的教学模型。真实的 batch 里还混着别的 prompt 和正在 decode 的请求，长 prompt 也会被切成几块算。最后一行看上去像在说“关掉重叠调度就好了”，可那样 CPU 和 GPU 就没法并行，这里没有测。

真实情况能直接核对。SGLang 给每个请求记了两个时刻：`forward_entry_time` 是请求第一次进入计算 batch 的时刻，紧跟在查缓存之后；`prefill_finished_time` 是 prefill 结束、状态马上要写进缓存的时刻。在 TP 2 下跑了 3 次，每个没用上缓存的样本，都是在同组第一个 prefill 结束之前就进了 batch；在那之后进去的样本，没有一个落空。每次还有 388–455 个落空的样本复用了 prompt 的前一段，最可能的解释是：兄弟样本的 prefill 还没跑完，但已经算好了前几块。

所以，这些样本重算并不是因为状态被淘汰，而是因为那时状态还不存在。这一层只和调度顺序有关。纯 attention 模型在同样的调度下应该也会这样，只是这里没有测。

## 第二层：先算一次 prompt，状态却等不到样本

最直接的改法，是先单独发一个只做 prefill 的请求，算完再发 8 个样本。SGLang 自己处理 `n > 1` 的请求时就是这么做的：先发一个 `max_new_tokens=0` 的请求，再发 n 个样本（[tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/tokenizer_manager.py#L2130-L2151)）。

这样一来，样本到的时候状态已经在缓存里了，可它们还得排队。一步里有约 1,000 个样本在排，前面的请求不停把新快照写进池子，这个 prompt 的状态在 LRU 顺序里越来越旧，很可能等不到自己的样本就被挤掉。

还是手算：4 个 prompt A–D，池子里只空着 3 个位置，每个 prompt 2 个样本。先把 A–D 都 prefill 一遍，再按顺序跑样本。prefill D 时池子已满，最久没用的 A 被挤掉。轮到 A 的样本，第一个查不到，只好重算，重算出的状态又挤掉了 B；第二个样本用上了。B、C、D 依次重演一遍。最后每个 prompt 的第一个样本都重算了，8 个里有 4 个。

实测也是这个样子。单卡上只先 prefill 一次、不加别的保护，128 个 prompt 里有 121–123 个的第一个样本照样重算。这一步反而比直接发样本慢了 10.6%–11.4%：每个 prompt 多了一个请求，状态却没留住。

## 改法：保住状态，再限制在途的组数

开了 `--enable-session-radix-cache` 之后，可以用 SGLang 的 session 保护一个 prompt 的状态，直到客户端调用 `/close_session`。这里的做法是：同组最后一个样本开始出 token 时，就关掉 session。

不过这种保护是软的，不是钉死的（[PR #29173](https://github.com/sgl-project/sglang/pull/29173)）。池子里全是受保护的状态时，再进来一个新状态，最旧的那个仍会被挤掉。所以光开 session 不够，还得限制同时在等样本的 prompt 组数，让受保护的状态放得进池子。

还是 A–D 的例子：

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

这也是教学模型，只数 prompt 的状态，不管正在跑的请求占的槽位、decode 快照和时间先后。

上限该定多少，可以拿真实的数算一遍。单卡时池子有 212 个槽位，42 个请求同时跑，实测占掉 132–143 个，剩下约 70–80 个。这放得下 32 组受保护的状态，放不下 128 组。

KV 池也得放得下。seed 0 的 prompt 平均 5.9k 个 token，32 个就需要约 188k 个 token 的 KV，默认的 KV 池有 373,823 个 token，够用。反过来，用 `--mamba-full-memory-ratio 4` 把更多显存分给状态池时，KV 池缩到 142,739 个 token，同样的改法反而慢了 4.3%。原因很可能就是 KV 放不下了，但没有单独验证。

## 实测：verl 默认设置下差多少 {#measured}

测试条件：

- A100 SXM 80 GB，SGLang 0.5.23，Qwen3.5-9B，bf16；
- `--mem-fraction-static 0.5 --max-running-requests 256`，即 verl 的 `gpu_memory_utilization` 和 `max_num_seqs` 默认值；另加 `--context-length 36864 --enable-session-radix-cache --enable-metrics`，其余用 SGLang 默认值；
- 模拟 GRPO 的一步：128 或 256 个随机 token 组成的 prompt，长度在 1k–16k 之间按对数均匀分布，各 seed 平均 4.8k–5.9k；
- 每个 prompt 8 个样本，每个样本生成 512 个 token；
- 所有请求按 prompt 顺序一次发出，和 verl 的 agent loop 一样。

verl 默认的 `prompt_length` 只有 512，这里的 prompt 要长得多。

表里的“改后”是三件事一起做：每个 prompt 先用 session 做一次 prefill，再发它的 8 个样本，同时最多 32 组在等样本开始。

两张 A100，TP 2，也就是 verl 的默认设置：

| Prompt 数, seed | 现状 | 算掉的 prompt token | 改后 | 算掉的 prompt token |
| --- | --- | --- | --- | --- |
| 128, 0 | 169.5 s | 去重后的 2.6 倍 | 130.7 s（−22.9%） | 1.07 倍 |
| 128, 1 | 159.6 s | 3.0 倍 | 128.7 s（−19.4%） | 1.11 倍 |
| 256, 0 | 332.3 s | 2.6 倍 | 243.5 s（−26.7%） | 1.08 倍 |

单张 A100，TP 1，3 个 seed。把三件事拆开单独用，效果都不大：

| 做法 | step 时间，相对现状 |
| --- | --- |
| 先 prefill + session + 限制组数 | −21.8% 到 −22.8% |
| 先 prefill + session | −0.9% 到 −6.3% |
| 先 prefill + 限制组数 | −1.8% 到 −3.1% |
| 只先 prefill（2 个 seed） | +10.6% 到 +11.4% |

单卡 seed 0 的“现状”和“只先 prefill”两次各少了 3 个样本（1,024 个里完成了 1,021 个），起因是客户端的文件句柄上限，所以这两个时间略微偏快。

原始数据、脚本和计时拆分都在 [hybrid-tree-state](https://github.com/tianyi-zhang-02/hybrid-tree-state/tree/main/runs)，可以从 `runs/verl-tp2` 和 `runs/verl-check3` 看起。

## 几种做法比一比

| 做法 | 省了什么 | 代价 | 实测，或在哪里失效 |
| --- | --- | --- | --- |
| 加大状态池（TP 2，或调 `--mamba-full-memory-ratio`） | 能存更多快照 | 并发上限跟着涨；KV 池可能变小 | 同时到的兄弟样本根本没有状态可用：TP 2 下仍有 72–73% 落空 |
| `--schedule-policy lpm` | 共享前缀的请求先放一个进 batch | 每次调度都要做前缀匹配 | 等待请求超过 128 个时整体退回 fcfs（[schedule_policy.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/schedule_policy.py#L304-L314)）：1 个 seed 下落空的样本少 16%，step 快 4.7% |
| 先 prefill 一次 | 不再并发重算 | 每个 prompt 多一个请求 | 状态在排队时被挤掉，step 反而变慢 |
| 再加 session | 状态不会被普通的 LRU 挤掉 | 受保护的状态占着槽位 | 受保护的状态比空位多时，照样被挤 |
| 再限制在途组数 | 受保护的状态放得下 | 上限要同时按状态池和 KV 池来定，还要改客户端 | 三件事一起：TP 2 下快 19%–27% |

## 没测的，和换版本要重查的

这几种情况没有测：

- 纯 attention 模型的对照，第一层的落空应该也会出现；
- 并发上限不变、只加大状态池；
- 关掉重叠调度（`--disable-overlap-schedule`）；
- 真实 prompt 和更长的回答，decode 占比变大，收益应该会变小；
- 样本数不是 8，或上限不是 32；
- 多轮 agent、其他混合模型、多个推理副本；
- 在 verl 内部跑：verl 在进程内调用引擎，这里用的是 HTTP 服务。

下面这些是 SGLang 0.5.23 的实现细节，换版本要重新查：并发上限按“池子大小 ÷ 5”来定，`extra_buffer` 的快照位置和 64 个 token 的对齐，以及 lpm 在等待请求超过 128 个时退回 fcfs。

核对日期：2026-11-25。依据上面各处链接的 SGLang v0.5.23 源码、session radix cache 的 [PR #29173](https://github.com/sgl-project/sglang/pull/29173)，以及上面的测量仓库。两个小例子由 [hybrid_prefix_cache.py](../code/hybrid_prefix_cache.py) 计算。

接着可以读 [KV cache 的共享前缀怎么算](kv-cache-and-inference.md#paged-budget)，或者回到[后训练基础设施](../../05-post-training/post-training-infrastructure.md#rollout-record)，看一条 rollout 记录还要留下什么。
