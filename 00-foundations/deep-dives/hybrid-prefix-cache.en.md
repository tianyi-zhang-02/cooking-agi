# Prefix caching in hybrid models: why did 8 samples of one prompt each compute it?

[中文](hybrid-prefix-cache.md) · **English**

> Last reviewed: 2026-11 · Prerequisites: [Gated DeltaNet](gated-deltanet.en.md), [KV cache](kv-cache-and-inference.en.md#paged-budget)

In one GRPO step, each prompt is sampled several times, say 8. In principle, the engine computes the prompt once, caches it, and the other 7 samples reuse it. This is prefix caching.

A measured step does not look like that. It ran SGLang serving Qwen3.5-9B at verl's default rollout settings; Qwen3.5 stacks Gated DeltaNet and attention layers. The results:

- 72% of the samples missed the cached prompt, each recomputing more than 512 of its tokens;
- in all, 2.6–3.0 times the unique prompt tokens were computed.

There are two causes:

1. A prompt's samples enter batches side by side, before any of them has cached the prompt. This depends on the scheduling order, not on the Gated DeltaNet layers.
2. Computing the prompt once before sending the samples avoids the first cause. But a hybrid model's state pool is small, and the state is often evicted before the samples run. This part is specific to hybrid models.

## Reusing a prefix needs one more thing in a hybrid model

An attention layer stores K and V for every token. To reuse the first $L$ tokens, take the first $L$ entries; any prefix length can be cut out.

A Gated DeltaNet layer keeps one fixed-size state and rewrites it at every token (see [Gated DeltaNet](gated-deltanet.en.md)). The state after $L+100$ tokens cannot be turned back into the state after $L$. Reusing a prefix needs a snapshot of the state taken there.

SGLang 0.5.23, with its default `extra_buffer` strategy, takes snapshots in two places:

- after each prefill chunk, rounded down to a 64-token boundary ([schedule_batch.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/schedule_batch.py#L3170-L3186));
- every 256 decode tokens.

Because of the rounding, a sample that hits the cache usually still recomputes a few dozen tokens at the end.

The snapshots live in a pool whose size is fixed at startup, evicting the least recently used (LRU) when full. A snapshot holds the state of every Gated DeltaNet layer, so it is large. These numbers are measured; the conditions are under [Measured](#measured).

- On one A100 80 GB with `--mem-fraction-static 0.5`, SGLang gives the pool 10.2 GB. That holds only 212 snapshots, about 48 MB each.
- The pool also sets the concurrency limit. SGLang budgets 5 slots per request and sets the running limit to the pool size ÷ 5 ([kv_cache_configurator.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/kv_cache_configurator.py#L3062-L3086)). So the 256 concurrent requests verl asks for become 42. In practice each running request used a little over 3 slots.
- With verl's default tensor parallelism of 2 (TP 2), each GPU holds half the weights and half of each state. The pool grows to 592 snapshots and the limit to 118, still below 256.

## The first way to miss: 8 samples arrive together, so none reuses another's prompt

A small example you can work by hand:

- the prompt is 2,000 tokens, and its 8 samples arrive at once;
- a prefill batch holds at most 8,192 tokens.

When the scheduler forms a batch, it first looks up how long a cached prefix each request can reuse.

1. **Batch 1** takes samples 1–4, 8,000 tokens. When the batch is formed, nothing is cached yet, so all four compute the prompt. The prompt's state is written to the cache only after this batch finishes.
2. **Batch 2** takes samples 5–8. SGLang uses overlap scheduling: while batch 1 runs on the GPU, the CPU is already forming batch 2 ([scheduler.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/scheduler.py#L2061)). So samples 5–8 find nothing either.

All 8 samples compute the prompt: 16,000 tokens where 2,000 would do.

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

Change one condition at a time:

```python
assert concurrent_misses(2000, 8, 8192) == 8          # two batches, all recompute
assert concurrent_misses(2000, 8, 4096) == 4          # smaller batches: only the first two miss
assert concurrent_misses(2000, 8, 8192, lag=0) == 4   # no overlap: only the first batch misses
```

This models only the scheduling order. Real batches also carry other prompts and running decodes, and long prompts are split into chunks. The last line seems to say that turning off overlap scheduling would help; it would also stop the CPU and GPU from working in parallel, and it was not tested.

The real case can be checked directly. SGLang records two times for each request:

- `forward_entry_time`: when it first entered a forward batch, right after its cache lookup;
- `prefill_finished_time`: when its prefill finished, just before its state is written to the cache.

In 3 runs at TP 2:

- Every sample that missed the cache entered a batch before the group's first prefill had finished. None that entered later missed.
- In each run, 388–455 of these samples reused the start of the prompt. Most likely a sibling's prefill was still running but had already finished a few chunks.

So these samples did not lose the state to eviction; the state did not exist yet. This part depends only on the scheduling order. An attention-only model scheduled the same way should miss in the same way, but that was not tested.

## Prefill once first: no duplicated work, but the state may be evicted

The direct fix is to send one prefill-only request first, then the 8 samples. SGLang does this itself for an `n > 1` request: a `max_new_tokens=0` request, then the n samples ([tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/tokenizer_manager.py#L2130-L2151)).

Now the state is cached when the samples arrive, but they still wait in the queue. About 1,000 samples are queued in one step, and the requests ahead keep writing new snapshots to the pool. The prompt's state grows older in the LRU order and may be evicted before its samples run.

Another hand-sized example:

- 4 prompts A–D, a pool with only 3 free slots, and 2 samples per prompt;
- all four are prefilled first, then the samples run in order.

1. Prefilling D evicts A, the least recently used.
2. A's first sample finds nothing and recomputes, and the recomputed state evicts B. A's second sample reuses it.
3. B, C and D go the same way.

The first sample of every prompt recomputes: 4 of 8.

The measurement shows the same pattern. On one GPU, prefilling once with nothing else made the first sample miss in 121–123 of 128 prompts. That step was 10.6%–11.4% slower than sending the samples as they are: each prompt cost an extra request and still lost its state.

## Hold the state, and limit how many groups are in flight

With `--enable-session-radix-cache`, an SGLang session protects a prompt's state until the client calls `/close_session`. Here the session was closed when the group's last sample returned its first token.

The protection is soft, not pinned ([PR #29173](https://github.com/sgl-project/sglang/pull/29173)): when the pool is full of protected states, a new one still evicts the oldest.

So a session alone is not enough. The number of prompt groups waiting for their samples also has to be capped, so that the protected states fit in the pool.

The same example:

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
assert queued_misses(four, 3, "warm") == 4                # every prompt's first sample misses
assert queued_misses(four, 3, "hold") == 2                # 4 protected states, 3 slots
assert queued_misses(four, 3, "hold_cap", cap=3) == 0     # at most 3 groups waiting: they fit
assert queued_misses(four, 3, "hold_cap", cap=4) == 2     # a cap above the free slots is back to hold
```

This is also a teaching model. It counts only prompt states; running requests' slots, decode snapshots and timing are left out.

To size the cap, use the real numbers.

- **State pool.** On one GPU it has 212 slots. With 42 requests running they used 132–143, which leaves about 70–80. That fits 32 protected states but not 128.
- **KV pool.** It has to fit too. Seed 0's prompts average 5.9k tokens, so 32 of them need about 188k tokens of KV. The default KV pool holds 373,823.
- **A counterexample.** With `--mamba-full-memory-ratio 4`, which gives the state pool more memory, the KV pool shrank to 142,739 tokens, and the same change was 4.3% slower. Running out of KV is the likely cause, but it was not checked separately.

## Measured: how much it matters at verl's defaults {#measured}

Conditions:

- **Hardware, software, model:** A100 SXM 80 GB, SGLang 0.5.23, Qwen3.5-9B in bf16.
- **Server:** `--mem-fraction-static 0.5 --max-running-requests 256`, which are verl's `gpu_memory_utilization` and `max_num_seqs` defaults, plus `--context-length 36864 --enable-session-radix-cache --enable-metrics`. Everything else is at SGLang's defaults.
- **Workload:** one simulated GRPO step:
  - 128 or 256 prompts of random tokens, log-uniform in length from 1k to 16k (mean 4.8k–5.9k by seed);
  - 8 samples per prompt, 512 new tokens each;
  - all requests sent at once in prompt order, as verl's agent loop does.

Note that verl's default `prompt_length` is 512, so these prompts are much longer.

"Changed" means all three together: each prompt is prefilled once on a session, then its 8 samples are sent, with at most 32 groups waiting for their samples to start.

Two A100s, TP 2 (verl's default):

| Prompts, seed | Current | Prompt tokens computed | Changed | Prompt tokens computed |
| --- | --- | --- | --- | --- |
| 128, 0 | 169.5 s | 2.6 × unique | 130.7 s (−22.9%) | 1.07 × |
| 128, 1 | 159.6 s | 3.0 × | 128.7 s (−19.4%) | 1.11 × |
| 256, 0 | 332.3 s | 2.6 × | 243.5 s (−26.7%) | 1.08 × |

One A100, TP 1, 3 seeds. Taken apart, no piece does much on its own:

| Policy | Step time vs. current |
| --- | --- |
| prefill once + session + cap | −21.8% to −22.8% |
| prefill once + session | −0.9% to −6.3% |
| prefill once + cap | −1.8% to −3.1% |
| prefill once only (2 seeds) | +10.6% to +11.4% |

On one GPU, seed 0's "current" and "prefill once only" runs each finished 1,021 of 1,024 samples because of the client's open-file limit, so those two times are slightly fast.

Raw runs, scripts and the timing split are in [hybrid-tree-state](https://github.com/tianyi-zhang-02/hybrid-tree-state/tree/main/runs); start with `runs/verl-tp2` and `runs/verl-check3`.

## Comparing the options

| Option | Saves | Costs | Measured / where it fails |
| --- | --- | --- | --- |
| A larger state pool (TP 2, or `--mamba-full-memory-ratio`) | Room for more snapshots | The running limit grows too; the KV pool may shrink | Siblings arriving together have no state to reuse: 72–73% still missed at TP 2 |
| `--schedule-policy lpm` | One request per shared prefix enters a batch first | Prefix matching on every schedule | Falls back to fcfs as a whole while more than 128 requests wait ([schedule_policy.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/schedule_policy.py#L304-L314)): with 1 seed, 16% fewer misses and a 4.7% faster step |
| Prefill once | No concurrent recompute | One extra request per prompt | The state is evicted while queued: the step gets slower |
| Add a session | Ordinary LRU cannot evict the state | Protected states hold slots | More protected states than free slots are still evicted |
| Add a cap on groups in flight | Protected states fit | The cap must fit both the state pool and the KV pool, and the client has to change | All three together: 19%–27% faster at TP 2 |

## What was not tested, and what to recheck

- **Not tested:**
  - an attention-only control, where the first kind of miss should also appear;
  - a larger state pool with the running limit held fixed;
  - turning off overlap scheduling (`--disable-overlap-schedule`);
  - real prompts, and longer responses (decode takes a larger share, so the gain should shrink);
  - sample counts other than 8, and caps other than 32;
  - multi-turn agents, other hybrid models, several inference replicas;
  - running inside verl, which calls the engine in-process; these runs used SGLang's HTTP server.
- **Version-specific behavior:** these are SGLang 0.5.23 implementation details, so check them again on a new version:
  - the running limit set to the pool size ÷ 5;
  - where `extra_buffer` takes snapshots, and the 64-token rounding;
  - lpm falling back to fcfs above 128 waiting requests.

Checked 2026-11-25 against:

- the SGLang v0.5.23 source linked in each section above;
- the session radix cache [PR #29173](https://github.com/sgl-project/sglang/pull/29173);
- the measurement repository linked above.

The small examples are computed by [hybrid_prefix_cache.py](../code/hybrid_prefix_cache.py).

Next, read [how shared prefixes are counted in the KV cache](kv-cache-and-inference.en.md#paged-budget), or go back to [post-training infrastructure](../../05-post-training/post-training-infrastructure.en.md#rollout-record) for what else a rollout record has to keep.
