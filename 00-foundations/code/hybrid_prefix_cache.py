"""When do a prompt's samples reuse its cached state in a hybrid model?

Teaching model for 00-foundations/deep-dives/hybrid-prefix-cache.md. It counts
samples that recompute the prompt in two situations:

  concurrent_misses  all samples arrive together, before any of them has
                     cached the prompt;
  queued_misses      the prompt is prefilled once, then its samples wait while
                     other prompts' states fill a small least-recently-used pool.

Both are hand-sized models of the scheduling order, not of SGLang's scheduler.
The measured numbers in the note come from real runs, linked there.

    python3 00-foundations/code/hybrid_prefix_cache.py
"""


def concurrent_misses(prompt_len, samples, batch_tokens, lag=1):
    """Samples that recompute the prompt when all of them arrive at once.

    A batch takes as many whole prompts as fit in batch_tokens. A sample reuses
    the prompt only if its batch is formed after the first batch's state has
    been written to the cache. With overlap scheduling, the next `lag` batches
    are formed before that, so they miss too.
    """
    per_batch = max(1, batch_tokens // prompt_len)
    return min(samples, per_batch * (1 + lag))


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


def checks():
    """The numbers quoted in the note."""
    four = list("ABCD")
    return {
        "concurrent_8k": concurrent_misses(2000, 8, 8192),
        "concurrent_4k": concurrent_misses(2000, 8, 4096),
        "concurrent_8k_no_overlap": concurrent_misses(2000, 8, 8192, lag=0),
        "warm": queued_misses(four, 3, "warm"),
        "hold": queued_misses(four, 3, "hold"),
        "hold_cap_3": queued_misses(four, 3, "hold_cap", cap=3),
        "hold_cap_4": queued_misses(four, 3, "hold_cap", cap=4),
    }


if __name__ == "__main__":
    for name, value in checks().items():
        print(f"{name}: {value}")
