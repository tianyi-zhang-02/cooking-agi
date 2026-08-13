# How big is your negative pool, really

[中文](negative-pool-size.md) · **English**

> Reading time: ~6 min · Type: practice note · Last reviewed: 2026-08

## In one sentence

In multi-GPU training, "the model is distributed" and "the loss is distributed" are two different claims. Parameters, gradients, and optimizer state can all be sharded across ranks while the contrastive loss's negative pool stays local to one device — and this asymmetry **raises no error**. It just lets you believe your negative count is world size times batch size.

## Two kinds of "distributed"

| Thing | Crosses ranks? | Who handles it |
| --- | --- | --- |
| Parameters | Yes, all-gathered when needed | The framework (a sharding config flag) |
| Gradients | Yes, reduce-scatter or all-reduce | The framework |
| Optimizer state | Yes, sharded with the parameters | The framework |
| Data | Yes, each rank takes its own slice | The framework |
| **The contrastive negative pool** | **Not necessarily** | **You, inside the loss** |

The first four rows are configuration. Turn the flag on and you get them. The fifth is not — the framework has no idea what your loss means, and it will not gather other ranks' vectors on your behalf. So this is entirely possible: model state evenly sharded across N devices, the collective communication written beautifully, and each device's contrastive loss seeing only its own handful of samples.

**You have to confirm this by reading the loss function, not the training config.** The config tells you the sharding strategy. It does not tell you how many terms are in the denominator.

## Why the denominator's size is the whole game

A contrastive loss looks like this:

$$\mathcal{L} = -\log \frac{\exp(s(u, c^+)/\tau)}{\exp(s(u, c^+)/\tau) + \sum_{c^- \in \mathcal{N}} \exp(s(u, c^-)/\tau)}$$

Nearly all of the learning signal lives in that sum. If $\mathcal{N}$ is the local batch, the negative count is $B-1$. If it is global, it is $NB-1$.

When $N$ is in the tens, that is **more than an order of magnitude** — with no new data, no new annotation, and no change to the model. Changes with that profile are rare, which is why it is worth checking whether you are leaving one on the table.

Intuitively: more negatives means more chances to put something that *looks right but isn't* into the denominator, which forces a finer boundary. With too few, any unrelated candidate is enough to fill the denominator, and the model only learns to separate obviously different things — a skill that is close to useless in retrieval, where the candidate pool is full of near-misses.

## Adding the cross-rank gather: the trap almost everyone hits first

The naive version:

```python
# wrong
z_all = all_gather(z)                    # collect candidate vectors from every rank
logits = z_u @ z_all.T / temperature
loss = cross_entropy(logits, labels)
```

The problem is that **a generic all-gather does not carry gradients**. It moves other ranks' tensors over, but on this rank they are leaves — constants. Gradients flow back only into the local slice, and the negatives contributed by the other $N-1$ ranks **push nothing**.

The symptom is deceptive: the denominator really is larger, the loss number really does look better, training really does run. But the model learned nothing from the added negatives. They lowered the loss without generating any force that pushes them away.

Two fixes:

1. Use an **autograd-aware** all-gather (every framework ships a differentiable variant, usually under a `distributed.nn`-style namespace);
2. Or gather manually, then splice the local slice back in **with its graph attached** — replace the rows of the gathered tensor that belong to this rank with the original, differentiable tensor.

Verifying beats reading:

```python
z_all = gather(z)
g = torch.autograd.grad(loss, z_all, retain_graph=True)[0]
# rows belonging to other ranks must have nonzero gradient
assert g[other_rank_rows].norm() > 0
```

If that part is zero, you are training an unchanged model against a bigger denominator.

## But it is coupled to sampling correction, and they must ship together

This is the step people skip.

In-batch negatives are sampled **in proportion to content frequency** — how likely an item is to appear in a batch depends on how common it is in the data. So enlarging the pool by $N$ also multiplies by $N$ how often popular items serve as negatives.

**A cross-rank gather amplifies popularity bias by a factor of $N$.**

Ship the gather without the correction and you get a debias that is stronger than before and still pointed in a direction you did not choose: the model systematically suppresses popular content, by an amount set by the data's frequency distribution rather than by you. Aggregate recall may still go up, but you have installed a prior you never designed — a bigger one than you had.

The correction is the standard logQ:

$$s'(u,c) = s(u,c) - \log Q(c)$$

where $Q(c)$ is the estimated probability of $c$ appearing in a batch; a streaming frequency estimate is enough. See [What counts as a positive, and as a negative](positive-negative-design.en.md).

**Land both in the same change.** Gather first, correct later, and everything trained in between carries an amplified prior nobody picked — and the offline metrics will not tell you.

## While you're in there: question the sharding granularity too

Since you are already reading the distributed config, there is a second common mismatch.

If the model itself is small, **sharding it across every device means paying cross-node bandwidth for memory you did not need**. Do the arithmetic: parameters plus gradients plus optimizer state for a sub-1B model, in bf16 with Adam, is a low tens of GB — it fits on one modern accelerator. The memory that full sharding saves is memory you had to spare, and the price is that every layer's all-gather now crosses the node boundary.

Intra-node interconnect is an order of magnitude faster than inter-node fabric. A hybrid strategy — shard within a node, replicate across nodes — keeps every all-gather inside the node and lets only gradient synchronization cross it. Same memory budget, very different traffic.

And one more that gets conflated: **sharding parameters does nothing for activations.** Under long context the real memory consumer is the intermediate tensors saved during the forward pass, which is what activation checkpointing is for. Cranking sharding to its maximum to fix an activation-driven OOM is reaching for the wrong knob.

The test is simple: total up parameters, gradients, and optimizer state, then compare against per-device memory. If it fits, full sharding is buying you something you do not need.

## Down to a checklist

1. Is my contrastive loss's negative pool local or global? — **read the loss function, not the training config**.
2. If it is global, does the gather carry gradients? Is the gradient norm on the non-local rows zero?
3. Now that the pool is bigger, did the sampling correction keep up — or did I just amplify popularity bias by $N$?
4. The things I am sharding: do they actually fit on one device?
5. Is my memory going to parameters or activations? Am I turning the knob that matches?

## Where to read next

- [What counts as a positive, and as a negative](positive-negative-design.en.md): why logQ correction is not optional
- [Two towers, and why the user side splits into several](two-tower-and-beyond.en.md): the model shape this loss trains
- [From noisy feedback to a servable retrieval system](noise-to-signal-retrieval.en.md): the whole chain
