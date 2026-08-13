# How big is your negative pool, really

[中文](negative-pool-size.md) · **English**

> Reading time: ~8 min · Type: practice note · Last reviewed: 2026-08

## In one sentence

Contrastive learning is, at bottom, making the model take a multiple-choice test. And in multi-GPU training it is very easy to end up here: you think you are setting a thousand-way question, and every device is actually answering a B-way freebie — **with no error raised anywhere**.

## Start with a multiple-choice question

Say I want to teach a model "what content suits this user."

I have one confirmed positive: user A genuinely liked item X. How do I train on that?

The most direct approach is to write a multiple-choice question. Show the model user A, then a pile of candidate items, one of which is X and the rest pulled at random. Ask it to pick X out. Reward a hit, penalize a miss.

That pile of random ones is the negatives. How many there are is how many options the question has.

Which raises the question that matters: **how many options does this question have?**

- 4-way: the model gets it right and you have learned almost nothing. Guessing scores 25%, and three of the four options are probably from a different planet than user A — all the model has to separate is "same universe or not."
- thousand-way: if the model still gets it right, that means something. With that many options, a few are guaranteed to look **a lot like X but be wrong**, and the model has to draw a far finer line.

Retrieval faces the second situation for real: the candidate pool is wall-to-wall near-misses. So **too few options in training means drilling a model on freebies and then sending it to take the hard exam.**

That is what the negative pool is. Not a hyperparameter you might nudge — it sets how hard the question is.

## So here is the thing that happens

In multi-GPU training, the framework will shard your model across devices and synchronize gradients for you. Those are configuration flags. Turn them on and you have them.

But **you write the question yourself, inside the loss function.** The framework has no idea what your loss is doing, and it will not go collect other devices' candidates to serve as your distractors.

Which can leave you here: N devices, per-device batch of B, and in your head the negative pool is N×B. In reality each device only uses its own B samples as each other's distractors. The question is B-way, not thousand-way.

| Thing | Crosses devices? | Who handles it |
| --- | --- | --- |
| Parameters | Yes, all-gathered back when needed | The framework (a sharding flag) |
| Gradients | Yes, reduce-scatter or all-reduce | The framework |
| Optimizer state | Yes, sharded along with the parameters | The framework |
| Data | Yes, each device takes its own slice | The framework |
| **The contrastive negative pool** | **Not necessarily** | **You, inside the loss** |

The first four rows are configuration. The fifth is not. So this is entirely possible: model state sharded beautifully across dozens of devices, collective communication written with care, and each device's contrastive loss seeing only its own handful of samples.

**You have to confirm this by reading the loss function; the training config will not show it.** The config tells you the sharding strategy. It does not tell you how many terms are in the denominator.

## Writing that question down as a formula

Now put the question above on paper. This is InfoNCE:

$$\mathcal{L} = -\log \frac{\exp\big(s(u, c^+)/\tau\big)}{\sum_{c \,\in\, \{c^+\} \cup \mathcal{N}} \exp\big(s(u, c)/\tau\big)}$$

What each symbol is:

| Symbol | Meaning | What it is in code |
| --- | --- | --- |
| $u$ | One query — here, "user A's current state" | the user tower's output vector, `z_u` |
| $c^+$ | The right answer: the confirmed positive | the positive candidate's vector |
| $\mathcal{N}$ | The distractor set, i.e. the negative pool | every candidate in the denominator except the answer |
| $s(u,c)$ | The scoring function; in a dual encoder, a dot product | `z_u @ z_c` |
| $\tau$ | Temperature, controlling how "peaked" the softmax is | a scalar; smaller means only the closest few matter |
| The denominator | Every option's score, summed | **number of terms = number of options = $\lvert\mathcal{N}\rvert + 1$** |

Read aloud, the formula says: **the right answer's score, as a share of every option's score summed — push that share up.**

So nearly all of the learning signal lives in that denominator sum. The numerator is one term and proves nothing on its own; only being higher *than a pile of distractors* counts as having learned something.

If $\mathcal{N}$ is the local batch, the option count is $B$. If it is global, it is $N \times B$ ($N$ devices, batch $B$ each).

When $N$ is in the dozens, that is more than an order of magnitude. And the change needs **no new data, no new annotation, no model change** — free wins like that are rare, which is exactly why it is worth checking whether you are sitting on one.

## Adding the cross-device gather: the trap almost everyone hits first

The naive version:

```python
# wrong
z_all  = all_gather(z)                   # collect candidate vectors from every device
logits = z_u @ z_all.T / temperature     # denominator now has N*B terms
loss   = cross_entropy(logits, labels)
```

It looks right — the denominator really does have $N \times B$ terms now. The problem is that **a generic all-gather does not carry gradients**.

It moved the other devices' tensors over, but in this device's graph they are constants. Backprop reaches them and stops.

The result is an asymmetry:

| Who | Gets gradient? | What that means |
| --- | --- | --- |
| Local `z_u` | Yes | the user vector is pushed away from all the new distractors ✓ |
| Local candidates | Yes | updated normally ✓ |
| **Other devices' candidates** | **No** | **they never learn they were used as distractors** ✗ |

Contrastive learning is supposed to push **both ways**: the user vector moves away from the distractor, and the distractor moves away from the user vector. Cut the gradient and only the first half survives.

In test terms: you turned a 4-way question into a thousand-way one, and students lose points for picking wrong — but the 764 new distractors are never sent back for revision for "reading too much like the right answer." The question got harder; you are only applying pressure to one side.

What makes this trap nasty is that **it looks like it works**: the denominator really is bigger, the loss number really does look better, training really does run. Metrics may even tick up, because the user tower genuinely did learn. Nothing prompts you to suspect it.

Two fixes:

1. Use an **autograd-aware** all-gather — every framework ships a differentiable variant, usually under a `distributed.nn`-style namespace;
2. Or gather manually, then splice the local slice back in **with its graph attached**: replace the rows of the gathered tensor that belong to this device with the original, differentiable tensor.

Do not judge by reading. Measure:

```python
z_all = gather(z)
g = torch.autograd.grad(loss, z_all, retain_graph=True)[0]
# rows belonging to other devices must have nonzero gradient
assert g[other_rank_rows].norm() > 0
```

If that part is zero, you are training a model on a harder question that it cannot feel.

## But it is coupled to sampling correction — ship them together

This is the step people skip, and skipping it moves you backwards.

Back to the test analogy. Where do the distractors come from? Borrowed from the other samples' candidates in the batch. And the batch is drawn at random from the data, so **an item's chance of appearing in a batch is proportional to how common it is.**

Popular items are common by definition, so they show up as somebody's distractor over and over. The model is trained to push distractors' scores down, and popular items serve as distractors the most, so **the model learns to penalize anything popular** — a bias you did not design, and a byproduct of the data's frequency distribution.

Now grow the pool by $N$ and that effect grows by $N$ too.

**A cross-device gather amplifies popularity bias by a factor of $N$.**

The standard fix is the logQ correction: subtract the log of the sampling probability from the score.

$$s'(u,c) = s(u,c) - \log Q(c)$$

| Symbol | Meaning | Where it comes from |
| --- | --- | --- |
| $s(u,c)$ | The raw score (the dot product) | the model computed it |
| $Q(c)$ | Estimated probability that $c$ shows up **in a batch** | a streaming frequency estimate is plenty |
| $-\log Q(c)$ | Subtracts more the more popular it is; barely touches the tail | cancels "popular items get used as distractors more often" |
| $s'(u,c)$ | The corrected score, and what goes into the softmax | replaces $s$ in the formula above |

Why subtract **$\log Q$** specifically? Because the softmax operates in log space, and subtracting $\log Q$ there is dividing by $Q$ in probability space — removing the part that says "it got sampled because it is common," recovering what it should have scored under uniform sampling.

**So both belong in the same change.** Gather now and correct in a month, and everything trained in between carries a prior nobody picked, amplified $N$ times — and the offline metrics will not tell you. They will just show recall going up.

## While you're in there: question the sharding granularity too

Since you are already in the distributed config, there is one more common mismatch.

**If the model itself is small, sharding it across every device means paying cross-node bandwidth for memory you did not need.**

The arithmetic settles it: parameters plus gradients plus optimizer state for a sub-1B model, in bf16 with Adam, comes to a low tens of GB — that fits on one modern accelerator. The memory full sharding saves is memory you had to spare, and the price is that every layer's all-gather now takes a trip across the node boundary.

Intra-node interconnect is an order of magnitude faster than inter-node fabric. A hybrid strategy — shard within a node, replicate across nodes — keeps every all-gather inside the node and lets only gradient sync cross it. Same memory budget, very different traffic.

And one that gets conflated constantly: **sharding parameters does nothing for activations.** Under long context the real memory consumer is the intermediate tensors saved in the forward pass, which is activation checkpointing's job and has nothing to do with how parameters are split. Maxing out sharding to fix an activation-driven OOM is reaching for the wrong knob.

The test is simple: total up parameters, gradients, and optimizer state, then compare to per-device memory. If it fits, full sharding is buying something you do not need.

## Down to a checklist

1. Is my contrastive loss's negative pool local or global? — **read the loss function, not the training config**.
2. If it is global, does the gather carry gradients? Is the gradient norm zero on the non-local rows?
3. Now that the pool is bigger, did the sampling correction keep up — or did I just amplify popularity bias by $N$?
4. The things I am sharding: do they actually fit on one device?
5. Is my memory going to parameters or activations? Am I turning the matching knob?

## Where to read next

- [What counts as a positive, and as a negative](positive-negative-design.en.md): where distractors come from, and why logQ correction is not optional
- [Two towers, and why the user side splits into several](two-tower-and-beyond.en.md): the model shape this loss trains
- [From noisy feedback to a servable retrieval system](noise-to-signal-retrieval.en.md): the whole chain
