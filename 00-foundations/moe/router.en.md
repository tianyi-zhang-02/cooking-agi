# MoE: how the router picks experts

[中文](router.md) · **English**

> Reading time: ~10 min · Level: advanced · Last reviewed: 2026-10-09

A token entering an MoE layer does not run through every expert. The router selects a few, then combines their outputs with learned weights. Selecting the largest scores is only the beginning: a token can take several branches, their results must return to the right position, and gradients must follow those paths back.

For a first pass, read the first two sections and the [dispatch example](#dispatch-example). For implementation work, expand the code and continue to the gradient checks. We start on one device before considering why distributed routing may restrict expert groups.

<span id="the-router-is-one-linear-layer"></span>

## A common router starts with a linear layer

A simple router produces one score per expert from the token's hidden state $x$, not its token ID. The same word can therefore choose different experts in different contexts. Multiplying $x$ by an $N \times d$ matrix gives $N$ scores: $s = W_r x$, where $N$ is the expert count and $d$ is the hidden width.

Row $i$ of $W_r$ scores expert $i$; it is not that expert's FFN weights. The router learns which experts to use, while each expert learns a feature transformation. They have separate parameters.

One common rule keeps the $k$ highest scores and normalizes over them to obtain gate weights. Other routers add grouping, biases, or different scoring rules:

$$g_i = \frac{e^{s_i}}{\sum_{j \in \mathrm{TopK}} e^{s_j}}, \qquad i \in \mathrm{TopK}$$

This is the same as a softmax over all $N$ followed by renormalising over the ones kept. Mixtral and gpt-oss do exactly this; DeepSeek-V3 scores with a sigmoid instead and normalises over the selected experts.

For scores `[2, 1, 0]`, keeping and renormalizing the top two gives weights about `[0.731, 0.269]`. This is not universal: top-1 in [Switch Transformer](https://www.jmlr.org/papers/v23/21-0998.html) keeps the selected probability from the full softmax rather than renormalizing it to 1. That gate preserves a task-gradient path to the router. The figure illustrates renormalization over selected experts.

These values are often called probabilities, but **top-k is not a random draw**. Without noise or ties, the same scores select the same experts deterministically. The gate values then weight their outputs.

<!-- widget:tx-moe-router -->

## Top-1, top-2, or top-8

- **Top-1**: Switch Transformer. One expert per token, using less expert compute when other factors are equal.
- **Top-2**: GShard, Mixtral. Two expert outputs can contribute, providing another path at additional cost.
- **Top-4**: gpt-oss.
- **Top-8**: DeepSeek-V3, Qwen3. Their experts are cut finer (next-but-one note), so each token picks more of them.

At fixed expert width, larger $k$ usually means more expert compute; cross-device traffic also depends on placement. Across models, top-8 with small experts need not cost more than top-2 with wide experts, or be more stable.

## Why early MoE added noise

The first sparsely-gated MoE (Shazeer et al., 2017) added Gaussian noise with a learned scale to the scores before taking the top k:

$$\begin{aligned}
\sigma_i &= \mathrm{Softplus}\big((xW_{\text{noise}})_i\big),\\
\epsilon_i &\sim \mathcal N(0,1),\\
H_i &= (xW_g)_i+\epsilon_i\sigma_i.
\end{aligned}$$

$\sigma_i$ sets the noise scale, $\epsilon_i$ is the current draw, and $H_i$ is the score used for expert selection.

[Noise](https://arxiv.org/abs/1701.06538) gives experts near the selection boundary a chance to run. It supports exploration, but a single perturbation need not change the selection, and balanced training is not guaranteed. The figure uses synthetic scores to show how noise changes decisions near a boundary; it does not identify actual word-to-expert assignments.

## Top-k has no gradient: how does the router learn?

Ordinary top-k implementations do not backpropagate through the selected indices. Gradients reach the router through selected gate weights $g_i$. Unselected expert **parameters** usually receive no task gradient from that token; unselected router logits may still receive gradients through normalization or auxiliary objectives. Full-softmax denominators involve other logits, whereas renormalizing over selected experts changes that relationship.

One possible feedback loop is that experts receiving more tokens early get more training and then get selected more often. The next note covers detecting and mitigating that imbalance, rather than assuming it must occur.

## Dispatch the inputs, then put the outputs back {#dispatch-example}

Start on one device with two tokens and three experts. The outputs below are made-up numbers for tracing the computation, not model measurements:

| Input position | Selected expert | Weight | Expert output |
| --- | --- | --- | --- |
| token 0 | expert 0 | 0.75 | `[2, 0]` |
| token 0 | expert 2 | 0.25 | `[0, 4]` |
| token 1 | expert 1 | 0.60 | `[1, 3]` |
| token 1 | expert 2 | 0.40 | `[5, 1]` |

Organizing inputs by expert gives token 0 to expert 0, token 1 to expert 1, and both tokens to expert 2. This is **dispatch**. After the experts run, use the saved token positions to return each weighted result:

$$\begin{aligned}
y_0 &= 0.75[2,0]+0.25[0,4]\\
&= [1.5,1],\\
y_1 &= 0.60[1,3]+0.40[5,1]\\
&= [2.6,2.2].
\end{aligned}$$

**Add the contributions; do not overwrite or concatenate them.** Assigning expert 2's result directly to `output[token_ids]` would erase earlier contributions. Shapes might still look correct even though the computation is wrong.

<details markdown="1">
<summary>A small PyTorch implementation with backpropagation</summary>

This function implements only the routed FFN branch. Inputs and outputs have shape `[tokens, width]`; the outer residual, shared experts, and padding handling are outside it. Each expert is a PyTorch module with the same input and output width.

```python
def sparse_moe(hidden, logits, experts, top_k, normalization="selected"):
    if hidden.ndim != 2 or logits.shape != (hidden.shape[0], len(experts)):
        raise ValueError("Expected [tokens, width] inputs and one logit per expert")
    indices, weights = selected_gates(logits, top_k, normalization)
    output = torch.zeros_like(hidden)
    for expert_id, expert in enumerate(experts):
        token_ids, slots = torch.where(indices == expert_id)
        if token_ids.numel() == 0:
            continue
        expert_inputs = hidden.index_select(0, token_ids)
        expert_outputs = expert(expert_inputs)
        weighted = expert_outputs * weights[token_ids, slots, None]
        output = output.index_add(0, token_ids, weighted)
    return output
```

Both `indices` and `weights` have shape `[tokens, top_k]`. `token_ids` identifies the original row; `slots` locates this expert among that token's choices. `index_select` gathers inputs and `index_add` accumulates outputs. In the [complete script](../code/moe_routing.py), `selected_gates` supports normalization over selected logits or retaining the selected entries of the full softmax.

This is a **dropless, single-device teaching implementation** with no capacity cap. Python loops, per-expert indexing, and allocating new output tensors are not a high-throughput design. Production implementations may sort tokens, use grouped GEMM, and exchange them across devices with all-to-all, while preserving dispatch and combination semantics. CUDA accumulation can also be nondeterministic; see the [PyTorch `index_add_` documentation](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.index_add_.html). These examples are tested on CPU.

</details>

## Check the gradients, not just the shapes {#routing-gradients}

Build a slow reference: run every expert on every token, assign zero weight to unselected experts, and sum. For deterministic experts without cross-token coupling, this **dense oracle** should match sparse dispatch in both outputs and gradients of the inputs, router, and expert parameters. It is a test reference, not a serving strategy.

Then isolate one token and deliberately leave an expert unselected:

| Gate rule | Task-gradient path to the router | Unselected expert parameters |
| --- | --- | --- |
| top-2, normalize selected scores | Through the two gates; logits outside the current set are absent from the denominator | No task gradient from this token |
| top-1, normalize the selected score | The sole weight is always 1, so this gate path has zero gradient | Same |
| top-1, keep the full-softmax probability | The selected probability varies; its denominator also depends on other logits | Same; logit gradients are not expert-parameter gradients |

These statements concern **this token, the task loss, and a neighborhood with fixed selection**. Other tokens, auxiliary losses, and shared parameters affect a real training step. A differentiable gate can still have zero gradient for particular expert outputs.

Keep finite-difference checks away from ties and selection boundaries. If the second and third scores are nearly equal, a tiny perturbation can swap experts; one smooth branch's derivative no longer describes both sides. PyTorch also does not guarantee [stable `topk` indices for tied values](https://docs.pytorch.org/docs/2.8/generated/torch.topk.html).

## Why distributed routing may choose groups first {#group-limited-routing}

If top-2 selects experts on two devices, the token representation has two destinations. Restricting candidates to fewer device groups can reduce destinations, at the cost of excluding higher-scoring experts elsewhere.

Take group A with scores `[0.90, 0.10]` and group B with `[0.80, 0.79]`. Unrestricted top-2 selects `0.90` and `0.80`. Selecting one group by its maximum first chooses A, leaving `0.90` and `0.10`. One fewer group is the benefit; losing the `0.80` expert is the tradeoff. Reduced communication alone does not establish unchanged quality.

[DeepSeek-V2 §2.2.2](https://arxiv.org/html/2405.04434v5#S2.SS2.SSS2) restricts device destinations. [V3 §2.1.2](https://arxiv.org/html/2412.19437v1#S2.SS1.SSS2) scores nodes by summing several of their highest affinities instead; grouped routing does not always mean taking a group maximum. In our example, summing each group's top two scores makes B's `1.59` beat A's `1.00`. The group scoring rule changes the result.

The script's `group_limited_topk` separates the group count, groups retained, and number of scores summed per group. It rejects insufficient candidates and masks excluded groups with negative infinity, not zero, so negative logits cannot accidentally select them. This tests selection logic, not network communication or a complete V2/V3 router.

From the repository root, run the CPU demo to check the output shape and gradient path, then compare two balance statistics:

```bash
python3 00-foundations/code/moe_routing.py
```

Continue to [load balancing](load-balancing.en.md#sequence-balance): choosing the same number of experts per token does not give every expert the same amount of work.

## Tokens choose experts, or experts choose tokens

Everything above has tokens choose experts. [Expert Choice](https://arxiv.org/abs/2202.09368) reverses that: each expert selects a fixed number of tokens from the batch. With enough candidates, assignment counts can balance by construction, though device runtimes need not. A token can be selected several times or not at all. Reported convergence gains apply to the paper's training settings, not arbitrary serving workloads.

For autoregressive tasks, ask whether the candidate set contains future tokens. If later tokens can change whether an earlier token is selected, selection itself can leak future information. Restrict candidate scope during training, and check whether batching changes a single request's behavior at deployment.
