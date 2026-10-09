# MoE: review questions

[中文](review.md) · **English**

> Reading time: ~4 min · Level: advanced · Last reviewed: 2026-10-09

## Interview questions

<details class="interview" markdown="1">
<summary>What sets the total and the active parameter count? Why is Mixtral 8x7B not 56B?</summary>

At fixed expert size, total expert parameters grow with N and main expert compute per token grows with k; the router still scores N experts. Mixtral 8x7B duplicates FFNs, not eight complete models, yielding 46.7B total and 12.9B active parameters rather than 8×7B.

</details>

<details class="interview" markdown="1">
<summary>How does the router compute gate weights? Top-k has no gradient, so how does the router learn?</summary>

A common router scores experts with a linear layer, then selects top-k. Mixtral renormalizes selected weights; Switch top-1 retains the selected full-softmax probability instead of turning its sole gate into one. Ordinary autograd does not differentiate discrete indices; task gradients flow through continuous gates. Unselected expert parameters receive no task gradient from that token, but their router logits may receive gradients through the full softmax or auxiliary objectives.

</details>

<details class="interview" markdown="1">
<summary>Why is load balancing needed? Why is the Switch auxiliary loss f_i times P_i?</summary>

More tokens provide more training opportunities, potentially reinforcing imbalance and leaving devices waiting. Switch uses $\alpha N \sum f_i P_i$: assignment fraction $f_i$ is nondifferentiable, while mean router probability $P_i$ is differentiable. Gradients flow through $P_i$. The value is $\alpha$ at perfect balance, not a strict lower bound over all routing distributions.

</details>

<details class="interview" markdown="1">
<summary>What is the capacity factor? Where do tokens over capacity go?</summary>

Switch top-1 sets capacity from $\frac{\text{tokens}}{N}\times\text{CF}$, rounded according to the implementation. This caps each expert's capacity rather than guaranteeing equal runtime per device. An overflowing expert branch can be skipped while the token continues through the residual path. Top-k and dropless implementations may behave differently; one capacity formula is not universal.

</details>

<details class="interview" markdown="1">
<summary>How does DeepSeek-V3's auxiliary-loss-free balancing work? Is there really no balance loss?</summary>

The bias affects top-k selection while gate weights use original affinities. Overloaded experts have their bias reduced; underloaded experts have it increased. This update needs no auxiliary-loss gradient, but can change the selected set, outputs, and task gradients. V3 still retains a small sequence-wise balance loss ($\alpha=0.0001$), so it is not entirely free of auxiliary objectives.

</details>

<details class="interview" markdown="1">
<summary>What do fine-grained experts and shared experts each solve?</summary>

Fine-grained experts allow more combinations of smaller experts at comparable expert matmul cost. More combinations do not guarantee better specialization or unchanged communication. Shared experts provide a common computation path; reducing duplication is a design aim, not a predetermined division of knowledge. DeepSeek-V3 uses both; Qwen3-235B-A22B / 30B-A3B have no shared experts.

</details>

<details class="interview" markdown="1">
<summary>Does MoE save memory? What are the main costs in deployment?</summary>

Sparse activation does not automatically shrink weight storage by k/N. All parameters must live somewhere: GPU residency, device sharding, and offloading have different costs. Expert parallelism typically adds token dispatch and return traffic, while large batches may use many experts. Comparing memory or speed with a dense baseline requires matching quality, precision, batch, and deployment conditions.

</details>

## Self-check

<div class="taste-check">
  <strong>You understand this if you can explain:</strong>
  <ol>
    <li>why a dense model's parameter count and its compute per token are tied, and how MoE unties them;</li>
    <li>what the normalisation after top-k runs over, and whether an unchosen expert gets any gradient from this token;</li>
    <li>what happens without load balancing, and what an auxiliary loss and a bias-only fix each cost;</li>
    <li>where tokens over capacity end up;</li>
    <li>why MoE saves compute rather than memory, and how batch size changes the arithmetic of decoding.</li>
  </ol>
</div>

## Papers

- [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538): the sparsely-gated MoE and noisy top-k
- [GShard](https://arxiv.org/abs/2006.16668): top-2 routing, expert capacity, expert parallelism
- [Switch Transformers](https://arxiv.org/abs/2101.03961): top-1, the auxiliary loss, capacity factor
- [ST-MoE](https://arxiv.org/abs/2202.08906): the router z-loss
- [Expert Choice Routing](https://arxiv.org/abs/2202.09368): experts choose tokens
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066): fine-grained and shared experts
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437): auxiliary-loss-free balancing, node-limited routing
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [gpt-oss model card](https://arxiv.org/abs/2508.10925)
