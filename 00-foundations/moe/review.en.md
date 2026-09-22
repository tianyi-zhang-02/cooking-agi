# MoE: review questions

[中文](review.md) · **English**

> Reading time: ~4 min · Level: advanced · Last reviewed: 2026-09

## Interview questions

<details class="interview" markdown="1">
<summary>What sets the total and the active parameter count? Why is Mixtral 8x7B not 56B?</summary>

Total parameters grow with the number of experts $N$; compute per token grows only with $k$. Mixtral 8x7B has 8 experts per layer and sends each token to 2: only the FFNs are copied, attention and embeddings exist once, so the total is 46.7B with 12.9B active per token.

</details>

<details class="interview" markdown="1">
<summary>How does the router compute gate weights? Top-k has no gradient, so how does the router learn?</summary>

The router is a linear layer that scores every expert; it keeps the top $k$ and takes a softmax over those $k$ to get the weights (DeepSeek-V3 uses a sigmoid and then normalises). Choosing has no gradient; the gradient reaches the router through the gate weights $g_i$ of the chosen experts, and an expert that was not chosen gets no gradient from this token.

</details>

<details class="interview" markdown="1">
<summary>Why is load balancing needed? Why is the Switch auxiliary loss f_i times P_i?</summary>

Imbalance reinforces itself: an expert that gets more tokens is trained better, gets more tokens, and eventually a few experts do all the work. Switch's auxiliary loss is $\alpha N \sum f_i P_i$: $f_i$ is the real load fraction but has no gradient, $P_i$ is the mean router probability and does; multiplied together, the gradient flows through $P_i$ with a strength set by $f_i$, and the loss is minimal at perfect balance.

</details>

<details class="interview" markdown="1">
<summary>What is the capacity factor? Where do tokens over capacity go?</summary>

Each expert takes at most $\frac{\text{tokens}}{N} \times \text{CF}$ tokens, which keeps compute per device fixed. Tokens over the cap are not deleted; they skip this layer's expert and pass on through the residual connection. Switch found CF between 1.0 and 1.25 worked better; DeepSeek-V3 drops no tokens.

</details>

<details class="interview" markdown="1">
<summary>How does DeepSeek-V3's auxiliary-loss-free balancing work? Is there really no balance loss?</summary>

Each expert has a bias that is added to its score only when choosing the top k, while the gate weight still uses the original score; after every step an overloaded expert's bias goes down by $\gamma$ and an idle one's goes up by $\gamma$. No extra gradient interferes with the language-model loss. It still keeps a very small sequence-wise balance loss ($\alpha = 0.0001$), so it is not literally none.

</details>

<details class="interview" markdown="1">
<summary>What do fine-grained experts and shared experts each solve?</summary>

Fine-grained: cut experts smaller and choose more of them; compute stays the same but the number of combinations explodes, so expertise separates more cleanly. Shared experts: every token passes through them, they hold common knowledge, and routed experts duplicate less. DeepSeek uses both; Qwen3's MoE has no shared expert.

</details>

<details class="interview" markdown="1">
<summary>Does MoE save memory? What are the main costs in deployment?</summary>

No. Any token may use any expert, so all parameters must be in memory (or sharded across GPUs). The main costs are memory, the two all-to-all exchanges, and, at large batch sizes, reading nearly every expert each step. What it saves is compute per token.

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
