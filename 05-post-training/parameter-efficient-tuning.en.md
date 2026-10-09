# Prompt, prefix, and adapter tuning: what can change around a frozen model?

[中文](parameter-efficient-tuning.md) · **English**

> Reviewed: 2026-10 · Prerequisites: [model adaptation](model-adaptation.en.md), [LoRA / QLoRA](lora-and-qlora.en.md)

Suppose an existing model should convert support records to a fixed format. Keep data and loss constant while comparing which parameters may change. Otherwise, gains from a new task can be mistaken for gains from a different update location.

PEFT is a family of parameter-efficient adaptation methods, not a loss. It can accompany SFT or other objectives. Freezing the base also does not imply that gradients never pass through it.

## Four locations where adaptation can happen

```text
Input embeddings: [learned soft prompt] + [text]
Layer attention:  [learned prefix K/V] + [text K/V]
Hidden states:    h → small nonlinear adapter → h'
Linear weights:   W0 → W0 + low-rank update BA
```

These are mechanism sketches. Placement, sharing, and parameterization vary by implementation, so the method name alone does not identify what was trained.

## Prompt tuning learns input vectors

For model width $d$, prepend $m$ trainable vectors $P\in\mathbb R^{m\times d}$:

$$
H_0=[P;E(x)],\qquad \hat y=f_{\theta_0}(H_0).
$$

Only $P$ updates; $\theta_0$ is frozen. This is not a human-written instruction, and soft vectors generally do not map one-to-one to readable text. [Prompt Tuning](https://arxiv.org/abs/2104.08691) studies this adaptation mechanism.

With $m=16,d=512$, the prompt has 8,192 parameters. The model still processes the extra positions, and gradients still travel from output to input. Activation memory does not shrink in proportion to trainable parameters.

In a decoder, some fixed-prompt states can be precomputed, subject to task, position, and implementation constraints. This does not remove their context occupancy or attention work.

## Prefix tuning supplies layerwise keys and values

A simplified direct-K/V prefix at each layer gives

$$
\operatorname{Attn}\left(Q,[K_P;K_x],[V_P;V_x]\right).
$$

Ordinary token queries can attend to those positions. This acts deeper than an input-only embedding prompt. [Prefix-Tuning](https://arxiv.org/abs/2101.00190) can use an additional network during training, so final prefix state size is not the trainable parameter count of every implementation.

With $L=6$ layers, $m=16$, and total K and V widths each equal to $d=512$, a direct prefix contains $2Lmd=98,304$ scalars. For GQA, use the actual KV width rather than query width.

The virtual prefix need not produce ordinary output tokens, but it still participates in attention and cache storage. Compare initialization, length, and layerwise parameterization as well as learning rate.

## Adapters add nonlinear bottlenecks within layers

One bottleneck adapter is

$$
h'=h+W_{\mathrm{up}}\phi(W_{\mathrm{down}}h),
$$

where $W_{\mathrm{down}}\in\mathbb R^{r\times d}$ and $W_{\mathrm{up}}\in\mathbb R^{d\times r}$. It compresses, transforms, and adds back to the input. [Houlsby et al.](https://arxiv.org/abs/1902.00751) provide a classic adapter design; placement can vary.

Ignoring biases, one module uses $2dr$ parameters: 8,192 for $d=512,r=8$. The nonlinearity generally prevents folding it into the original matrix as a linear LoRA update, so inference includes additional computation.

A narrow bottleneck may underfit; widespread insertion increases overhead. Near-identity initialization can help, but zeroing both matrices may prevent the branch from learning. Check initialization gradients.

## Compare all four on the same small model

Use $L=6,d=512,m=16,r=8$. This is parameter accounting, not a quality ranking:

| Method | Assumption | Trainable parameters / state size |
| --- | --- | ---: |
| Prompt | One input $m\times d$ table | 8,192 |
| Prefix | Direct per-layer K/V, each width $d$ | 98,304 |
| Adapter | One bottleneck per layer, no biases | 49,152 |
| LoRA | One adapted $d\times d$ matrix per layer | 49,152 |

Equal adapter and LoRA counts do not imply equal function classes. Adapting Q, K, V, O, and multiple FFN projections increases LoRA parameters. List target modules when making comparisons.

## Why must gradients pass through frozen weights?

An input prompt precedes the model; its loss follows the model. Computing $\partial\mathcal L/\partial P$ requires the chain rule through the intervening operations. Putting the entire model under no-grad usually severs that path. Freezing controls parameter updates, not arbitrary graph disconnection.

Adapters and LoRA also rely on gradient propagation through surrounding operations. Some intermediates can be avoided or recomputed, but savings depend on the graph; see [precision and memory](../00-foundations/deep-dives/precision-and-memory.en.md).

## Make the small comparison fair

Fix data, objective, tokenizer, training tokens, and tuning budget. Record task quality, general-capability regressions, trainable parameters, peak training memory, inference latency, and per-task storage.

For formatting-only tasks, compare against a good ordinary prompt first. For major capability changes or substantial new domain knowledge, do not assume a tiny adapter suffices. Multi-task serving also needs tests for cache invalidation, batching, and version binding.

A useful choice explains the behavior to change, where to intervene, and what resources are actually saved. The smallest checkpoint is not automatically the best option.
