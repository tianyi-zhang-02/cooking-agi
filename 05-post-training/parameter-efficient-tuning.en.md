# Prompt, prefix, and adapter tuning: what can change around a frozen model?

[中文](parameter-efficient-tuning.md) · **English**

> Reviewed: 2026-10 · Prerequisites: [model adaptation](model-adaptation.en.md), [LoRA / QLoRA](lora-and-qlora.en.md)

Consider the same task: adapting an existing model to organize support records into a fixed format. Keep the data and loss unchanged while varying which parameters can learn. That lets you compare update locations without confusing their effects with a change in the training task.

Parameter-efficient fine-tuning (PEFT) covers methods that update only a small set of parameters; it is not a loss function. These methods can be used for supervised fine-tuning or other objectives. The approaches below freeze most or all of the original model parameters, but gradients may still need to pass through the frozen parts.

## Four locations where adaptation can happen

```text
Input embeddings: [learned soft prompt] + [text embeddings]
Layer attention:  [learned prefix keys/values] + [text keys/values]
Hidden states:    h → small nonlinear adapter → h'
Linear weights:   W0 → W0 + low-rank update BA
```

The last approach is low-rank adaptation, or LoRA. This sketch shows where each method changes the computation. Exact placement, parameter sharing, and parameterization vary across implementations; the method name alone does not tell you what was trained.

## Prompt tuning learns input vectors

Prompt tuning prepends a set of learned vectors to the text input. For a model of width $d$ and a prompt with $m$ positions, write these vectors as $P\in\mathbb R^{m\times d}$:

$$
H_0=[P;E(x)],\qquad \hat y=f_{\theta_0}(H_0).
$$

Here, $E(x)$ is the sequence of text embeddings, $[P;E(x)]$ concatenates the two sequences, and $\theta_0$ denotes the frozen model parameters. Only $P$ is trained. These vectors form a soft prompt, not a human-written instruction: individual vectors generally do not correspond to readable words. See [Section 2 of Prompt Tuning (2021)](https://arxiv.org/abs/2104.08691).

With $m=16,d=512$, the soft prompt has 8,192 parameters. The model still processes the extra positions, and gradients still travel from output to input. Activation memory therefore does not shrink in proportion to the trainable parameter count.

At inference time, a decoder with causal attention can precompute some soft-prompt states if the model weights, prompt, and position handling remain fixed. What can be cached depends on the task and implementation. Those positions still occupy context space and participate in subsequent attention computation. During training, the prompt changes, so cached states cannot simply be reused across parameter updates.

## Prefix tuning supplies layerwise keys and values

In a simplified version of prefix tuning, each layer directly learns $m$ extra key/value pairs, $K_P,V_P$, that the text queries $Q$ can attend to:

$$
\operatorname{Attn}\left(Q,[K_P;K_x],[V_P;V_x]\right).
$$

Here, $K_x,V_x$ are the keys and values from the text. The queries can attend to the added prefix positions, which supply keys and values directly at each layer rather than only adding vectors at the input. [Section 4.3 of Prefix-Tuning (2021)](https://arxiv.org/abs/2101.00190) uses an additional network to generate the prefix during training. After training, only the generated prefix states need to be saved, so their size need not equal the training parameter count.

With $L=6$ layers, $m=16$, and total key and value widths each equal to $d=512$, the stored prefix contains $2Lmd=98,304$ scalars. With grouped-query attention, where several query heads share keys and values, use the actual key/value width rather than the total query width.

The virtual prefix positions need no ordinary output predictions, but they still participate in attention and occupy key/value-cache space. Experiments should compare initialization, prefix length, and how prefix parameters are organized across layers—not just learning rate.

## Adapters add nonlinear bottlenecks within layers

A bottleneck adapter maps a layer's representation to a lower dimension and then back to its original width. Omitting biases, a common form is:

$$
h'=h+W_{\mathrm{up}}\phi(W_{\mathrm{down}}h),
$$

Here, $r$ is the bottleneck width, $W_{\mathrm{down}}\in\mathbb R^{r\times d}$, $W_{\mathrm{up}}\in\mathbb R^{d\times r}$, and $\phi$ is a nonlinear activation function. The input is projected down, transformed, projected back up, and added to the original input. [Section 2 of Houlsby et al. (2019)](https://arxiv.org/abs/1902.00751) describes this design; placement can vary by implementation.

Without biases, one module has $2dr$ parameters: 8,192 for $d=512,r=8$. The nonlinearity generally prevents merging it into the original matrix as a linear LoRA update could be, so inference requires additional computation.

A bottleneck that is too narrow may limit what the adapter can learn; adding adapters to every layer increases overhead. Initializing the module close to an identity mapping lets training begin near the original model's behavior. However, in the bias-free formulation above, if $\phi(0)=0$, setting both matrices to zero also makes their gradients zero. Check that the chosen initialization allows the branch to learn.

## Compare all four on the same small model

Use $L=6,d=512,m=16,r=8$. The table counts parameters or stored states under explicit assumptions; it does not compare task performance:

| Method | Assumption | Trainable parameters / state size |
| --- | --- | ---: |
| Prompt | One input $m\times d$ table | 8,192 |
| Prefix | Direct per-layer keys and values, each with total width $d$ | 98,304 |
| Adapter | One bottleneck per layer, no biases | 49,152 |
| LoRA | One adapted $d\times d$ matrix per layer | 49,152 |

The adapter row is a teaching example, not the complete Houlsby configuration, which uses two adapters per layer and also trains layer-normalization parameters and a task-specific classification head. This count includes only one bias-free adapter per layer, not those additional parameters.

Equal adapter and LoRA parameter counts do not mean they can represent the same functions. Adding LoRA to the attention query, key, value, and output projections, as well as multiple feed-forward matrices, increases the count. Always list the target modules when comparing configurations.

## Why must gradients pass through frozen weights?

The soft prompt is at the model's input, while the loss $\mathcal L$ is computed from its output. Obtaining $\partial\mathcal L/\partial P$ requires applying the chain rule through the intervening operations. Wrapping the whole model in `no_grad()` usually cuts that path. Freezing should prevent updates to the model weights, not disconnect the loss from the prompt.

Training bottleneck adapters and LoRA can likewise require gradients to pass through surrounding operations. Implementations can avoid saving intermediates that backward computation does not need, or recompute some results with activation checkpointing. The savings depend on the computation graph; see [precision and memory](../00-foundations/deep-dives/precision-and-memory.en.md).

## Make the small comparison fair

Fix the data, training objective, tokenizer, total training tokens, and hyperparameter-tuning budget. Equal tuning budgets do not require a shared learning rate: tune each method within the same budget. Record task quality, regressions in general capabilities, trainable parameter count, peak training memory, inference latency, and the additional parameter storage needed for each task.

For a formatting-only task, first compare against a well-designed text prompt. Do not assume a tiny adapter will suffice for major capability changes or substantial new domain knowledge. When serving several tasks, also check whether caches remain valid after switching adaptation parameters, whether tasks can share a batch, and whether the base model and adaptation parameters have compatible versions.

Choose based on the behavior you need to change, where the model can best be adapted, and the resources saved in practice. The smallest per-task file is not necessarily the best overall choice.
