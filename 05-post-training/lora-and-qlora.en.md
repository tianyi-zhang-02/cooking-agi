# LoRA and QLoRA: fewer parameters, but still out of memory?

[中文](lora-and-qlora.md) · **English**

> Last reviewed: 2026-10 · Prerequisites: [One training step](../00-foundations/deep-dives/training-step.en.md), [Model adaptation](model-adaptation.en.md)

You want a model to organize support records into a fixed format. The data and loss for supervised fine-tuning (SFT) are ready, but full fine-tuning exceeds your GPU memory budget. You switch to low-rank adaptation (LoRA), training only a small set of added parameters. Memory use drops, but longer sequences still trigger out-of-memory (OOM) errors.

LoRA mainly reduces the number of trainable parameters and the gradients and optimizer states associated with them. It does not eliminate all the intermediate results needed for the forward and backward passes. A single linear layer is enough to see what it saves—and what it does not.

## SFT and LoRA answer different questions

SFT describes training on example responses; LoRA determines which parameters can change. The relevant comparison is **full-parameter SFT versus SFT with LoRA**, not “LoRA versus SFT.” LoRA also works with other training objectives.

If the only problem is occasional invalid formatting, consider output validation or constrained decoding first. Retrieval may be a better fit for missing facts that change frequently. Choose a parameter-update method once you have established that training is needed to change the model's behavior.

## Replace one large update with two small matrices

A linear layer originally computes $y=W_0x$. LoRA freezes $W_0$ and learns a small additional branch:

$$
y=W_0x+sBAx,\qquad s=\alpha/r.
$$

Here, $r$ controls the rank of the update and $\alpha$ is a scaling hyperparameter; $s=\alpha/r$ is the scaling used in the original LoRA formulation. For input dimension $d_{in}$ and output dimension $d_{out}$, the matrix shapes are:

| Object | Shape | Updated? |
| --- | --- | --- |
| $W_0$ | $d_{out}\times d_{in}$ | No |
| $A$ | $r\times d_{in}$ | Yes |
| $B$ | $d_{out}\times r$ | Yes |
| $BA$ | $d_{out}\times d_{in}$ | Computed from the two factors; need not be stored as a separate parameter |

The low-rank constraint applies to the **update** $BA$, not to the original $W_0$. Matrix $A$ projects the input into $r$ dimensions, and $B$ maps it back to output space, so the update has rank at most $r$. Training adjusts both mappings, but the rank still limits the updates they can represent. See [Section 4.1 of the original LoRA paper (2021)](https://arxiv.org/abs/2106.09685).

A 4096 × 4096 matrix has 16,777,216 parameters. At $r=8$, LoRA uses:

$$
8\times4096+4096\times8=65{,}536,
$$

which is **0.390625%** of the original matrix's parameter count. This ratio applies to one adapted matrix, not to total training memory.

## Follow one small input

Set $s=1$ and examine only the update branch:

$$
A=\begin{bmatrix}1&0&-1\end{bmatrix},\quad
B=\begin{bmatrix}2\\-1\end{bmatrix},\quad
x=\begin{bmatrix}3\\1\\1\end{bmatrix}.
$$

First $Ax=2$, then $B(Ax)=[4,-2]^\top$. Add this vector to the original output $W_0x$.

```text
x=[3,1,1] ─────────────→ Original W₀ ─────→ Original output
    │                                       + → New output
    └→ A: project to 1 dimension, 2 → B: [4,-2] ┘
```

The correction depends on the input; it is not a fixed offset added to every example. This rank-1 branch passes the correction through a single intermediate direction. Increasing the rank allows a wider range of updates, but does not guarantee better test performance. With limited data, it can also make noise easier to fit.

## Why not initialize both A and B to zero?

Let $\mathcal L$ be the loss and $g=\partial\mathcal L/\partial y$ its gradient with respect to the output. For one example, the two parameter gradients are:

$$
\frac{\partial\mathcal L}{\partial B}=s\,g(Ax)^\top,\qquad
\frac{\partial\mathcal L}{\partial A}=s\,B^\top g x^\top.
$$

With $A=B=0$, both gradients are zero, so the branch cannot start learning. A common choice is to initialize $A$ randomly and $B$ to zero. Initially, $BA=0$ leaves the original output unchanged. Matrix $B$ can receive a nonzero gradient on the first step; once it changes, $A$ can receive one too. This explains why the initialization works, not why it would be the only valid choice.

Freezing a layer's weights does not necessarily eliminate its backward computation. Training an earlier LoRA module may still require gradients to pass through intervening frozen layers. Freezing usually means neither updating those weights nor storing their parameter gradients. Wrapping the entire computation in `no_grad()` can cut a gradient path that training needs.

## Where does memory go?

Break memory use down by component rather than relying on the trainable parameter count:

| Component | What ordinary LoRA changes |
| --- | --- |
| Base weights | Still stored; ordinary LoRA does not quantize them |
| Base-weight parameter gradients | Usually unnecessary when frozen |
| Base-weight optimizer states | Usually unnecessary |
| LoRA weights, gradients, optimizer states | Required, but smaller |
| Activations needed for backward | Can remain large, depending on sequence length, batch size, and implementation |
| Workspaces and communication buffers | Depend on kernels and parallel configuration |

When activations dominate, reducing the number of examples processed in each forward/backward pass—the microbatch size—may help more than trimming the rank. Other options include shorter sequences, grouping similar-length sequences into batches, and activation checkpointing. Checkpointing saves memory by recomputing some intermediates during the backward pass. That costs additional computation, so measure time as well as memory.

## What does QLoRA save next?

QLoRA adds quantization to LoRA: it stores the frozen base weights at low bit width while training higher-precision LoRA parameters. Writing the quantized weights as $W_q$ and dequantization as $\operatorname{dequant}$ gives:

<!-- paper-figure:qlora-memory -->

$$
y=\operatorname{dequant}(W_q)x+sBAx.
$$

Not every training tensor becomes 4-bit. Storage, computation, gradients, and optimizer states can use different data types. Matrix operations typically use higher-precision approximations reconstructed from the quantized weights. The original [QLoRA paper (2023)](https://arxiv.org/abs/2305.14314) uses 4-bit NormalFloat (NF4), double quantization, and paged optimizers to address weight representation, quantization constants, and memory spikes, respectively.

For an idealized estimate, seven billion (7B) weights at 2 bytes each take about 14 GB. At exactly 4 bits each, the raw codes take about 3.5 GB, using decimal GB. **3.5 GB is not total training memory.** Quantization metadata, unquantized layers, LoRA parameters, activations, and workspaces still take up space. Smaller storage requirements do not guarantee faster training, either.

### What do NF4, double quantization, and paging each save?

These techniques often appear together, but save memory in different ways:

| Technique | What it stores | Caveat |
| --- | --- | --- |
| NF4 | Weights in nonuniform codes designed for a normal distribution | Unlike uniform 4-bit integer quantization; actual layer weights need not follow a perfectly normal distribution |
| Double quantization | Quantized block-scale constants | Those constants still need their own scales and metadata |
| Paged optimizer | Optimizer-state pages moved between host and GPU memory using unified memory | Can relieve memory spikes, but transfers cost time; this is not attention key/value-cache paging |

One 32-bit floating-point scale per 64 weights costs $32/64=0.5$ bits per weight. Quantizing those scales to 8 bits reduces the first-level overhead to $8/64=0.125$ bits per weight, before adding second-level scales and other metadata. A checkpoint with “4-bit weights” therefore does not occupy exactly four bits per parameter overall.

See [QLoRA Section 3](https://arxiv.org/html/2305.14314v1#S3). Whether training fits still depends on sequence length, batch size, and checkpointing.

## Can the adapter be merged for deployment?

For ordinary linear LoRA at inference time, with training dropout disabled, form $W_{merged}=W_0+sBA$ and run a conventional linear layer. In real arithmetic, this is equivalent to computing the two branches separately and adding their outputs; floating-point rounding can introduce numerical differences. Keeping LoRA parameters separate may be more convenient when switching tasks.

Quantization adds another source of error: dequantizing, adding the update, and requantizing can change the weight values again. Tools and backends may also support different merging procedures. Successful QLoRA training does not imply lossless loading into any 4-bit inference engine. Check outputs and memory on the deployment backend, and consult the [documentation for Hugging Face's parameter-efficient fine-tuning library, PEFT](https://huggingface.co/docs/peft/developer_guides/lora), for support in the version you use.

## Design an experiment you can interpret

Start with a few controlled comparisons rather than dozens of configurations. Fix the dataset, train/validation split, input/output template, positions included in the loss, and total training-token budget. When comparing the best tuned performance of different methods, give each the same tuning budget rather than forcing them to share a learning rate.

| Question | Change only | Record together |
| --- | --- | --- |
| Is rank too restrictive? | Rank, such as 8 → 16, with an explicit scaling rule | Validation task success, regressions on existing capabilities, memory |
| Which layers need adaptation? | Attention projections only versus adding feed-forward projections | Parameters, throughput, quality |
| What does quantization cost? | LoRA versus QLoRA with otherwise matched configuration | Quality, peak memory, effective tokens per second |
| Are activations the bottleneck? | Length or microbatch, one at a time | OOM boundary, time, actual token count |

For the first two comparisons, see [LoRA Without Regret (Schulman et al., Thinking Machines, 2025-09-29)](https://thinkingmachines.ai/blog/lora/). In its module-selection experiments with Llama-3.1-8B and Qwen3-30B-A3B-Base, attention-only LoRA underperformed adaptation of the feed-forward multilayer perceptrons (MLPs). Adding attention adaptation to MLP adaptation produced no clear further gain.

LoRA's best learning rate was roughly 10× that of full fine-tuning under $\alpha/r$ scaling, $\alpha=32$, and separate learning-rate searches—not a parameterization-independent ratio. Rank-1 LoRA matched full fine-tuning in policy-gradient reinforcement learning experiments rewarding correct mathematical answers, not arbitrary tasks.

Low-rank configurations also fell behind as data volume grew, which the authors attribute to limited capacity; these results should not be extrapolated to pretraining-scale data. On an OpenThoughts3 subset, larger batches hurt LoRA more than full fine-tuning, and increasing rank did not close the gap. Other setups need their own measurements.

Lower training loss does not automatically mean a better user experience. For the support-record task, check whether the format is valid, fields are correct, missing information leads to fabrication, and previously reliable tasks regress.

LoRA can make experiments cheaper, but it cannot repair bad labels. For a different source of supervision, read [what a student learns from a teacher](distillation.en.md). For guidance on choosing training examples, return to [SFT](sft-and-its-ceiling.en.md).
