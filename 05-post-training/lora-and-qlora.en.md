# LoRA and QLoRA: fewer parameters, but still out of memory?

[中文](lora-and-qlora.md) · **English**

> Last reviewed: 2026-10 · Prerequisites: [One training step](../00-foundations/deep-dives/training-step.en.md), [Model adaptation](model-adaptation.en.md)

You want a model to organize support records into a fixed format. The examples and SFT loss are ready, but full fine-tuning does not fit in memory. You switch to LoRA and train far fewer parameters. Then you increase sequence length—and still hit OOM.

There is no contradiction. LoRA mainly reduces the weights being updated and their associated gradients and optimizer states. Intermediate results needed during forward and backward computation do not all disappear. Start with one linear layer.

## SFT and LoRA answer different questions

SFT means training on demonstrated responses; LoRA specifies which parameters may change. Compare **full-parameter SFT with LoRA-SFT**, not “LoRA versus SFT.” LoRA can also be used with other objectives.

If the only issue is occasional invalid formatting, consider validation or constrained decoding first. If facts change frequently, consider retrieval. Choosing a parameter-update method comes after deciding that model behavior needs training.

## Replace one large update with two small matrices

A linear layer originally computes $y=W_0x$. LoRA freezes $W_0$ and learns a small additional branch:

$$
y=W_0x+sBAx,\qquad s=\alpha/r.
$$

For input and output dimensions $d_{in},d_{out}$:

| Object | Shape | Updated? |
| --- | --- | --- |
| $W_0$ | $d_{out}\times d_{in}$ | No |
| $A$ | $r\times d_{in}$ | Yes |
| $B$ | $d_{out}\times r$ | Yes |
| $BA$ | $d_{out}\times d_{in}$ | Composed from the factors, not necessarily stored as a separate parameter |

The low-rank constraint applies to the **update** $BA$, not to the original $W_0$. A projects the input into $r$ dimensions and B maps it back to output space. Training adjusts this branch rather than freely changing every direction in the layer. This is the parameterization introduced by [LoRA](https://arxiv.org/abs/2106.09685).

A 4096 × 4096 matrix has 16,777,216 parameters. At $r=8$, LoRA uses:

$$
8\times4096+4096\times8=65{,}536,
$$

or **0.390625%** of the original matrix's parameter count. That is a ratio for one targeted matrix, not for total training memory.

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

The correction changes with the input; it is not a fixed offset for every example. This rank-1 branch has one intermediate direction. Increasing rank relaxes the constraint but does not guarantee better test performance: it can also make fitting noise easier when data are limited.

## Why not initialize both A and B to zero?

Let $g=\partial\mathcal L/\partial y$. For one example:

$$
\frac{\partial\mathcal L}{\partial B}=s\,g(Ax)^\top,\qquad
\frac{\partial\mathcal L}{\partial A}=s\,B^\top g x^\top.
$$

With $A=B=0$, both gradients are zero, so this branch cannot start learning. A common initialization makes A random and B zero: $BA=0$ initially preserves the original behavior, while B can receive a gradient at the first step and A can update later. Other initializations make different choices; this is not the only valid scheme.

“Frozen” also does not mean “no backward computation.” Training an earlier LoRA module may still require gradients to pass through intervening frozen layers. Freezing usually means not updating those weights or storing their parameter gradients—not wrapping the entire computation in `no_grad()`.

## Where does memory go?

Use this accounting rather than trainable parameter count alone:

| Component | What ordinary LoRA changes |
| --- | --- |
| Base weights | Still stored; ordinary LoRA does not quantize them |
| Base-weight parameter gradients | Usually unnecessary when frozen |
| Base-weight optimizer states | Usually unnecessary |
| LoRA weights, gradients, optimizer states | Required, but smaller |
| Activations needed for backward | Can remain large, depending on length, batch, and implementation |
| Workspaces and communication buffers | Depend on kernels and parallel configuration |

When activations dominate, reducing the microbatch, shortening or bucketing sequences, or using activation checkpointing may help more than trimming rank. Checkpointing trades recomputation for storage, so measure time as well as memory.

## What does QLoRA save next?

QLoRA stores the frozen base in low precision while training higher-precision LoRA parameters. Schematically:

$$
y=\operatorname{dequant}(W_q)x+sBAx.
$$

This does not make every training tensor 4-bit. Storage, computation, gradients, and optimizer states can use different dtypes; matrix operations typically use reconstructed weight values at higher precision. The original [QLoRA paper](https://arxiv.org/abs/2305.14314) introduces NF4, double quantization, and paged optimizers to address weight representation, quantization constants, and memory spikes.

For an idealized estimate, 7B weights at 2 bytes each take about 14 GB; exactly 4 bits each would take about 3.5 GB for raw codes. **3.5 GB is not total training memory.** Quantization metadata, unquantized layers, adapters, activations, and workspaces remain. Lower storage alone also does not establish faster training.

### What do NF4, double quantization, and paging each save?

They address different storage costs:

| Technique | What it stores | Caveat |
| --- | --- | --- |
| NF4 | Weights in nonuniform codes designed for a normal distribution | Not uniform INT4; actual layers need not be exactly normal |
| Double quantization | Quantized block-scale constants | Those constants still need their own scales and metadata |
| Paged optimizer | Optimizer-state pages migrated between CPU and GPU using unified memory | Migration costs time; this is not KV-cache paging |

One FP32 scale per 64 weights costs $32/64=0.5$ bit/weight. Quantizing those scales to 8 bits reduces that first-level cost to $8/64=0.125$, before second-level metadata. Thus “4-bit weights” does not mean an entire checkpoint occupies exactly four bits per parameter.

See [QLoRA Section 3](https://arxiv.org/html/2305.14314v1#S3). Whether training fits still depends on sequence length, batch size, and checkpointing.

## Can the adapter be merged for deployment?

For ordinary linear LoRA, form $W_{merged}=W_0+sBA$ and run a conventional linear layer. This equals the two-branch computation in real arithmetic, with small floating-point differences possible. Keeping adapters separate may be more convenient when switching tasks.

Quantization complicates merging: dequantizing, adding the update, and requantizing can introduce new error. Tools and backends may support different merge paths. “QLoRA training works” does not imply lossless loading in any 4-bit engine. Check outputs and memory on the actual deployment backend; consult [PEFT documentation](https://huggingface.co/docs/peft/developer_guides/lora) for version-specific support.

## Design an experiment you can interpret

Do not start with dozens of configurations. Fix the dataset, split, template, loss mask, and training-token budget, then compare:

| Question | Change only | Record together |
| --- | --- | --- |
| Is rank too restrictive? | Rank, such as 8 → 16, with an explicit scaling rule | Validation task success, regressions, memory |
| Which layers need adaptation? | Attention projections only versus adding MLP projections | Parameters, throughput, quality |
| What does quantization cost? | LoRA versus QLoRA with otherwise matched configuration | Quality, peak memory, effective tokens per second |
| Are activations the bottleneck? | Length or microbatch, one at a time | OOM boundary, time, actual token count |

Lower training loss is not automatically better user experience. For the support task, check valid formatting, correct fields, fabrication when information is missing, and regressions on tasks the model already handled.

LoRA can make experiments cheaper; it cannot repair bad labels by itself. To change the source of supervision, read [what a student learns from a teacher](distillation.en.md). To select demonstrations, return to [SFT](sft-and-its-ceiling.en.md).
