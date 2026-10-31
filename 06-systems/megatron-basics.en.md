# From one update to Megatron: distributed training basics

[中文](megatron-basics.md) · **English**

Begin with one process on one GPU. Read tokens, predict outputs, compute a loss, differentiate it, then let the optimizer update parameters. A central problem addressed by Megatron is: **when this becomes too large or too slow for one GPU, how can multiple GPUs cooperate?**

We will calculate one update, then divide the work across GPUs. The memory figures, rank layouts and batch sizes are worked examples, not recommended training configurations or measured results. Sources were checked on 2026-10-30.

## What happens in one parameter update? {#one-update}

### Work through one parameter first

Set the Transformer aside for a moment. Let the model have one learnable number $w$, input $x=3$, and prediction rule $\hat y=wx$. The target is $y=10$ and the current parameter is $w=2$.

The **forward pass** produces $\hat y=2\times3=6$. Choose half the squared error as the loss:

$$
L=\frac12(\hat y-y)^2=\frac12(6-10)^2=8.
$$

The **backward pass** asks how the loss changes as the parameter changes. The chain rule gives:

$$
\frac{\partial L}{\partial w}=(wx-y)x=(6-10)\times3=-12.
$$

The negative gradient tells us that a small increase in $w$ reduces the loss here. An **optimizer step** using ordinary gradient descent with learning rate 0.1 gives:

$$
w_{\rm new}=w-0.1\frac{\partial L}{\partial w}=2-0.1\times(-12)=3.2.
$$

The new prediction is 9.6 and the loss is 0.08. This is a worked teaching example. Large models use many parameters and more elaborate losses and optimizers, while retaining the sequence: predict, measure error, differentiate, update.

### Put the terms back into that example

| Term | Meaning | Common mistake |
| --- | --- | --- |
| Tensor | A multidimensional array with shape, dtype and device | Equal shapes do not imply equal meaning |
| Parameter | A tensor learned through updates, such as a linear-layer weight | Checkpoint size is not peak training memory |
| Forward | Apply parameters to produce logits or other outputs | Training may retain intermediates for backward |
| Loss | Turn predictions and targets into an optimization signal | Padding and response tokens need appropriate treatment |
| Backward | Differentiate through the graph to obtain parameter gradients | Parameters have not automatically changed yet |
| Optimizer step | Modify parameters using gradients and historical state | Adam and similar optimizers retain additional state |

Here $w$ is the parameter and $-12$ its gradient for this sample. Computing the gradient does not change the parameter; the update rule still has to run. Optimizers such as Adam also use statistics of past gradients, which is why resuming training commonly requires optimizer state.

In PyTorch, `loss.backward()` accumulates gradients through the graph; `optimizer.step()` applies the optimizer's update rule. The training loop decides when to clear gradients. Use the [official autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) to translate the hand calculation into code.

### From one number to language-model tensors

A language model commonly maps token IDs `[B,S]` to hidden states `[B,S,H]`, then vocabulary logits `[B,S,V]`. These are logical shapes: packing, fused losses and sharding may avoid materializing the full logits tensor.

B is the number of sequences in the batch, S the sequence length, H each position's hidden-vector dimension, and V the vocabulary size. Logits are unnormalized scores for candidate tokens; softmax converts them into the corresponding probabilities.

Take B=2, S=1024, H=4096 and V=128000, with two-byte BF16 elements and four-byte FP32 elements. Multiply the element count by bytes per element, then divide by `2^20` to express the result in MiB:

| Tensor | Byte calculation | Size |
| --- | --- | --- |
| BF16 hidden states `[B,S,H]` | `2 × 1024 × 4096 × 2` | 16 MiB |
| BF16 full logits `[B,S,V]` | `2 × 1024 × 128000 × 2` | 500 MiB |
| FP32 full logits `[B,S,V]` | `2 × 1024 × 128000 × 4` | 1000 MiB |

Replacing H with the much larger V takes the BF16 tensor from 16 MiB to 500 MiB. That is still the size of just one tensor; we also need to count the other state held in memory at the same time.

### What else occupies device memory?

| State | Why it is needed | Memory question |
| --- | --- | --- |
| Parameters | Execute the model | Full replicas or local shards? |
| Activations | Retain some forward intermediates for backward | Which are retained, and which recomputed? |
| Gradients | Supply the optimizer's update direction | What are the dtype, accumulation and sharding rules? |
| Optimizer state | Retain history required by the update rule | Gradient moments, high-precision master weights, or state offloaded to the CPU or elsewhere? |

Add temporary tensors, communication buffers, operator workspaces and allocator reservations. **A weights file accounts for only part of this total.** Inference generally omits this backward/optimizer chain; autoregressive generation instead retains runtime state such as the KV cache, so it needs its own memory estimate.

Keep asking the same question as we introduce more GPUs: are we dividing data, computation, or one of these kinds of state?

## A GPU and a rank are different concepts {#processes}

A node here is a machine, a GPU is a device, a process is a running program, and a rank identifies a process within a distributed process group. One process per GPU is common, but is a deployment choice. The same process can belong to TP, PP and DP groups and have a corresponding rank in each group. Rank 0 is not inherently the cluster's first physical GPU.

Identify which process owns which parameters and processes which data before choosing parallelism. For rank and data-layout conventions in communication primitives, see [NCCL's collective documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).

## Different strategies split different things {#parallelism}

### If the model fits, start with data parallelism

Data parallelism (DP) runs model replicas on different data. Two GPUs each hold a model; one reads sample A and the other sample B. They synchronize corresponding gradients for updates so the replicas remain consistent. PyTorch's DistributedDataParallel (DDP) is a common implementation.

Return to the one-parameter model. Both replicas start at $w=2$ and use the same squared loss, but each reads a different sample:

| Replica | Sample | Prediction | Local gradient |
| --- | --- | --- | --- |
| A | $x=3,\ y=10$ | $wx=6$ | $(6-10)\times3=-12$ |
| B | $x=2,\ y=2$ | $wx=4$ | $(4-2)\times2=4$ |

For the mean loss over the two samples, both replicas update with the mean gradient and learning rate 0.1:

$$
\begin{aligned}
g&=\frac{-12+4}{2}=-4,\\
w_{\rm new}&=2-0.1\times(-4)=2.4.
\end{aligned}
$$

Local gradients would move A to 3.2 and B to 1.6. Synchronizing gives both 2.4, matching a single-GPU update on the mean loss of these equally weighted samples. “Sum, then divide by 2” explains normalization; DDP's default reduction already averages gradients, so do not divide again unconditionally. See the [DDP documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

This increases the amount of data processed concurrently, but each GPU still needs room for its model state. If parameters and optimizer state are too large, consider sharding with FSDP or ZeRO. FSDP2 can distribute parameters and other state across ranks, gathering a parameter group when computation needs it. See the lifecycle in [training frameworks](training-frameworks.en.md#fsdp-lifecycle).

### Splitting a layer differs from splitting a stack of layers

Tensor parallelism (TP) divides computation within a layer. Two GPUs might compute different output columns of a linear layer, then communicate as required by subsequent computation. They cooperate on the same model computation.

Pipeline parallelism (PP) places layers in different stages. Imagine four Transformer blocks split into two stages: stage 0 computes the first two and transfers intermediate activations to stage 1 for the remaining two. Backward sends gradients in the opposite direction. Another problem is how to schedule microbatches so that multiple stages have work to do.

### Compare the common partitions

| Strategy | What is divided | Work and tradeoff |
| --- | --- | --- |
| DP / DDP | Data | Model replicas process different data and synchronize gradients; model state is replicated |
| FSDP / ZeRO | Some or all training state | Shard parameters, gradients and optimizer state according to the strategy; communicate required state for execution |
| TP | Computation inside a layer | For example, compute different linear-layer columns, then combine or reduce as required |
| PP | Layers or stages | Transfer activations forward and corresponding gradients backward; fill and drain the pipeline |
| CP | A long sequence's context | Own different sequence portions while exchanging information needed by attention |
| EP | MoE experts | Dispatch and return routed tokens, and balance expert load |

After the first four rows, consider context parallelism (CP) and expert parallelism (EP). CP targets long sequences whose partitions still cooperate on attention. EP routes tokens to the relevant experts in a mixture-of-experts (MoE) model. Sequence parallelism within TP has its own activation partitioning rules; continue with [tensor parallelism](tensor-parallel.en.md).

TP and FSDP can both use all-gather while partitioning different things. FSDP typically gathers parameters to compute on local data; TP divides a layer's computation.

Continue with the [Megatron parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html), [CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html), [MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html) and [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html). Supported combinations depend on the version; this table is not a compatibility matrix.

## Draw actual groups for eight GPUs {#eight-gpus}

Consider a dense model with TP=2, PP=2, DP=2, without CP/EP: $2\times2\times2=8$ ranks. The numbering below is chosen for teaching, not Megatron's default rank order:

```text
Data replica A: stage 0 [0,1]  →  stage 1 [2,3]
Data replica B: stage 0 [4,5]  →  stage 1 [6,7]

TP: two ranks inside each bracket cooperate on a layer
PP: 0↔2, 1↔3, and 4↔6, 5↔7 transfer stage data
DP: 0↔4, 1↔5, 2↔6, 3↔7 synchronize corresponding gradients
```

Read replica A from left to right: ranks 0 and 1 cooperate on its first stage, and 2 and 3 on its second. Together these four ranks execute one model replica. Replica B uses four more ranks for a different batch. The diagram uses global ranks 0–7 to identify each process consistently; its rank within a subgroup may differ.

Then read vertically: ranks 0 and 4 own corresponding model portions in different data replicas, so they synchronize corresponding gradients. Ranks 0 and 2 belong to successive stages of one replica and exchange stage data. **The links between eight GPUs do not all carry the same kind of tensor.**

### How many sequences does one update process?

Each DP replica receives two sequences per microbatch and accumulates four microbatches per update. Label the inputs A1–A8 and B1–B8, then trace them through the diagram:

| Progress | Replica A: `[0,1] → [2,3]` | Replica B: `[4,5] → [6,7]` |
| --- | --- | --- |
| First microbatch | Process A1 and A2; pass activations to the second stage | Process B1 and B2; pass activations to the second stage |
| Backward for that microbatch | Activation gradients return from `[2,3]` to `[0,1]`; each stage computes parameter gradients | Activation gradients return from `[6,7]` to `[4,5]`; each stage computes parameter gradients |
| Four microbatches accumulated | A1–A8: 8 sequences | B1–B8: 8 sequences |
| This update | Synchronize gradients with B's corresponding shards, normalize and update | Synchronize gradients with A's corresponding shards and update by the same rule |

The global batch is therefore $2\times2\times4=16$ sequences: sequences per microbatch × DP replicas × accumulated microbatches. TP and PP cooperate on these sequences; multiplying by them again to obtain 64 is incorrect. Variable lengths mean 16 sequences still do not specify the effective token count or loss weighting. This traces data and gradient flow; communication can overlap backward rather than waiting for the final row.

Pipeline microbatches let stage 0 work on later data while stage 1 handles earlier data. For example, stage 0 can process A3 and A4 while stage 1 handles A1 and A2. Too few microbatches can leave substantial idle time; too many affect memory and scheduling costs. PP=2 does not imply a 2× speedup. Nor should this simple product be extended blindly to every MoE/EP topology.

## Megatron-LM, Core and Bridge {#megatron-family}

| Name | Boundary to understand |
| --- | --- |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | A large-scale Transformer training repository, including training entry points and implementations |
| [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html) | Reusable models, parallelism, scheduling, distributed optimizers and other training components |
| [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/) | Core-based training and model adaptation tools, including Hugging Face conversion |
| [NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html) | Post-training orchestration for rollout, rewards and training, with a Megatron training path |
| vLLM / SGLang | Here, focus on their rollout-generation role and how they receive training weights |

In the eight-GPU diagram, Core supplies parallel-training components, Megatron-LM organizes training, and cluster tools allocate devices. Core itself is not a new language model; Megatron-LM does not take over Kubernetes' responsibilities. Training and inference partitions can differ, so transferring weights may require resharding, conversion and synchronization beyond copying a Python object. The [Bridge conversion guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/bridge-guide.html) explains this adaptation layer.

Also distinguish Megatron's distributed optimizer from full parameter sharding. It addresses optimizer-state partitioning and update communication across DP ranks; resident parameter and gradient layouts depend on configuration. Its name does not make it equivalent to PyTorch FSDP2. See the [official explanation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html).

## The first exercise needs no eight-GPU rental {#exercise}

Redraw the groups, labeling arrows as activations, gradients or parameters. Explain why four ranks processing A1 and A2 still count as two sequences. Inspect a tiny model's shapes, dtypes, devices and state around forward, backward and step: when do gradients appear, and when do parameters change? Then validate normalization and synchronization on a small multi-GPU run. Record unexecuted configurations as reading or planned experiments.

Continue: [training frameworks and FSDP2](training-frameworks.en.md), [tensor parallelism](tensor-parallel.en.md), [GPU and cluster scheduling](execution-boundaries.en.md#scheduling), then [RL systems](rl-systems.en.md).
