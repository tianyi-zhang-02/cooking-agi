# Choosing a training stack: DDP, FSDP2, and the surrounding tools

[中文](training-frameworks.md) · **English**

> Checked 2026-10-16. FSDP2 mechanics were first written against PyTorch v2.8.0 source; in October 2026 they were re-checked against v2.14.1 (released 2026-09-30), and the grouping, DTensor sharding and `reshard_after_forward` behaviour are unchanged. No multi-GPU performance benchmark was run for this note.

Will switching to FSDP2 make training faster? First distinguish a model that fits on one GPU but trains too slowly from a model that does not fit at all. The first may benefit from processing more data in parallel. The second also needs a different way to store or partition state. Identify the bottleneck before choosing the stack.

Start with [data, tensor, and pipeline parallelism](distributed-training.en.md) if these are new. Here we look at what a framework takes over—and what remains your responsibility.

## The names belong to different layers {#layers}

Follow a small batch of sequences through one update: the data loader supplies inputs, model modules produce predictions, a loss function reduces predictions and targets to a scalar, autograd computes gradients, and the optimizer updates parameters using those gradients and its own state. The training loop connects these steps; the distributed strategy determines state placement and communication. Replacing one layer does not replace the whole workflow.

For why these interfaces are separate, see [CMU's *Neural Network Library Abstractions*, slides 14–16 and 19–20](https://dlsyscourse.org/slides/7-nn-framework.pdf#page=14), focusing on the relationships between modules, losses, optimizers, and data loading.

| Layer | Examples | Responsibility | What it does not decide |
| --- | --- | --- | --- |
| Model and objective | PyTorch, Transformers, custom losses | Forward computation, trainable parameters, autograd | Whether labels or objectives are appropriate |
| Training loop | Custom loop, Trainer / TRL | Batches, loss, updates, validation | Data selection and test separation |
| Distributed strategy | DDP, FSDP2, DeepSpeed ZeRO; tensor parallelism (TP) / pipeline parallelism (PP) at larger scales | State placement, computation, and communication | Whether extra GPUs are worthwhile or results correct |
| Launch and configuration | torchrun, Accelerate | Processes, devices, backend configuration | These are not model architectures |
| Post-training orchestration | NeMo RL, verl | Training, generation, rewards, weight synchronization | Reward validity and off-policy corrections |
| Generation engine | vLLM, SGLang | Efficient evaluation or rollout generation | The trainer's optimizer step |

A run can use TRL, FSDP2, and vLLM together: one for the loop, one for sharding, one for generation. Map the calls before choosing dependencies. See [SFT and framework choices](../practice/post-training/training-plan.en.md) for a project example.

## When DDP is enough {#ddp}

DDP means Distributed Data Parallel; FSDP means Fully Sharded Data Parallel. Both can process different data on each rank. One key distinction is whether model state remains fully replicated.

In a common DDP setup, each training process, identified by a rank, holds the full model, processes different data, and synchronizes gradients. If the model, optimizer state, and a useful microbatch fit on each device, this is a straightforward baseline to inspect.

Two GPUs processing four sequences each are not splitting one sequence in half. Accumulating eight microbatches gives $2\times4\times8=64$ sequences per update. Effective target tokens still depend on lengths, padding, and loss masks. Matching “64 sequences” alone does not match the objective.

FSDP primarily reduces replicated state. Gathering parameters introduces additional communication, so it is **not an unconditional upgrade in speed**. If CPU preprocessing, remote storage, or padding dominates, changing sharding strategy may do very little.

## FSDP2 gathers parameters when a block needs them {#fsdp-lifecycle}

How can a GPU compute a whole layer when it stores only a shard? Imagine four GPUs each holding a parameter shard. Before a Transformer block runs, an all-gather assembles that block's parameters. The GPUs can release the full copies afterward, gather again for backward, and use reduce-scatter to retain only their assigned portions of the reduced gradients.

<figure class="worked-update" lang="en">
<ol>
<li><small>01 / Forward</small><strong>All-gather parameters</strong><span>Ranks process their own batches, using the full parameters of the current block.</span></li>
<li><small>02 / Backward</small><strong>Gather again; reduce-scatter gradients</strong><span>If forward released the full parameters, backward gathers them again. Reduced gradient shards stay local.</span></li>
<li><small>03 / Update</small><strong>Update local shards</strong><span>The next gather uses updated parameters. This is still data parallelism, not a fixed set of layers assigned to each rank.</span></li>
</ol>
<figcaption>This path assumes reshard_after_forward=True. Retention, prefetching, and mixed precision change the memory and communication timeline.</figcaption>
</figure>

[PyTorch v2.14.1 fully_shard](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/distributed/fsdp/_fully_shard/_fully_shard.py) groups the parameters covered by each call for communication and represents shards with DTensor. A common approach applies it to blocks first, then to remaining root parameters. A root-only call puts many parameters in one group, limiting opportunities to reduce transient memory and overlap communication with computation.

These are **FSDP parameter communication groups**, not attention head groups or optimizer hyperparameter groups.

## Memory is not simply divided by GPU count {#memory}

Consider one billion parameters with FP32 persistent parameters and gradients, two FP32 Adam moments, and no separate master copy. These components total $10^9\times(4+4+8)=16$ GB, or an ideal 4 GB per rank over four ranks. GB here is decimal, not GiB.

Does that mean 4 GB of GPU memory is enough? No: peak memory also includes activations, temporary full parameters, communication buffers, kernel workspace, and allocator overhead:

$$M_{\text{peak}}=\max_t\left(M_{\text{persistent}}(t)+M_{\text{activation}}(t)+M_{\text{temporary}}(t)\right).$$

Apply this formula to the billion-parameter example, using assumed values throughout. The model has 20 equal blocks of 50 million parameters each. When backward reaches one block, each rank holds 4 GB of persistent shards. That block's full FP32 parameters have just been gathered, taking $5\times10^7\times4=0.2$ GB, and its full gradient takes another 0.2 GB before reduce-scatter.

If 3 GB of activations and 0.5 GB of workspace are also live, the total at that moment is $4+3+0.2+0.2+0.5=7.9$ GB—already nearly twice the 4 GB counted from persistent shards alone. $M_{\text{peak}}$ is the largest total across all moments.

Independent component peaks may occur at different times; summing them is not an exact measurement. Counting only persistent shards is also insufficient. Inspect which allocations coexist. Mixed precision requires a fresh accounting of parameter, gradient, and communication dtypes.

| Memory source | Possible intervention | Cost |
| --- | --- | --- |
| Replicated parameters, gradients, optimizer | FSDP / ZeRO | State gathering and communication scheduling |
| Saved forward activations | Activation checkpointing | Recomputation during backward |
| Padding | Length bucketing, appropriate packing | Preserve boundaries, masks, and objective weights |
| GPU-resident state | CPU offload | Host memory and transfer overhead |
| A compute group still too large | Smaller groups or model parallelism such as TP | More small messages or layer-level communication |

FSDP does not automatically divide long-sequence activations across GPUs. Adding data-parallel ranks does not shrink a fixed per-rank microbatch either.

Capacity and memory traffic are also separate concerns. [CMU's *GPU Acceleration*, slides 10–17](https://dlsyscourse.org/slides/12-gpu-acceleration.pdf#page=10), uses the memory hierarchy and tiled matrix multiplication to explain reuse within one GPU—a different level of optimization from sharding state across GPUs.

## Group size and parameter retention {#grouping}

Return to the 20-block example: gather the whole model at once, or gather each block when it is needed? A large group reduces message count but creates a larger transient gather. Many tiny groups offer finer control but can spend too much time on message latency. Transformer-block grouping is a starting point to test, not a universal optimum.

`reshard_after_forward=True` releases full parameters and pays for another gather in backward. `False` retains them longer to avoid that gather. **Root defaults can differ from inner-block defaults**, so the diagram states its assumption explicitly. [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

Prefetching can fetch the next block while the current one computes. It also keeps more full parameters alive. It needs memory headroom and cannot create bandwidth on a saturated network.

## Comparing DDP, FSDP, ZeRO, and Megatron {#choices}

| Situation | Path worth testing first | Additional checks |
| --- | --- | --- |
| A useful batch fits; scale data throughput | DDP | Loading, global token weighting, synchronization |
| Replicated state does not fit; retain a PyTorch model | FSDP2 | Grouping, dtypes, peaks, save and restore |
| An established DeepSpeed or offload setup exists | The relevant ZeRO stage | Actual sharding, versions, checkpoint compatibility |
| Layer compute or scale needs more partitioning | TP / PP and implementations such as Megatron | Model support, topology, communication, layouts |

[ZeRO](https://arxiv.org/abs/1910.02054) studies state sharding; [Megatron-LM](https://arxiv.org/abs/1909.08053) studies Transformer parallelism. Their concerns overlap, but they are not mutually exclusive entries in a speed ranking. A real stack may combine data, tensor, and pipeline parallelism.

## Check the objective before measuring speed {#verify}

Compare one update on fixed small inputs. Align sample order, valid tokens, masks, gradient normalization, clipping, and optimizer settings. Reduction order can cause numerical differences; declare tolerances rather than promising bitwise equality.

Then measure effective target tokens/s after warm-up, step-time distributions, peak memory, and communication waits. High GPU utilization can include recomputation, padding, and communication; why MFU and HFU count different numerators is covered in [Performance: MFU and HFU](../12-infra/performance.en.md#mfu). Ask how long it takes to complete a reliable experiment at the required quality, not just how busy the devices look.

Finally, interrupt a run deliberately. Loading weights is not full training recovery: inspect optimizer, scheduler, RNG, data cursor, and update count. Resharding a checkpoint to another world size depends on the format and loader and needs testing. Successful loading does not guarantee the same training trajectory. Continue with [saving and resuming](../practice/post-training/checkpoint-and-resume.en.md).

For the surrounding product, see [a complete ML system](../practice/ml-system/README.en.md). For generation after training, see [vLLM and SGLang](inference-frameworks.en.md).
