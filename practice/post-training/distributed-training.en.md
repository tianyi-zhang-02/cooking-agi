# Choose multi-GPU training around the bottleneck

[中文](distributed-training.md) · **English**

> Original teaching project · Checked: 2026-10-09. Memory and batch figures are worked examples, not GPU benchmarks. No multi-node training was executed.

A run that starts on one GPU may still be impractical, while a single-GPU OOM does not automatically call for tensor parallelism. Separate **capacity** from **time to completion** before selecting a strategy.

## Measure a representative batch first

Do not profile only an unusually short sample. Cover short, typical, and long sequences from the actual distribution. After warmup, measure forward, backward, optimizer updates, and saving. Optimizer state is often allocated at the first `step`, so post-load memory is not the training peak.

| Observation | First inspection | Potential intervention |
| --- | --- | --- |
| OOM during model loading | Weights, initialization copies, placement | Quantization, sharded initialization, or a smaller model |
| OOM during long-sequence backward | Activations, attention implementation, temporary buffers | Smaller microbatches, activation checkpointing, defensible length budgets |
| OOM at the first optimizer step | Gradients, optimizer state, master weights | LoRA, state sharding, or offload when justified |
| Periodic GPU idle time | Data reads, CPU tokenization, synchronization, checkpoint I/O | Cache preprocessing and inspect profiles before adding GPUs |

Activation checkpointing trades retained activations for backward recomputation. A disk checkpoint saves training state for recovery. Similar names, different problems.

## Estimate a lower bound, then measure the peak

A simplified full-parameter Adam budget might use 2 bytes for BF16 weights, 2 for BF16 gradients, 4 for FP32 master weights, and 8 for two FP32 moments: approximately 16 bytes per parameter. These are explicit assumptions; implementations can retain FP32 parameters or gradients, or avoid a separate master copy.

For 0.6B parameters, those states alone occupy $0.6\times10^9\times16=9.6\times10^9$ bytes, about 8.94 GiB. **Activations, logits, communication buffers, allocator reservations, and fragmentation are excluded.** LoRA has a different trainable-state budget; this full-fine-tuning estimate does not transfer unchanged.

Parameter count alone cannot explain why a small model uses many GPUs. Long contexts, throughput goals, and iteration deadlines matter—but do not automatically justify the allocation. Compare time, peak memory, and cost for the same task.

## Compare what each strategy partitions

| Strategy | Partitioned work or state | When to consider it | Main cost |
| --- | --- | --- | --- |
| DDP | Data; each GPU retains a complete training replica | The model fits and more data throughput is useful | Replicated state and gradient synchronization |
| FSDP2 | Parameters, gradients, optimizer state | State capacity is limiting and layerwise gathering is feasible | Parameter all-gathers and gradient reduce-scatters; peak memory is not simply total state divided by GPU count |
| ZeRO 1 / 2 / 3 | Progressively optimizer, gradient, and parameter state | Increasing memory savings within a compatible training stack | Deeper sharding generally adds state exchange; offload adds transfer costs |
| TP | Tensor computation within a layer | Large layers and sufficient interconnect bandwidth | Frequent intra-layer communication |
| PP | Layers assigned to pipeline stages | Layer partitioning with suitable microbatch scheduling | Pipeline bubbles, stage imbalance, activation transfers |

See [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) for sharding behavior. Old FSDP1 configurations are not drop-in FSDP2 APIs. Strategies can be combined, but establish why one dimension is needed before adding another.

For this small LoRA project, establish a single-GPU baseline if it fits, then measure DDP when throughput matters. Compare FSDP/ZeRO if full-fine-tuning state does not fit. If long-sequence activations dominate, investigate lengths, attention, and recomputation first; parameter sharding alone may offer limited relief.

## Check whether scaling changed the batch or objective

Without packing and with fixed example counts:

$$
B_{\mathrm{update}}=B_{\mathrm{micro}}\times A\times D.
$$

$B_{\mathrm{micro}}$ is examples per data-parallel replica per microstep, $A$ is accumulation steps, and $D$ is data-parallel degree. Eight GPUs with TP=2 and DP=4, microbatch=2, and accumulation=8 process $2\times8\times4=64$ examples per update—not 128. This example excludes PP, duplicated samples, and incomplete final batches.

Equal example counts are not enough. Suppose one DP rank has 1 valid target with loss sum 0.2, while another has 3 with sum 3. Averaging local means gives $(0.2+1)/2=0.6$; the global token mean is $3.2/4=0.8$.

If the backend averages gradients over $D$ ranks, an update contains $N$ valid targets, and local loss sums are $S_{r,a}$, backpropagating $D S_{r,a}/N$ for each accumulation microstep produces the global token-mean gradient after rank averaging and microstep summation.

**This derivation assumes no other automatic scaling.** Do not duplicate a trainer's token normalization, accumulation division, or distributed scaling. $N$ must cover the entire accumulation window, including a recomputed count for a partial final window. Compare gradients against one concatenated single-process batch; similar displayed losses are not sufficient evidence.

## Measure useful work, not only utilization

Assume identical data and valid-target totals. A single GPU processes 1,000 valid target tokens/s; four GPUs process 2,800. Speedup is 2.8 and scaling efficiency is $2.8/4=70\%$, while GPU-hours rise by roughly $4/2.8=1.43$. These are teaching assumptions, not a hardware recommendation.

Report both non-pad input tokens/s and valid target tokens/s, plus step time, peak memory, and saving pauses. Prompt tokens still require computation even when excluded from SFT loss. Changing prompt/answer proportions changes the relationship between throughput measures; dropping difficult examples is not a fair efficiency gain.

| Priority | Comparison |
| --- | --- |
| Get an answer sooner | End-to-end time with fixed task and validation frequency |
| Control cost | GPU-hours and storage/I/O at comparable quality |
| Avoid OOM | Length tails, first optimizer step, and saving peaks |
| Preserve training semantics | Sample IDs, target denominators, and a small gradient comparison |

The project's [standard-library checks](code/training_contracts.py) recompute batch and padding quantities; tests also verify unequal-length normalization using analytic gradients. They are not NCCL/FSDP benchmarks. Next: [checkpointing and recovery](checkpoint-and-resume.en.md). A distributed run that starts is not necessarily one that can resume.
