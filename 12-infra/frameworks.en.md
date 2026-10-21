# The framework layer: from one line of PyTorch to kernels on the GPU

[中文](frameworks.md) · **English**

> Reading time: ~13 min · Last reviewed: 2026-10

You write `y = torch.relu(x @ w + b)`. Python returns at once while the GPU may still be working; wrap it in `torch.compile` and the same line gets noticeably faster. What sits in between? **In eager mode this line becomes 3 separate ATen operators, at least 3 kernels and two memory round-trips that could have been avoided; a compiler mostly saves per-operator CPU overhead and the reads and writes of intermediate results; it does not remove a single multiplication.**

For what kernels, streams and SMs are, see [CUDA](cuda.en.md) and [GPU architecture](gpu-architecture.en.md); for how to measure the bottleneck with a profiler, see [Performance](performance.en.md). How to choose training and inference frameworks is already covered in [Training frameworks](../06-systems/training-frameworks.en.md) and [vLLM and SGLang](../06-systems/inference-frameworks.en.md); this note only puts them back on one map. Source links are pinned to PyTorch v2.14.1 (released 2026-09-30), documentation is the 2.14 version, checked on 2026-10-15.

## Fix the example first

```python
import torch

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
b = torch.zeros(4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)

y = torch.relu(x @ w + b)
```

Think of $x$ as activations for 4096 tokens of width 4096, $w$ as one layer's weight and $b$ as its bias. All three tensors live on the same GPU in BF16, 2 bytes per element. $w$ and $b$ require gradients, so while this line computes the forward pass it is also recording for the backward pass. The shapes are a teaching setup, not any particular model.

## The stops this line passes through

| Stop | What happens | Source |
| --- | --- | --- |
| Python | `@` calls `torch.matmul`, `+` calls `torch.add`, then `torch.relu`; each goes through the Python bindings into C++ | Operator overloading |
| Dispatcher | Looks at the dispatch keys carried by the input tensors and at thread-local state, and picks the highest-priority key: the Autograd layer runs first, then hands off to the CUDA implementation | [Dispatcher tutorial](https://docs.pytorch.org/tutorials/advanced/dispatcher.html) |
| ATen operators | `matmul` is a composite operator and calls `mm` directly for two 2-D inputs; `add.Tensor` and `relu` are one operator each | [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/native_functions.yaml#L3781), [`_matmul_impl`](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/LinearAlgebra.cpp#L2024) |
| CUDA implementation | The CUDA version of `mm` goes to `addmm_out_cuda_impl`, which calls cuBLASLt or cuBLAS depending on conditions; `add` and `relu` each launch one elementwise kernel | [cuda/Blas.cpp](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/Blas.cpp#L785) |
| Memory | Each of the three outputs ($xw$, the result after adding the bias, and $y$) takes a block from PyTorch's caching allocator | [CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html) |
| Stream | These kernels are queued in order on the current stream; Python returns without waiting for the GPU | Same |
| Autograd | Records `ReluBackward0 → AddBackward0 → MmBackward0` and the tensors the backward pass will need | [Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html) |

**The dispatcher splits by layer first, then by device.** The official [dispatcher tutorial](https://docs.pytorch.org/tutorials/advanced/dispatcher.html) (marked deprecated as a custom-operator guide since PyTorch 2.4, though its description of the dispatcher is still a useful reference) says an operator is split into several kernels by dispatch key, such as CPU, CUDA and Autograd; a call runs the Autograd layer first and then redispatches to the backend kernel for the inputs' device.

**`matmul` has no backward formula of its own.** It is registered in `native_functions.yaml` as `CompositeImplicitAutograd`, meaning it is assembled from other operators, which also provide its gradients. Two 2-D inputs take the `mm` path, so the backward graph shows `MmBackward0` rather than a matmul node. For a 3-D input such as `[batch, seq, hidden]`, `matmul` may fold the first two dimensions into one and then call `mm`; the exact conditions are in `_matmul_impl` linked above.

**Matrix multiply goes to a library; each elementwise op gets its own kernel.** The CUDA implementation of `mm` calls `addmm_out_cuda_impl` with $\beta=0$; when conditions allow it takes the cuBLASLt path, otherwise cuBLAS GEMM, and both are issued on the current stream returned by `getCurrentCUDAStream()`. Which GEMM kernel is finally chosen depends on the GPU model, dtype, shape and library version, so no kernel name is given here.

**Write it differently and the operator count changes.** If you write `torch.nn.functional.linear(x, w.T, b)`, the v2.14.1 [`at::linear`](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/Linear.cpp#L85) calls `addmm(bias, input, weight.t())` directly for 2-D inputs, with a source comment saying the fused op is "marginally faster". On CUDA, when conditions are met, `addmm` passes the bias pointer to the cuBLASLt epilogue, so the bias is added as the GEMM writes its result; PyTorch also has an internal operator, `_addmm_activation`, that can put ReLU or GELU into the epilogue as well. Note that an `nn.Linear` weight has shape `[out, in]`, the transpose of `w` here.

**Python returning does not mean the GPU has finished.** The 2.14 CUDA semantics note states that GPU operations are asynchronous by default: a call only queues work on the device. Each device has its own default stream, and operations within one stream run in the order they were submitted. `.item()`, `print(y)` and `.cpu()` all need the result, so they make the CPU stop and wait; timing must synchronize first or use CUDA events. For data transfers and multiple streams, see [the GPU data flow in the training loop](../00-foundations/pytorch/training-loop.en.md#gpu-data-flow).

**Autograd keeps only what the backward pass actually needs.** The autograd mechanics note says the forward pass records a directed acyclic graph whose leaves are the inputs and whose roots are the outputs, `grad_fn` is the entry into that graph, and the graph is rebuilt every iteration. According to the v2.14.1 [derivatives.yaml](https://github.com/pytorch/pytorch/blob/v2.14.1/tools/autograd/derivatives.yaml#L2131), ReLU's backward uses its own output (the formula reads `threshold_backward(grad, result, 0)`), the gradient of `mm` with respect to $w$ needs $x$, and the backward of addition needs no input values. So the two intermediates $xw$ and $xw+b$ can go back to the allocator once used; what stays is $y$ itself plus $x$, which was there anyway. For the rules in detail, see [Autograd](../00-foundations/pytorch/autograd.en.md).

The snippet below checks the operator sequence and the backward graph on a CPU:

```python
import torch
from torch.profiler import profile, ProfilerActivity

x = torch.randn(4, 3)
w = torch.randn(3, 2, requires_grad=True)
b = torch.randn(2, requires_grad=True)

with profile(activities=[ProfilerActivity.CPU]) as prof:
    y = torch.relu(x @ w + b)

print([e.name for e in prof.events() if e.name.startswith("aten::")])
print(y.grad_fn.name(), [f.name() for f, _ in y.grad_fn.next_functions])
```

On a local CPU with PyTorch 2.8.0, the first line lists `aten::matmul`, `aten::mm`, `aten::add` and `aten::relu`, interleaved with helper calls such as `aten::resolve_conj` and `aten::clamp_min`; the second line is `ReluBackward0 ['AddBackward0']`. That is an older version than the 2.14.1 source cited above. On a GPU you need to add `ProfilerActivity.CUDA` to see the kernels, whose names vary with the GPU and library versions.

## Where eager mode leaves performance on the table

The first leak is **fixed CPU-side cost**. Every operator goes through a Python call, the dispatcher, argument checks, output allocation and kernel submission. With large tensors, GPU compute hides this time; with small ones, the GPU finishes an operator in a few microseconds and the CPU cannot keep up. [PyTorch's CUDA Graphs blog post](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) (by NVIDIA and Facebook engineers, written around PyTorch 1.10) gives Mask R-CNN from MLPerf as an example: at a per-GPU batch size of 1, the CPU was fully loaded while the GPU sat idle for long stretches between kernels.

The second leak is **memory round-trips for intermediate results**. The official [performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) puts it this way: pointwise operations are usually memory-bound, and eager mode launches a separate kernel for each one, each of which reads from memory, computes and writes back.

### Do the memory arithmetic

Count only the elementwise part after the GEMM output. Let $M=N=4096$ and $s=2$ bytes; one intermediate is $MNs=33{,}554{,}432$ bytes, or 32 MiB. In eager mode `add` reads once and writes once, then `relu` reads once and writes once; fused, you read the GEMM output once and write $y$; inside the GEMM epilogue there is no extra traffic:

$$
T_{\rm eager}=4MNs,\qquad T_{\rm fused}=2MNs,\qquad T_{\rm epilogue}=0 .
$$

This ignores the 8 KiB of $b$ itself. The table estimates time at a **teaching bandwidth of 2 TB/s**:

| How it is written | Elementwise reads and writes | Estimated at 2 TB/s |
| --- | ---: | ---: |
| Eager: `mm` → `add` → `relu` | 128 MiB | about 67 µs |
| After the GEMM, one fused bias+ReLU kernel | 64 MiB | about 34 µs |
| Bias in the GEMM epilogue (as in `addmm`), ReLU as its own kernel | 64 MiB | about 34 µs |
| Both bias and ReLU in the GEMM epilogue | 0 | 0 |

<!-- widget:step-fusion-traffic -->

Now the GEMM itself: it reads at least $x$ and $w$ and writes one output, 96 MiB in total, about 50 µs at 2 TB/s; its work is $2MNK\approx1.37\times10^{11}$ FLOP, about 275 µs at a **teaching compute rate of 500 TFLOP/s**. At this shape the GEMM is compute-bound; the eager elementwise part costs about 24% of the GEMM time, and one fusion brings that to about 12%. 2 TB/s and 500 TFLOP/s are round numbers for hand calculation, not the spec of any card; real numbers depend on the SKU, the precision and the efficiency actually achieved, see [NVIDIA GPU generations](nvidia-generations.en.md) and the roofline discussion in [Performance](performance.en.md#roofline).

**Change one condition:** reduce $M$ from 4096 to 8, as in decoding with only 8 tokens at a time. An intermediate shrinks to 64 KiB, and the eager elementwise part totals 256 KiB, about 0.13 µs at 2 TB/s. The GEMM still has to read all of $w$ (32 MiB, about 17 µs) while its arithmetic takes only about 0.5 µs, so it is now bound by reading weights. Assume each operator costs 5 µs of CPU time to prepare and submit (**a teaching assumption**; the real value varies with the CPU, PyTorch version and operator, and must be measured with a profiler). The two elementwise operators then spend 10 µs of CPU time to submit 0.13 µs of GPU work. Over the whole layer, the CPU needs 15 µs to submit three operators and the GPU about 17.1 µs to run them, so the GPU stays ahead by only about 2 µs. A real layer also has small operators such as norms and residual adds, each another 5 µs of CPU time for far less than 1 µs of GPU time, so the CPU soon falls behind and the GPU starts waiting for it. Here fusion mainly saves submissions, not bytes; CUDA Graphs go further and turn a whole sequence of submissions into one.

| What to try | What to observe | What you can NOT conclude |
| --- | --- | --- |
| Run the eager version with the profiler recording both CPU and CUDA | At least 3 kernels; with small $M$, gaps between kernels | That every gap is launch overhead; synchronization and allocation leave gaps too |
| Replace `x @ w + b` with `F.linear` | 3 operators become 2 | That the bias went into the epilogue on every card; check the actual kernel |
| Use `torch.compile` with `TORCH_LOGS=output_code` | How many kernels were generated and which operations were fused | That compiled code is always faster; compile time, graph breaks and recompilation count too |
| Change $M$ from 4096 to 8 | The bottleneck moves from GEMM arithmetic to reading weights and CPU submission | That small batches should never be compiled |

## torch.compile: capture a graph, then generate kernels

According to the [2.14 torch.compiler documentation](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html), `torch.compile` has three parts:

- **TorchDynamo captures the graph.** It uses CPython's frame evaluation API (PEP 523) to rewrite bytecode just before it runs, extracting PyTorch operations into an FX graph and leaving the remaining Python in residual bytecode. While capturing it generates guards that record assumptions about dtype, device, shape, strides, `requires_grad` and so on; every later call checks the guards first and [recompiles](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html) if they fail. On unsupported Python, such as a data-dependent `if` or a call to `torch.save`, the default is a graph break: compile the part captured so far, run the unsupported code in ordinary Python, then resume capturing after it.
- **AOTAutograd captures the backward pass ahead of time.** It captures backpropagation in advance, so both the forward and the backward pass can be handed to the backend for optimization.
- **TorchInductor generates code.** It is the default backend; the [Dynamo overview](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_dynamo_overview.html) says it generates Triton for GPUs and C++/OpenMP for CPUs, and the tuning guide says it fuses kernels automatically.

The [`torch.compile` modes](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html) map onto the two leaks above: `reduce-overhead` uses CUDA Graphs to reduce Python overhead and suits small batches, but may use more memory; `max-autotune` tries Triton or template-based matrix multiplications and enables CUDA Graphs by default on GPU.

```python
compiled = torch.compile(lambda x, w, b: torch.relu(x @ w + b))
y = compiled(x, w, b)
```

Run it with `TORCH_LOGS=output_code` to see what Inductor actually generated (see the [logging docs](https://docs.pytorch.org/docs/2.14/logging.html); `graph_breaks` and `recompiles` are also common options). I ran a small-shape version on a local CPU with PyTorch 2.8.0: the generated code first calls an external `addmm`, folding the bias into the library call, then runs one generated ReLU kernel that writes back in place, which corresponds to the third row of the table above. **This is one observation on a CPU; nothing was run on a GPU.** On a GPU, Inductor may generate Triton kernels, call cuBLAS, or try other matrix-multiply templates under `max-autotune`; read output_code in your own environment.

Compilation has costs too: the first call pays for compiling; a shape change may trigger recompilation; graph breaks cut the graph into pieces and leave fewer chances to fuse. To make sure the whole function lands in one graph, add `fullgraph=True`, which raises an error on any graph break.

## Two other compiler routes: XLA/JAX and TensorRT

**JAX and XLA.** [`jax.jit`](https://docs.jax.dev/en/latest/jit-compilation.html) wraps arguments in tracer objects, and the recorded operations form a jaxpr; per the [AOT docs](https://docs.jax.dev/en/latest/aot.html), the jaxpr is then lowered to StableHLO, XLA's input language, compiled into an executable for CPU, GPU or TPU, and cached. Traced values can influence control flow only through static attributes such as shape and dtype; a value-dependent Python `if` raises an error under jit, and an argument marked static triggers a recompilation for every new value. Unlike Dynamo, which graph-breaks and falls back to Python on unsupported code, jit requires the whole function to be traceable. The [XLA architecture page](https://openxla.org/xla/architecture) says XLA first runs hardware-independent optimizations (common subexpression elimination, operation fusion, buffer analysis) and then backend-specific ones; the CPU and GPU backends use LLVM for code generation, with NVIDIA GPUs going through LLVM's NVPTX backend.

**TensorRT.** It is NVIDIA's inference SDK and works in two phases: in the [build phase](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html) a builder times candidates on the target GPU and picks the fastest kernel for each layer, producing a serialized engine (a plan file); the runtime phase loads the engine and executes it. The docs say that by default an engine is only guaranteed to work with the same OS, CPU architecture, GPU model and TensorRT version used to build it, and that version- and hardware-compatibility options can relax this. The common entry is to export ONNX and build the network with the ONNX parser; Torch-TensorRT compiles in-process instead (see [capabilities](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/capabilities.html)), and `torch.compile` also has an inference-only `"tensorrt"` backend. So an engine is a build artifact tied to the hardware: after changing GPU type or upgrading versions, rebuild or re-validate it. As of 2026-10-15, the docs list 11.4.0 as the latest version.

The three routes side by side (only what the docs cited above say):

| | `torch.compile` (default Inductor) | `jax.jit` | TensorRT |
| --- | --- | --- | --- |
| When the graph is captured | On the first call, Dynamo captures an FX graph before the bytecode runs | On the first call, tracers record a jaxpr, which is lowered to StableHLO and handed to XLA | In a separate build phase: the builder times each layer on the target GPU and picks kernels |
| Python it cannot capture | Graph break by default, and that part runs as plain Python; with `fullgraph=True` it raises an error | A value-dependent `if` raises an error; the whole function must be traceable | Takes no Python directly: export ONNX first, or compile in-process with Torch-TensorRT |
| What it produces | Triton code on GPU, C++/OpenMP code on CPU | An XLA executable for CPU, GPU or TPU | A serialized engine (plan file) |
| What forces a redo | A failed guard: dtype, device, shape, stride or `requires_grad` changed | Only inputs with the same shape and dtype reuse it ([tracing docs](https://docs.jax.dev/en/latest/tracing.html)); a new value of a static argument recompiles | By default, a different OS, CPU architecture, GPU model or TensorRT version means rebuilding or re-validating |

From left to right, there is less and less room for arbitrary Python; only TensorRT also fixes the GPU model into its output.

## Multi-GPU: only a map of NCCL and the distributed APIs {#nccl-map}

[NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html) (documentation version 2.32.3, checked 2026-10-15) implements collectives such as AllReduce, AllGather, ReduceScatter and AlltoAll, plus point-to-point send/recv, over PCIe, NVLink, InfiniBand Verbs and IP sockets; PyTorch's [distributed docs](https://docs.pytorch.org/docs/2.14/distributed.html) recommend the NCCL backend for training on CUDA GPUs. An NCCL call is enqueued on the CUDA stream the caller passes in and runs asynchronously on the device ([stream semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)), so communication can compete with compute for resources and can also overlap with it; to debug a hang, turn on debug output with [`NCCL_DEBUG`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

| PyTorch API | What it splits | Main communication | Where this site covers it |
| --- | --- | --- | --- |
| DDP | Data; every GPU holds a full model | Gradient all-reduce, bucketed and overlapped with backward | [Multi-GPU training](../06-systems/distributed-training.en.md), [Training frameworks](../06-systems/training-frameworks.en.md) |
| FSDP2 ([`fully_shard`](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)) | Parameters, gradients and optimizer state, sharded per parameter as DTensor | All-gather parameters before compute, reduce-scatter gradients after backward | [The FSDP2 lifecycle](../06-systems/training-frameworks.en.md#fsdp-lifecycle) |
| TP ([`tensor.parallel`](https://docs.pytorch.org/docs/2.14/distributed.tensor.parallel.html)) | Matrices within a layer, built on DTensor | All-reduce or all-gather in each layer | [Tensor parallelism](../06-systems/tensor-parallel.en.md) |
| PP ([`pipelining`](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html), marked alpha in the 2.14 docs) | Layers cut into stages | Activations and gradients between neighbouring stages | [Multi-GPU training](../06-systems/distributed-training.en.md) |

These APIs decide **where state lives and when to communicate**; the compute on each GPU still follows the dispatcher → ATen → kernel path described earlier.

## Inference engines: also only a map

Inference engines sit on top of the framework: model code and many kernels still come from the PyTorch ecosystem, but request queuing, batching, KV-cache management and CUDA Graph capture are the engine's job.

| Engine | Version checked in October 2026 | Where this site covers it |
| --- | --- | --- |
| vLLM | v0.31.0 (2026-10-05) | [vLLM and SGLang](../06-systems/inference-frameworks.en.md), [How one request runs](../06-systems/llm-serving.en.md) |
| SGLang | v0.5.21 (2026-10-02) | Same |
| TensorRT-LLM | Latest stable release on GitHub v1.2.1 (2026-04-20); 1.3 and 1.4 are still release candidates | Not covered separately yet |

The [TensorRT-LLM README](https://github.com/NVIDIA/TensorRT-LLM) (checked 2026-10-15) describes it as an open-source library for optimizing LLM inference on NVIDIA GPUs, with custom kernels for attention, GEMM, MoE and more plus runtime optimizations such as speculative decoding, and says it is "Architected on PyTorch". The name contains TensorRT, but do not assume from the name that every path builds a TensorRT engine first. Whether the two open-source engines enable `torch.compile` by default was checked against pinned versions in [the existing note](../06-systems/inference-frameworks.en.md#bottlenecks).

## When something goes wrong, which layer to check first

| Layer | What it decides | Typical failure | Where to look |
| --- | --- | --- | --- |
| Python and model code | Which operators, shapes, dtypes, whether gradients are needed | `.item()` in a loop causing implicit syncs; dtype silently becoming FP32 | CPU call stacks in the profiler; print dtype and device of key tensors |
| Dispatcher and ATen | Which implementation each call lands on | A tensor accidentally left on the CPU; non-contiguous tensors possibly triggering extra copies | `aten::` events in the profiler, with `record_shapes=True` |
| Libraries: cuBLAS, NCCL and others | Concrete algorithm, workspace, communication method | Shapes unfavourable for GEMM; workspace using extra memory; collectives hanging or timing out | CUDA kernel names and durations; `NCCL_DEBUG` |
| Compiler: Dynamo and Inductor | How large a graph to capture, how to fuse, when to recompile | Graph breaks, repeated recompilation, long compile times, extra memory from `reduce-overhead` | `TORCH_LOGS=graph_breaks,recompiles,output_code`; `fullgraph=True` |
| CUDA runtime and streams | Submission order, synchronization, memory pool | Timing without synchronizing; cached memory making `nvidia-smi` look full | CUDA events; compare `memory_allocated()` with `memory_reserved()` |
| Distributed layer: DDP, FSDP2, TP, PP | Where state lives, when to communicate | One slow rank making everyone wait; peak memory still too high after sharding | Per-rank timelines; whether communication overlaps compute |
| Inference engine | Queuing, batching, KV cache | Slow first token while the GPU is not busy | Engine metrics and traces, see [How one request runs](../06-systems/llm-serving.en.md) |

This table only tells you which layer to search for evidence first; it does not replace a profiler. The framework turns one line of code into work on a GPU; the next question is how those GPUs are handed out to different jobs and teams: continue with [Kubernetes and GPU clusters](kubernetes.en.md).
