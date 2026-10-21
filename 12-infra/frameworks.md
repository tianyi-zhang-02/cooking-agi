# 框架这一层：从一行 PyTorch 到 GPU 上的 kernel

**中文** · [English](frameworks.en.md)

> 阅读时间：约 13 分钟 · 草稿，待审核

写下 `y = torch.relu(x @ w + b)`，Python 立刻返回，GPU 却可能还在算；套上 `torch.compile`，同一行又快了一截。中间到底经过了哪几层？**在 eager 模式下，这一行会变成 3 个独立的 ATen 算子、至少 3 个 kernel 和两次本可以省掉的显存往返；编译器省下的主要是每个算子的 CPU 调度开销和中间结果的读写，乘法一次也没少。**

> **待审核草稿。** 这一篇还没有经过站主审阅，事实、取舍和措辞都可能修改；引用前请回到文中链接的一手资料核对。

kernel、stream 和 SM 是什么，见 [CUDA](cuda.md) 和 [GPU 架构](gpu-architecture.md)；怎样用 profiler 量出瓶颈，见[性能](performance.md)。训练与推理框架怎么选，已经在[训练框架](../06-systems/training-frameworks.md)和 [vLLM 与 SGLang](../06-systems/inference-frameworks.md) 讲过，这里只把它们放回同一张图。源码链接固定在 PyTorch v2.14.1（2026-09-30 发布），文档对照 2.14 版，2026-10-15 核对。

## 先把例子固定下来

```python
import torch

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
b = torch.zeros(4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)

y = torch.relu(x @ w + b)
```

把 $x$ 想成 4096 个 token、每个 4096 维的激活，$w$ 是一层权重，$b$ 是 bias。三个张量在同一张 GPU 上，都是 BF16，每个元素 2 字节。$w$ 和 $b$ 需要梯度，所以这一行在算前向的同时，也在为反向传播做记录。形状是教学设定，不对应某个具体模型。

## 这一行经过了哪几站

| 站点 | 发生了什么 | 依据 |
| --- | --- | --- |
| Python | `@` 调用 `torch.matmul`，`+` 调用 `torch.add`，再调用 `torch.relu`；每次都经 Python 绑定进入 C++ | 运算符重载 |
| Dispatcher | 看输入张量带的 dispatch key 和线程局部状态，选出优先级最高的 key：先跑 Autograd 那一层，再转给 CUDA 实现 | [dispatcher 教程](https://docs.pytorch.org/tutorials/advanced/dispatcher.html) |
| ATen 算子 | `matmul` 是组合算子，两个二维输入时直接调用 `mm`；`add.Tensor` 和 `relu` 各是一个算子 | [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/native_functions.yaml#L3781)、[`_matmul_impl`](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/LinearAlgebra.cpp#L2024) |
| CUDA 实现 | `mm` 的 CUDA 版本转到 `addmm_out_cuda_impl`，按条件调用 cuBLASLt 或 cuBLAS；`add` 和 `relu` 各启动一个逐元素 kernel | [cuda/Blas.cpp](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/Blas.cpp#L785) |
| 显存 | 三个输出（$xw$、加 bias 后的结果、$y$）各从 PyTorch 的 caching allocator 取一块 | [CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html) |
| Stream | 这些 kernel 按顺序排进当前 stream；Python 不等 GPU 算完就返回 | 同上 |
| Autograd | 记下 `ReluBackward0 → AddBackward0 → MmBackward0`，以及反向要用的张量 | [autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html) |

**Dispatcher 先分层，再分设备。** 官方的 [dispatcher 教程](https://docs.pytorch.org/tutorials/advanced/dispatcher.html)（作为自定义算子指南自 PyTorch 2.4 起标为过时，但对 dispatcher 的描述仍可参考）说，一个算子按 dispatch key 拆成多个 kernel，例如 CPU、CUDA、Autograd；调用时先执行 Autograd 那一层，再按输入所在设备转发给后端 kernel。

**`matmul` 自己没有反向公式。** 它在 `native_functions.yaml` 里注册为 `CompositeImplicitAutograd`，意思是由别的算子拼出来，梯度也由那些算子提供。两个二维输入走 `mm`，所以反向图里出现的是 `MmBackward0`，而不是某个 matmul 节点。输入是三维的 `[batch, seq, hidden]` 时，`matmul` 可能把前两维折成一维再调 `mm`，具体条件见上面链接的 `_matmul_impl`。

**矩阵乘交给库，逐元素运算各起一个 kernel。** `mm` 的 CUDA 实现以 $\beta=0$ 调用 `addmm_out_cuda_impl`，满足条件时走 cuBLASLt，否则走 cuBLAS 的 GEMM，调用都发在 `getCurrentCUDAStream()` 返回的当前 stream 上。最终选中哪个 GEMM kernel，由 GPU 型号、dtype、形状和库版本决定，这里不写具体名字。

**换个写法，算子数就变了。** 若写成 `torch.nn.functional.linear(x, w.T, b)`，v2.14.1 的 [`at::linear`](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/Linear.cpp#L85) 对二维输入直接调用 `addmm(bias, input, weight.t())`，源码注释说融合版本“稍快”。CUDA 上的 `addmm` 在条件满足时把 bias 指针交给 cuBLASLt 的 epilogue，bias 就在 GEMM 写回结果时顺手加上；PyTorch 还有一个内部算子 `_addmm_activation`，能把 ReLU 或 GELU 也放进 epilogue。注意 `nn.Linear` 的权重形状是 `[out, in]`，和这里的 `w` 是转置关系。

**Python 返回，不代表 GPU 算完了。** 2.14 的 CUDA semantics 写明，GPU 操作默认是异步的：调用只是把工作排进设备队列。每个设备有自己的默认 stream，同一 stream 里的操作按提交顺序执行。`.item()`、`print(y)`、`.cpu()` 都要等结果，会让 CPU 停下来；计时也必须先同步或用 CUDA event。数据搬运和多 stream 的细节见[训练循环里的 GPU 数据流](../00-foundations/pytorch/training-loop.md#gpu-data-flow)。

**Autograd 只保存反向真正需要的东西。** autograd mechanics 说，前向时记录的是一个有向无环图，叶子是输入、根是输出，`grad_fn` 是进入这张图的入口，而且每次迭代重新建图。按 v2.14.1 的 [derivatives.yaml](https://github.com/pytorch/pytorch/blob/v2.14.1/tools/autograd/derivatives.yaml#L2131)，ReLU 的反向用的是它自己的输出（公式写作 `threshold_backward(grad, result, 0)`），`mm` 对 $w$ 求梯度要用到 $x$，加法的反向不需要输入的值。所以 $xw$ 和 $xw+b$ 这两个中间结果用完就能还给分配器，留下来的是 $y$ 本身和本来就在的 $x$。规则细节见[自动求导](../00-foundations/pytorch/autograd.md)。

下面这段代码在 CPU 上就能核对算子序列和反向图：

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

在本地 CPU、PyTorch 2.8.0 上，第一行列出 `aten::matmul`、`aten::mm`、`aten::add`、`aten::relu`，夹着 `aten::resolve_conj`、`aten::clamp_min` 等辅助调用；第二行是 `ReluBackward0 ['AddBackward0']`。这是旧一点的版本，和上面引用的 2.14.1 源码不是同一版；GPU 上加 `ProfilerActivity.CUDA` 才能看到 kernel，名字随 GPU 和库版本变化。

## 没有编译，性能漏在哪儿

第一处是 **CPU 侧的固定开销**。每个算子都要走一遍 Python 调用、dispatcher、参数检查、分配输出和提交 kernel。张量大时，这点时间被 GPU 计算盖住；张量小时，GPU 几微秒就算完一个算子，CPU 反而跟不上。[PyTorch 的 CUDA Graphs 博客](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)（NVIDIA 与 Facebook 工程师，写于 PyTorch 1.10 时期）举过 MLPerf 里的 Mask R-CNN：每卡 batch 为 1 时，CPU 满载，GPU 却在 kernel 之间大段空闲。

第二处是 **中间结果的显存往返**。官方[性能调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)的说法是：逐元素运算通常受内存带宽限制，eager 模式为每个运算启动单独的 kernel，每个都要从显存读、算、再写回去。

### 算一笔显存账

只数 GEMM 输出之后的逐元素部分。记 $M=N=4096$，$s=2$ 字节，一个中间结果是 $MNs=33{,}554{,}432$ 字节，即 32 MiB。eager 下 `add` 读一遍写一遍，`relu` 再读一遍写一遍；融合后只读 GEMM 输出、写出 $y$；放进 GEMM 的 epilogue 就没有额外读写：

$$
T_{\rm eager}=4MNs,\qquad T_{\rm fused}=2MNs,\qquad T_{\rm epilogue}=0 .
$$

这里忽略了 $b$ 本身的 8 KiB。下表按 **2 TB/s 的教学带宽**估算时间：

| 写法 | 逐元素部分的读写 | 按 2 TB/s 估算 |
| --- | ---: | ---: |
| eager：`mm` → `add` → `relu` | 128 MiB | 约 67 µs |
| GEMM 之后，一个融合的 bias+ReLU kernel | 64 MiB | 约 34 µs |
| bias 进 GEMM epilogue（如 `addmm`），ReLU 单独一个 kernel | 64 MiB | 约 34 µs |
| bias 和 ReLU 都进 GEMM epilogue | 0 | 0 |

<!-- widget:step-fusion-traffic -->

再看 GEMM 本身：至少读 $x$、$w$，写一份输出，共 96 MiB，按 2 TB/s 约 50 µs；运算量是 $2MNK\approx1.37\times10^{11}$ FLOP，按 **500 TFLOP/s 的教学算力**约 275 µs。在这个形状下，GEMM 受算力限制；eager 的逐元素部分约相当于 GEMM 时间的 24%，融合一次后约 12%。2 TB/s 和 500 TFLOP/s 只是方便手算的整数，不是某张卡的规格；真实数字要看具体 SKU、精度和实际达到的效率，见 [NVIDIA 各代 GPU](nvidia-generations.md) 和[性能](performance.md#roofline)里的 roofline。

**换一个条件：** 把 $M$ 从 4096 改成 8，像 decode 时一次只有 8 个 token。一个中间结果只剩 64 KiB，eager 逐元素部分共 256 KiB，按 2 TB/s 约 0.13 µs。GEMM 仍要读完整个 $w$（32 MiB，约 17 µs），运算只要约 0.5 µs，变成读权重受限。假设每个算子在 CPU 侧要花 5 µs 准备和提交（**教学假设**，实际值随 CPU、PyTorch 版本和算子变化，要用 profiler 测），两个逐元素算子就花 10 µs CPU 时间，去提交 0.13 µs 的 GPU 工作。整层算下来，CPU 提交三个算子要 15 µs，GPU 执行约 17.1 µs，只领先 2 µs 左右；真实的层里还有 norm、残差相加这类小算子，每个又是 5 µs 的 CPU 时间，GPU 上却远不到 1 µs，CPU 很快落在后面，GPU 开始等 CPU。这时融合省下的主要是提交次数，不是字节；CUDA Graphs 则把一整段提交合并成一次。

| 先试什么 | 应该观察什么 | 不能据此推出什么 |
| --- | --- | --- |
| 用 profiler 同时记录 CPU 和 CUDA，跑 eager 版本 | 至少 3 个 kernel；小 $M$ 时 kernel 之间有空隙 | 空隙全是 launch 开销；同步和分配也会留下空隙 |
| 把 `x @ w + b` 换成 `F.linear` | 算子从 3 个变成 2 个 | 每张卡上 bias 都进了 epilogue；要看实际 kernel |
| 用 `torch.compile` 并打开 `TORCH_LOGS=output_code` | 生成了几个 kernel，哪些运算被融合 | 编译后一定更快；编译时间、graph break 和重编译要另算 |
| 把 $M$ 从 4096 改成 8 | 瓶颈从 GEMM 计算转向读权重和 CPU 提交 | 小 batch 就不该编译 |

## torch.compile：先抓图，再生成 kernel

按 [2.14 的 torch.compiler 文档](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html)，`torch.compile` 由三部分组成：

- **TorchDynamo 抓图。** 它通过 CPython 的 frame evaluation API（PEP 523）在字节码执行前改写，把 PyTorch 运算抽成 FX graph，剩下的 Python 留在残余字节码里。抓图时会生成 guard，记录 dtype、设备、形状、stride、`requires_grad` 等假设；之后每次调用先检查 guard，不满足就[重新编译](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html)。遇到不支持的 Python，例如依赖数据的 `if` 或调用 `torch.save`，默认会 graph break：先编译已经抓到的部分，用普通 Python 跑不支持的代码，再从后面接着抓。
- **AOTAutograd 提前抓反向。** 它把反向传播也提前捕获下来，前向和反向就都能交给后端优化。
- **TorchInductor 生成代码。** 它是默认后端；[Dynamo 概览](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_dynamo_overview.html)说它为 GPU 生成 Triton、为 CPU 生成 C++/OpenMP，调优指南说它会自动融合 kernel。

[`torch.compile` 的 mode](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html) 也对应前面两处漏洞：`reduce-overhead` 用 CUDA Graphs 降低 Python 开销，适合小 batch，但可能多占显存；`max-autotune` 会尝试 Triton 或模板实现的矩阵乘，并在 GPU 上默认启用 CUDA Graphs。

```python
compiled = torch.compile(lambda x, w, b: torch.relu(x @ w + b))
y = compiled(x, w, b)
```

以 `TORCH_LOGS=output_code` 运行，可以看到 Inductor 实际生成了什么（[logging 文档](https://docs.pytorch.org/docs/2.14/logging.html)；`graph_breaks` 和 `recompiles` 也是常用选项）。我在本地 CPU、PyTorch 2.8.0 上跑了一个小形状版本：生成代码先调用外部的 `addmm`，把 bias 并进了库调用，再接一个生成的 ReLU kernel 原地写回，相当于上表第三行。**这是 CPU 上的一次观察，没有在 GPU 上运行。** GPU 上 Inductor 可能生成 Triton kernel、调用 cuBLAS，或在 `max-autotune` 下尝试别的矩阵乘模板，要在自己的环境里看 output_code。

编译也有代价：第一次调用要花编译时间；形状一变可能触发重编译；graph break 把图切碎，融合机会随之变少。想确认整段都被抓进一张图，可以加 `fullgraph=True`，遇到 graph break 就直接报错。

## 另外两条编译路线：XLA/JAX 与 TensorRT

**JAX 和 XLA。** [`jax.jit`](https://docs.jax.dev/en/latest/jit-compilation.html) 用 tracer 对象包住参数，记录下来的运算组成 jaxpr；按 [AOT 文档](https://docs.jax.dev/en/latest/aot.html)，jaxpr 再降到 XLA 的输入语言 StableHLO，然后编译成 CPU、GPU 或 TPU 上的可执行程序，并缓存起来。被追踪的值只能通过 shape、dtype 这类静态属性影响控制流，依赖数值的 Python `if` 在 jit 里会报错；标成 static 的参数，每换一个值就重新编译一次。和 Dynamo 遇到不支持的代码就 graph break、回退到 Python 不同，jit 要求整个函数能被追踪。[XLA 架构文档](https://openxla.org/xla/architecture)说，它先做与硬件无关的优化（公共子表达式消除、算子融合、buffer 分析），再由后端按硬件优化；CPU 和 GPU 后端用 LLVM 生成代码，NVIDIA GPU 走 LLVM 的 NVPTX 后端。

**TensorRT。** 它是 NVIDIA 的推理 SDK，分两个阶段：[构建阶段](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html)由 builder 在目标 GPU 上计时，为每一层挑最快的 kernel，输出序列化的 engine（plan 文件）；运行阶段加载 engine 执行。文档说，默认情况下 engine 只保证在构建时相同的操作系统、CPU 架构、GPU 型号和 TensorRT 版本上正确运行，可用版本兼容和硬件兼容选项放宽。常见入口是先导出 ONNX 再用 ONNX parser 建网络，也可以用 Torch-TensorRT 在进程内编译（[能力说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/capabilities.html)），`torch.compile` 也有仅用于推理的 `"tensorrt"` backend。所以 engine 是和硬件绑定的构建产物，换卡型或升级版本后要重新构建或验证。截至 2026-10-15，文档列出的最新版本是 11.4.0。

把三条路线并排放（只列上面引用过的文档说法）：

| | `torch.compile`（默认 Inductor） | `jax.jit` | TensorRT |
| --- | --- | --- | --- |
| 什么时候抓图 | 第一次调用时，Dynamo 在字节码执行前抓出 FX graph | 第一次调用时，用 tracer 记下 jaxpr，降到 StableHLO 交给 XLA | 在单独的构建阶段：builder 在目标 GPU 上给每层计时、挑 kernel |
| 遇到抓不了的 Python | 默认 graph break，那一段回到普通 Python 跑；`fullgraph=True` 时直接报错 | 依赖数值的 `if` 直接报错，整个函数必须能被追踪 | 不直接接 Python：先导出 ONNX，或用 Torch-TensorRT 在进程内编译 |
| 产出什么 | GPU 上的 Triton、CPU 上的 C++/OpenMP 代码 | CPU、GPU 或 TPU 上的 XLA 可执行程序 | 序列化的 engine（plan 文件） |
| 什么情况要重来 | guard 不满足：dtype、设备、形状、stride、`requires_grad` 变了 | 只有 shape、dtype 相同的输入才复用（[tracing 文档](https://docs.jax.dev/en/latest/tracing.html)）；static 参数换值就重编译 | 默认换操作系统、CPU 架构、GPU 型号或 TensorRT 版本，就要重新构建或验证 |

从左到右，留给 Python 写法的余地越来越小；只有 TensorRT 把 GPU 型号也固定进了产物。

## 多卡：NCCL 和分布式 API 只画地图 {#nccl-map}

[NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)（2026-10-15 核对的文档版本为 2.32.3）实现 AllReduce、AllGather、ReduceScatter、AlltoAll 等集合通信和点对点 send/recv，可走 PCIe、NVLink、InfiniBand Verbs 和 IP sockets；PyTorch 的[分布式文档](https://docs.pytorch.org/docs/2.14/distributed.html)建议 CUDA GPU 训练用 NCCL 后端。NCCL 调用排进调用方传入的 CUDA stream，在设备上异步执行（[stream 语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)），所以通信既会和计算争资源，也可以和计算重叠；排查挂起时用 [`NCCL_DEBUG`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) 打开调试输出。

| PyTorch API | 切什么 | 主要通信 | 本站在哪里讲 |
| --- | --- | --- | --- |
| DDP | 数据；每张卡一份完整模型 | 梯度 all-reduce，按 bucket 与反向重叠 | [多卡训练](../06-systems/distributed-training.md)、[训练框架](../06-systems/training-frameworks.md) |
| FSDP2（[`fully_shard`](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)） | 参数、梯度、optimizer 状态按参数分片，用 DTensor 表示 | 计算前 all-gather 参数，反向后 reduce-scatter 梯度 | [FSDP2 的生命周期](../06-systems/training-frameworks.md#fsdp-lifecycle) |
| TP（[`tensor.parallel`](https://docs.pytorch.org/docs/2.14/distributed.tensor.parallel.html)） | 层内矩阵，建在 DTensor 上 | 每层的 all-reduce 或 all-gather | [张量并行](../06-systems/tensor-parallel.md) |
| PP（[`pipelining`](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html)，2.14 文档标为 alpha） | 按层切成 stage | 相邻 stage 之间传激活和梯度 | [多卡训练](../06-systems/distributed-training.md) |

这些 API 决定的是**状态放在哪、什么时候通信**；每张卡上的计算仍然走前面那条 dispatcher → ATen → kernel 的路径。

## 推理引擎：也只画地图

推理引擎站在框架之上：模型代码和很多 kernel 仍来自 PyTorch 生态，但请求排队、组 batch、KV cache 管理和 CUDA Graph 捕获由引擎自己负责。

| 引擎 | 2026-10 核对的版本 | 本站在哪里讲 |
| --- | --- | --- |
| vLLM | v0.31.0（2026-10-05） | [vLLM 与 SGLang](../06-systems/inference-frameworks.md)、[一次请求怎样跑完](../06-systems/llm-serving.md) |
| SGLang | v0.5.21（2026-10-02） | 同上 |
| TensorRT-LLM | GitHub 上最新正式版 v1.2.1（2026-04-20）；1.3、1.4 仍是 release candidate | 本站暂未单独展开 |

[TensorRT-LLM 的 README](https://github.com/NVIDIA/TensorRT-LLM)（2026-10-15 核对）说它是在 NVIDIA GPU 上优化 LLM 推理的开源库，提供 attention、GEMM、MoE 等定制 kernel 和推测解码等运行时优化，并写着“Architected on PyTorch”。名字里有 TensorRT，但不要凭名字认定它的每条路径都先构建 TensorRT engine。两套开源引擎是否默认启用 `torch.compile`，[已有笔记](../06-systems/inference-frameworks.md#bottlenecks)按固定版本核对过。

## 出了问题，先查哪一层

| 层 | 它决定什么 | 典型故障 | 去哪里看 |
| --- | --- | --- | --- |
| Python 与模型代码 | 调哪些算子、形状、dtype、是否需要梯度 | 循环里的 `.item()` 引起隐式同步；dtype 意外变成 FP32 | profiler 的 CPU 调用栈；打印关键张量的 dtype 和设备 |
| Dispatcher 与 ATen | 每次调用落到哪个实现 | 张量意外留在 CPU；非连续张量可能触发额外拷贝 | profiler 里的 `aten::` 事件，加 `record_shapes=True` |
| 库：cuBLAS、NCCL 等 | 具体算法、workspace、通信方式 | 形状不利于 GEMM；workspace 多占显存；集合通信挂起或超时 | CUDA kernel 名称与耗时；`NCCL_DEBUG` |
| 编译器：Dynamo 与 Inductor | 抓多大的图、怎么融合、何时重编译 | graph break、反复重编译、编译太久、`reduce-overhead` 多占显存 | `TORCH_LOGS=graph_breaks,recompiles,output_code`；`fullgraph=True` |
| CUDA 运行时与 stream | 提交顺序、同步、显存池 | 计时前没同步；缓存的显存让 `nvidia-smi` 看起来已满 | CUDA event；`memory_allocated()` 与 `memory_reserved()` 对比 |
| 分布式层：DDP、FSDP2、TP、PP | 状态放哪、何时通信 | 一个 rank 变慢，全体等待；分片后峰值仍然超出 | 每个 rank 的时间线；通信是否与计算重叠 |
| 推理引擎 | 排队、batch、KV cache | 首 token 慢，GPU 却不忙 | 引擎指标与 trace，见[一次请求怎样跑完](../06-systems/llm-serving.md) |

这张表只是告诉你先去哪一层找证据，不代替 profiler。框架把一行代码变成了 GPU 上的工作，下一个问题是这些 GPU 怎样分给不同的任务和团队：接着读 [Kubernetes 与 GPU 集群](kubernetes.md)。
