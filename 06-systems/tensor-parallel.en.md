# Tensor parallelism: two GPUs, one layer

[中文](tensor-parallel.md) · **English**

> Reviewed: 2026-10 · Prerequisites: [tensors and shapes](../00-foundations/pytorch/README.en.md), [distributed training](distributed-training.en.md)

Cutting a weight matrix in half is easy. The useful question is whether each GPU can continue computing with its half. Does it hold a slice of the answer, or a partial sum that another GPU must contribute to? That determines where communication is necessary.

Start with a tiny FFN, then follow attention and backward. The first half needs only matrix multiplication. The expandable PyTorch example checks the arithmetic without a distributed setup.

## Work out the answer before choosing a collective {#two-ranks}

Our input is `[2, −1]`. The first layer produces four channels; the second returns to two. We use ReLU for easy arithmetic and leave out bias and dropout.

Split the first layer by **output channels**, then the second by the matching **input channels**. We call these column and row splits relative to the weight in `input @ weight`.

<figure class="worked-update">
<ol>
<li><small>01 · Replicated input</small><strong>Both ranks start with [2, −1]</strong><span>Rank 0 owns the first two intermediate channels; rank 1 owns the last two. Neither receives just one input coordinate.</span></li>
<li><small>02 · Local computation</small><strong>[1, 0] and [0, 4]</strong><span>Each rank applies its first layer and ReLU, then uses its own second-layer weights. No need to gather all four channels.</span></li>
<li><small>03 · Sum, not concatenate</small><strong>[1, 2] + [8, 4]</strong><span>Both partial outputs have width two. A sum AllReduce gives each rank the complete output [9, 6].</span></li>
</ol>
<figcaption>A rank identifies a process within a communication group. Here one rank corresponds to one GPU. All numbers are teaching examples.</figcaption>
</figure>

Each rank owns one $A_i$ and one $B_i$:

$$
A_0=\begin{bmatrix}1&0\\1&1\end{bmatrix},\quad
A_1=\begin{bmatrix}-1&2\\0&0\end{bmatrix}.
$$

$$
B_0=\begin{bmatrix}1&2\\0&1\end{bmatrix},\quad
B_1=\begin{bmatrix}1&0\\2&1\end{bmatrix}.
$$

Rank 0 computes `[2, −1] @ A₀ = [1, −1]`, applies ReLU to get `[1, 0]`, then multiplies by `B₀` to get `[1, 2]`. Rank 1 gets `[−2, 4]`, then `[0, 4]`, then `[8, 4]`. Adding the partial outputs gives `[9, 6]`.

Concatenation would produce four values, `[1, 2, 8, 4]`. Averaging would produce `[4.5, 3]`. Neither is the original layer's output.

## Why pair a column split with a row split? {#column-row}

Let $N$ be the number of tokens computed this round, $H$ the hidden size, and $F$ the FFN width. Assume $F$ is divisible by two.

| Tensor | Full shape | What each rank holds |
| --- | --- | --- |
| Input $X$ | $N\times H$ | The same complete input |
| First weight $A$ | $H\times F$ | A column shard of $H\times(F/2)$ |
| Activation $U$ | $N\times F$ | A channel shard of $N\times(F/2)$ |
| Second weight $B$ | $F\times H$ | A row shard of $(F/2)\times H$ |
| Local output $Z_i$ | $N\times H$ | A partial sum, despite having the full output shape |

Each rank computes $U_i=\phi(XA_i)$ and $Z_i=U_iB_i$, then

$$Y=Z_0+Z_1.$$

The intermediate $U$ is the useful part: the next operation needs exactly the channels already on that rank. No AllGather is needed between the layers. [Megatron-LM §3](https://arxiv.org/html/1909.08053v4#S3) uses this pairing so the FFN forward pass reduces only at its output.

If the first weight were split along its input dimension instead, each rank would produce a partial sum for the same intermediate channel. Those sums must be combined before a nonlinearity. For instance,

$$\operatorname{ReLU}(3-4)=0,$$

$$\operatorname{ReLU}(3)+\operatorname{ReLU}(-4)=3.$$

A row-first arrangement is not mathematically forbidden. It needs to combine contributions before activation, and the second layer's placement can introduce more communication. Column-then-row is a useful arrangement, not a rule for every matrix in a model.

SwiGLU needs matching channel partitions too: keep the gate and up values for a given intermediate channel together for their elementwise product, then apply the corresponding down shard. Matching dimensions alone does not establish a correct partition.

## AllGather versus AllReduce {#collectives}

Ask **whether values must be added**, and **which ranks need the result**. Reductions below use SUM; other reduction operators also exist.

| Operation | Two-rank example | Result |
| --- | --- | --- |
| Broadcast | Root holds `[2, 5]` | Every rank gets `[2, 5]` |
| Scatter | Root holds `[2, 5, 7, 9]` | Rank 0 gets `[2, 5]`; rank 1 gets `[7, 9]` |
| Gather | Ranks hold `[2, 5]` and `[7, 9]` | Only root receives `[2, 5, 7, 9]` |
| AllGather | Same inputs | Every rank receives the concatenated four values |
| Reduce | Ranks hold `[2, 5]` and `[7, 9]` | Only root receives the sum `[9, 14]` |
| AllReduce | Same inputs | Every rank receives `[9, 14]` |
| ReduceScatter | Same inputs | Rank 0 receives `[9]`; rank 1 receives `[14]` |

AllGather does not add values; AllReduce does not concatenate channels. ReduceScatter followed by AllGather can produce the same reduced result as AllReduce, without requiring an implementation to issue two separate API calls. [NCCL's collective documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) defines these result semantics; ring and tree are algorithms used to implement communication.

## Where backward needs communication {#backward}

Write the output gradient as $G=\partial L/\partial Y$. Since $Y$ is a sum of local outputs, each $Z_i$ receives the same $G$. Each rank can then compute its second-layer gradient, intermediate gradient, and first-layer gradient locally:

$$\nabla B_i=U_i^\top G,$$

$$D_i=(GB_i^\top)\odot\phi'(XA_i),$$

$$\nabla A_i=X^\top D_i.$$

The **input gradient** is different. The input contributed to both paths, so both contributions matter:

$$\nabla X=D_0A_0^\top+D_1A_1^\top.$$

That is the FFN's backward AllReduce. With `loss = output.sum()` in our example, the two input-gradient contributions are `[3, 3]` and `[6, 0]`. Their sum is `[9, 3]`.

These are **different paths through the same computation**, not gradients from two different data batches. Applying a data-parallel averaging factor would change the result.

<details markdown="1">
<summary>Check outputs and gradients with PyTorch on CPU</summary>

Both logical ranks run in one process for comparison. This tests algebra, not NCCL, networking, or multi-GPU performance. We store weights in the mathematical orientation; `torch.nn.Linear.weight` actually uses `[out_features, in_features]`, so account for the transpose when slicing real parameters.

```python
import torch

def split_ffn(inputs, first_weight, second_weight, parts=2):
    if type(parts) is not int or parts < 1:
        raise ValueError("parts must be a positive integer")
    if any(tensor.ndim != 2 for tensor in (inputs, first_weight, second_weight)):
        raise ValueError("expected matrices")
    if inputs.shape[1] != first_weight.shape[0]:
        raise ValueError("input width does not match the first layer")
    if first_weight.shape[1] != second_weight.shape[0]:
        raise ValueError("intermediate widths must match")
    if first_weight.shape[1] == 0 or first_weight.shape[1] % parts:
        raise ValueError("intermediate width must divide evenly into parts")
    first_shards = first_weight.chunk(parts, dim=1)
    second_shards = second_weight.chunk(parts, dim=0)
    partials = [torch.relu(inputs @ first) @ second
                for first, second in zip(first_shards, second_shards)]
    return torch.stack(partials).sum(dim=0)

inputs = torch.tensor([[2., -1.]], dtype=torch.float64, requires_grad=True)
first_weight = torch.tensor([[1., 0., -1., 2.], [1., 1., 0., 0.]],
                            dtype=torch.float64, requires_grad=True)
second_weight = torch.tensor([[1., 2.], [0., 1.], [1., 0.], [2., 1.]],
                             dtype=torch.float64, requires_grad=True)
parameters = (inputs, first_weight, second_weight)
full = torch.relu(inputs @ first_weight) @ second_weight
split = split_ffn(*parameters)
torch.testing.assert_close(split, full)
torch.testing.assert_close(split, torch.tensor([[9., 6.]], dtype=torch.float64))
full_grads = torch.autograd.grad(full.sum(), parameters)
split_grads = torch.autograd.grad(split.sum(), parameters)
for full_grad, split_grad in zip(full_grads, split_grads):
    torch.testing.assert_close(full_grad, split_grad)

with torch.no_grad():
    upstream = torch.ones_like(full)
    input_contributions = []
    for first, second in zip(first_weight.chunk(2, 1), second_weight.chunk(2, 0)):
        preactivation = inputs @ first
        local_gradient = (upstream @ second.T) * (preactivation > 0)
        input_contributions.append(local_gradient @ first.T)
    combined = torch.stack(input_contributions).sum(0)
    torch.testing.assert_close(combined, full_grads[0])
    torch.testing.assert_close(combined, torch.tensor([[9., 3.]], dtype=torch.float64))
```

A distributed implementation also needs correct dtypes, devices, random-number handling, bias, and communication. For example, add the second-layer bias once after reducing partial outputs. Adding the entire bias on each rank before summation would count it repeatedly.

</details>

## When does a Decoder layer communicate four times? {#decoder-count}

Specify the configuration first: a dense decoder block, standard multi-head attention (MHA), replicated input within the TP group, no sequence/context parallelism, and no activation recomputation. Count logical collectives inside the block, not network packets, DP gradient synchronization, the vocabulary head, or pipeline transfers.

Attention can be partitioned by complete heads. Q/K/V projections produce local heads, those heads compute attention locally, and the output projection is split along its input channels. Eight heads with TP=2 gives four heads per rank, not necessarily one head per GPU. Each head retains its original context access and attention mask; splitting heads does not isolate the first half of a text from the second. The output projection yields partial sums to combine.

| Branch | Forward | Backward |
| --- | --- | --- |
| Attention | One sum after output projection | One sum of input-gradient contributions |
| FFN | One sum after the second layer | One sum of input-gradient contributions |
| Total | Two | Two |

This gives four collectives for training, but two for forward-only inference. Do not multiply the training count into an inference estimate, or treat one AllReduce as one network transfer.

Why not reduce again in backward at the forward reduction boundary? Here the downstream complete output and its gradient are replicas within the TP group, not independent losses. Summing them again would count the same contribution repeatedly. Megatron's [mapping implementation](https://github.com/NVIDIA/Megatron-LM/blob/4603a836261fb39fd0050342d58dc66aecdf6f74/megatron/core/tensor_parallel/mappings.py) distinguishes a forward identity with backward reduction from a forward reduction with backward identity. There is no universal backward rule to apply without knowing the tensor layout.

## How sequence parallelism changes the trace {#sequence-parallel}

Ordinary TP can still replicate residual and normalization activations. Megatron-style sequence parallelism (SP) shards those regions over tokens, AllGathers before column-parallel computation, and ReduceScatters row-parallel outputs to retain local token shards.

For one forward FFN:

<figure class="worked-update">
<ol>
<li><small>Input · AllGather</small><strong>Collect the token slices</strong><span>An N/2 × H token shard becomes N × H so the local column shard can process all tokens.</span></li>
<li><small>Middle · Local computation</small><strong>Compute local channels</strong><span>The intermediate remains N × F/2. Token sharding and channel sharding are different dimensions.</span></li>
<li><small>Output · ReduceScatter</small><strong>Sum, then retain local tokens</strong><span>Reduce partial outputs and partition over tokens. Each rank returns to N/2 × H.</span></li>
</ol>
<figcaption>This describes a specific layout, not every framework's use of “SP.” It does not remove attention dependencies across the context.</figcaption>
</figure>

The paired backward swaps the AllGather / ReduceScatter roles; an implementation can overlap input-gradient communication with weight-gradient computation. In the [pinned linear-layer implementation](https://github.com/NVIDIA/Megatron-LM/blob/4603a836261fb39fd0050342d58dc66aecdf6f74/megatron/core/tensor_parallel/layers.py), read `sequence_parallel`, `gather_output`, and `input_is_parallel` together rather than searching for AllReduce alone. This revision was checked on 2026-10-09; other backends or later versions require another check.

GQA / MQA also require checking KV-head placement: few heads can lead to replication. MoE adds expert dispatch, context parallelism exchanges context information, and activation recomputation repeats some forward work. “Four” describes a useful baseline, not an immutable property of a Decoder.

## Before scaling up {#before-scaling}

Draw one block and label each input shape, partitioned dimension, and output layout: shard or partial sum. Compare outputs, input gradients, weight gradients, and one update against an unpartitioned baseline. Start without dropout, then verify the real random-number policy; check that bias and residuals are not added twice.

Only then compare performance: bytes transferred, dependency-driven waits, and whether smaller matrices still compute efficiently. **Fewer collectives need not mean lower latency, and more GPUs need not mean higher throughput.** For an eight-GPU TP-versus-replica choice, continue with [deployment layouts](distributed-training.en.md#eight-gpus). For slow first tokens, read [requests and scheduling](llm-serving.en.md).
