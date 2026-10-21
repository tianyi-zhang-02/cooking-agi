# How to read a model report

[中文](how-to-read.md) · **English**

Remembering that a model uses GQA or MoE is easy. More useful questions are: why here, which bottleneck would appear without it, and did the experiments actually isolate that change?

This is not a model ranking. We'll connect the prose, tensors, and experiments using a simplified decoder.

## Step 1: identify the exact model

One family may contain different generations, sizes, base and instruct checkpoints, and modalities. Record the report version, checkpoint, public config, and code revision. Don't use one version's architecture to explain another's results.

| Record first | Check later |
| --- | --- |
| Task and constraints | Is the goal quality, context length, memory, or latency? |
| Configuration | Layers, hidden size, Q / KV heads, FFN type |
| Training | Did data, objective, budget, and post-training change together? |
| Evaluation | Are prompts, samples, output budgets, and tool permissions comparable? |

## Step 2: follow one token through the computation

```mermaid
flowchart TD
    A["Token IDs"] --> B["Embedding: B × T × D"]
    B --> C["Norm → Attention → Residual"]
    C --> E["Norm → FFN → Residual"]
    E --> F["Repeat for L layers"]
    F --> G["Output projection → vocabulary logits"]
```

This sketches a common pre-norm decoder, not a universal template. Let $B$ denote batch, $T$ length, and $D$ hidden size. At each module ask: does the shape change, does information cross token boundaries, and what must persist for the next generation step?

For example, the FFN usually acts token by token, while attention aggregates information from visible tokens. Distinguishing their jobs is more useful than memorizing acronyms.

## Step 3: calculate a GQA example

Suppose there are 16 query heads, 4 KV heads, and head dimension 64. There are still 16 queries per position, with each KV head shared by 4 query heads. Take one request of 4,096 tokens, 24 layers and BF16, and compute the KV cache as $2BTLH_{\mathrm{KV}}d_hs$: 16 KV heads need 384 MiB and 4 need 96 MiB. [Original GQA paper](https://arxiv.org/abs/2305.13245)

The skill to practise here is to find which cost a change reduces, then ask what other costs remain. The KV cache becomes one quarter of its previous size, **not the whole model's memory use or request latency**. The [GQA section of the Llama walkthrough](llama.en.md#gqa-cache) works through the full calculation: what each symbol means, code you can change, and why weights and activations do not all shrink by the cache reduction factor. Sliding windows, quantization, or specialized caches need different accounting.

## Step 4: separate prefill and decode

- **Prefill:** process the existing prompt. Many tokens enter together; inspect sequence length, batch, attention implementation, and time to first token.
- **Decode:** generate further tokens while reading cached state. Memory access can matter especially with long contexts and small batches. For why a larger batch shifts decoding toward compute-bound, see [Performance: decode batching](../../12-infra/performance.en.md#decode-batching).

Reuse the four-KV-head configuration from step 3. Each token's K/V across 24 layers takes $2\times24\times4\times64\times2=24{,}576$ bytes, or 24 KiB. Prefill a 4,096-token prompt, then run 256 decode steps that each feed in one new token:

| Stage | What happens | KV-cache traffic |
| --- | --- | --- |
| Prefill | One forward pass over all 4,096 positions | Writes 96 MiB |
| Decode step 1 | The new token reads every cached position, then appends its own K/V | Reads 96 MiB, writes 24 KiB |
| Decode step 256 | The cache has grown by 255 positions | Reads about 102 MiB, writes 24 KiB |
| All 256 steps | — | Read about 24.7 GiB in total |

Prefill wrote 96 MiB once; the 256 decode steps read about 264 times that amount from the cache, and every step also reads the model weights again. With small batches, decoding is often limited by how many bytes move rather than by how many multiplications happen.

```python
def decode_kv_traffic(prompt_tokens, decode_steps, bytes_per_token):
    if min(prompt_tokens, decode_steps, bytes_per_token) < 0:
        raise ValueError("counts must be non-negative")
    written = prompt_tokens * bytes_per_token
    cached_positions = sum(prompt_tokens + step for step in range(decode_steps))
    return written, cached_positions * bytes_per_token

bytes_per_token = 2 * 24 * 4 * 64 * 2
written, read = decode_kv_traffic(4096, 256, bytes_per_token)
assert bytes_per_token == 24 * 2**10
assert written == 96 * 2**20
assert read == 1_081_216 * bytes_per_token
assert round(read / written) == 264
```

This is a logical byte count: it leaves out weight reads, activations and how kernels actually access memory, and it is not a measured bandwidth. When 8 requests decode together, they can share each step's weight reads; without shared prefixes, each request still reads its own KV cache. That is where the accounting changes as the batch grows.

Fewer theoretical FLOPs do not guarantee lower latency. Parallelism, bandwidth, and kernels affect the result. Record workload, input/output lengths, batch, and hardware before comparing measurements.

## Step 5: separate observation from explanation

| Reported observation | Possible explanation | Missing evidence |
| --- | --- | --- |
| Higher scores after an architecture change | Architecture may improve learning | Same-data, same-budget ablation |
| Better long-context behavior | Long-context training may help | Rule out data and evaluation changes |
| Faster measured inference | Smaller caches may help | Stage-level profiling, not just total time |

“Uncertain” is a better answer than attributing every gain to the newest component. Close the report and explain in two minutes: the constraint, the key change, its cost, and an experiment that could falsify your explanation.

Now try a family: [Llama](llama.en.md) · [Qwen](qwen.en.md) · [DeepSeek](deepseek.en.md) · [Gemma](gemma.en.md).
