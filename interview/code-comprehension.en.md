# Code comprehension: understand, change, and verify

[中文](code-comprehension.md) · **English**

> Last reviewed: 2026-10 · Original teaching exercises, not interview questions from any company.

Writing a solution from scratch and taking over someone else's code need different practice. With existing code, first establish its behavior: how inputs move, where state lives, and why a result is wrong. Running successfully—or passing most tests—does not prove that you understand it.

We will use a tiny embedding cache and a metric-aggregation example. No GPU, model, or external service is needed. Read the code and predict the result before running it; the reference fixes are further down.

<span id="google"></span>

## Ask how this round works {#rules}

Code-reading and debugging rounds vary a lot. In some you can run code and tests; in others you trace by hand. Some allow a specified AI tool; others allow none. Different rounds with the same team can have different rules. Once you have the invitation, ask your recruiter:

- Will you implement from scratch, or read and modify existing code?
- Can you run code and tests? Which language, editor, and references are available?
- Is AI explicitly allowed? In which round, using which tool, and with which material?

**Unless AI is explicitly allowed, do not use it in a live interview.** Code-reading practice does not replace algorithm preparation either; keep practicing LeetCode.

## Where do you start in unfamiliar code? {#read}

Establish the intended behavior, then trace one input all the way through. You do not need to narrate every line in file order or refactor the entire file after noticing one suspicious expression.

<figure class="worked-update" lang="en">
<ol>
<li><small>01 / BEHAVIOR</small><strong>State inputs and expectations</strong><span>Parameters, return values, permitted empty values, callers, and state shared across calls.</span></li>
<li><small>02 / EVIDENCE</small><strong>Trace a concrete example</strong><span>Follow branches, mutations, and cache hits. Find a small counterexample rather than relying on a hunch.</span></li>
<li><small>03 / CHANGE</small><strong>Make a small fix and test it</strong><span>Explain the changed behavior, preserve working cases, then discuss cost and remaining risks.</span></li>
</ol>
<figcaption>This is our practice sequence, not a company's scoring rubric. Run tests when available; otherwise walk through the same inputs manually.</figcaption>
</figure>

A reasonable opening: “Should the same text be recomputed after the model version changes? If so, I will trace these two calls and check whether the key includes the version.”

## Exercise one: a new model, the same old result {#cache}

First, state the requirements. This is a single-threaded, in-process cache that takes a version and text, both strings. The encoder returns a flat numeric list, and results may be reused for the same `(model_version, text)`. The version must identify immutable model, tokenizer, and preprocessing settings, not an alias that can point to new weights. Mutating a returned list must not change the cache. This teaching interface permits an empty list; a real embedding service may not.

This implementation is **deliberately broken**. Predict the final two results before fixing it.

```python
class BuggyEmbeddingCache:
    def __init__(self, encode):
        self.encode = encode
        self.values = {}

    def get(self, model_version, text):
        if not self.values.get(text):
            self.values[text] = self.encode(model_version, text)
        return self.values[text]


def encode(model_version, text):
    return [len(text), 1 if model_version == "v1" else 2]


cache = BuggyEmbeddingCache(encode)
first = cache.get("v1", "cat")
second = cache.get("v2", "cat")
```

The encoder is a stub, not a meaningful embedding model. The contract requires `[3,1]` for `first` and `[3,2]` for `second`. Both actually return `[3,1]`: the second call hits an entry keyed only by text.

Now consider two boundaries: what happens after `first[0] = 99`? If the encoder returns `[]`, does a repeated request run it again? Confusing a missing value with a false-valued result and exposing mutable cached state are separate problems.

<details markdown="1">
<summary>A small fix, and what it still does not do</summary>

```python
class EmbeddingCache:
    def __init__(self, encode):
        self.encode = encode
        self.values = {}

    def get(self, model_version, text):
        key = (model_version, text)
        if key not in self.values:
            self.values[key] = list(self.encode(model_version, text))
        return self.values[key].copy()
```

Include the version, test membership explicitly, and copy both when storing and returning the list. A shallow copy suffices because we assume the numeric elements are immutable. With nested mutable objects, you would need to reconsider what the interface allows callers to change.

Test repeat hits, changed versions, empty lists, and mutations by both the caller and encoder after storage. Tests with different text every time will not catch the missing version.

This is not a production cache: it has no size bound, expiry, concurrent request coalescing, or permission isolation. Define the sharing and security requirements before deciding whether tenant, permission, or preprocessing context belongs in the key. Adding a tuple does not solve multi-tenant security.

</details>

**Be precise about complexity.** Average O(1) dictionary lookup excludes key construction / text hashing and copying a vector of length D. A hit still copies O(D) elements, and initial string hashing can grow with text length. A miss also includes encoding. Storage grows with entries, text, and vectors; it is not O(1).

## Exercise two: correct batch scores, incorrect overall score {#aggregation}

The required metric is mean loss across all valid tokens; padding has already been excluded. With batch losses `[2,2]` and `[8]`, the target is `(2+2+8)/3 = 4`.

```python
def buggy_token_mean(batches):
    means = [sum(batch) / len(batch) for batch in batches if batch]
    return sum(means) / len(means)
```

This returns `(2+8)/2 = 5`. It is not rounding error: equal batch weights give tokens in the shorter batch more influence. If the desired metric were a mean of batch means, this would not be a bug. Here it conflicts with the metric defined above.

<details markdown="1">
<summary>The fix and tests to add</summary>

```python
def token_mean(batches):
    total_loss = 0.0
    token_count = 0
    for batch in batches:
        total_loss += sum(batch)
        token_count += len(batch)
    if token_count == 0:
        raise ValueError("No valid tokens")
    return total_loss / token_count
```

Regrouping the same tokens as `[[2],[2,8]]` must still give 4. Test empty batches, entirely empty input, and unequal batch sizes. This example explicitly raises on no valid tokens instead of silently reporting zero loss.

To report this metric across ranks, reduce the loss sum and valid-token count before dividing; do not unconditionally average rank means. This concerns **metric aggregation**. Correct distributed backpropagation additionally requires the gradient-reduction and scaling rules of the chosen framework.

</details>

Continue: [Training loops](../00-foundations/pytorch/training-loop.en.md) · [Data and objectives](../practice/post-training/data-and-objectives.en.md) · [FSDP and the training stack](../06-systems/training-frameworks.en.md).

## How can AI-enabled practice be useful? {#ai}

State your hypothesis before asking AI for a counterexample: “I think the cache key omits the model version. Give me inputs to test that, without editing the code.” Trace its proposed example yourself. If it violates the contract, explain why you reject it.

You can also compare two suggested fixes for changes in semantics. Adding a TTL, for example, does not prevent two models from sharing the wrong key within the same time window. A more elaborate change is not necessarily a relevant fix.

In the actual interview, use only approved tools and materials. Do not paste confidential questions or internal code into an unapproved external service. AI assistance does not remove your responsibility to explain and test a change, or replace independent algorithm practice.

## A 45-minute practice session {#practice}

| Time | Action | What you can check |
| --- | --- | --- |
| 5 minutes | Read the contract and draw calls between two or three functions | Inputs, outputs, and shared state |
| 10 minutes | Trace versions, mutations, and empty cache values | Where actual behavior differs from expectation |
| 10 minutes | Fix the cache and add regression tests | The old implementation fails; the corrected one passes |
| 10 minutes | Fix token aggregation and change batch boundaries | Regrouping does not change the metric |
| 10 minutes | Explain complexity and limitations; reject a bad proposed fix | Understanding without rewriting the whole system |

Run from the repository root. The [teaching code](code/comprehension_lab.py) contains both broken and reference implementations; try your own fix before reading them.

```bash
python3 interview/code/comprehension_lab.py
python3 -m unittest discover -s site/tests -p test_speedrun.py -v
```

The second command also checks site content and requires the site's normal test dependencies. Without the full checkout, run the first file alone; it uses only the Python standard library. Passing these small tests does not validate a real model, concurrent cache, or production service.

## Where to go next {#next}

- Short on preparation time: [Speedrun](speedrun.en.md).
- Still struggling with algorithm rounds: [LeetCode patterns](leetcode.en.md).
- Want to read more realistic code: follow one part of the [vLLM / SGLang request path](../06-systems/inference-frameworks.en.md#engine-map), rather than reading a repository from its first line.
- [Google's code-review guidance](https://google.github.io/eng-practices/review/reviewer/looking-for.html) helps practice reviewing design, behavior, and tests. It is engineering guidance, not official validation of these exercises or their interview format.
