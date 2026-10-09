# ML coding: PyTorch & implementations

<span id="pytorch-ml-implementations"></span>

[中文](README.md) · **English**

There are two routes here: use PyTorch to connect data, gradients, and a training loop, or implement attention, losses, and sampling as small functions. Run them, then inspect each step rather than relying on whiteboard recall alone.

## Pick the skill you want to practice

| Practice | Entry | What to check |
| --- | --- | --- |
| Write a correct training loop | [PyTorch: tensors to training](../../00-foundations/pytorch/README.en.md) | Storage aliasing, gradient accumulation, and loss denominators |
| Turn a formula into code | [Whiteboard implementations](../../00-foundations/hand-write-kit.en.md) | Shapes, masks, edge cases, numerical stability |
| Debug shapes and broadcasting | [Tensor operations](../../00-foundations/pytorch/operations-and-shapes.en.md) | What dimensions mean and whether broadcasting changes the intended operation |
| Check gradients | [Autograd](../../00-foundations/pytorch/autograd.en.md) | Graph breaks and whether gradients reach the intended parameters |

For conceptual explanations and derivations, use [ML / LLM fundamentals review](../../interview/basics/README.en.md). This route focuses on implementation rather than treating explanation and coding as the same skill.

## Practice one question three ways

For attention: explain how Q, K, and V enter the computation; implement a small causal-masked version; then introduce padding, a longer sequence, or caching and identify what must change.

Return to the [foundations](../../00-foundations/study-guide.en.md) when needed. These exercises check understanding; they aren't predictions of a particular company's interview.

## Self-check

<details><summary>Why isn't a correct output a complete explanation?</summary><p>You also need the input conditions and assumptions. A small example may miss empty inputs, mask direction, precision, or complexity problems.</p></details>

<details><summary>How do you check that you haven't just memorized the answer?</summary><p>Change one constraint: scale, missing data, duplicate values, or access to future information. Predict what changes before testing.</p></details>
