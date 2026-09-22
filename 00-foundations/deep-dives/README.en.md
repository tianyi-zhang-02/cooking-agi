# Advanced reading: from mechanisms to model families

[中文](README.md) · **English**

The core section draws the map. The advanced section handles the places where “I understand the diagram, but something still feels wrong.” There are two ways through it: **go inward into mechanisms, or read sideways across model families.**

Start with mechanisms:

- [Sequence gradients, BPTT, and gates](recurrent-dynamics.en.md): why information disappears and what the LSTM additive path changes.
- [Transformer architecture](../transformer.en.md): $Q/K/V$, masks, normalization, RoPE, GQA, SwiGLU, and KV cache.
- [Language-model objectives, training, and generation](language-model-objective.en.md): why one model has a parallel training path and a sequential decode path.

Read each note with a falsifiable question: **if this component is removed, which invariant should fail first?**

Then read sideways:

- [Model family deep dives](../model-families/README.en.md): compare Llama, Qwen, DeepSeek, and Gemma with the same questions.
- [MoE series](../moe/README.en.md): isolate routing, load balance, shared experts, and systems cost.
- [Looped Transformers](../looped/README.en.md): understand parameter sharing, recurrent depth, and adaptive compute.

When reading across families, do not ask “who has more components?” Ask: **which constraint is it optimising, where does capability come from, and where did the cost move?**
