# Deep dives: mathematics, objectives, and execution

[中文](README.md) · **English**

This level decomposes the mechanisms that determine behavior rather than repeating architecture labels:

- [Sequence gradients, BPTT, and gates](recurrent-dynamics.en.md): why information disappears and what the LSTM additive path changes.
- [Transformer architecture](../transformer.en.md): $Q/K/V$, masks, normalization, RoPE, GQA, SwiGLU, and KV cache.
- [Language-model objectives, training, and generation](language-model-objective.en.md): why one model has a parallel training path and a sequential decode path.

Read each note with a falsifiable question: if this component is removed, which invariant should fail first?
