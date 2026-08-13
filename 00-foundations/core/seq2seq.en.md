# Seq2Seq: encode and generate

[中文](seq2seq.md) · **English**

> Reading time: ~8 min · Level: core · Last reviewed: 2026-08

## In one sentence

Seq2Seq separates reading from writing: an encoder represents the source, and a decoder generates the target one token at a time under that representation.

The earliest version compressed the whole source into the final encoder state $c=h_S$, creating a fixed-vector bottleneck. Attention replaces that single vector with a fresh weighted read over all encoder states at every output step:

$$e_{tj}=\text{score}(s_{t-1},h_j), \quad \alpha_{tj}=\text{softmax}_j(e_{tj}), \quad c_t=\sum_j\alpha_{tj}h_j$$

## Teacher forcing

During training, step $t$ receives the true previous token $y_{t-1}$. During inference it receives its own previous prediction. The objective is

$$\mathcal L=-\sum_{t=1}^{T}\log p_\theta(y_t\mid y_{<t},x)$$

The mismatch means generation errors can move later prefixes away from the training distribution.

## What remained unsolved

- recurrent encoders and decoders are still sequential;
- distant positions still communicate through long paths;
- attention fixes dynamic reading, not recurrent throughput.

The Transformer keeps attention and removes recurrence. Run the `reverse` task in [`../code/sequence_torch.py`](../code/sequence_torch.py), then continue to [Vanilla Transformer](vanilla-transformer.en.md).
