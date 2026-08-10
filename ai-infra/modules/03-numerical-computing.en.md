# Module 03 · Numerical Computing and Mixed Precision

[中文](03-numerical-computing.md) · **English** · [Back to AI Infra](../README.en.md)

## What this module solves

Lower precision can increase throughput and reduce memory and communication, but it sacrifices numerical range or resolution. This module builds a decision framework: **which values can use lower precision, which operations need higher precision, and how is damage detected?**

## Learning goals

- Understand the sign, exponent, and mantissa of a floating-point number.
- Compare FP32, TF32, FP16, BF16, FP8, FP4, and integer formats.
- Explain overflow, underflow, rounding, and accumulation error.
- Understand automatic mixed precision and loss scaling.
- Design an experiment comparing quality, memory, and speed after quantization.

## Core notes

### Range and precision are different

Exponent bits mainly determine the range of magnitudes; mantissa bits mainly determine resolution within a magnitude.

| Format | Exponent intuition | Mantissa intuition | Main property |
| --- | --- | --- | --- |
| FP32 | large | high | stable but expensive in storage and bandwidth |
| FP16 | smaller | higher than BF16 | vulnerable to overflow and underflow |
| BF16 | similar to FP32 | lower than FP16 | wide dynamic range suited to training |
| FP8 E4M3 | smaller | higher within FP8 | favors precision |
| FP8 E5M2 | larger | lower | favors range |
| FP4 / INT4 | very limited | very limited | depends heavily on scaling and calibration |

TF32 is normally a Tensor Core matrix-computation mode, not a new 19-bit storage dtype for model weights. “BF24” is not common in mainstream LLM workflows; check the exact paper or hardware definition when it appears.

### Mixed precision

Different operators need different precision. A common pattern is:

```text
matrix inputs: BF16 / FP16 / FP8
accumulation: FP32 or higher internal precision
reductions, normalization, some softmax work: higher precision
optimizer state: higher precision when required for stability
```

Autocast chooses dtypes by operator. FP16 training often uses loss scaling to prevent small gradients from underflowing during backpropagation. BF16's wider exponent range usually reduces that need, although actual training behavior still must be measured.

### Quantization

Quantization maps continuous values onto a finite discrete set:

```text
q = clamp(round(x / scale) + zero_point)
x_hat = scale × (q - zero_point)
```

Important choices include:

- symmetric versus asymmetric;
- per-tensor, per-channel, or per-group scales;
- weight-only versus weight-and-activation;
- static calibration versus dynamic quantization;
- post-training quantization versus quantization-aware training.

Outliers force a scale to cover a wider range, reducing effective resolution for ordinary values. Quantization is therefore a data-distribution problem as well as a format conversion.

## Quantities to calculate

Basic tensor storage is:

```text
memory = number of elements × bytes per element
```

A first approximation of compression is:

```text
compression ratio ≈ original bits / quantized bits
```

Real savings must account for scales, zero points, padding, temporary buffers, and unquantized layers.

Measure error at three levels:

1. tensor level: absolute error, relative error, and cosine similarity;
2. model level: loss, perplexity, and task metrics;
3. system level: latency, throughput, memory, and cost.

## Hands-on work

1. Compare FP16 and BF16 representations of very large and very small values.
2. Run the same matrix multiplication in FP32, FP16, and BF16; measure error and time.
3. Record operator dtypes in a small model under autocast.
4. Implement simple per-tensor INT8 quantize/dequantize.
5. Compare BF16, INT8, and INT4 inference quality, memory, and throughput.

## Common misconceptions

- Fewer bits do not guarantee faster execution on a given device.
- A model running successfully does not prove its output quality is preserved.
- Mean error can hide a small number of severe outliers.
- Theoretical compression does not equal end-to-end memory reduction.
- Kernel availability, layout, and hardware support often matter more than the dtype name.

## Mastery check

- Why does BF16 have a wider dynamic range than FP16?
- Why can matrix multiplication use low-precision inputs and higher-precision accumulation?
- What problem does loss scaling solve?
- Why can per-channel scales outperform one per-tensor scale?
- What evidence would justify adopting a low-precision optimization?

Next: [Module 04 · Distributed Training](04-distributed-training.en.md)
