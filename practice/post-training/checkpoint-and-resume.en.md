# A saved model is not always a resumable run

[中文](checkpoint-and-resume.md) · **English**

> Original teaching project · Checked: 2026-10-09. Includes a CPU standard-library recovery experiment, not a complete distributed checkpoint backend.

A run is interrupted, but the weights survived. After loading them, loss jumps and data starts over. This may not be a slightly noisy recovery: you may have restored the model without restoring the experiment.

Distinguish **resuming the same run** from **warm-starting a new experiment**. A warm start may change data or reset the optimizer, but should receive a new run identity.

## Separate recovery and inference artifacts

| Artifact | Contents | Use |
| --- | --- | --- |
| Recovery checkpoint | Model/adapter, optimizer, scheduler, counters, RNG, data state, manifest; AMP scaler and other state when used | Continue interrupted training |
| Inference export | Weights or adapter plus exact base, tokenizer, template, configuration and generation conventions | Offline evaluation and serving |

A small LoRA adapter does not identify its base by itself. Record an immutable revision, not only a model name. After merging, quantization, or a backend change, rerun fixed-input checks. Generation inside the training process does not validate the exported artifact.

Recovery bundles can contain data paths and sample state; do not publish them as model exports without inspection. Load trusted artifacts only, rather than disabling deserialization safeguards to bypass compatibility errors.

## Which instant does the checkpoint represent?

The simplest boundary is after an optimizer update and scheduler update, with gradients cleared, before starting the next accumulation window.

<figure class="worked-update">
<ol>
<li><small>01 / COMPUTE</small><strong>Finish the microbatch group</strong><span>The target denominator covers the entire accumulation window.</span></li>
<li><small>02 / UPDATE</small><strong>Advance parameters and counters</strong><span>Keep optimizer, scheduler and data position consistent.</span></li>
<li><small>03 / SAVE</small><strong>Write a complete snapshot first</strong><span>Do not advertise a partial write as the latest recoverable version.</span></li>
<li><small>04 / VERIFY</small><strong>Restart and take the next step</strong><span>Compare the next sample, learning rate and updated state.</span></li>
</ol>
<figcaption>The snapshot describes completed updates and the next action—not merely model weights.</figcaption>
</figure>

A mid-accumulation save additionally needs unapplied gradients, the microstep index, and corresponding data state. This project avoids that complexity initially. Asynchronous saving also needs a consistent snapshot rather than background reads from tensors still being updated.

Data position is particularly subtle: workers may prefetch examples the model has not consumed. Recovery should follow consumed data, not the furthest read offset. Dynamic packing requires retained buffers or deterministic reconstruction. `epoch=2` alone is insufficient.

## Verify recovery in a small experiment

[training_contracts.py](code/training_contracts.py) trains a one-weight regression toy with momentum, a step-dependent learning rate, shuffled data, and random input perturbations. It is not an LLM, but it exposes missing training state.

Compare 12 uninterrupted steps with 5 steps, a save, a fresh object restored from disk, and 7 more steps. In the same Python environment, complete recovery should produce identical state. Restoring only weights or deliberately clearing momentum follows a different trajectory.

```bash
python3 practice/post-training/code/training_contracts.py
python3 -m unittest discover -s site/tests -p 'test_training_contracts.py'
```

The program writes JSON inside a temporary directory and loads no external model files. It writes a same-directory temporary file, flushes/fsyncs it, then atomically replaces the destination. This prevents readers from observing a partly written destination; **it is not a power-loss durability guarantee, object-storage protocol, or multi-rank commit protocol**.

Tests reject a missing RNG field or changed data revision, and check that a leftover temporary file does not replace the last recoverable state. Actual GPU runs additionally require per-rank CPU/CUDA RNG handling and attention to nondeterministic operators. Exact equality in this toy does not generalize to all hardware.

## What changes with multiple GPUs?

For sharded training, rank 0 cannot simply assume it holds everything. [PyTorch Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) supports distributed state I/O and compatible resharding. Tensor layout recovery does not automatically restore application-level data order or experiment configuration.

| Change | Check first | Not automatically guaranteed |
| --- | --- | --- |
| Same-topology recovery | Required shards and manifest describe the same completed step | Existing files form a complete usable checkpoint |
| Different GPU count | Backend compatibility for model/optimizer layout; recomputed DP batch | Identical sample order, dropout, or bitwise update trajectory |
| Framework upgrade | State format, parameter names and optimizer mappings | Arbitrary cross-version recovery |
| Asynchronous saving | Consistent snapshots, completed background writes, extra memory | Returning from submission means every file is durable |
| Object storage | Unique version prefixes; publish completion manifest after successful writes | Local rename semantics transfer unchanged |

Before recovery, inspect the file inventory, sizes/checksums, and completion marker. Hashes detect corruption, not trustworthiness. Keep an older valid checkpoint when a new save fails; do not delete the only usable copy first.

## Save frequency, best, and latest

Frequent saves reduce potential lost work but add pauses and storage. If a save pauses training for 30 seconds after every 10 minutes of training, its time fraction is $30/(600+30)\approx4.76\%$, ignoring other costs. Under uniformly timed failures between saves, average replay is about 5 minutes of training. These are illustrative assumptions, not a universally optimal interval.

`latest` supports continued training; `best` records the candidate selected by development-set rules. They need not be the same step. Declare metrics, frequency, and retention policy beforehand; do not select best on the test set. Retain a verified-loadable older version until its replacement is complete and checked.

Test recovery early, before the first actual device failure. Next, [compare and replace models](experiments-and-release.en.md): resumability is necessary, not evidence that a model deserves deployment.
