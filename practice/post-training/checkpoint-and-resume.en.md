# A saved model is not always a resumable run

[中文](checkpoint-and-resume.md) · **English**

> Original teaching project · Checked: 2026-10-09. Includes a CPU standard-library recovery experiment, not a complete distributed checkpoint backend.

A run stops halfway through, but you saved the weights. After loading them, predictions match—yet the next training update does not. The missing piece may be the optimizer's history, the learning rate, or which batch comes next.

We will start with two updates you can calculate by hand, then work through what to save and how to test recovery. The first example explains the idea; the later sections are for building a training loop that can actually resume.

## Three uses of the word checkpoint {#checkpoint-meaning}

| Phrase | What it means | Does it resume a run? |
| --- | --- | --- |
| “Download a model checkpoint” | Obtain a version of the weights and supporting configuration | It can initialize training; full training state may be absent |
| “Save a checkpoint every 1,000 steps” | Persist the training state at that point | Only if required state is complete and compatible |
| “Enable activation checkpointing” | Retain fewer intermediate activations and recompute them during backward | This saves memory, not interrupted progress |

A **warm start** begins a new experiment from existing weights. A **resume** continues an interrupted one. Both are useful. Changing the data or resetting the optimizer is fine as a new run; it just should not be mistaken for uninterrupted continuation.

## Same weights, different next update {#momentum-restore}

Use one weight initialized to 1, a learning rate of 0.1, and momentum of 0.9. Keep the gradient at 1 on each step, with no weight decay or other corrections, so momentum is the only difference.

Each step computes $v_{t+1}=0.9v_t+g_t$, then $w_{t+1}=w_t-0.1v_{t+1}$. After the first step, velocity is 1 and weight is 0.9. Now stop and reload:

<figure class="worked-update worked-update--pairs">
<ol>
<li><small>RESTORE BOTH</small><strong>0.9 → 0.71</strong><span>Restored velocity is 1. The next velocity is 0.9 × 1 + 1 = 1.9.</span></li>
<li><small>RESTORE WEIGHTS ONLY</small><strong>0.9 → 0.80</strong><span>The new optimizer starts with zero velocity. Its next velocity is just 1.</span></li>
</ol>
<figcaption>Identical starting weights and gradients produce different updates. These are illustrative calculations.</figcaption>
</figure>

```python
def momentum_step(weight, velocity, gradient, rate=0.1, momentum=0.9):
    next_velocity = momentum * velocity + gradient
    return weight - rate * next_velocity, next_velocity

saved_weight, saved_velocity = momentum_step(1.0, 0.0, 1.0)
resumed_weight, resumed_velocity = momentum_step(saved_weight, saved_velocity, 1.0)
warm_weight, warm_velocity = momentum_step(saved_weight, 0.0, 1.0)
assert abs(resumed_weight - 0.71) < 1e-12
assert abs(warm_weight - 0.80) < 1e-12
```

Matching predictions mostly checks model state. **Taking the next update** also exercises the optimizer, learning rate, and next batch. Adam has more state to restore, but the test has the same purpose.

## Separate recovery and inference artifacts

| Artifact | Contents | Use |
| --- | --- | --- |
| Recovery checkpoint | Model/adapter, optimizer, scheduler, counters, RNG, data state, manifest; AMP scaler and other state when used | Continue interrupted training |
| Inference export | Weights or adapter plus exact base, tokenizer, template, configuration and generation conventions | Offline evaluation and serving |

A small LoRA adapter does not identify its base by itself. Record an immutable revision, not only a model name. After merging, quantization, or a backend change, rerun fixed-input checks. Generation inside the training process does not validate the exported artifact.

Recovery bundles can contain data paths and sample state; do not publish them as model exports without inspection. Load trusted artifacts only, rather than disabling deserialization safeguards to bypass compatibility errors.

For a first implementation, ask what changes if each piece is missing:

| Missing state | Possible symptom | What to compare |
| --- | --- | --- |
| Optimizer | Predictions match; the next weights do not | The momentum example above |
| Scheduler / step | Learning rate restarts or is shifted by one update | Rate actually used for the next update |
| RNG | Dropout or augmentation draws different randomness | Post-restart random values and gradients |
| Data position / packing buffer | Samples repeat or disappear | Next batch IDs, tokens, and masks |
| Data or tokenizer version | The same ID now produces different input | Versions and a fixed sample's encoding |

A small loss jump does not prove a bad restore, and a smooth curve does not prove a good one. First compare the next step under fixed conditions; across hardware or versions, distinguish numerical variation from missing state. [PyTorch's saving tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) likewise separates continued-training state from model weights alone.

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
