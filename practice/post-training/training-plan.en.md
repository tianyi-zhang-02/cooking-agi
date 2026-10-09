# Decide what the SFT experiment should change

[中文](training-plan.md) · **English**

> Original teaching project · Checked: 2026-10-09. We continue the knowledge-assistant example; this is an experiment design, not a reported model improvement.

The correct documents are in context, but the assistant often fails to cite them. With a limited training budget, should we change the model, run SFT, or improve the prompt first?

Before choosing a framework, keep a few failing inputs and identify the intended change: **how the model uses supplied evidence**, rather than memorizing facts that change daily. Those require different data and acceptance tests.

## Establish a no-training baseline

Hold questions, evidence, and generation budgets fixed. Try clearer instructions and output constraints. Requiring `answer` and `citations` can reduce missing fields, but it cannot establish that a citation supports the answer. Test format and evidence separately.

| Failure | First intervention | Why not train immediately? |
| --- | --- | --- |
| Evidence is outdated | Fix retrieval snapshots and revision selection | SFT cannot automatically discover yesterday's policy update |
| Citation fields are missing | Prompting, structured output, parsing checks | New weights may be unnecessary |
| Fields exist but cite the wrong passage | Grounded demonstrations, missing/conflicting-evidence examples | Behavior learning may help, provided labels are reliable |
| Two answers differ only in preferred detail | Define preferences and check reviewer agreement | More examples cannot reconcile contradictory standards by itself |

For this experiment, assume prompting has not reliably fixed missing or incorrect citations. Keep unanswerable questions in evaluation: otherwise the model might merely learn to invent a citation-shaped string every time.

## LoRA, QLoRA, or full fine-tuning?

SFT describes the demonstration-based objective. LoRA and full fine-tuning specify which parameters change; QLoRA also involves quantized base weights. These are not three competing objectives.

| Approach | Why consider it here? | Cost or limitation | When to reconsider |
| --- | --- | --- | --- |
| LoRA | Test whether the data helps with a smaller trainable parameter set | Rank and insertion points restrict updates; adapters require base-version management | Data issues are ruled out and reasonable tuning still underfits |
| QLoRA | Base weights dominate memory; train adapters over a quantized base | Quantization error, kernel/export compatibility; not necessarily faster than unquantized LoRA | More memory is available, or quantization hurts a critical task |
| Full fine-tuning | Allow broader parameter adaptation with adequate resources | Larger gradient, optimizer, and checkpoint costs; overfitting remains possible | Data, task shift, or results do not justify the additional cost |

A $4096\times4096$ linear layer has 16,777,216 weights. Rank-16 LoRA adds $16(4096+4096)=131{,}072$ trainable parameters, roughly 0.78% of that matrix. **This does not mean total training memory falls to 0.78%.** Base weights and forward activations still occupy memory.

QLoRA typically freezes the quantized base and trains extra parameters; it does not put every training state in 4-bit. Check quantization, compute precision, and merge support in the [PEFT guide](https://huggingface.co/docs/peft/en/developer_guides/quantization). For the mechanism, read [LoRA and QLoRA](../../05-post-training/lora-and-qlora.en.md).

## Assign responsibilities before comparing frameworks

Start with a training path whose batches are easy to inspect. Add more orchestration when rollouts, rewards, and weight synchronization are needed—not merely to make the stack look complete.

| Layer | Options | Responsibility in this project |
| --- | --- | --- |
| Model and tokenizer | Transformers | Pinned revisions, chat templates, forward passes and generation |
| SFT loop | TRL SFTTrainer or an auditable custom loop | Batch construction, loss, optimization and validation |
| Parameter-efficient updates | PEFT | Adapter configuration, trainable parameters and artifacts |
| Launch and distributed backend | Accelerate launch configuration; compatible DDP, FSDP or DeepSpeed integration | Precision, processes and state partitioning—not task correctness |
| Multi-role post-training | NeMo RL, verl | Organize training, generation and rewards; check model/backend support |
| Generation service | vLLM, SGLang | Evaluation or rollout generation, not a replacement for a backward-pass trainer |

[NeMo RL's backend guide](https://docs.nvidia.com/nemo/rl/latest/design-docs/training-backends.html) distinguishes DTensor/FSDP2 and Megatron paths. [verl](https://verl.readthedocs.io/en/latest/index.html) focuses on composing training and generation components. Neither implies a universal speed ranking: model, length distribution, hardware, and objective must be controlled. Existing team tooling matters too; reliable data and recovery paths can be worth more than a shorter launch command.

## Make the first configuration explainable

This is an **experiment manifest, not a runnable TRL configuration**. Replace placeholders with actual revisions and retain a dependency lock file. Existing mask examples in this project were checked against TRL v0.29.0; that is not a claim that it is the only suitable version in 2026.

```json
{
  "run_id": "citation-lora-pilot",
  "base_revision": "<immutable-model-revision>",
  "tokenizer_revision": "<immutable-tokenizer-revision>",
  "dataset_snapshot": "citation-demo-v1",
  "split_rule": "question-family-v1",
  "template_hash": "<sha256>",
  "objective": "assistant-target-token-mean",
  "adaptation": "lora",
  "packing": false,
  "resume_at": "optimizer-step-boundary"
}
```

Why disable packing initially? Not because it is undesirable, but because individual boundaries and labels should be easy to inspect first. Why start with LoRA? We are testing the demonstrations while limiting training-state cost, not declaring full fine-tuning pointless.

This manifest cannot choose a learning rate, rank, or maximum length for us. Inspect sequence lengths and valid target counts, run stability checks, then compare configurations within a declared development-set tuning budget. Do not repeatedly tune against the test set.

## What should pass before scaling?

| Stage | Evidence to retain | First checks when it fails |
| --- | --- | --- |
| Data inspection | Decoded inputs, target spans and provenance | Template, concatenation, truncation and labels |
| Tiny overfit | Training loss and generated responses on a few samples | Gradients, learning rate, conflicting targets; memorization is not generalization |
| Interruption/recovery | Continuous-versus-resumed sample sequences and states | Optimizer, scheduler, RNG and data cursor |
| Development comparison | Citation correctness, refusal slices, time and memory | Data coverage and objectives, not only model size |
| Fixed test evaluation | Paired results against the previous model | Decide whether to adopt, rather than continuing to tune |

Moving to RL needs its own justification: can we sample useful trajectories, distinguish outcomes with rewards, and detect reward exploitation? An unsuccessful SFT experiment does not automatically justify GRPO.

Next, turn these decisions into [data structures and batches](data-pipeline.en.md). For a multi-GPU question, jump to [distributed choices](distributed-training.en.md).
