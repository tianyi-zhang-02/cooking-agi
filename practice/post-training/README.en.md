# Post-training in practice: start with a concrete failure

[中文](README.md) · **English**

> Original teaching project · Reviewed: 2026-10. Synthetic samples and loss values, not a completed SFT or RL training run.

The assistant retrieves the right policy and gets the date right, but often omits its source. You could revise the prompt, constrain the output format, or train. First distinguish **missing information from failure to use information already available**.

We keep the knowledge-assistant setting but focus on training. Daily-changing rules are poor candidates for repeated memorization through fine-tuning. More stable behaviors—citing evidence and acknowledging conflicting documents—are better candidates for learning from demonstrations.

## Start with an observable failure

Give the model the current policy and explicitly request an answer with citations. Collect inputs where citations are missing and test clearer prompts or output constraints. If those work reliably across varied questions, stop there. If failures persist, prepare targeted demonstrations.

| Observed failure | First action | When training becomes relevant |
| --- | --- | --- |
| Outdated policy in the answer | Inspect the retrieval snapshot and evidence | Behavior training cannot replace knowledge updates |
| Evidence is present but formatting fails | Clear prompts and structured-output constraints | Behavior remains inconsistent across varied inputs |
| Both answers are valid, but one is preferable | Define preferences and inspect annotator agreement | Reliable paired preferences justify preference learning |
| Multi-step tool tasks go off course | Inspect state, interfaces, and trajectories | An executable environment, useful reward, and budget make RL testable |

Supervised fine-tuning (SFT) learns from demonstrations; preference learning uses comparisons between responses; reinforcement learning (RL) optimizes a policy through sampling and rewards. These approaches overlap—for example, RLHF can use preference data to train a reward model. They are not three independent upgrade tiers. Define the target behavior before choosing data and methods.

## From failure examples to an evaluation report

<figure class="worked-update">
<ol>
<li><small>01 / FIND A FAILURE</small><strong>Evidence present, citation missing</strong><span>Keep the input and output; rule out missing information first.</span></li>
<li><small>02 / PREPARE DEMONSTRATIONS</small><strong>When to answer and cite</strong><span>Include examples with missing and conflicting evidence.</span></li>
<li><small>03 / SPLIT DATA</small><strong>Keep paraphrases together</strong><span>Keep near-identical questions out of separate training and test splits.</span></li>
<li><small>04 / CHECK THE TARGET</small><strong>Hand-check a batch</strong><span>Inspect the template, loss mask, valid tokens, and denominator.</span></li>
<li><small>05 / COMPARE</small><strong>Change one main factor</strong><span>Evaluate training on paired questions and fixed evidence.</span></li>
<li><small>06 / DECIDE</small><strong>Keep, revise, or reject</strong><span>Inspect critical slices such as missing-evidence behavior.</span></li>
</ol>
<figcaption>Keep examples, data revisions, loss checks, and evaluation results alongside the weights, so the experiment can be repeated and debugged.</figcaption>
</figure>

A checkpoint is an intermediate artifact. Without tokenizer, chat-template, and preprocessing versions, the same conversation may become different tokens on another machine. Without a fixed evaluation set, “the second run looks better” is hard to substantiate.

Reproducibility here first means reconstructing data, objectives, and evaluation—not guaranteeing bitwise equality across hardware or library versions. The [PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.8/notes/randomness.html) distinguish those conditions too.

## Change as few variables as possible

A first comparison can use two settings: the original model with the existing prompt, and a model initialized from the same weights and fine-tuned on targeted demonstrations. Evaluate both with the same prompt, evidence, and generation budget to isolate the effect of training.

Fixing the starting checkpoint and comparison settings **does not mean freezing every parameter**. Full fine-tuning or LoRA determines which parameters are updated. Changing the base model, evidence, and prompt at the same time makes the effect of SFT harder to isolate.

For LoRA versus full fine-tuning, use comparable documented tuning budgets and report actual tokens, time, and resources—not just equal epochs. Different sequence lengths can make an epoch very different computationally.

## Continue with the actual checks

In order, these notes follow one experiment from method selection to delivery. For a specific problem, jump to the relevant chapter; learning every distributed framework is not a prerequisite.

| Chapter | Work at this stage | Decision to understand |
| --- | --- | --- |
| [1 · Training plan](training-plan.en.md) | Decide whether to train; choose adaptation and framework | Whether the behavior warrants SFT and added cost |
| [2 · Data to batches](data-pipeline.en.md) | Records, group splits, templates and labels | Traceability, truncation, padding and packing |
| [3 · Data and objectives](data-and-objectives.en.md) | Hand-check masked loss and token/example means | Supervised positions and length-dependent weighting |
| [4 · Multi-GPU choices](distributed-training.en.md) | Separate capacity from speed; calculate batches and denominators | DDP/FSDP/ZeRO/TP/PP costs |
| [5 · Save and resume](checkpoint-and-resume.en.md) | Compare interrupted recovery; separate resume and export | State, save timing, and changed-topology guarantees |
| [6 · Experiments and release](experiments-and-release.en.md) | Paired evaluation and a rollback-ready bundle | Overall gains versus slice regressions; offline versus deployment evidence |

For a first pass, read 1 → 2 → 6. Before implementation, work through 3 → 4 → 5 too. Two standard-library programs check objectives/evaluation and data/batches/toy recovery without downloading a model. Real trainer integration still needs tokenizer, forward-pass, gradient, and distributed-recovery checks. Passing the teaching programs is not a completed SFT run.

For the algorithms, read [SFT](../../05-post-training/sft-and-its-ceiling.en.md) and the [RLHF introduction](../../05-post-training/rlhf/README.en.md). Those explain optimization methods; this project explains how to establish what an experiment actually optimized.
