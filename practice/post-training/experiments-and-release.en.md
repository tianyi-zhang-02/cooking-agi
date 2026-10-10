# Compare experiments and replace models safely

[中文](experiments-and-release.md) · **English**

> Original teaching examples · Checked: 2026-10-10. Responses, pass rates, and traces below are illustrative, not model measurements. The code reproduces data processing, not model inference.

After training, the tempting question is “How much did it improve?” An aggregate move from 60% to 70% doesn't tell you which questions improved, which regressed, or whether evaluation added an extra hint.

We'll follow an experiment from saved settings to paired results, then open a failure and work out what to change. For debugging, start with [the missing-evidence example](#trace-a-failure). For a new experiment, start with the manifest below.

## Save enough information to repeat the experiment

You don't need a large platform first. A manifest saved with the results prevents many avoidable mix-ups:

```json
{
  "run_id": "citation-sft-demo-01",
  "base_revision": "local-fixture-v1",
  "data_snapshot": "examples-v3",
  "split_rule": "question-family-v1",
  "template_revision": "chat-template-v2",
  "objective": "assistant-target-token-mean",
  "eval_snapshot": "fixed-evidence-v2",
  "decode": {"sampling": false, "max_new_tokens": 128}
}
```

These are illustrative fields, not downloadable checkpoint names. Real runs also need content hashes, dependency versions, seeds, optimizer settings, training steps, and valid-token counts. A run ID indexes artifacts; it cannot replace a data snapshot.

If A gets 128 output tokens and B gets 512, the comparison can still be useful—but it compares systems with different budgets, not training alone.

## Six questions reveal what an aggregate hides

`1` means passing a predefined check; `0` means failing. Put old and new outcomes for the same question on one row:

| Question | Slice | Old | New | Change |
| --- | --- | --- | --- | --- |
| q1 | Evidence available | 0 | 1 | Win |
| q2 | Evidence available | 0 | 1 | Win |
| q3 | Evidence available | 1 | 1 | Tie |
| q4 | Evidence available | 1 | 1 | Tie |
| q5 | Evidence missing | 1 | 0 | Loss |
| q6 | Evidence missing | 1 | 1 | Tie |

Overall performance rises from `4/6` to `5/6`. But q5, previously handled correctly without evidence, now fails. That slice drops from `2/2` to `1/2`. The new model might answer better—or simply be more willing to answer. Inspect q5's actual output, then check the quantity and quality of missing-evidence demonstrations.

A **paired comparison** retains the change on each question: two wins, one loss, three ties. Six cases illustrate regression detection, not a stable improvement estimate. Real comparisons need sample sizes and uncertainty; resampling should respect related questions from the same user or document.

## The retriever found it. Why was the answer wrong? {#trace-a-failure}

Suppose someone asks, “When does North Hall close on Sunday?” The knowledge base says 18:00 on Sundays and 20:00 on weekdays. The assistant answers 20:00. It's tempting to add more date-related training examples. First, look at what happened to this request:

<figure class="worked-update" lang="en" id="failure-trace">
<figcaption>A fictional debugging trace: finding a passage doesn't mean it reached the model.</figcaption>
<ol>
<li><small>01 / Retrieved</small><strong>Both passages found</strong><span>Weekdays: 20:00<br>Sundays: 18:00</span></li>
<li><small>02 / Final input</small><strong>Only weekdays included</strong><span>That passage came first and used the evidence budget. The Sunday passage never reached the prompt.</span></li>
<li><small>03 / Illustrative answer</small><strong>“20:00 on Sunday.”</strong><span>The answer is wrong. It doesn't establish that the model cannot learn the distinction.</span></li>
</ol>
</figure>

We have found an **evidence-assembly problem**: retrieval logs contain the right passage, but the final input doesn't. Giving a confident answer without Sunday evidence is another behavior worth testing. What this case cannot show is that the model ignores correct evidence when it receives it. Both failures may exist; investigate them separately.

A screenshot of the answer isn't enough to replay the request. Keep the original question, retrieved document revisions, assembled input, actual output, and reason for the verdict. Where available, include token IDs, truncation boundaries, generation stop reasons, and tool state. Logs containing user data need access controls, redaction, and retention limits; don't paste them into public issues.

### Change one part and replay the request {#controlled-replay}

Hold the model and decoding settings fixed, then try small interventions. These are proposed experiments, not reported results:

| Change only this | If the outcome changes, investigate | What it doesn't establish |
| --- | --- | --- |
| Manually supply the Sunday passage within the same budget | Evidence selection and assembly | Better real-world retrieval, or success on every question |
| Correct an outdated evaluation answer without changing the input | An evaluation defect | An improvement in the model |
| Supply the right evidence, then change the chat template | Training/serving format mismatch | That arbitrary template changes help |
| Replace the checkpoint while keeping template, evidence, and budget fixed | Model differences under these conditions | The effect of a particular training trick |

Manually supplying the needed passage is an **oracle-context comparison**. It asks whether the downstream component can work when upstream supplies the right material. Keep that result separate from end-to-end scores. One correct answer isn't enough; repeat across cases. Continued failure isn't a capacity diagnosis either—conflicting instructions, formatting, and grading may still be wrong.

For sampled generation, keep repeated outcomes rather than selecting one successful screenshot. Greedy decoding still requires pinned versions and execution settings; hardware, kernels, or service updates can change results. Single-component changes help isolate clues, but the final combination still needs an end-to-end test.

<details markdown="1">
<summary>Reproduce the evidence-budget failure in Python</summary>

This uses `pack_evidence` from our RAG project. It counts whitespace-separated English words for easy inspection; **it is not a tokenizer**, nor a suitable counter for unsegmented Chinese. Each passage costs 7 words, and only one fits. Production budgeting must use actual tokens and reserve room for instructions, the question, role markers, and output.

Run from the repository root:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("practice/rag/code").resolve()))
from evidence_pipeline import Chunk, pack_evidence

weekday = Chunk(
    "weekday", "hours", 1,
    "North Hall closes at 20:00 on weekdays.",
    frozenset({"public"}),
)
sunday = Chunk(
    "sunday", "hours", 1,
    "North Hall closes at 18:00 on Sundays.",
    frozenset({"public"}),
)
retrieved = [weekday, sunday]
provided, used = pack_evidence(retrieved, budget=7)
oracle, oracle_used = pack_evidence([sunday, weekday], budget=7)

assert sunday in retrieved and sunday not in provided
assert sunday in oracle
assert used == oracle_used == 7
print("retrieved:", [chunk.chunk_id for chunk in retrieved])
print("provided:", [chunk.chunk_id for chunk in provided])
print("oracle:", [chunk.chunk_id for chunk in oracle])
```

The lists are `['weekday', 'sunday']`, `['weekday']`, and `['sunday']`. This demonstrates an input-selection effect, not an accuracy gain. No answer is generated here. An automatic method for choosing the relevant passage needs its own validation.

</details>

## Before training more, inspect the targets {#inspect-training}

Return to q5, where the model answers without enough evidence. Read a small batch of actual training examples, not just the config. Do missing-evidence demonstrations acknowledge uncertainty? Do similar inputs have conflicting labels? After templating and truncation, which answer tokens still contribute to loss?

Suppose the intended demonstration says, “The material doesn't give Sunday hours, so I can't confirm.” A long input truncates away the answer, leaving no valid target tokens. The run is called SFT, but that example contributes none of the intended answer supervision. An inverted mask can be equally misleading: loss falls on user questions while the answers you wanted to teach are ignored.

Decode a batch and inspect **inputs, targets, loss masks, and valid-target counts** together. Causal-LM logits and labels generally require a one-position shift; count the positions actually used by the loss. This project supervises assistant answers. That is a task choice, not a requirement for every pretraining or fine-tuning objective.

[TRL v0.29.0](https://huggingface.co/docs/trl/v0.29.0/en/sft_trainer#train-on-assistant-messages-only) requires a compatible template that returns an assistant mask for `assistant_only_loss=True`. A flag in a config is not evidence that the resulting batch is correct. See [data to batches](data-pipeline.en.md) and [masks and loss denominators](data-and-objectives.en.md).

### Loss falls, but generation doesn't improve {#loss-versus-behavior}

Start with a tiny, internally consistent training set. Check trainable parameters, nonzero finite gradients, and weight changes. If the model cannot learn those examples, inspect the objective, optimization, and implementation first. Once it can, test unseen cases. Memorization and generalization answer different questions.

- **It predicts well after correct prefixes but fails when generating.** Teacher forcing supplies the correct history; free generation may already have taken a wrong turn. Measure held-out loss and generated behavior separately.
- **It works locally but fails in the service.** Verify the base model, adapter, tokenizer, template, and stopping settings. A local checkpoint name doesn't identify the revision actually being served.
- **It repeats itself or stops early.** Inspect EOS handling, stop reasons, output limits, and the actual context before changing decoding. A higher temperature can add variation; it cannot restore missing evidence or correct a bad label.

One inexpensive check is whether templating and tokenization added special tokens twice. The [Transformers chat-template documentation](https://huggingface.co/docs/transformers/chat_templating) warns about this exact mismatch. Checking the input path gives you a clearer next step than immediately changing the training algorithm.

## Let the failure determine the fix {#choose-a-fix}

You don't have to progress through prompt changes, then SFT, then RL. If a document is stale, update it. If a tool executes twice, fix execution. Training isn't a substitute for those components doing their jobs.

| Diagnosed cause | Reasonable next experiment | Cost or risk to check |
| --- | --- | --- |
| Missing, stale, or dropped material | Version selection, retrieval, reranking, context budgeting | Larger top-k costs more and may add distracting evidence |
| Evidence is present, but behavior or format is unreliable | Clear instructions and format constraints; targeted SFT if needed | Valid JSON doesn't establish factuality; more refusal examples may increase false refusals |
| Plausible answers receive inconsistent preferences | Agree on the labeling criterion before preference learning | DPO needs meaningful comparisons, not arbitrary responses labeled rejected |
| Multi-step decisions have verifiable outcomes | Repair the environment and tools, then compare SFT/RL | Rollout cost, reward exploits, and failure recovery |
| Input and implementation check out, but the task remains too hard | A better-suited model, tools, or task decomposition | Size alone may not solve it; tools add latency and permission risks |

LoRA, QLoRA, and full-parameter updates are a separate decision. A serious failure doesn't imply that full fine-tuning is required. The [LoRA paper](https://arxiv.org/abs/2106.09685) reports competitive results on its evaluated tasks, not a guarantee across all tasks. Compare key slices and resource costs under controlled data and tuning budgets; see [training choices](training-plan.en.md).

Input and output checks have tradeoffs too. Rules are inspectable and cheap but cover limited cases. Another model offers flexibility at the cost of latency, missed violations, and false rejections. Tool permissions should be enforced reliably by the execution layer, not granted solely because a second model approves an action.

## Fix the case, then test nearby cases {#regression-families}

Once the assistant answers 18:00 correctly, try a new phrasing or change one condition. The following original examples use the behavioral test types in [CheckList (ACL 2020)](https://aclanthology.org/2020.acl-main.442/):

| Test | North Hall example | Expectation |
| --- | --- | --- |
| Minimum functionality (MFT) | Supply only the Sunday rule and ask for closing time | Handle the simplest case |
| Invariance (INV) | Rename North Hall to South Hall in both question and evidence, keeping rules unchanged | Still answer 18:00; citation IDs may change |
| Directional expectation (DIR) | Keep both rules, but change Sunday to a weekday in the question | Change the time from 18:00 to 20:00 |

Add missing evidence, conflicting rules, and irrelevant long text to test when an answer is justified. Ten cosmetic paraphrases may be less useful than one change to a condition that should alter the answer.

Keep repaired failures in a regression set to catch recurrence. Once used during development, they are no longer independent test evidence. Group close paraphrases by document or question template when splitting. Validate the final choice on untouched cases; success on a rehearsed question is not a generalization result.

Don't rank fixes by counts from a hand-collected failure folder alone. Ten complaints are not ten randomly sampled requests. Estimate frequency on data with a known denominator, then consider impact, cost, and strength of evidence. Severe permission violations need separate treatment rather than being traded away in an average score.

## Record incomplete cases separately

If the old model completes 100 questions and the new one completes only 80, averaging each model's successful responses can introduce selection bias. Our small implementation rejects mismatched IDs so incomplete pairs require explicit handling.

A real evaluator can retain `ok / timeout / invalid_output` per case and report two views:

- **Did the user get a usable result?** Use all requests as the denominator; timeouts and malformed outputs affect task success.
- **Which model answered better when both completed validly?** Compare that subset, with counts and reasons for exclusions. This is conditional performance, not an estimate automatically valid for all requests.

Both views matter. A timeout isn't a factual error, but it cannot disappear from the report. If the task requires JSON, invalid formatting can itself count as task failure.

```bash
python3 practice/post-training/code/experiment_checks.py
```

The [code](code/experiment_checks.py) prints the six-case overall and slice results. Removing a new-model result causes a missing-pair error; assigning a question family to both training and testing causes a leakage error.

## Increase cost in stages

| Stage | First check | Only then move to |
| --- | --- | --- |
| One example / batch | Tokens, masks, loss, finite gradients | Tiny-dataset training |
| Tiny dataset | Learn a few clear examples without breaking unsupported-answer behavior | More data and longer runs |
| Save and restore | Output comparisons under fixed settings, optimizer/scheduler and required random state | Long runs and interruption recovery |
| Full offline evaluation | Fixed protocol, slice regressions, resource cost | Controlled release with explicit rollback |

If a few examples cannot be learned, inspect inputs, objectives, and implementation before adding GPUs. Memorizing those examples doesn't establish generalization either; it is a training-path check.

See [training choices](training-plan.en.md), [multi-GPU batches](distributed-training.en.md), and [recovery comparisons](checkpoint-and-resume.en.md) for implementation decisions. GPU count alone is not a resource report: include valid target tokens, end-to-end time, peak memory, and failed-work replay costs.

If only offline evaluation is available, keep the conclusion offline. User simulation can stress-test behavior, but it cannot establish real-user product benefit.

## Release a compatible bundle

Weights belong with the tokenizer, template, generation settings, preprocessing, and tool/output schemas. Rolling back weights while retaining a new template may not restore the old system.

First run a shadow version on matching inputs without changing user-visible outputs, checking format, latency, and failure types. Then decide whether controlled online validation is appropriate. Shadow traffic cannot measure changed user behavior because users never saw the new outputs. Define stop conditions in advance, such as key-slice regressions, excessive invalid outputs, or latency beyond the product budget. Thresholds follow risk, not this six-question table.

“Don't release this run” is a valid conclusion. Explaining why, and what to test next, is more useful than presenting every training run as a win.

[Project introduction](README.en.md) · [Compare the recommender release path](../recommender-systems/08-serving-lifecycle.en.md)
