# Before choosing a benchmark, define the comparison

[中文](benchmark-protocols.md) · **English**

A knowledge-question score may be a poor way to choose an assistant for long reports. Knowledge, evidence retrieval, cross-document reasoning, citation, and abstention are different capabilities. One attractive average cannot establish them all.

This page is not a latest-model leaderboard. It gives a reusable process: choose the question, fix the protocol, then interpret scores. Checked 2026-10-08. Historical benchmarks remain useful for comparison, but age and familiarity do not establish present-day discriminative power.

## 1. What question does each test answer?

| Question | Example reference | What remains to test |
| --- | --- | --- |
| Subject knowledge and multiple-choice reasoning | [MMLU](https://github.com/hendrycks/test) | Open responses, citations, real tasks, contamination |
| Chinese subject knowledge | [C-Eval](https://github.com/hkust-nlp/ceval) | Conversation, domain language, mixed-language work |
| Code passing tests | [LiveCodeBench](https://livecodebench.github.io/) | Dependencies, repository edits, security, maintainability |
| Long-context lookup, tracing, aggregation | [RULER](https://arxiv.org/abs/2404.06654) | Ambiguous natural documents and actual workflows |
| More complex realistic long-input tasks | [LongBench v2](https://arxiv.org/abs/2412.15204) | Your files, costs, and user requirements |
| Helpful conversation | Human review or a calibrated judge | Separate stylistic preference, facts, and execution |

Multiple-choice accuracy is not conversational usefulness, and passing code tests does not establish security. HELM's useful framing is to evaluate [scenarios and multiple dimensions](https://arxiv.org/abs/2211.09110) together rather than rank all behavior with one number.

## 2. The same benchmark name can hide different tests

Save at least this manifest. Revisions should identify actual artifacts, not just say `latest`.

```text
model / model_revision / tokenizer_revision
dataset / dataset_revision / split / item_ids
prompt_template / few_shot_examples / chat_template
context_limit / truncation_policy / output_budget
temperature / top_p / seed / samples_per_item
tools / network_access / retries / timeouts
answer_extraction / scorer_version / invalid_output_policy
hardware / engine_version / dtype / quantization
```

Suppose one model emits an option directly while another reasons for 8,000 tokens first. The second may be more useful, but this is not a fixed-compute comparison. Report equal-budget and recommended-configuration results separately rather than silently mixing them.

Define timeout, parse-failure, and refusal handling before running. Accuracy calculated only over successful responses rewards systems that turn hard items into timeouts. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) standardizes parts of the protocol, but still requires a pinned commit, task configuration, and run settings.

## 3. A needle test starts with finding one record

Create a small task: find the value associated with `target` among many records. Move the same evidence between beginning, middle, and end while retaining the other records.

```python
def lookup_case(record_count, position, include_evidence=True):
    if type(record_count) is not int or record_count < 2:
        raise ValueError("Expected at least two record slots")
    if type(position) is not int or not 0 <= position < record_count:
        raise ValueError("Position must be inside the record slots")
    if type(include_evidence) is not bool:
        raise ValueError("Evidence flag must be boolean")
    records = [f"item-{index:04d}: value-{index:04d}" for index in range(record_count - 1)]
    evidence = "target: maple-47" if include_evidence else "unrelated: birch-82"
    records.insert(position, evidence)
    return "\n".join(records), "maple-47" if include_evidence else None

contexts = [lookup_case(9, position)[0] for position in (0, 4, 8)]
assert all(context.count("target: maple-47") == 1 for context in contexts)
assert all(set(context.splitlines()) == set(contexts[0].splitlines()) for context in contexts)
assert lookup_case(9, 4, False)[1] is None
```

This moves **record slots**, not exact model-token positions. Count with the actual tokenizer and verify that truncation has not removed the evidence. Repeated filler makes a controllable synthetic test, not a representative natural document.

| Added difficulty | Original test idea | Alternative explanation to rule out |
| --- | --- | --- |
| Position | Move identical evidence | Reliance on clues near the end |
| Multiple pieces | Find a target alias, then the alias's value | Copying one local string without linking facts |
| Aggregation | Combine records into a total | Missing or double-counting records |
| Distractors | Similar IDs, stale values, quotations | Confusing evidence with irrelevant matches |
| No answer | Remove the only evidence | Guessing a familiar answer anyway |

RULER broadens single-needle testing; LongBench v2 supplies another kind of evidence from more realistic long-input tasks. An all-green needle heatmap does not establish comprehensive context understanding.

## 4. Separate quality from cost in the result table

Cross length, evidence position, and task difficulty, with multiple distinct examples per cell. This is a recording template, not fabricated results:

| Length / position / task | Items | Accuracy | False answers on unanswerable items | Input/output tokens | TTFT / total latency |
| --- | --- | --- | --- | --- | --- |
| Short / middle / single evidence | To measure | To measure | To measure | To measure | To measure |
| Long / middle / multiple evidence | To measure | To measure | To measure | To measure | To measure |

TTFT means time to first token. Cache hits, prefill, decode, and tool waiting affect different costs and deserve separate records. An almost-empty failed response can be fast without completing the task.

Maximum accepted context, context length retaining task quality, and context length meeting a latency budget are often three different numbers.

## 5. Are two more correct answers worth a model change?

On 200 items, 160 versus 164 correct answers establishes a sample improvement of two percentage points. It does not yet identify which tasks improved, whether the change is stable, or its extra cost.

Retain paired per-item results and use [paired comparisons](metric-robustness.en.md#paired-comparison) to estimate uncertainty. Questions derived from the same document should be grouped at a sensible unit rather than treated as independent. Repeated generation measures sampling variation; independent training runs measure training variation.

Predefine the main metric, protected slices, and acceptable costs. Selecting the largest post-hoc improvement can create an unstable story. An interval containing zero does not prove equivalence, and statistical significance does not automatically justify deployment.

## 6. Contamination and evaluation updates

Old items can enter pretraining, SFT, synthetic data, or prompt demonstrations. Exact and near-duplicate checks help, but failing to find overlap does not prove cleanliness. Temporal splits require actual training and data-update dates; changing a date label does not remove leaked answers.

For a continually updated test, fix a release snapshot and date window rather than combine different periods into one supposedly comparable score. Treat a claimed model cutoff cautiously; it is not an auditable inventory of every training source.

Keep a final holdout unused for prompt changes, judge tuning, or model selection. When a benchmark changes, rerun both old and new systems; otherwise the apparent gain may come from the changed test. The code here creates test fixtures, not model calls, and no real benchmark run is claimed.
