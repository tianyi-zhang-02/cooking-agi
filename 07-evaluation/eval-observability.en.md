# Taking evaluation online: observability for LLM applications

[中文](eval-observability.md) · **English**

> Reading time: ~14 min · Last reviewed: 2026-10

A new prompt scores 92 on the offline eval set. Two weeks after launch, complaints that answers "miss the question" go up. The offline set did not change; the traffic did: new topics, new languages, longer conversations, and perhaps a provider that swapped the model behind the same alias. Production has no reference answers, and nobody can read a hundred thousand requests a day. **Online evaluation has to answer a question you can compute: if you judge only a small fraction of requests, how large a regression can you detect, how quickly, and with what confidence? The answer is usually worse than intuition suggests, especially for small slices.**

## How offline eval sets differ from production traffic

| Aspect | Offline eval set | Production traffic |
| --- | --- | --- |
| Inputs | Fixed, known distribution | Drifts over time; long-tail and new topics keep appearing |
| Reference answers | Usually available, or an executor can verify | Almost never; only reference-free checks are possible |
| Versions | Pinned while the eval runs | Prompt, index, tools and the model behind an alias can all change |
| Context | Mostly single-turn and clean | Multi-turn, with memory and retrieved results, sometimes truncated |
| Scale | Hundreds to thousands of items, all of them scored | Tens of thousands a day or more; only a sample can be scored |
| Feedback | Labels from annotators | Sparse, biased user behavior: thumbs-down, regenerate, abandoning midway |

So online evaluation does not replace offline evaluation. It does two things the offline set cannot: it finds failures the offline set never covered, and it estimates whether quality is moving when there are no reference answers. The failures it finds go back into the offline set as regression items that every future change has to pass.

Also keep monitoring and experiments apart. Monitoring answers "did quality change?"; an A/B experiment answers "did this change cause it?". A curve that drops on release day may be caused by the release, or by a shift in traffic on the same day. Attribution still needs a [controlled comparison](ablation-and-slices.en.md).

## What to record for each request

[Agent observability](../06-systems/agent-observability.en.md) already covers how a trace links model, retrieval and tool calls, and how to replay and redact it. Here the only question is: **which fields must survive so this request can be evaluated later?** In other words, whatever a judge or a person needs to understand the answer again: which prompt template and version, which model and parameters, what was retrieved, which tools were called, what came out, how much time and how many tokens it took, and what the user did afterwards.

OpenTelemetry (OTel) defines GenAI semantic conventions for these fields. The names are still moving, so first their status as of 2026-10-15:

- The conventions have status **Development** (OTel used to call this level Experimental). Per the Development section of [OTel's versioning and stability document](https://opentelemetry.io/docs/specs/otel/versioning-and-stability/#development), this level allows breaking changes, and "Long-term dependencies SHOULD NOT be taken against signals in Development."
- Starting with core semantic conventions [v1.42.0](https://github.com/open-telemetry/semantic-conventions/releases/tag/v1.42.0) (2026-06-12), all `gen_ai.*` definitions are deprecated and moved to the separate [semantic-conventions-genai repository](https://github.com/open-telemetry/semantic-conventions-genai); from v1.42.0 through the latest core release, v1.44.0 (2026-08-04), both the core repository and the [GenAI page on opentelemetry.io](https://opentelemetry.io/docs/specs/semconv/gen-ai/) carry only a "moved" notice.
- As of 2026-10-15 the new repository has no release or tag at all; its CHANGELOG has only an "Unreleased" section, which already holds about 10 breaking changes.

So the table below has two kinds of information: the names in v1.41.0, the last core release with complete GenAI documents ([spans](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-spans.md), [metrics](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-metrics.md), [events](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-events.md); v1.41.1 left these documents unchanged), and, in the notes column, known changes on the new repository's main branch (as of 2026-10-14, unreleased).

| What to record | Name in v1.41.0 | Notes (including changes on main) |
| --- | --- | --- |
| Operation type, provider | `gen_ai.operation.name`, `gen_ai.provider.name` | Operation values include `chat`, `retrieval`, `execute_tool`, `invoke_agent`, `embeddings` and others |
| Requested model, model that actually answered | `gen_ai.request.model`, `gen_ai.response.model` | The spec's examples are `gpt-4` and `gpt-4-0613`; record both, or you cannot see a version change behind an alias |
| Generation parameters | `gen_ai.request.temperature`, `gen_ai.request.top_p`, `gen_ai.request.max_tokens` | Record the values actually sent |
| Prompt template and version | `gen_ai.prompt.name` exists only in the attribute registry and is not attached to any span | Main adds `gen_ai.prompt.version`; use custom attributes until it is released |
| Conversation | `gen_ai.conversation.id` | Cluster multi-turn analysis by conversation |
| System instructions, input and output text | `gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.output.messages` | All Opt-In; the spec says they should not be captured by default |
| Retrieval | Span for the `retrieval` operation; `gen_ai.data_source.id`; `gen_ai.retrieval.documents` | Document content is Opt-In |
| Tool calls | Span for the `execute_tool` operation; `gen_ai.tool.name`, `gen_ai.tool.call.id` | Arguments and results, `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result`, are Opt-In |
| Finish reason | `gen_ai.response.finish_reasons` | A rising share of `length` often means truncation |
| Tokens | `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`; histogram `gen_ai.client.token.usage` | Main removes this histogram in favour of per-direction metrics such as `gen_ai.client.inference.usage.input_tokens` and `…output_tokens` |
| Latency | Span duration; metrics `gen_ai.client.operation.duration`, `gen_ai.client.operation.time_to_first_chunk` | On main, inference calls use `gen_ai.client.inference.duration` and `gen_ai.client.inference.time_to_first_chunk` instead |
| Errors | `error.type` | This attribute is Stable |
| Evaluation result | Event `gen_ai.evaluation.result`: `gen_ai.evaluation.name`, `gen_ai.evaluation.score.value`, `gen_ai.evaluation.score.label`, `gen_ai.evaluation.explanation` | Parent it to the evaluated span; when the span is unavailable, correlate via `gen_ai.response.id` |
| Cost, user feedback | No standard attribute | Compute cost downstream from token counts and a dated price table; record feedback as a custom event linked by trace id or response id |

Three practical points. First, record which version of the conventions you emit, and keep a mapping layer between dashboard queries and raw fields; names may change when you upgrade instrumentation (v1.41.0 says instrumentations that already emitted v1.36.0 or earlier keep the old names by default and switch to the latest experimental version only when `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` is set). Second, capture model name, parameters and prompt version when the request is sent; reconstructing them from configuration afterwards tends to go wrong. Third, a judge call is itself a GenAI call: record its model, parameters and rubric version too, which the section on drift relies on.

## How online evaluation runs: from sampling to a regression set

```text
production traffic ──(100%)──► errors, latency, tokens, cost, finish reasons: every request
    │
    └─(random sampling by trace id)──► traces to evaluate
                                 ├─ deterministic checks: schema, citation IDs, tool args, PII detection
                                 └─ LLM judge: grounded? answered the question? should it have refused?
                                          │
                       score events ──► aggregate by slice ──► alerts
                                          │
           random audit ─┬─ judge unsure or high risk ─┬─ negative user feedback
                         └──────► human annotation queue ◄──┘
                                          │
                   confirmed failures ──► regression set ──► offline eval and CI
```

### Sample at random first, then add targeted samples

Estimating a proportion such as a pass rate needs a sample drawn from the same distribution as all traffic. The most common form of head sampling in the OTel docs is consistent probability sampling on the trace id: a trace is either selected whole or not at all. Tail sampling that keeps errors or slow requests is good for debugging, but it changes the distribution, and rates computed from it are biased; the 10/19 versus 10/100 example in [Agent observability](../06-systems/agent-observability.en.md) shows exactly this. Both kinds of samples can coexist as long as they are labeled separately and only the random one feeds rate estimates.

Decide the sampling unit too. Requests from the same user or conversation are correlated; sampling and computing intervals per request overstates the effective sample size.

### What to check without reference answers

Production allows reference-free checks. Run the deterministic ones first: is the JSON valid, are the cited document IDs among the retrieved results, do tool arguments pass the schema, does the output contain likely personal data, is `finish_reasons` equal to `length`. These are cheap and deterministic and can run on 100% of traffic.

Hand what rules cannot express to a judge: are the answer's claims supported by the context retrieved for this request (which must be recorded, or retrievable by reference), does it address what the user actually asked, did it refuse when it should have. Give each judge one criterion and score them separately; see [layered evaluation](evaluation-stack.en.md) and [LLM-as-a-Judge](llm-as-a-judge/README.en.md).

### Which traces go to human review

There are usually four entry points: the judge is unsure or the score sits near the threshold; a high-risk category (money, health, sending something externally); the user gave a thumbs-down or asked for a human; and **a random audit that does not depend on the judge's verdict**.

The first three select samples that "look problematic", so they only tell you how many flagged samples are truly bad. To estimate how many bad answers the judge let through, you must randomly pick some of the samples the judge passed and have a person read them. Without a random audit, the judge's miss rate stays unknown. For when human involvement is worth it, see [Human-in-the-loop](../06-systems/human-in-the-loop.en.md).

### Turn failures into regression items

A failure confirmed by a reviewer enters the offline set in these steps:

1. Pin what reproduces it (the input, a context snapshot or reference, the version of each component), redact it, and confirm it may be copied out of production;
2. Write a checkable expected behavior, such as "must cite the 2026 policy", not just a reference answer; tag the slice and failure type, and deduplicate;
3. Put part of it in the CI regression set and keep part as a holdout that is never used for tuning.

Do not delete the item after the fix; but tuning prompts against the same items again and again slowly turns the regression set into a training set. For how to split data, see [judge calibration](llm-as-a-judge/calibration.en.md); for a CI gate that blocks regressions, see [eval reports and CI](../practice/web-lab/eval-report.en.md).

## Compute it once: how large a regression can 1% sampling see?

All numbers below are made-up teaching assumptions:

- 100,000 requests a day; 1% are randomly sampled by trace id for the judge, so $n=1000$ are judged per day.
- On a human-labeled calibration set, among answers humans pass, the judge also passes a share $s=0.95$; among answers humans fail, the judge also fails a share $t=0.85$.
- By the human standard, the true pass rate is $p=0.90$.

First, a single "the judge agrees with humans 94% of the time" is not enough. At $p=0.90$ agreement is $0.90\times0.95+0.10\times0.85=0.94$, but agreement moves whenever the pass rate moves. Correcting the online estimate requires $s$ and $t$ separately.

The pass rate the judge sees, $q$, is not $p$:

$$
q = s\,p + (1-t)(1-p) = 0.95\times0.90 + 0.15\times0.10 = 0.87.
$$

Solving for $p$ gives the classic correction for estimating prevalence from a screening test ([Rogan & Gladen, 1978](https://pubmed.ncbi.nlm.nih.gov/623091/)):

$$
\hat p = \frac{\hat q + t - 1}{s + t - 1},\qquad
\mathrm{SE}(\hat p) \approx \frac{1}{s+t-1}\sqrt{\frac{\hat q(1-\hat q)}{n}}.
$$

$\hat q$ is the share of the day's judged samples that the judge passed. The denominator $s+t-1=0.80$ means that when the true pass rate moves by 1 point, the judge sees only 0.8 points, so the error is inflated by $1/0.8=1.25$. Plugging in $\hat q=0.87$: $\hat p=(0.87+0.85-1)/0.80=0.90$, $\mathrm{SE}(\hat p)\approx0.0106/0.80=0.0133$, and the 95% interval is about 90% ± 2.6 points.

How small a regression can you detect? Comparing today with yesterday, 1,000 samples each, the standard error of the difference is $\sqrt2\,\mathrm{SE}(\hat p)$. With a two-sided 5% test and 80% power, the minimum detectable effect (MDE) is roughly:

$$
\mathrm{MDE}\approx(1.96+0.84)\sqrt{2}\,\mathrm{SE}(\hat p)=2.8\times1.414\times0.0133\approx0.053.
$$

So the true pass rate has to fall from 90% to about 84.7% before a day-over-day comparison alarms with 80% probability. Now change one condition at a time:

| Setup | Judged per day | 95% interval on the pass rate | Day-over-day MDE |
| --- | --- | --- | --- |
| All traffic, 1% sampled | 1,000 | ±2.6 points | about 5.3 points |
| All traffic, 5% sampled | 5,000 | ±1.2 | about 2.4 |
| A slice with 2% of traffic, 1% sampled | 20 | ±18 (normal approximation no longer reliable) | about 37 |
| Same slice, 5% sampled | 100 | ±8.2 | about 17 |
| Same slice sampled separately at 25% | 500 | ±3.7 | about 7.4 |

```python
from math import sqrt

def judge_monitor(n, q, s=0.95, t=0.85, z=1.96, z_power=0.84):
    """n judged traces; q = judge pass rate; s, t = judge agreement on human pass / human fail."""
    j = s + t - 1                         # share of a true difference that survives the judge
    p_hat = (q + t - 1) / j               # misclassification-corrected pass rate
    se = sqrt(q * (1 - q) / n) / j        # sampling error only; s and t treated as known
    mde = (z + z_power) * sqrt(2) * se    # day over day, two-sided 5%, about 80% power
    return p_hat, z * se, mde

rows = {n: judge_monitor(n, q=0.87) for n in (1000, 5000, 20, 100, 500)}
assert abs(rows[1000][0] - 0.90) < 1e-9
assert round(rows[1000][1], 3) == 0.026 and round(rows[1000][2], 3) == 0.053
assert round(rows[5000][2], 3) == 0.024 and round(rows[20][2], 2) == 0.37
for n, (p_hat, half_width, mde) in rows.items():
    print(f"n={n:5d}  p={p_hat:.3f} ±{half_width:.3f}  MDE≈{mde:.3f}")
```

The table shows three things.

**Five times the sampling rate narrows the error only by $\sqrt5\approx2.2$.** Judge cost grows in proportion to the sampling rate; precision improves only with the square root.

**Small slices are nearly invisible.** A slice with 2% of traffic, say one language or one new feature, gets only 20 judged samples a day at 1% sampling, and the normal approximation breaks down: with 17 of 20 passing, the Wilson 95% interval for the judge pass rate runs from 64% to 95%. Suppose this slice falls from 90% to 60%. The overall pass rate drops by only $0.02\times30=0.6$ points, far below an MDE of 5.3 or even 2.4, and the overall curve stays almost flat. This is the problem from [population mixtures](metric-robustness.en.md#population-mix) and [weighted slices](ablation-and-slices.en.md), as it shows up in production.

**Stratified sampling puts the budget where it matters.** Sample this slice separately at 25% (500 a day) and the other 98% of traffic at 1% (980 a day): 1,480 judge calls a day in total, and the slice's MDE drops to about 7.4 points. The price is that the overall pass rate must be weighted by true traffic shares: the slice is about one third of the sample but only 2% of traffic, so averaging the 1,480 samples directly gives it about 17 times its real weight.

These numbers cover sampling error only. Not yet included:

- $s$ and $t$ were themselves estimated from a finite calibration set. If $t$ came from 100 samples humans failed, its standard error is about 0.036; the sensitivity of $\hat p$ to $t$ is $(s-q)/(s+t-1)^2=0.125$, which works out to about 0.45 points. This mostly shifts the absolute level; as long as $s,t$ stay fixed, most of it cancels in a day-over-day comparison.
- The judge's $s,t$ may differ by slice. Correcting a slice in a language the judge handles poorly with global $s,t$ can shift the whole estimate.
- Requests from the same user or conversation are correlated, so the effective sample is smaller than the count; estimate with clustering by user or conversation, see [metric robustness](metric-robustness.en.md).
- Watch 20 slices and 5 metrics at once and a few will look "significant" every day even when nothing changed.

An alternative does not rely on fixed $s,t$: in each reporting period, have people label a small part of the random sample, and subtract the judge's average error on that small labeled part from the judge's average on the large sample. This is the mean estimator of [prediction-powered inference](https://arxiv.org/abs/2301.09633) (Angelopoulos et al., 2023); the paper's setting has the human-labeled samples and the judged traffic drawn independently from the same distribution (a random audit satisfies this), and under that assumption the intervals are valid without assuming anything about how accurate the judge is. The closer the judge is to humans, the narrower the interval. The cost is fresh human labels every period.

<!-- paper-figure:ppi-intervals -->

## Which signals should alert, and how fast

| Signal | Computed on how much traffic | How to alert |
| --- | --- | --- |
| Error rate (grouped by `error.type`), format and tool-argument validation failures | All | Minute-level thresholds |
| Latency p50/p95, time to first chunk when streaming | All | Minute-level, split by model and route |
| Tokens and cost per request | All | Hourly; also track cost per successful task, which rises when retries increase |
| Share of `length` finish reasons | All | A rise often means truncation or longer outputs |
| Refusal rate, safety flag rate | All (rules or a small classifier) or sampled (judge) | Two-sided: too high is over-refusal, too low may mean a failing guard; a single safety flag may need human review |
| Judge pass rate (corrected) | Sampled | Daily or weekly windows, thresholds from the MDE in the previous section |
| Share of thumbs-down and regenerate | All, but sparse and biased | Watch the trend; no thumbs-down does not mean satisfied, see [from feedback to training objectives](../01-data-and-feedback/feedback-to-objectives.en.md) |

Signals computed on every request can alert quickly. Quality signals from a judge are sampled and delayed, so their thresholds should come from the calculation in the previous section, not from a round number. Every slice you want to watch on its own needs its own sample-size calculation; otherwise its alert is noise.

## Judges drift too

Continue with the numbers from the worked example. A month later, the judge's provider updates the model behind the same alias, and the new version is more lenient on bad answers: $t$ drops from 0.85 to 0.75. At the same time the product really does regress a little, with the true pass rate falling from 90% to about 88.6%. The judge now sees $0.95\times0.886+0.25\times0.114\approx0.870$, still 87%. The two changes cancel and the curve stays flat.

<!-- widget:step-judge-sampling -->

What you can do:

- Pin the judge's model version (record `gen_ai.response.model` of the judge call), the rubric version and the parsing rules. If any of them changes, treat it as a new ruler: rebuild the baseline instead of splicing the before and after curves into one.
- Keep an anchor set: a fixed batch of human-labeled samples, re-judged on a schedule and whenever anything about the judge changes. Report $s$ and $t$ separately, not just agreement.
- Put enough failing samples in the anchor set. Random traffic is only 10% failures, so 200 anchors give about 20 for estimating $t$, with a standard error near 0.08, which cannot tell 0.85 from 0.75. Deliberately include more failing samples, and fill in per slice.
- A fixed anchor set goes stale. Keep adding new items from recent random audits; this is the fresh audit in [judge calibration](llm-as-a-judge/calibration.en.md). For the perturbations that can push a judge off course, see [bias checks](llm-as-a-judge/bias-and-workflow.en.md).

## Privacy and retention: what not to log

The OTel GenAI conventions take a conservative default: system instructions, user inputs and model outputs are treated as sensitive and large, and instrumentations **should not capture them by default** but let users opt in; for production environments with high telemetry volume or sensitive data, the recommended pattern is to store content externally and record only references on spans (see the section on capturing instructions, inputs and outputs in the v1.41.0 [spans spec](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-spans.md)). [OTel's guidance on sensitive data](https://opentelemetry.io/docs/security/handling-sensitive-data/) likewise asks you to collect only data that serves an observability purpose.

In online evaluation, the common traps are:

- **What should not be logged.** Credentials, session tokens and API keys (often hidden in tool arguments, request headers and URLs), payment and identity-document data, and contact or health details unrelated to the evaluation. Tool arguments and results are free text, so filtering top-level fields is not enough.
- **Judges and reviewers are data recipients too.** Sending a conversation to a judge sends it to one more model service, and its explanation may quote personal data verbatim; redact inside the annotation queue, and limit who can see raw text per task.
- **Tiered retention.** Keep aggregate metrics long and raw text and full traces short. Samples copied into a regression set escape the original retention clock, so redact them first and record their origin; when a user asks for deletion, traces, judge events, annotation queues and regression sets must all be findable.
- **Check what your tools actually do.** For example, as of October 2026 Langfuse's [licensing page](https://langfuse.com/self-hosting/license-key) lists Data Retention Policies and Server-Side Data Masking as self-hosted features that require an enterprise license; with the open-source edition, those must be handled in the application or SDK.

When choosing tools, first separate the categories: instrumentation and collection (OTel SDKs and the Collector), a general tracing backend (such as [Jaeger](https://www.jaegertracing.io/docs/latest/)), an LLM platform that runs judges on sampled production traces (such as [Langfuse](https://langfuse.com/docs/evaluation/get-started/online)), and an offline eval runner that plugs into CI (such as [promptfoo](https://www.promptfoo.dev/docs/intro/)); these are examples, not recommendations. Then ask questions you can verify: which version of the GenAI conventions does it accept? Where is content stored, who can see it, and for how long? Can the judge sampling rate be set per slice? Can annotation results be exported as a regression set?

## Finally: what each signal can and cannot tell you

| Signal | What it tells you | What it cannot tell you |
| --- | --- | --- |
| Deterministic checks (schema, citation IDs) | This response clearly broke a specific rule | That responses passing the checks are correct |
| A single judge score | Whether, under this rubric, this response probably has a problem | That this response is definitely right or wrong |
| Corrected, sampled judge pass rate | Overall quality changes larger than the MDE | Regressions in small slices; whether the judge itself changed |
| Random human audit | The judge's miss and false-alarm rates; an unbiased quality estimate | Anything cheap, fast, or covering every slice |
| Triggered human review | How many flagged samples are truly bad | The overall failure rate, or what the judge missed |
| Thumbs-down, regenerate | That some users are unhappy, and the trend | Whether the silent majority is satisfied |
| One complete trace of a failure | Which steps this failure went through, and a hypothesis for why | How common this kind of failure is |

Read next: [Agent observability](../06-systems/agent-observability.en.md) · [Judge calibration](llm-as-a-judge/calibration.en.md) · [Metric robustness](metric-robustness.en.md) · [Evaluation overview](README.en.md)
