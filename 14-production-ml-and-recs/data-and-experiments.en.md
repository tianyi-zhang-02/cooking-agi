# Data and experiments: why didn't offline gains reach users?

[中文](data-and-experiments.md) · **English**

A recommender's inputs are more than numbers. Each value carries assumptions about time, provenance, and sampling. Had the event happened? Could the service know about it yet? Did the user dislike an item, or never see it? A model can learn artifacts of data processing remarkably well without learning the intended task.

This chapter continues the [recommendation request](ranking-system.en.md) with three worked examples: time, probabilities, and experiment results. All numbers are teaching examples, not production measurements.

## Define one training example first {#labels}

Suppose the target is “a click within 24 hours of an actual impression.” At minimum, distinguish these records:

| Record | What it needs to establish | What it does not establish |
| --- | --- | --- |
| Request | Request ID, time, model and experiment versions | That an impression occurred |
| Candidate | Item ID, source, eligibility result | A negative label: not selected does not mean disliked |
| Impression | Impression ID, item ID, position and display time | A click; returning an item to a client does not prove it was displayed |
| Click | Associated impression, event time and arrival time | A new positive example for every report: reports may be retried |
| Feature | Entity, value, event time, availability time and transform version | That today's historical query returns what serving used then |

A request displaying ten items can produce ten impression examples. Reporting the same impression twice should not create another example. Fix the click-attribution rule too, or training and reporting may count different outcomes.

**An impression whose 24-hour window is still open is not yet a negative.** An impression at 23:00 yesterday still has 15 hours left at 08:00 today. Wait for labels to mature, or explicitly model delayed feedback; do not silently label immature examples as zero and evaluate on mature ones. Waiting also sacrifices fresh signal, so acknowledge that tradeoff.

Keeping join keys for debugging does not mean retaining complete personal profiles. The examples below use invented IDs. Real systems still need access controls, retention limits, and deletion procedures.

## An event can be old but unavailable {#time}

Consider reconstructing an activity-count feature for a 10:00 request:

| Snapshot | Event time | Available to serving at | Count |
| --- | --- | --- | --- |
| A | 09:50 | 09:51 | 3 |
| B | 09:58 | 10:04 | 7 |
| C | 10:02 | 10:02 | 9 |

Taking the latest event timestamp no later than 10:00 chooses B: a value of 7. But B arrived at 10:04. At decision time, the service could only use A's 3. C is too late on both clocks.

Reconstructing what was knowable requires at least:

\[
t_{\rm event}\leq t_{\rm decision},\qquad
t_{\rm available}\leq t_{\rm decision}.
\]

Among eligible snapshots, apply the same version-selection rule as serving. Here we choose the latest event time, breaking ties by latest availability. Check the TTL as well: a month-old value may satisfy both inequalities but still be unusably stale.

This is more specific than finding an old row. [Feast's point-in-time join documentation](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) describes event-time lookbacks and TTLs. It also notes that, by default, `created_timestamp_column` deduplicates records sharing an event time rather than enforcing an additional availability-time cutoff. A historical join alone does not settle late-arrival and backfill handling.

Where the deployed version and offline store support it, `filter_by_created_timestamp=True` adds that cutoff. The column must actually represent online availability, not merely arrival in an offline table. The documentation also notes that null timestamps are excluded and that not every backend supports the option.

Two approaches can cross-check each other:

- Retain effective-dated feature history and replay the selection logic. This requires tracking backfills, withdrawals, and transform versions; the cost is maintaining that history.
- Log a permitted sample of the features actually used by serving and compare them with offline reconstruction. This gives direct evidence, but coverage depends on sampling, field restrictions, and retention.

Shared transformation code removes one source of skew, not late data, missing values, or stale features. Inject each failure into a test rather than merely comparing script names.

## After negative sampling, what does the score mean? {#sampling}

Suppose all positives are retained and each negative is independently retained with probability \(q\). Conditional on the label, inclusion does not further depend on features. Let \(p\) be click probability among original impressions and \(p_s\) the probability in the sampled distribution. Conditional probability gives:

\[
p_s=\frac{p}{p+q(1-p)}.
\]

Multiply through and rearrange:

\[
p=\frac{q p_s}{1-p_s+q p_s}.
\]

If one tenth of negatives are retained, \(q=0.1\). A prediction calibrated to the sampled distribution of \(p_s=0.5\) corresponds to \(0.05/0.55\approx9.09\%\) in the original distribution, not 50%. Sampling made positives more common.

Alternatively, weight retained negatives by \(1/q\) and positives by 1 to estimate the original loss sum. Keep normalization and mini-batch construction consistent. **This does not guarantee a calibrated fitted model**: fitting error, regularization, and distribution shifts remain. Check calibration on an unsampled or appropriately weighted validation set.

| Situation | Is the fixed-\(q\) correction sufficient? | What else is needed? |
| --- | --- | --- |
| Every negative is independently sampled at the same rate | It corrects the prior shift under the stated conditions | Still check calibration and implementation |
| Popular items are more likely to be retained | No | Feature-dependent inclusion probabilities and support |
| Only model-selected hard negatives are retained | A fixed \(q\) generally does not apply | An explicit selection mechanism; keep representative evaluation data |
| Higher positions get more exposure and clicks | No | The display policy and position bias are a different selection process |

Multiplying negative weights by ten does not remove position bias. Estimating the effect of a different display policy also requires the behavior policy, adequate coverage, and the relevant off-policy evaluation assumptions; see [offline RL](../05-post-training/deep-rl/offline-and-ope.en.md). Without them, state the limitation rather than describing click prediction as a causal effect.

## Overall CTR rose. Who got worse results? {#experiment}

Here are invented experiment counts. Each arm has 20,000 impressions, with new users accounting for one tenth:

| Segment | Control: clicks / impressions | New model: clicks / impressions | CTR change |
| --- | --- | --- | --- |
| New users | 100 / 2,000 = 5% | 60 / 2,000 = 3% | −2 percentage points |
| Returning users | 900 / 18,000 = 5% | 1,140 / 18,000 ≈ 6.33% | +1.33 percentage points |
| Overall | 1,000 / 20,000 = 5% | 1,200 / 20,000 = 6% | +1 percentage point |

The new model gained 200 clicks overall but lost 40 among new users. This is not Simpson's paradox: the segment proportions are unchanged, and there is no common within-segment direction reversed by aggregation. It simply shows that **an overall average can hide different effects on different groups**.

These counts support descriptive arithmetic, not significance or long-term benefit on their own. With user-level randomization, repeated impressions from the same user are correlated. Do not treat all 40,000 impressions as independent trials for a confidence interval. Preserve analysis data at the assignment unit and use uncertainty estimates appropriate to the design; also check allocation ratios, missing logs, and cross-arm contamination.

| Stage | What it can tell you | What it cannot establish |
| --- | --- | --- |
| Offline replay | Output quality and slice regressions on the same inputs | How users behave after seeing new results |
| Shadow traffic | Compatibility, resource use, and latency on real inputs | That outputs never displayed to users improve clicks |
| Canary release | Obvious service failures at small exposure | Causal attribution if traffic was not randomized |
| Randomized experiment | Effects under the specified experiment design | Equal effects across every population, season, and future period |

Specify the primary metric, important slices, guardrails, and stopping rule beforehand. Repeatedly checking and stopping as soon as the result looks positive requires an appropriate sequential-analysis procedure, not an unchanged fixed-horizon test. Continue with [controls and slices](../07-evaluation/ablation-and-slices.en.md); [Microsoft's large-scale experimentation paper](https://exp-platform.com/large-scale/) also discusses common interpretation problems.

## Separate possible causes before changing the model {#diagnosis}

| Observation | A testable explanation | Evidence to inspect next |
| --- | --- | --- |
| Offline improves; new users decline online | More missing features or poorer candidate coverage | Missingness and retrieval coverage by user stage |
| Clicks fall without changed model outputs | More timeouts, slower display, or lost logs | Request-to-impression traces, end-to-end latency and loss rates |
| A launch starts well, then deteriorates | Freshness, index updates, or traffic changed | Hold the model version fixed; replay by time and compare inputs |
| Validation looks good; recent traffic does not | Time leakage, near-duplicate examples, or distribution shift | Availability timestamps, deduplication keys, and time splits |

Do not change the model, label window, and index together and leave only “results improved.” Preserve the data snapshot, transform version, model, experiment assignment, and evaluation rules for each comparison. Otherwise the next iteration cannot tell which change was worth keeping.

## Run a check without training a model {#exercise}

The small program in the repository uses the same time and sampling examples. The site test below requires Python 3.11 or newer:

```bash
python3 14-production-ml-and-recs/code/data_checks.py
python3 -m unittest discover -s site/tests -p 'test_public_learning_routes.py'
```

It should select feature value 3, correct sampled probability 0.5 to about 0.0909, and reproduce overall CTR of 5% → 6% with new-user CTR of 5% → 3%. The time example handles one feature for one entity; real data must first be grouped by entity keys and feature definitions. It checks arithmetic under the stated assumptions; it does not train a recommender or run an A/B test.

Then change one condition. Move B's availability to 09:59: the selected value should become 7. Shorten the TTL to five minutes: the original example should return missing rather than silently reuse 3. Real pipelines need small examples that fail this way too.

If data and experiments agree but the service is still unreliable, continue to [AI Infra](../15-ai-infrastructure-and-distributed-systems/README.en.md). Queueing, caching, and fallback behavior also change what users actually receive.
