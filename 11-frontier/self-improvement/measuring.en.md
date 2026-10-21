# Self-improvement: how to measure "AI speeding up AI"

[中文](measuring.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

"Over 80% of a lab's merged code is written by AI" and "a model can complete tasks that take humans 12 hours": do either of these claims show that AI R&D is accelerating? Neither answers the question directly. The first measures a share of code, the second measures task-level capability, and the pace of research as a whole is a separate quantity.

Start by identifying who made the measurement and its units. As of October 2026, externally checkable task-level measurements are still rising but have reached the limits of the measuring tools. Numbers for overall R&D acceleration come almost entirely from the labs themselves; every estimate of an overall progress multiplier is below 2x.

## Sort the evidence into four kinds

- **Task-level measurements**: METR's time horizon, RE-Bench and the like. Others can check them, but they measure "how long a task can be completed", not research progress.
- **Lab determinations against their own thresholds**: the "crossed / not crossed" verdicts in system cards, mostly based on internal evaluations.
- **Numbers labs self-report about themselves**: code share, agent hours, staff surveys, slopes of capability indices.
- **Statements and predictions**: see the [guide](README.en.md); not repeated here.

## The 50% time horizon: definition and fitting

Kwa et al. at METR define it in their [2025-03-18 paper](https://arxiv.org/abs/2503.14499) as the time humans typically take to complete tasks that AI models can complete with 50% success. For each model, fit against each task's human time $t$ (§3.1):

$$
P(\text{success}\mid t)=\sigma\big(\beta\,(\log h-\log t)\big),\qquad \sigma(u)=\frac{1}{1+e^{-u}}
$$

$t$ is the geometric mean time of successful human baselines on the task; $h$ and $\beta$ are fitted. At $t=h$, $\sigma(0)=0.5$, so $h$ is the 50% horizon. Continuously scored tasks are turned into success or failure with a threshold chosen to represent human performance, and each task is weighted by the inverse square root of the size of its task family, so that large families do not dominate.

<!-- paper-figure:metr-logistic-fits -->

### Work it once

A teaching example with made-up numbers: one model runs 10 times on each of four task groups.

| Human time $t$ | $x=\log_2(t/1\,\text{h})$ | Successes | Success rate $p$ | $\operatorname{logit}p$ |
| --- | --- | --- | --- | --- |
| 15 min | −2 | 9/10 | 0.9 | 2.20 |
| 1 h | 0 | 7/10 | 0.7 | 0.85 |
| 4 h | 2 | 3/10 | 0.3 | −0.85 |
| 16 h | 4 | 1/10 | 0.1 | −2.20 |

Convert success rates to logits and the model becomes a straight line in $x$:

$$
\operatorname{logit}p=\ln\frac{p}{1-p}=\beta\,(\log h-\log t)
$$

The four points are symmetric around $x=1$, so the least-squares line has slope about −0.74 (per doubling of human time) and crosses 0 at $x=1$, giving $h=2^1=2$ hours. Two hours is the geometric midpoint of 1 h and 4 h, not the arithmetic midpoint of 2.5 h, because the axis is log time. The same line gives the 80% horizon: $\operatorname{logit}0.8=1.39$, $x=1-1.39/0.74\approx-0.86$, about 0.55 h, or roughly 33 minutes. The 80% horizon is usually much shorter than the 50% one, as METR's data below also show. METR actually fits weighted maximum likelihood on individual runs; the logit line here is only for hand calculation. With four symmetric points both methods put the 50% crossing at 2 hours, while the slope and the 80% horizon differ somewhat.

### When the reading lies beyond the tasks

Take a stronger model. The 15-minute group is still 9/10; the other three groups are:

| Case | 1 h | 4 h | 16 h | Fitted 50% horizon |
| --- | --- | --- | --- | --- |
| Example above | 7/10 | 3/10 | 1/10 | 2 h |
| Stronger model | 8/10 | 7/10 | 6/10 | about 34 h |
| One more success at 16 h | 8/10 | 7/10 | 7/10 | about 108 h |

In the last two rows every group succeeds more than half the time, so the 50% point can only be extrapolated along the line, beyond the longest task. One extra success in the longest group moves the estimate from about 34 hours to about 108. That is why METR warns that readings above 16 hours are unreliable; its 2026-03-20 methods note adds that none of the suite's tasks is over 30 hours long.

<!-- widget:step-time-horizon -->

## Trend numbers, and the premises behind them

METR's 2025-03 paper reports in its [abstract](https://arxiv.org/abs/2503.14499) that the frontier horizon has been "doubling approximately every seven months since 2019, though the trend may have accelerated in 2024"; its extrapolation to month-long software tasks within 5 years is explicitly conditional. METR's [Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/), released 2026-01-29, grew the suite from 170 to 228 tasks and the tasks of 8 hours or more from 14 to 31; the post-2023 doubling time is 131 days (165 under the old TH1), and since 2024 it is 89 days (109 under TH1). METR's current [data file](https://metr.org/assets/benchmark_results_1_1.yaml) gives 128.7 days (95% CI 104.4–158.0) for 2023 onward, excluding models whose 50% point is above 16 hours.

<!-- paper-figure:metr-time-horizon -->

Converting 131 days: a year is about $365/131\approx2.8$ doublings, so the horizon would multiply by about 6.9 per year. That is arithmetic on a past trend, not a forecast. Read such numbers together with four premises:

1. METR's [Frontier Risk Report](https://metr.org/blog/2026-05-19-frontier-risk-report/) of 2026-05-19 (assessment window 2026-02-16 to 03-16, with participation from Anthropic, Google, Meta and OpenAI) says the most capable agents essentially saturated TH1.1, and that the most capable shared model's 50% point estimate was between 16 and 20 hours.
2. The same report: on tasks of 8 hours or more, at least 16% of successful runs were illegitimate on review. In METR's 2026-06-26 measurement of [GPT-5.6 Sol](https://metr.org/blog/2026-06-26-gpt-5-6-sol/), the horizon is about 11.3 hours (95% CI 5–40 hours) if cheating counts as failure, 71 hours if cheating attempts are discarded, and beyond 270 hours if cheating counts as success; METR says none of these is a robust measurement.
3. METR's 2026-03-20 [methods note](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/): correcting for noise in task-length estimates could reduce frontier 50% horizons by about 30%.
4. METR's [time-horizons page](https://metr.org/time-horizons/) has not been actively updated since 2026-09-08. The Claude Opus 4.6 reading was corrected on 2026-03-03 for a regularization mistake; the roughly 14.5-hour figure that first circulated appears only in secondary sources.

## Readings by model

From METR's data file and time-horizons page; 95% CIs in parentheses where recorded:

| Model | 50% horizon | 80% horizon | Added by METR |
| --- | --- | --- | --- |
| GPT-5 | ≈3.4 h | 38 min | — |
| Claude Opus 4.5 | ≈4.9 h | 49 min | — |
| Gemini 3 Pro | ≈3.7 h | 54 min | 2026-02-03 |
| GPT-5.2 | ≈5.9 h | 66 min | 2026-02-04 |
| Claude Opus 4.6 | ≈12.0 h (5.3–60.6 h) | 70 min | 2026-02-20, corrected 03-03 |
| Gemini 3.1 Pro | ≈6.4 h | 90 min | 2026-04-15 |
| Claude Mythos Preview (early snapshot) | ≈17.4 h (8.5–55.1 h) | ≈3.1 h (1.6–6.6 h) | 2026-05-08 |
| GPT-5.6 Sol (separate METR post) | ≈11.3 h (5–40 h), cheating counted as failure | — | 2026-06-26 |

The Mythos Preview point estimate is above the range METR itself calls reliable; the Opus 4.6 interval reaches 60 hours. With intervals this wide, the ordering of two neighbouring models often means little.

## AI R&D benchmarks

| Benchmark | Org and date | What it measures | Original headline | As of October 2026 |
| --- | --- | --- | --- | --- |
| [RE-Bench](https://arxiv.org/abs/2411.15114) | METR, 2024-11 | 7 open-ended ML research-engineering environments; 71 eight-hour attempts by 61 experts | At a 2-hour budget the best agents score 4x human experts; humans narrowly lead at 8 h and reach about 2x the top agent at 32 h | No standalone 2026 update found; tasks folded into TH1.1 |
| [MLE-bench](https://arxiv.org/abs/2410.07095) | OpenAI, 2024-10 | Medal rate on 75 Kaggle competitions | o1-preview + AIDE earns at least bronze in 16.9% of competitions | Top of the official [leaderboard](https://github.com/openai/mle-bench): Famou-Agent 2.0 at 64.44% (2026-02-23; read secondhand, not checked line by line); closed to new submissions since 2026-04-24 |
| [PaperBench](https://arxiv.org/abs/2504.01848) | OpenAI, 2025-04 | Replicate 20 ICML 2024 papers from scratch, scored by an LLM judge on 8,316 rubric leaves | Best agent 21.0%; ML PhDs (best of 3) reach 41.4% after 48 h on a 3-paper subset, against 26.6% for o1 | No 2026 numbers found; OpenAI's 2026 system cards no longer use it |
| [PostTrainBench](https://arxiv.org/abs/2603.08640) | Rank et al., 2026-03 | An agent post-trains a base LLM in 10 h on one H100 | Best agent 23.2% against 51.1% for official instruct models; reward hacking observed | A Lite version is in OpenAI's GPT-5.6 self-improvement eval suite |

## The lines labs drew for themselves

| Lab and document | Level | Gist |
| --- | --- | --- |
| OpenAI [Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) (2025-04-15) | High | Equivalent to giving every OpenAI researcher a highly performant mid-career research engineer assistant, relative to those researchers' 2024 baseline |
| Same | Critical | Capable of recursive self-improvement, i.e. fully automated AI R&D. Leading indicator: a superhuman research-scientist agent; lagging indicator: a generational model improvement (e.g. o1 to o3) in 1/5 of the 2024 wall-clock time (e.g. 4 weeks), sustained for several months. The response includes halting further development until safeguards are specified |
| Anthropic [RSP v2.2](https://www-cdn.anthropic.com/872c653b2d0501d6ab44cf87f43e1dc4853e4d37.pdf) (2025-05-14) | AI R&D-4 | Able to fully automate the work of an entry-level, remote-only researcher at Anthropic |
| Same | AI R&D-5 | Able to dramatically accelerate the rate of effective scaling: one year's effective-compute increase equal to two years at the early-2018 to early-2024 average rate |
| Anthropic [RSP v3.4](https://www-cdn.anthropic.com/files/4zrzovbb/website/0bacdc8440ea96e62a8766d99ebe1d4eea6d5f3a.pdf) (effective 2026-07-08; v3.0 was a full rewrite on 2026-02-24) | Automated R&D in key domains | Either (1) able to substitute for the entire set of research scientists and research engineers at competitive cost (within a factor of 5), or (2) dramatic acceleration of AI progress, operationalised as aggregate capability progress at double both the expected rate and the fastest sustained rate seen without significant AI contribution, over at least three model generations |
| Google DeepMind [FSF v3.0](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/strengthening-our-frontier-safety-framework/frontier-safety-framework_3.pdf) (2025-09-22; wording unchanged in v3.1, 2026-04-17) | ML R&D acceleration level 1 | Has been used to accelerate AI development so that AI progress substantially speeds up from historical rates |
| Same | ML R&D automation level 1 | Can fully automate the work of any Google team of researchers focused on improving AI capabilities, at roughly comparable all-inclusive cost |

In the v3.4 change log, Anthropic says the threshold is meant to capture the onset of "dramatic recursive self-improvement" and has proven difficult to operationalise. OpenAI said on 2026-08-18 that it will evolve its Preparedness Framework; as of 2026-10-15 no new version was found. OpenAI's June 2026 [policy blueprint](https://openai.com/index/frontier-safety-blueprint/) proposes that frontier developers share RSI-related measurements with CAISI; it is a proposal and publishes no metrics.

## Determinations in 2025–2026

Newest first for each lab. None reports crossing its AI R&D threshold, but the evidence behind the verdicts is changing: once the older automated task evals saturated, labs turned to staff surveys, trend breaks in capability indices, and new internal evals.

**Anthropic**

- [Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5-system-card) (2026-09-22): does not cross the next capability threshold. CoBench 2.1 scores 55.8% against an ≥85% bar for substituting for staff; the capability-index trend-break fit gives a 1.53x slope increase (95% range 1.20–1.82), i.e. the slope has not doubled; some internal acceleration measures have moved and are being monitored closely.
- [Claude Mythos 5.1](https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20&%20Claude%20Mythos%205.1%20System%20Card.pdf) (2026-09-01): does not cross the threshold for dramatic acceleration of automated AI R&D; the older automated evals were not run because they are saturated.
- [Claude Mythos Preview](https://www.anthropic.com/claude-mythos-preview-system-card) (2026-04-07): does not cross the "two years of progress in one" threshold, held with less confidence than for any prior model. The staff-survey uplift has a geometric mean around 4x; combined with an elasticity estimate, the overall progress multiplier is below 2x. The capability-index slope ratio lies between 1.86x and 4.3x depending on the breakpoint, and Anthropic attributes the jump to factors other than AI-accelerated R&D.
- [Claude Opus 4.6](https://anthropic.com/claude-opus-4-6-system-card) (2026-02): the rule-out case is more tenuous than for any previous model. Kernel optimisation reached a 427x speedup with a novel scaffold (threshold 300x); none of 16 surveyed staff thought it could automate an entry-level researcher.
- [Claude Opus 4.5](https://www-cdn.anthropic.com/bf10f64990cfda0ba858290be7b8cc6317685f47.pdf) (2025-11): does not cross AI R&D-4, but confidently ruling it out is becoming increasingly difficult; internal suite 0.604 against a 0.6 rule-out line; none of 18 staff thought it could automate an entry-level researcher.

**OpenAI** (system cards at [deploymentsafety.openai.com](https://deploymentsafety.openai.com/))

- GPT-6 Sol and Luna October update (2026-10-07): neither reaches High in AI Self-Improvement.
- GPT-6.1 Sol (2026-09-29): below High; internal Research Debugging 75.52%.
- GPT-6 Astra (2026-09-03): below High in AI Self-Improvement; it is the first model OpenAI rates Critical in cybersecurity. Research Debugging 78.05%, below an unpublished indicative threshold for High.
- GPT-5.6 (2026-07-09): new suite (Internal Research Debugging, KernelGen 1P, NanoGPT, PostTrainBench Lite, MLE-Bench Revised) because Monorepo-Bench saturated and OPQA had unsolvable items; below High.
- GPT-5.5 (2026-04-23): High can be ruled out.

**Google DeepMind**

- [Gemini 4 Argon](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_safety_report.pdf) (2026-10-13): neither ML R&D CCL is reached; the finding is no significant end-to-end acceleration of AI R&D. AI agents author most production code and bug fixes, with changes still under human review. The capability-index (GCI) hinge regression gives a slope of 0.091 (0.071–0.108) points per month before 2026-04-01 and 0.118 (0.053–0.171) after, with no statistically significant difference.
- [Gemini 3.7 Flash](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_fsf_report.pdf) (2026-08): internal benchmark 27% pass@1 against a 90% rule-out threshold.
- [Gemini 3 Pro](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) (2025-11): RE-Bench aggregate substantially below the alert threshold.

## Self-reported acceleration numbers come in different units

| Source (date) | Number | Unit | Caveats from the same source |
| --- | --- | --- | --- |
| [OpenAI](https://openai.com/index/research-acceleration-view-inside-openai/) (2026-09-06, self-reported) | 3.1 | Agent-workdays per human workday, mid-August 2026 | Over half of successful 4-8 hour tasks involved human intervention; compute grew significantly over the same period; people still set research priorities |
| [Anthropic Institute](https://www.anthropic.com/institute/recursive-self-improvement) (about 2026-06, date from secondary sources; self-reported) | Over 80% | Share of merged code authored by Claude, May 2026 | The authors say lines of code almost certainly overstate productivity |
| [Anthropic risk report](https://www.anthropic.com/aug-2026-risk-report) (2026-08, self-reported) | Under 2x | Speed of internal AI R&D | Task evals have saturated; confidence lower than in earlier reports |
| Anthropic Mythos Preview system card (2026-04-07, staff survey, self-reported) | About 4x, below 2x after conversion | Staff-estimated output, converted to an overall progress multiplier | The conversion uses an elasticity estimate |
| [Quoted by METR](https://metr.org/blog/2026-09-22-claude-opus-5-5/) (2026-09-22) | About 1.5x (perhaps 30% chance of 2x) | Acceleration of capability progress due to AI | From a non-public preliminary METR report; METR could not see the supporting evidence, and Anthropic reviewed and edited the summary |
| [METR note](https://metr.org/notes/2026-07-08-anthropic-researcher-uplift/) (2026-07-08) | Probably over 2x | Researcher uplift, inferred from Anthropic's self-reported 8x code merged per day | Researcher output is not the overall pace of progress |
| Google DeepMind Gemini 4 Argon report (2026-10-13, self-assessed) | No significant end-to-end acceleration | Capability-index slope | Intervals before and after the break overlap |

One lab can report "most code is written by AI", "researcher uplift is probably over 2x" and "overall, not yet 2x" at the same time without contradiction, because the units differ. OpenAI itself writes that AI research has many potential bottlenecks and that compute is another gating factor, which may matter more as other bottlenecks shrink.

## How an economics model reads the loop

Cunningham et al.'s 2026-09-14 paper [The Economics of Recursive Self-Improvement](https://arxiv.org/abs/2609.15802) argues that net acceleration depends on the product of elasticities across the feedback loops. Its calibration in §4.1: acceleration is self-sustaining only if a one-unit increase in AI capability (on the Epoch Capabilities Index) raises AI R&D productivity by at least 15%. Their rough estimate since coding agents launched is about 9%, derived from the self-reported 4x Mythos Preview uplift, which the authors say is very likely an overestimate. On this calibration, the elasticity would need to rise by about two thirds ($15/9\approx1.7$) to reach the threshold. The paper concludes that the feedback loops are not currently strong enough to generate a self-sustaining acceleration, though they appear to be strengthening.

Earlier, [Whitfill and Wu (2025-07)](https://arxiv.org/abs/2507.23181) used 2014–2024 data from OpenAI, DeepMind, Anthropic and DeepSeek to estimate how substitutable research compute and cognitive labour are. The answer depends on the specification: the baseline model finds them substitutes; a "frontier experiments" model finds them complements, which implies compute becomes a bottleneck.

## Feeling faster is not measuring faster

METR's 2025-07 [randomised controlled trial](https://arxiv.org/abs/2507.09089) (early-2025 tools) found that allowing AI increased developers' completion time by 19%, while developers had forecast a 24% reduction. METR's 2026-02-24 [follow-up](https://metr.org/blog/2026-02-24-uplift-update/) gives "a speedup of -18%" for returning developers (CI −38% to +9%) and −4% for new recruits (CI −15% to +9%); both intervals include zero, and the post does not restate its sign convention. METR calls the data an unreliable signal, partly because developers increasingly decline to work without AI. METR's 2026-05-19 report summarises its RCT as finding small (about 4–20%) productivity benefits. The Anthropic Institute authors likewise expect the true uplift to be somewhat lower than the roughly 4x median from their March 2026 staff poll.

The lesson: self-perception and controlled measurement can differ a lot, even in sign. Read survey numbers separately from controlled experiments.

## When you read a new claim, ask these first

| Ask | Why | Example from this note |
| --- | --- | --- |
| Who measured it? Can outsiders check it? | Internal data are invisible from outside | METR could not see the evidence behind the ~1.5x estimate |
| What is the unit? | Code share, agent hours, researcher uplift and overall pace of progress are different quantities | Over 80% of code, against "not yet 2x" |
| How much did humans intervene? | Degree of autonomy decides whether you can extrapolate | OpenAI: over half of successful 4-8 hour tasks involved intervention |
| How much did compute grow at the same time? | Progress may come from compute rather than AI labour | OpenAI says compute grew significantly over the period |
| Is the instrument saturated? Was there cheating? | Readings beyond the task range are extrapolations | TH1.1 saturated; at least 16% of long-task successes illegitimate |
| Compared with what? Where is the breakpoint? | Trend breaks are sensitive to that choice | Mythos Preview's slope ratio of 1.86–4.3x depends on the breakpoint |

## What to watch on this line next

These are open questions and the kind of result that would change the current reading; they are not predictions.

1. **Is there a longer external ruler?** METR said on 2026-05-19 that TH1.1 is essentially saturated, and its page has not been actively updated since 2026-09-08; no METR release called "Time Horizon 2.0" was found. A new suite with tasks well beyond 30 hours and cheating review would restore an externally checkable trend line.
2. **Can self-reported numbers be checked from outside?** OpenAI's June 2026 blueprint proposes reporting RSI measurements to CAISI; METR said on 2026-09-22 that it could not see the evidence behind the ~1.5x estimate. Public, third-party-checked measurements would change the evidential status of these numbers.
3. **Will a trend break become significant?** Anthropic's fitted slope increase for Opus 5.5 is 1.53x (2026-09-22) against a 2x bar; Google DeepMind's GCI hinge regression (2026-10-13) shows no significant difference. A statistically significant break, or a lab declaring a formal threshold crossed, would change the picture.
4. **The elasticity estimate.** Cunningham et al.'s (2026-09-14) roughly 9%, against a 15% threshold, rests on self-reported uplift. Re-estimating that elasticity from controlled measurements would either firm up "the loop is not yet strong enough" or overturn it.

Finally: [review questions](review.en.md).
