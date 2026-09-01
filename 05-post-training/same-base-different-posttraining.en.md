# How far one base model can go

[中文](same-base-different-posttraining.md) · **English**

> Reading time: ~6 min · Type: case study · Last reviewed: 2026-08

## In one sentence

Post-training is hard to price on its own, because it almost always ships alongside a bigger base and more pretraining data, and you can't tell which side the gains came from. GLM-5.3 offers a rare control: **the base weights are identical to 5.2; only post-training changed.** That gives "what is post-training worth" a readable number for once — and a more interesting conclusion: **most of the gain came from infrastructure, not from the algorithm.**

## Why this is a natural experiment

The vendor's stated position: parameter count, pretraining corpus, context length, and base weights are all unchanged from the previous version. No new architecture, no new pretraining run.

That is exactly the control group we normally don't get. **Usually you see "the new model is better" while base, corpus, and post-training all changed at once**, so any attribution is guesswork. Here two of the three are pinned.

⚠️ One qualifier up front: the numbers below are the vendor's own, not independently reproduced. Read them with that in mind.

## The numbers

| Benchmark | 5.2 | 5.3 |
| --- | --- | --- |
| Terminal-Bench 3.0 | 4.6% | 28.3% |
| DeepSWE v1.1 | 46.2% | 66.9% |
| ExploitBench | 24.4% | 54.4% |
| CyberGym | 77.2% | 84.5% |

Meanwhile token consumption on comparable tasks fell from roughly 96k to roughly 50k — **accuracy went up while output got shorter**. That line carries more information than the accuracy itself: it says the change isn't "thinking longer so getting it right more often," it's a cleaner path.

## But read that first row carefully

4.6% → 28.3% is a fivefold relative gain. It's the most striking number here and **the least informative one**.

A baseline of 4.6% on a benchmark usually means the model was **essentially never trained on that kind of task** — terminal operation and long-horizon agent work don't show up naturally in general conversational data. Going from "almost none of this capability" to "some of it" is far easier than going from 46% to 67%.

So the correct reading is: **a huge relative gain on a near-zero baseline tells you about a coverage gap, not about method strength.** What actually measures the method is the mid-baseline move — 46.2 → 66.9, twenty points. That one is solid.

Generalized: **look at the absolute baseline before you look at the relative gain.** The closer the baseline is to zero, the less the relative number means.

## What they actually changed is the interesting part

Of the three published changes, only one is arguably a "training method." The other two are infrastructure:

**One: asynchronous RL rollout.** Parallelize and decouple sampling from agent environments. This doesn't touch the objective; it changes **how much experience you can collect per unit time**. For long-horizon tasks, sampling is the bottleneck — a terminal task runs for minutes, and under synchronous waiting the GPUs idle through most of it.

**Two: context compaction for long sessions.** Multi-hour coding sessions overflow the context. Compaction exists so long-horizon tasks **can finish at all**, not so they finish better.

**Three: training-inference alignment.** This one deserves its own section, below.

**Not one of them is a new loss function.** For the instinct that post-training means "pick a better algorithm," this is a counterexample.

## Training-inference alignment: probably the important one

The stated change is driving the log-probability divergence between the training and inference sides down to the $10^{-7}$ range. It reads like engineering fastidiousness. It isn't — it touches the foundation of the RL loop.

Rollout usually runs on an inference engine; training runs in a training framework. The two differ in kernel implementations, precision, and fusion, so **the same text through the same model can produce different logprobs on each side**.

That matters because the importance ratio is:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

The denominator is **reported by the inference engine**; the numerator is **computed by the training framework**. When the two carry a systematic numerical offset, the ratio absorbs a factor that has nothing to do with the policy changing — it comes from the two implementations disagreeing.

The consequence: you believe you are on-policy, while a gap you never modeled sits between the sampling policy and the optimized one. Clipping still runs, but it clips a contaminated ratio. **And nothing errors** — loss descends normally, metrics move normally, and every update carries a small systematic tilt.

This is the same species of problem as in [how big is your negative pool](../practice/recommender-systems/negative-pool-size.en.md): **numerically correct, semantically no longer what you think it is, and silent.**

Closing that gap to $10^{-7}$ doesn't show up in any formula. It shows up as **every previous update finally meaning what it claimed to mean.**

## What this experiment shows

**One: post-training's ceiling is higher than most people assume.** Same base, multiples on a specific task family. The base sets the upper bound on capability, but most models sit well below their own bound.

**Two: capability gets *exposed*, not *injected*.** Their account is putting the model into ten times more long-horizon task environments. Much of the capability already existed in the base; post-training made it reliably callable. That's also why the lowest-baseline task moved most — the capability had rarely been elicited before.

**Three: the bottleneck is often sampling and numerics, not the loss.** Two of three changes are infrastructure. **This is the most underrated fact in post-training: public discussion concentrates on the algorithm taxonomy, while real gains often come from "how much experience per hour" and "are both sides computing the same number."**

## What it does not show

Drawing the boundary honestly:

- **One vendor, one base, one task family.** Gains concentrate in coding and agent work and don't extrapolate to other capabilities.
- **Self-reported, not independently reproduced.** Releases don't volunteer regressions — **"no regressions mentioned" is not "no regressions"**, especially for general dialogue, diversity, and safety, none of which were the launch's focus.
- **No comparable cost accounting.** Ten times the environments means the post-training run itself got much more expensive. "Only post-training changed" is not "cheap."
- **You can't invert it into "pretraining doesn't matter."** This went far precisely because the base was already strong. What the same post-training does on a weak base is a question this experiment cannot answer.

## Down to a checklist

1. For the gain I'm looking at, what was the baseline? Near zero means the story is coverage, not method.
2. Did output get longer or shorter alongside the gain? Longer means ruling out "thought longer, guessed better" first.
3. In my RL loop, do sampling and training share numerics? Have I measured the logprob divergence between them?
4. Is my bottleneck "the algorithm isn't good enough" or "I can't collect enough experience per hour"? Those have completely different fixes.
5. The dimensions the release didn't mention — did I measure them myself?

## Where to read next

- [After PPO: every algorithm deletes one of its parts](after-ppo.en.md): the algorithm half
- [The three stages of RLHF](rlhf-pipeline.en.md): what each model does
- [How big is your negative pool, really](../practice/recommender-systems/negative-pool-size.en.md): the same species of silent numerical mismatch

## Sources

Numbers and technical claims come from the vendor's release and reporting on it; not independently reproduced:

- [GLM-5.3 didn't change the base model — where did its coding gains come from?](https://thenewstack.io/glm-5-3-post-training-coding/) — The New Stack
- [GLM-5.3 vs GLM-5.2: Complete Benchmark Comparison & Post-Training Analysis](https://codingfleet.com/blog/glm-5-3-vs-glm-5-2-complete-benchmark-comparison/) — CodingFleet
