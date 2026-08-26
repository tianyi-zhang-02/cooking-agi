# Where preferences come from, and what a reward model actually learns

[中文](where-preferences-come-from.md) · **English**

> Reading time: ~7 min · Type: chapter · Last reviewed: 2026-08

## In one sentence

The **H** in RLHF happens entirely at this step. Everything downstream — PPO, GRPO, DPO — **amplifies the signal this step produced**, and an amplifier cannot produce what isn't in the signal. Whatever the reward model learned wrong, the rest of the pipeline will faithfully optimize toward.

## Why comparisons instead of scores

The obvious approach is to have people score each response and fit a regression. It doesn't work, because humans are unreliable at **absolute scores**: the same response gets a 7 today and a 5 tomorrow, different annotators use entirely different scales, and scores drift — after a run of bad responses, a mediocre one gets rated too high.

People are much more reliable at **relative comparison**. "Which of these two is better" has far higher agreement than "how many points is this worth."

So preference data comes in pairs: for a prompt $x$, a preferred response $y_w$ and a worse one $y_l$.

## Bradley-Terry: turning comparisons into a score

To get from pairwise comparisons to a function that scores any response, you need a model that translates **order** into a **scalar**. Bradley-Terry is that translator:

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

| Symbol | Meaning |
| --- | --- |
| $r(x,y)$ | the reward model's scalar score for response $y$ |
| $\sigma$ | sigmoid, squashing the score gap into a probability |
| the **difference** | the only thing that enters the formula — note this |

Training maximizes the likelihood of the observed preferences:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\big(r(x,y_w) - r(x,y_l)\big)\right]$$

In practice this is a pretrained model with a scalar head that reads the whole response and emits one number.

## A consequence: the reward model learns order, not scale

Only the **difference** in $r$ appears above. Add the same constant to every $r$ and the loss is unchanged.

**A reward model's absolute values mean nothing. Only relative ones do.**

That isn't theoretical fastidiousness; it has direct engineering consequences:

- "this response scored 3.2" says nothing on its own — only higher or lower than another;
- scores are not comparable across reward models, or even across training runs of the same one;
- so the RL stage almost always normalizes rewards — **that isn't a tuning trick, it's supplying a scale the model never defined**.

## Four traps in preference data

**One: annotators disagree with each other to begin with.** Show the same pair to different people and agreement is well below 100%. That agreement rate is the **ceiling** on reward model accuracy — you cannot learn to be more accurate than the labels. So when RM accuracy sits at 70%, check human agreement before touching the model.

**Two: length.** The most famous and most stubborn one. Longer responses are systematically preferred — they look more complete, more effortful. So part of what the reward model learns is "longer is better," and RL will faithfully amplify that into "write more." See [the alignment tax](alignment-tax.en.md) for the downstream shape of this.

**Three: what annotators can't check, the reward model can't learn.** Whether a fact is right, whether the code actually runs, whether a citation exists — if annotators had no way to verify, their preference can only rest on **whether it looks right**. So the reward model learns *credible-looking*, not *correct*. This is a structural ceiling that more data does not fix, and it's why [verifiable rewards](verifiable-rewards.en.md) exist.

**Four: presentation order and formatting.** Position bias, Markdown structure, whether things were bolded — all of it seeps into the labels. These aren't noise (noise averages out); they're **systematic bias**, and they get learned in full.

## The one people miss: reward models expire

A reward model is trained on **a distribution** — typically the SFT model's outputs.

The moment RL starts, the policy moves. As training proceeds, its responses drift further from that distribution, and the reward model starts scoring **a kind of response it has never seen**. Its scores stop being trustworthy — but it **still returns a number**.

This is the first reason the KL penalty exists. The textbook line is "keep the model from drifting too far." The more accurate one:

> **The KL penalty confines the policy to the region where the reward model is still valid.**

It's not a moral constraint, it's a **validity constraint**. Which is also why the KL coefficient is so hard to tune — it is really asking "how wide is my reward model's radius of trust," and you have never measured that radius.

It also explains the mechanism of reward hacking. Those high-scoring terrible responses the policy discovers are usually out-of-distribution oddities — **the reward model was never supervised there, and the high score is an extrapolation artifact.**

## Down to a checklist

1. What's my annotator agreement rate? Is my RM accuracy already pressed against that ceiling?
2. In my preference pairs, how strongly does length correlate with the label? How much signal survives after regressing length out?
3. Could annotators verify facts and code? If not, is my RM learning *right* or *looks right*?
4. During RL, am I monitoring how far policy outputs have drifted from the RM's training distribution?
5. Is my reward normalization supplying a missing scale, or papering over a distribution shift I haven't understood?

## Where to read next

- [The three stages of RLHF](rlhf-pipeline.en.md): what happens to this reward model next
- [Verifiable rewards](verifiable-rewards.en.md): which of these traps disappear when the reward isn't learned
- [The alignment tax](alignment-tax.en.md): what the bias in preferences grows into downstream
- [Data and feedback](../01-data-and-feedback/): label quality in general

## Starting papers

- [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) — the origin of training a reward model from pairwise preferences
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — early evidence on KL penalties and reward hacking
- [InstructGPT](https://arxiv.org/abs/2203.02155) — the three-stage pipeline and annotation guidelines
