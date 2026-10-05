# Imitation and rewards: what demonstrations do not tell us

[中文](imitation-and-rewards.md) · **English**

> Reading time: ~8–10 min · Last reviewed: 2026-10

Learning steering from expert driving looks like supervised learning. But one small steering error changes the road you see next. **Small one-step prediction error can coexist with poor full-trajectory behavior.**

## Behavior cloning has a direct objective

Learn demonstrated actions:

$$
L_{\rm BC}=-\mathbb E_{(s,a)\sim\mathcal D_{\rm expert}}
[\log\pi_\theta(a\mid s)].
$$

No reward model or extensive agent trial and error is required. But training states come from the expert, while deployment states come from the learner. Drift toward the road edge changes subsequent observations, and the demonstrations may never show how to recover there.

This does not mean more data is always useless. More samples from the same narrow coverage may fail to teach recovery in new states.

## Why errors can accumulate over time

Consider a simplified analysis: before the first deviation from the expert, each step has error probability at most $\epsilon$, and recovery after deviation is not guaranteed. A union bound puts the probability of an error by step $t$ at most $\min(1,t\epsilon)$, without requiring independent errors.

Compare expert and learner trajectories with the same initial conditions and coupled environmental randomness; they coincide until the first action deviation. If excess loss after deviation is at most 1 per step, **expected cumulative excess loss** over horizon $T$ satisfies:

$$
\mathbb E[\text{cumulative excess loss}]\le\sum_{t=1}^{T}\min(1,t\epsilon)
\le\frac{T(T+1)}{2}\epsilon.
$$

This explains the $O(T^2\epsilon)$ term: one error can affect many later steps. It is a worst-case bound under these simplified conditions, not a formula predicting real task loss. Environments that permit recovery can behave much better.

## More data can still teach the wrong behavior

| Problem | Example | What to change |
| --- | --- | --- |
| Insufficient observation | One frame does not reveal whether an object is approaching or receding | Add history or memory; do not equate an observation with a full state |
| Multiple valid action modes | Left -1 and right +1 both avoid an obstacle, but squared error may predict unsafe straight-ahead 0 | Represent a multimodal action distribution through mixtures, discrete choices, or other expressive policies rather than only its mean |
| Shortcut learning | A corner color in toy navigation data always indicates the expert's next turn, but is unreliable at deployment | Change or mask the cue and test whether behavior depends on task-relevant information |

The third example illustrates causal confusion. History can reveal useful state information or create more shortcuts. More complete input does not automatically produce reliable behavior, and DAgger is not an automatic causal-identification algorithm.

## DAgger labels states the learner visits

Train an initial BC policy, run the learner, ask an expert for actions at visited states, aggregate those labels with prior data, and retrain. Early rollouts can mix expert and learner actions, gradually reducing expert execution.

<details markdown="1">
<summary>Why is collecting more expert-only trajectories different?</summary>

The expert may never reach the road edge and therefore never demonstrate recovery there. DAgger changes the distribution of states being labeled: it queries the expert where the learner actually ends up. It requires a queryable expert and safe collection, not free supervision from deployment failures.

</details>

Classical analyses show that BC errors can compound over the horizon in adverse settings, while DAgger improves cumulative-loss bounds under assumptions such as no-regret learning. The familiar $O(T^2\epsilon)$ and $O(T\epsilon)$ are not fixed empirical performance multipliers for every environment.

## IRL infers objectives, not minds

Inverse RL asks which reward could explain expert behavior. Many reward functions can explain the same policy. Adding identical trajectory constants or suitable potential shaping can preserve policy rankings.

Additional assumptions or inductive biases are necessary: a reward family, a maximum-entropy model, or preference comparisons. Experts also face capability, information, and cost constraints; observed behavior need not reveal perfectly rational optimal preferences. A complete MaxEnt IRL optimization derivation is outside this introduction.

## Where preference-based rewards can go wrong

For two outcomes on the same task, a common pairwise model is:

$$
P(y^+\succ y^-\mid x)=\sigma(r_\phi(x,y^+)-r_\phi(x,y^-)).
$$

This supervises a reward difference, but does not automatically separate quality from length, style, or annotator preference. A policy optimizing the learned reward can exploit regions beyond label coverage. Keep independent task validation rather than relying on the same judge to approve its own influence.

## How does one preference pair update a reward model?

For one question, an annotator prefers A to B. Current scores are $r_A=1,r_B=0$. Predicted preference probability is $\sigma(1)\approx0.731$, and negative log-likelihood is about 0.313.

$$
L=-\log\sigma(r_A-r_B),\qquad
\frac{\partial L}{\partial r_A}=p-1,\quad
\frac{\partial L}{\partial r_B}=1-p.
$$

Gradients are approximately −0.269 and +0.269, so descent raises A and lowers B. Adding 100 to both scores leaves probability and loss unchanged. The supervision constrains differences; an absolute reward of 100 is not a hundred times the quality of reward 1.

If annotators favor length, the model may learn a length shortcut. Test equal-quality answers of different lengths and similar-length answers of different factual quality. When annotators disagree, inspect genuine preference differences rather than treating all disagreement as labeling error.

## After demonstrations, which data is worth adding?

| Failure | More specific intervention than collecting more good demonstrations | Limitation |
| --- | --- | --- |
| Cannot recover from a small deviation | Query the expert at deviated states visited by the learner | Requires safe, reliable expert answers |
| Several valid actions at one state | Retain alternatives and conditions instead of averaging them | The policy must express multimodal choices |
| Uses a background shortcut | Change backgrounds while preserving task conditions | Do not remove genuinely useful context |
| High reward, poor task completion | Independent outcomes and difficult preference comparisons | Do not rely only on the same scoring model |

A DAgger round executes the learner, queries expert actions at those states, aggregates data, and refits. It does not label the learner’s own action as correct. Limited annotation budgets may prioritize risky, disputed, or uncertain states, but selection changes coverage too; confident successful trajectories alone are insufficient.

## Connecting back to language models

SFT resembles learning demonstrations; RL changes behavior using outcome feedback. Neither simply replaces the other. Once a policy can perform basic tasks, ask which feedback adds information beyond demonstrations. Human preferences, programmatic verification, and environmental rewards each have blind spots.

References: [DAgger](https://arxiv.org/abs/1011.0686) · [Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979). Continue with [RLHF](../rlhf/README.en.md), or [what fixed logs can support](offline-and-ope.en.md).
