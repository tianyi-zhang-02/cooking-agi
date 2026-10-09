# DAPO: how sampling, length, and clipping change training

[中文](dapo.md) · **English**

> Last reviewed: 2026-10-08 · Prerequisite: [GRPO's group-relative signal](rlhf/after-rlhf.en.md)

Two training scripts both say GRPO, yet behave very differently. What should we inspect besides learning rate and model size? Small batches make it easier to see when an apparent implementation detail changes the objective itself.

## DAPO's 4 changes

[DAPO](https://arxiv.org/html/2503.14476v1) combines separate clipping bounds, continued sampling to obtain prompts with within-group reward variation, valid-token loss aggregation, and treatment of overlong-response reward noise. Its mathematical-reasoning setup also removes Reference KL. That is a particular recipe, not a claim that every task can discard KL.

## What clipping actually clips

For token ratio $\rho$ and advantage $A$, maximize:

$$
j(\rho,A)=\min\left(\rho A,\operatorname{clip}(\rho,1-\epsilon_l,1+\epsilon_h)A\right).
$$

Let $A=1$ and the current ratio be 1.25. With an upper bound of 1.2, the objective is 1.2 and offers no direct gain for increasing this ratio further. With an upper bound of 1.3, it remains 1.25 and still offers that gain. The lower bound can stay unchanged.

For $A=-1$, the same ratio gives -1.25, retaining a gradient to correct the bad move. **PPO does not hard-constrain model probabilities to the interval**: shared parameters, gradients from other tokens, and repeated updates can move ratios outside it.

Absolute changes can be tiny for rare tokens. Moving from 0.01 to 0.012 is already a factor of 1.2, just like moving from 0.5 to 0.6. Clip-Higher relaxes a multiplicative margin, not an equal additive probability allowance or a guarantee against entropy decline.

## Dynamic sampling retains groups, not only correct answers

With 4 samples per prompt, binary rewards might be:

| Prompt | Rewards | Distinguishing group-relative policy-gradient signal? |
| --- | --- | --- |
| A | `[0, 0, 0, 0]` | No |
| B | `[0, 1, 0, 1]` | Yes; retain successes and failures |
| C | `[1, 1, 1, 1]` | No |
| D | `[0, 0, 1, 0]` | Yes |

If the target is 4 informative prompt groups, this round supplies only 2. More generation is needed. Sampling cost already spent on A and C still counts.

For $G$ independent binary samples with equal success probability $p$, the probability of a mixed group is:

$$
P(\text{mixed group})=1-p^G-(1-p)^G.
$$

At $p=0.5,G=4$, it is 0.875; at $p=0.99$, only about 0.0394. Real samples need not be independent. The calculation illustrates why resampling becomes expensive on very easy or very hard tasks.

Filtering also changes the prompt distribution toward tasks the current model sometimes solves. Accuracy on retained groups is no longer accuracy on the original task distribution. Keep evaluation on a fixed set.

## Two averages, two different answers

Suppose a short response has 2 valid tokens, each with surrogate 1, and a long response has 6, each with surrogate 3.

- Average within responses, then across responses: $(1+3)/2=2$.
- Average all 8 valid tokens: $(2\times1+6\times3)/8=2.5$.

The first gives each response equal total weight. The second gives each token equal base weight, so the longer response has more total weight. This changes weighting; it is not merely a more accurate division operation.

```python
import math

def aggregate_tokens(values, masks):
    if not values or len(values) != len(masks):
        raise ValueError("a nonempty batch with matching masks is required")
    selected_rows = []
    for row, mask in zip(values, masks):
        if len(row) != len(mask) or any(type(flag) is not bool for flag in mask):
            raise ValueError("one Boolean mask per token is required")
        selected = [value for value, active in zip(row, mask) if active]
        if not selected or not all(math.isfinite(value) for value in selected):
            raise ValueError("each response needs finite action tokens")
        selected_rows.append(selected)
    response_mean = sum(sum(row) / len(row) for row in selected_rows) / len(selected_rows)
    token_mean = sum(sum(row) for row in selected_rows) / sum(map(len, selected_rows))
    return response_mean, token_mean

values = [[1, 1, 999], [3, 3, 3, 3, 3, 3]]
masks = [[True, True, False], [True] * 6]
assert aggregate_tokens(values, masks) == (2.0, 2.5)
```

999 represents padding or a non-action value that should not affect the result. All-masked samples need a defined policy: this helper rejects them; production code may skip them, but should not silently count them as zero-score responses.

### Another weighting step across devices

Rank 0 has 2 tokens with local mean 1; rank 1 has 6 with local mean 3. Averaging local means gives 2, not the global token mean 2.5.

If DDP averages rank gradients, world size is $W$, and the global valid-token count is $N$, each rank can multiply its differentiable local sum by $W/N$ before the DDP average. A framework using sum reduction, gradient accumulation, or another convention needs matching scaling. The denominator must cover the batch you claim to average.

This is a general distributed-averaging issue, not unique to DAPO. A numerical unit test can catch it before a multi-GPU run.

## A truncated response is not necessarily incorrect reasoning

A length cap makes training manageable, but an unfinished correct approach differs from a completed wrong answer. DAPO discusses both filtering overlong samples and soft overlong punishment. The latter gradually adds a penalty near the cap instead of treating every boundary case as one abrupt failure.

With an illustrative limit of 100 and buffer of 20: no penalty through 80, -0.5 at 90, and -1 at or beyond 100. This length term is added to task reward. It neither replaces correctness checking nor removes the generation cap.

```python
def length_penalty(length, limit, buffer):
    if not all(type(value) is int for value in (length, limit, buffer)):
        raise ValueError("lengths must be integers")
    if length < 0 or not 0 < buffer <= limit:
        raise ValueError("invalid lengths")
    return -min(1.0, max(0.0, (length - (limit - buffer)) / buffer))

assert [length_penalty(value, 100, 20) for value in (70, 80, 90, 100, 110)] == [0, 0, -0.5, -1, -1]
```

Mean reward alone can improve when the model writes less and pays a smaller penalty without solving more tasks. Log correctness, length penalties, natural completion, and truncation separately.

## An ablation that can be interpreted

| Experiment | Hold fixed | Additional records |
| --- | --- | --- |
| Change only the clipping upper bound | Rollouts and advantages | Clipping by advantage sign; update norms |
| Change only loss reduction | Token values and masks | Agreement between single-batch and sharded gradients |
| Enable dynamic sampling | Total generation budget and independent evaluation | Discarded groups, generated tokens, retained prompt mix |
| Add a length term | Maximum output budget | Correctness and penalty separately, not only total reward |

Check these local properties before training comparisons. The examples test objectives and data handling, not a reproduction of the paper's training run. When reading [GSPO / ASPO](policy-ratios.en.md) and [SAO](async-policy-learning.en.md) in 2026, ask the same question: did the method change sampling, weighting, or when data reaches the trainer?

Implementation entry points: [DAPO project](https://dapo-sia.github.io/) and [verl DAPO recipe](https://github.com/volcengine/verl/tree/main/recipe/dapo). Record a commit when reproducing a result; a changing main branch is not a fixed version.
