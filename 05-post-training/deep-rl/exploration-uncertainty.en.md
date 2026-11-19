# Exploration and uncertainty: what should we try next?

[中文](exploration-uncertainty.md) · **English**

> Reading time: ~12–15 min · Last reviewed: 2026-11 · Prerequisite: [Exploration and hierarchy](exploration-and-hierarchy.en.md)

Option A succeeded in 8 of 10 trials. Option B succeeded in 1 of only 2. Should the next trial use A again? If this were the last decision, A would look more attractive. With many decisions ahead, two observations may not tell us enough about B.

Exploration is more than adding noise to an action. We will use these small counts to compare upper confidence bounds, posterior sampling, and information gain. Each handles what we do not know, but in a different way.

## Start with a bandit {#bandit}

In a multi-armed bandit, an agent chooses one action per round and observes only that action's outcome. Unlike an MDP, the action does not move the agent into a state that changes future opportunities. Here each action has a fixed unknown success probability, and rewards are either zero or one.

| Action | Successes | Failures | Empirical success rate |
| --- | --- | --- | --- |
| A | 8 | 2 | 0.8 |
| B | 1 | 1 | 0.5 |

A greedy rule selects A by its empirical mean. The problem is not that this choice must be wrong. It is that never selecting B again leaves no way to correct the estimate of B.

If action $a$ has true expected reward $\mu_a$, and the best mean is $\mu^*$, expected pseudo-regret measures cumulative opportunity cost:

$$
\mathcal R_T=\mathbb E\!\left[\sum_{t=1}^{T}(\mu^*-\mu_{A_t})\right].
$$

This compares means, not whether one particular trial happened to fail. Because the true means are unknown, the expression is not directly available as a training reward. Real systems can also have user variation, delayed feedback, and drift; conclusions from this fixed-rate example do not transfer automatically.

## UCB: give an underexplored choice a chance {#ucb}

An upper confidence bound (UCB) adds an allowance that shrinks with sample count. In a common UCB1 convention, after $t$ observations and at least one trial of each action, choose:

$$
A_{t+1}=\arg\max_a\left[\hat\mu_a+\sqrt{\frac{2\log t}{N_a}}\right].
$$

Our data have $t=12$:

| Action | Empirical mean | Exploration term | UCB index |
| --- | --- | --- | --- |
| A | 0.8 | $\sqrt{2\log12/10}\approx0.7050$ | About 1.5050 |
| B | 0.5 | $\sqrt{2\log12/2}\approx1.5764$ | About 2.0764 |

UCB chooses B this time, not because B has been established as better, but because its estimate is less settled. An index above one is not a contradiction: this is an upper-bound-style decision index, not a predicted success probability.

Why the $1/\sqrt N$ dependence? For independent bounded rewards, a Hoeffding bound gives a tail such as $P(\mu_a-\hat\mu_a>b)\le e^{-2N_ab^2}$. At a fixed sample count, setting the right side to $\delta$ gives $b=\sqrt{\log(1/\delta)/(2N_a)}$. UCB1 tightens its confidence level over time. A full regret proof must also handle adaptive action choices and the possible sample counts; this fixed-sample tail bound alone is not that proof.

[Auer et al. (2002)](https://doi.org/10.1023/A:1013689704352) establish guarantees under the relevant bounded, stationary bandit assumptions. Adding a heuristic bonus to a neural-network output does not automatically inherit them.

## Thompson sampling: draw a plausible model, then act on it {#thompson}

Posterior sampling, also called Thompson sampling, maintains a distribution over possible success probabilities rather than adding a fixed allowance.

For Bernoulli rewards, start with a $\operatorname{Beta}(1,1)$ prior. After $s$ successes and $f$ failures:

$$
p(\theta\mid\mathcal D)
\propto\theta^s(1-\theta)^f
\quad\Longrightarrow\quad
\theta\mid\mathcal D\sim\operatorname{Beta}(1+s,1+f).
$$

A's posterior is therefore $\operatorname{Beta}(9,3)$; B's is $\operatorname{Beta}(2,2)$. Their posterior means are 0.75 and 0.5. A's mean is no longer exactly 0.8 because the prior contributes to the update.

At each round, draw a success probability from each posterior and select the action with the larger draw. For example, draws of 0.68 for A and 0.81 for B select B. These are illustrative sample values, not newly calculated posterior means. After observing B's result, update the posterior supported by that observation. Here the action parameters have independent priors, so observing B does not also update A.

Comparing posterior means alone would still be greedy. Thompson sampling explores through posterior uncertainty; a badly misspecified prior or likelihood can also discourage trying a genuinely good action. The [authors' tutorial, §§3, 6, and 8](https://arxiv.org/html/1707.02038v3) covers the algorithm, modeling choices, and limitations.

## Information gain: how much will an observation teach us? {#information-gain}

Different uncertainties need different treatment. A machine's fixed but unknown state can be learned from observations. A screen that independently flickers at random remains unpredictable even if its rule is perfectly known. The first involves epistemic uncertainty; the second contains irreducible aleatoric randomness.

Consider a binary hidden state $\Theta$, left or right, with equal prior probabilities. A probe reports the state correctly 80% of the time. A “left” report gives posterior $[0.8,0.2]$; a “right” report is symmetric. Using base-two entropy:

$$
I(\Theta;Y)=H(\Theta)-\mathbb E_Y[H(\Theta\mid Y)]
=1-H_2(0.8)\approx0.2781\text{ bits}.
$$

Here $H_2(p)=-p\log_2p-(1-p)\log_2(1-p)$. A second action returning an independent fair coin is also unpredictable, but the posterior remains $[0.5,0.5]$ after either outcome. Its information gain about the hidden state is zero.

Equivalently, expected information gain is the mean KL divergence from posterior to prior:

$$
\mathbb E_Y\left[D_{\rm KL}\big(p(\Theta\mid Y,a)\,\|\,p(\Theta)\big)\right].
$$

Large prediction error is therefore not the same as a useful learning opportunity. Information itself is not always useful either: on the final step, probing something with no time left to exploit it may be worse than taking a reward. Information-driven exploration still needs to account for the task, remaining horizon, and cost of experimentation. [VIME](https://arxiv.org/abs/1605.09674) approximates information gain about a dynamics model as an exploration bonus; it does not settle all of these tradeoffs.

## What does a visit count mean when pixels rarely repeat? {#pseudo-count}

A small lighting change can create a new image. Counting exact images may therefore call every frame novel. Pseudo-counts let a density model share information across related observations.

Suppose the model assigns observation $x$ probability $p$ before an update and $p'$ after learning from one more occurrence of $x$. Ask what counts would explain this change if it came from a counting model. Let the local pseudo-count be $\hat N(x)$ and its corresponding total be $\hat n$:

$$
p=\frac{\hat N(x)}{\hat n},\qquad
p'=\frac{\hat N(x)+1}{\hat n+1}.
$$

Substitute $\hat N(x)=p\hat n$ into the second equation. This gives $(p'-p)\hat n=1-p'$, hence:

$$
\begin{aligned}
\hat n&=\frac{1-p'}{p'-p},\\
\hat N(x)&=\frac{p(1-p')}{p'-p}.
\end{aligned}
$$

With $p=0.20$ and $p'=0.25$, we obtain $\hat n=15$ and $\hat N(x)=3$. Substitution checks both equations: $3/15=0.20$ before the update and $4/16=0.25$ afterward. **The total and the local count differ by a factor of $p$; they are not the same number.** Tabular counts recover actual frequencies. A neural density model supplies an equivalent count tied to its updates, not evidence of exactly three real visits.

[Bellemare et al. (2016), §3.1](https://arxiv.org/html/1606.01868v2#S3.SS1) derive this construction and its conditions. We use $0<p<p'<1$ to avoid degenerate cases. If learning makes $p'<p$, the formula becomes negative and cannot be used naively as a visit count. If $p'=p$, the model may simply have failed to learn; that is not automatically evidence of complete knowledge.

A bonus might then take a form such as $\beta/\sqrt{\hat N(x)+c}$, with $c>0$ avoiding a zero denominator. Features, the density model, and update size all affect the result. A low pseudo-count is not a calibrated estimate of danger or task value; track original task success separately.

## Long trajectories also need consistency over time {#deep-exploration}

Moving from bandits to MDPs requires more than independent randomization at each step. [Bootstrapped DQN, §3](https://arxiv.org/html/1602.04621v3#S3) maintains several Q estimates, chooses one Q head at the start of an episode, and acts according to that head for the episode. Different estimates supply different exploration directions that can persist over time.

Resampling a head after every step is different: it can keep reversing direction. Averaging all heads before taking the maximizing action is different too: averaging can remove the disagreement driving exploration. Nor does disagreement between heads automatically represent an exact Bayesian posterior. Initialization, training data, shared features, and target networks all affect it.

Q-learning can learn from other behavior policies, but that does not make the data distribution irrelevant. Missing coverage, function approximation, and bootstrapping can still make its estimates unreliable.

## For review, compare what each method changes {#review}

| Method | Extra information used for decisions | Common confusion |
| --- | --- | --- |
| $\epsilon$-greedy | A random exploration switch | Random actions need not produce useful long trajectories |
| UCB | A mean plus a count-dependent allowance | The index is not a success probability; bandit guarantees do not directly cover deep RL |
| Thompson sampling | A sampled parameter or value hypothesis | Comparing posterior means is not posterior sampling |
| Information gain | How observations could change beliefs | Unpredictable outputs need not be informative |
| Pseudo-counts / RND | Generalized novelty estimates | Novelty is neither usefulness nor calibrated uncertainty |
| Bootstrapped DQN | A Q hypothesis followed throughout an episode | Multiple heads do not ensure meaningful disagreement; inspect how they are trained |

Reproduce three calculations: why UCB chooses B, why the 80%-accurate probe yields about 0.2781 bit, and why $p=0.20\to0.25$ corresponds to a local count of 3 rather than the total of 15. The [companion checks](code/feedback_exploration_checks.py) verify this arithmetic too.

Then return to [exploration and hierarchy](exploration-and-hierarchy.en.md) to compare intrinsic rewards, potential shaping, and option timescales. They can be combined, but they do not solve the same problem.
