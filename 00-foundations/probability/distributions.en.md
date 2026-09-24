# Probability: drawing a distribution

[中文](distributions.md) · **English**

> Reading time: ~5 min · Level: beginner · Last reviewed: 2026-09

<div class="lesson-recipe">
  <div><span>The problem</span><strong>what a random variable looks like, and how to describe and draw it</strong></div>
  <div><span>Prerequisites</span><strong>random variables · indicators</strong></div>
  <div><span>Core mechanism</span><strong>bars for discrete (PMF), area for continuous (PDF), and a CDF that works for both</strong></div>
  <div><span>Common mistakes</span><strong>reading a density f(x) as a probability; thinking a continuous variable has probability f(x) at a point</strong></div>
</div>

## What a distribution is

The **distribution** of a random variable X is the probability that X lands in each set. The whole thing fits in one picture; how you draw it depends on what kind of variable X is.

## Discrete: a row of bars

When X takes countably many values, give each value its probability. That is the **PMF** (probability mass function):

$$p(k) = P(X = k), \qquad \sum_k p(k) = 1$$

Drawn, it is a bar chart. Flip a coin 8 times with heads probability p, and the number of heads X is Binomial(8, p): nine bars on the integers 0 to 8. The expectation is the weighted average of the bars: $\mathbb{E}[X] = \sum_k k\,p(k) = 8p$.

## Continuous: area under a curve

When X can take a whole interval of real values, giving each point a probability stops working — there are too many points, so each one can only get 0. Instead you use a **density** f (the PDF, probability density function), and probability is the area under the curve:

$$P(a \le X \le b) = \int_a^b f(x)\,dx$$

Two things are easy to get wrong:

- **f(x) is not a probability and can exceed 1.** Uniform(0, 1/2) has density 2 everywhere, and the area is still 1.
- **Every single point has probability 0.** So for a continuous variable $P(X \le b)$ and $P(X < b)$ are the same.

The expectation becomes an integral: $\mathbb{E}[X] = \int x\,f(x)\,dx$.

## The CDF: the one picture that always works

Discrete, continuous or otherwise, the **cumulative distribution function** is always defined:

$$F(x) = P(X \le x)$$

It collects probability from left to right, so it never decreases, tends to 0 on the left and 1 on the right, and is right-continuous.

- The CDF of a discrete variable is a **staircase**: each step is exactly as tall as the PMF at that point;
- the CDF of a continuous variable is a **smooth ramp** whose slope is the density, $F' = f$;
- reading probabilities takes two rules: $P(a < X \le b) = F(b) - F(a)$, and $P(X = x) = F(x) - F(x^-)$ — **how far it jumps** at x.

Move x and watch F(x) collect everything to its left:

<!-- widget:tx-prob-dist -->

## Mixed distributions: bars and a curve at once

The third kind is the fun one. Take an insurance payout: with probability π the year has no claim and pays exactly 0; otherwise the amount is roughly normal, but capped at 5.5. So 0 and 5.5 **each carry real probability**, and the stretch in between is a density.

- A PMF alone fails: every point in the middle has probability 0;
- a PDF alone fails too: the probability sitting on 0 and on 5.5 cannot be written as a density;
- the CDF works: a ramp that **jumps** at 0 and at 5.5, each jump as tall as the probability at that point.

The expectation adds the pieces:

$$\mathbb{E}[X] = 0 \cdot P(X = 0) + 5.5 \cdot P(X = 5.5) + \int_0^{5.5} x\,f(x)\,dx$$

Machine learning is full of these. Pass a standard normal Z through a ReLU, $\mathrm{ReLU}(Z) = \max(Z, 0)$: half the probability lands exactly on 0 and the other half spreads over the positive axis — one spike plus half a bell curve. Clipping a reward or a gradient to $[-c, c]$ does the same: everything clipped piles up on the two points $\pm c$, and the CDF jumps there.

Conditional probability and independence come next: [conditional probability and independence](conditional.en.md).

## Common interview questions

<details class="interview" markdown="1">
<summary>Can a PDF take values larger than 1?</summary>

Yes. A density is not a probability; area is. Uniform(0, 1/2) has density 2 across the whole interval and still integrates to 1. A normal with a small σ peaks far above 1 as well.

</details>

<details class="interview" markdown="1">
<summary>X is continuous. What is P(X = 3), and why is it not f(3)?</summary>

It is 0: a single point has zero area. f(3) is probability per unit length near 3: $P(3 \le X \le 3 + \varepsilon) \approx f(3)\,\varepsilon$, which goes to 0 as ε does.

</details>

<details class="interview" markdown="1">
<summary>Z is standard normal. What is the distribution of ReLU(Z), and its expectation?</summary>

Mixed: $P(\mathrm{ReLU}(Z) = 0) = P(Z \le 0) = 1/2$, a spike at 0; for x > 0 the density is just $\varphi(x)$. The expectation is $\mathbb{E}[\mathrm{ReLU}(Z)] = \int_0^\infty x\,\varphi(x)\,dx = \varphi(0) = 1/\sqrt{2\pi} \approx 0.399$.

</details>

<details class="interview" markdown="1">
<summary>Given only the CDF, how do you find P(2 &lt; X ≤ 5) and P(X = 3)?</summary>

$P(2 < X \le 5) = F(5) - F(2)$. $P(X = 3) = F(3) - F(3^-)$, the height of the CDF's jump at 3; no jump means 0.

</details>
