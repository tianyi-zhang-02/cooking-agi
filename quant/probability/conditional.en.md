# Probability: conditional probability and independence

[中文](conditional.md) · **English**

> Reading time: ~6 min · Level: beginner · Last reviewed: 2026-09

<div class="lesson-recipe">
  <div><span>The problem</span><strong>how learning that one event happened changes the probability of another</strong></div>
  <div><span>Prerequisites</span><strong>the three axioms · intersection, union, complement</strong></div>
  <div><span>Core mechanism</span><strong>conditioning = shrinking the sample space to A and renormalising</strong></div>
  <div><span>Common mistakes</span><strong>treating disjoint as independent; thinking "can happen together" means independent</strong></div>
</div>

## Conditional probability: shrink the world to A

**Definition** (a definition, not a theorem): when $P(A) > 0$,

$$P(B \mid A) = \frac{P(A \cap B)}{P(A)}$$

The intuition: once you know A happened, throw away every outcome outside A, and A becomes the new sample space. What is left of B in that smaller world is $A \cap B$, and dividing by $P(A)$ brings the total probability of the new world back to 1.

In the figure below the whole square is Ω and area is probability. Switch to "given A" and A is stretched to fill the frame — the share of the new frame that B still covers is $P(B \mid A)$.

<!-- widget:tx-prob-events -->

## The chain rule: always true

Rearrange the definition:

$$P(A \cap B) = P(A)\,P(B \mid A)$$

This holds **always**, with no assumptions. When events are dependent you still multiply; you just multiply by the conditional.

Rain and snow: $P(\text{rain and snow}) = P(\text{rain}) \cdot P(\text{snow} \mid \text{rain})$. If snow is more likely on rainy days, this is larger than $P(\text{rain}) \cdot P(\text{snow})$.

## Independence: the conditional collapses to the plain probability

**Definition**: $P(A \cap B) = P(A)\,P(B)$. It says the same thing as $P(B \mid A) = P(B)$, in two forms:

- if $P(A \cap B) = P(A)P(B)$, put it into the definition of conditional probability: $P(B \mid A) = P(A)P(B) / P(A) = P(B)$;
- conversely, multiply both sides of $P(B \mid A) = P(B)$ by $P(A)$ and you are back at $P(A \cap B) = P(A)P(B)$.

By symmetry $P(A \mid B) = P(A)$ as well. So independence is just the special case of the chain rule where the conditional happens to equal the plain probability.

One thing that is often said wrong: **"can happen together" is not independence.** Rain and snow can fall on the same day, yet they are almost certainly dependent — both are driven by the same weather system. Overlap is necessary for independence, not sufficient. Independence is an equation you compute and check, not a feeling that two events are unrelated. In real problems it usually comes from the setup — two separate coins, two separate days — not from what the sets look like.

## Disjoint is not independent

This is the single most common error in probability interviews.

| | Disjoint | Independent |
| --- | --- | --- |
| About | the sets: $A \cap B = \varnothing$ | the probabilities: $P(A \cap B) = P(A)P(B)$ |
| Meaning | if A happens, B cannot | learning A tells you nothing about B |
| Rule it pairs with | addition: $P(A \cup B) = P(A) + P(B)$ | multiplication: $P(A \cap B) = P(A)P(B)$ |

The key fact: **disjoint events with positive probability are never independent.** Disjoint says $P(A \cap B) = 0$; independent needs $P(A \cap B) = P(A)P(B) > 0$. They contradict each other. Disjointness is in fact extreme dependence: seeing A tells you B is impossible.

One roll of a die, three examples:

- $A = \{1, 2\}$, $B = \{5, 6\}$: disjoint. $P(B \mid A) = 0$ but $P(B) = 1/3$, so not independent.
- $A = \text{even} = \{2, 4, 6\}$, $B = \text{at most 4} = \{1, 2, 3, 4\}$: they overlap. $P(A) = 1/2$, $P(B) = 2/3$, $P(A \cap B) = P(\{2, 4\}) = 1/3 = 1/2 \times 2/3$, so independent. That is a numerical coincidence, which is why you always compute $P(A \cap B)$.
- A and $A^c$: disjoint, and never independent unless $P(A)$ is 0 or 1.

## Total probability

Split on whether A happens, weight each case, and add:

$$P(B) = P(B \mid A)\,P(A) + P(B \mid A^c)\,P(A^c)$$

Reach for it whenever you need B without knowing how A turned out.

## Pairwise independent is not mutually independent

Flip two fair coins. A = the first is heads, B = the second is heads, C = the two match.

- $P(A) = P(B) = P(C) = 1/2$;
- $P(A \cap B) = P(A \cap C) = P(B \cap C) = 1/4$, so every pair is independent;
- but $P(A \cap B \cap C) = P(\{HH\}) = 1/4 \ne 1/8 = P(A)P(B)P(C)$, so the three together are **not** mutually independent.

The reason: know any two of them and the third is fixed. It is the classic "pairwise but not mutually independent" example, and it comes up in interviews.

## Common interview questions

<details class="interview" markdown="1">
<summary>Flip two coins: A = first is heads, B = second is heads, C = both match. Are A and B disjoint? Independent? And A and C?</summary>

A and B are not disjoint (HH satisfies both) and they are independent: $P(A \cap B) = 1/4 = 1/2 \times 1/2$. A and C are not disjoint either; $A \cap C = \{HH\}$, $P(A \cap C) = 1/4 = P(A)P(C)$, so they are independent too. All three together: see above — pairwise independent, not mutually independent.

</details>

<details class="interview" markdown="1">
<summary>Draw two cards without replacement. A = first is an ace, B = second is an ace. Find P(A ∩ B) and P(B). Are A and B independent? And with replacement?</summary>

Without replacement: by the chain rule, $P(A \cap B) = \frac{4}{52} \cdot \frac{3}{51} = \frac{1}{221}$. By total probability, $P(B) = \frac{3}{51} \cdot \frac{4}{52} + \frac{4}{51} \cdot \frac{48}{52} = \frac{1}{13}$. And $P(A)P(B) = \frac{1}{169} \ne \frac{1}{221}$, so they are not independent.

The surprise is $P(B) = 1/13$, exactly the same as the first card: when you do not know what the first card was, the second is symmetric with it.

With replacement: $P(A \cap B) = \frac{1}{169}$ and $P(B) = \frac{1}{13}$, so they are independent.

</details>

<details class="interview" markdown="1">
<summary>What is the difference between disjoint and independent? Can two disjoint events be independent?</summary>

Disjoint is about the sets, $A \cap B = \varnothing$, and pairs with addition; independent is about the probabilities, $P(A \cap B) = P(A)P(B)$, and pairs with multiplication. Two disjoint events with positive probability are never independent: disjointness forces $P(A \cap B) = 0$, independence needs it to equal a positive number.

</details>

<details class="interview" markdown="1">
<summary>Why is P(A ∩ B) = P(A)P(B) the same statement as P(B | A) = P(B)?</summary>

Use the definition $P(B \mid A) = P(A \cap B) / P(A)$: substituting the first gives the second; multiplying both sides of the second by $P(A)$ gives the first (with $P(A) > 0$).

</details>
