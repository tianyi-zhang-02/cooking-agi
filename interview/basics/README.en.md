# ML / LLM fundamentals review

[中文](README.md) · **English**

Use this section after studying a topic: try explaining a question, then revisit the part you cannot explain. For a first encounter, start in [Foundations](../../learn/README.en.md) rather than learning through follow-up questions alone.

## Pick a subject to review

| Subject | Entry | What to practice |
| --- | --- | --- |
| Common ML / LLM concepts | [Basic questions](../../00-foundations/interview-basics.en.md) | Explain the mechanism in your own words, beyond the definition |
| Probability, estimation, and optimization | [ML mathematics](../../00-foundations/ml-math-interview.en.md) | Define variables and assumptions, and justify each step |
| Transformer design choices | [Transformer follow-ups](../transformer-followups.en.md) | Compare what changes, what it fixes, and what it costs |
| A broader check | [Site question bank](../questions.en.md) | Find gaps without the chapter order giving you hints |

## How far to take an explanation

Consider “Why divide attention scores by $\sqrt{d_k}$?”

- **Start with the intuition:** if dot-product scale grows with dimension, the softmax can become very sharp.
- **State the assumptions:** under simplifying assumptions such as independent, zero-mean, unit-variance components, the dot product has variance $d_k$. Dividing by $\sqrt{d_k}$ returns the variance to 1.
- **State the limit:** a real model's Q and K need not satisfy these assumptions. This motivates the scaling; it does not guarantee unit variance throughout training.

For the calculation, read [multi-head attention](../../00-foundations/core/multi-head-attention.en.md). To check your implementation skills, try [ML coding](../../learn/ml-exercises/README.en.md). Understanding, explaining, and implementing the same concept can be practiced separately.

## When to move on

If you can explain the mechanism, give a small example, and state the assumptions, move to the next question. When a follow-up exposes a gap, record it specifically: “cannot derive the variance” or “unclear about mask direction” is easier to revisit than “weak at attention.”

This is not a script of ideal answers. Roles call for different depths. You do not need to expand every derivation aloud, but you should know which steps you omitted. Return to [Interview preparation](../README.en.md) for coding and algorithm routes.
