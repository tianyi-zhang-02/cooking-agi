# ML questions & implementations: explain it, then build it

[中文](README.md) · **English**

An answer can feel obvious while you're reading it. Try closing the page, explaining an example, and writing a few lines before checking it again.

## Pick the skill you want to practice

| Practice | Entry | What to check |
| --- | --- | --- |
| Explain a foundation | [Basic questions](../../00-foundations/interview-basics.en.md) | State assumptions, not just terminology |
| Turn a formula into code | [Whiteboard implementations](../../00-foundations/hand-write-kit.en.md) | Shapes, masks, edge cases, numerical stability |
| Reason about probability and estimation | [ML mathematics](../../00-foundations/ml-math-interview.en.md) | Identify the denominator and sampling assumptions |
| Explain architectural choices | [Transformer follow-ups](../../interview/transformer-followups.en.md) | Name the bottleneck and the new cost |
| Review a random question | [Site question bank](../../interview/questions.en.md) | Continue the reasoning after a condition changes |

## Practice one question three ways

For attention: explain how Q, K, and V enter the computation; implement a small causal-masked version; then introduce padding, a longer sequence, or caching and identify what must change.

Return to the [foundations](../../00-foundations/study-guide.en.md) when needed. These exercises check understanding; they aren't predictions of a particular company's interview.

## Self-check

<details><summary>Why isn't a correct output a complete explanation?</summary><p>You also need the input conditions and assumptions. A small example may miss empty inputs, mask direction, precision, or complexity problems.</p></details>

<details><summary>How do you check that you haven't just memorized the answer?</summary><p>Change one constraint: scale, missing data, duplicate values, or access to future information. Predict what changes before testing.</p></details>
