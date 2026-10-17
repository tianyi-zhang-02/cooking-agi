# Grading rubric

Score every answer 0-4 against the source note. The score is about what the user said, not what
they probably know.

| Score | Meaning | Typical signal |
| --- | --- | --- |
| 0 | No answer, or fundamentally wrong | Wrong mechanism; confuses the concept with another one |
| 1 | Fragments | Right keywords, no causal explanation; key error left uncorrected |
| 2 | Partially correct | Core idea right, but misses an important condition, edge case, or trade-off; cannot go one level deeper |
| 3 | Solid | Correct, structured, handles the follow-up; minor imprecision only |
| 4 | Strong hire signal | Correct and precise, gives numbers/shapes/complexities unprompted, names trade-offs and failure modes, knows how to verify it |

What to check, depending on the question type:

- **Concept / "why"**: the mechanism (cause, not just the name), the conditions under which it
  holds, and what breaks without it.
- **Derivation / math**: the setup, each step justified, the result, a sanity check (shape,
  limit, small example).
- **Numbers / systems**: an order-of-magnitude estimate with units, what dominates (compute,
  memory, bandwidth), which knob changes it.
- **Design / trade-off**: requirements first, options compared on named axes, a decision, how it
  would be evaluated and rolled back.
- **Code**: correctness, shapes and dtypes, edge cases (masking, empty inputs, numerics), a test
  that would catch the bug.

Common deductions: claiming something holds universally when the note says it depends on
budget/data/model; mixing up training-time and inference-time behaviour; correct conclusion with
an incorrect reason.

## Feedback format

Keep it tight:

```
Score: 2/4 — core idea right, missing the masking detail.

✓ What you got right: ...
✗ Missing / wrong: ...
Model answer: (3-6 sentences, the way you would say it in an interview)
Source: <note path> → "<section heading>"
```

Then either ask the follow-up or move to the next question.
