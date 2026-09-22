# LLM-as-a-Judge: how to write a criterion

[中文](criteria.md) · **English**

A good criterion usually has five properties.

## 1. Judge one thing at a time

"Is this a high-quality response?" is too vague. Split it into correctness, faithfulness, relevance, completeness, and style.

If several dimensions must be merged into a total score, still judge them separately first, then state explicitly how they are combined.

## 2. Point to observable evidence

Don't just write "helpful." Say what helpful looks like in the current task, for example: does the response give actionable steps, does it cover the user's explicit constraints, does it avoid irrelevant content?

## 3. State the scope

Faithfulness can mean "relies only on the given context," or it can mean "consistent with real-world facts." The two tasks need completely different evidence.

## 4. Give boundaries and counterexamples

Say which cases must fail, which cases may still pass, and whether `unknown` is allowed when information is insufficient.

## 5. Give each score a behavioral anchor

`4 = good`, `3 = okay` does not help much. A better rubric is:

```text
5  Fully correct, covers every necessary constraint, no substantive omission
4  Core is correct, with only small omissions that do not affect use
3  Partly correct, but misses one important requirement or needs the user to supply a correction
2  Has major errors; only a small part of the content is usable
1  Wrong, irrelevant, or violates a key constraint
```

## Common criteria

Different tasks should choose different combinations, rather than running every metric by default.

### RAG / question answering

- **Answer correctness**: is the answer correct?
- **Faithfulness / groundedness**: can the conclusion be supported by the evidence provided?
- **Answer relevance**: does it directly answer the user's question?
- **Context relevance**: is the retrieved evidence related to the question?
- **Completeness**: does it cover the key points needed to complete the task?

### Agent

- **Task completion**: was the real task completed?
- **Tool correctness**: were the tool choice and the arguments correct?
- **State integrity**: does the external state after execution match expectations?
- **Policy compliance**: were permissions and inviolable constraints respected?
- **Efficiency**: are there loops that make no progress, repeated searches, or runaway cost?
- **Recovery quality**: after a tool failure, did it correctly explain, retry, or escalate to a human?

### Personalization / Model Experience

- **Intent fit**: does it fit the current intent, rather than only matching long-term preference?
- **Memory appropriateness**: is the memory it used relevant, still valid, and open to correction?
- **Breadth**: are the results overly repetitive, or do they trap the user in one narrow topic?
- **Control**: can the user understand and intervene in the key decisions?
- **Longitudinal improvement**: do corrections actually take effect in later sessions?

### Open-ended generation

- **Correctness**, **coherence**, **instruction following**, **tone**, **clarity**, **originality**.

The more subjective these dimensions are, the more they need an explicit rubric and human calibration.
