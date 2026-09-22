# Evaluation: an evaluation stack, from cheap to expensive

[中文](evaluation-stack.md) · **English**

No single evaluator can answer every question. Ordered from cheapest to most expensive, the tools fall into five layers.

## 1. Deterministic checks

Good for schemas, formats, state changes, tool arguments, permissions, and business rules. Whatever can be decided in code should not be handed to a probabilistic model first.

## 2. Reference or executor

Code can run tests, a math problem can be checked against the answer, retrieval results can be checked against the evidence, and an agent's task completion can be verified. Signals of this kind are usually closer to ground truth than language-based scoring.

## 3. LLM-as-a-Judge

Good for relevance, helpfulness, coherence, style, and open-ended quality. It scales to large numbers of samples, but it is affected by the prompt, the model version, position, and writing style.

## 4. Human review

Good for defining rubrics, checking edge cases, and discovering new failures the automatic evaluators never anticipated. Humans are inconsistent too, so they need calibration and recorded rationales.

## 5. Online and long-term outcomes

In the end you still have to look at real task completion, user corrections, repeat use, and long-term behavior. An improvement in offline metrics does not necessarily turn into a better experience.

## How to use an LLM judge

Think of an LLM judge as a **semantic sensor**, not as the final truth on the judge's bench.

More robust practice includes:

- judge only one clear question at a time;
- provide a reference whenever you can;
- swap the A/B order when comparing two options;
- allow `tie` and `both bad`; don't force a winner;
- use human-labeled samples to check on which slices the judge fails;
- store the prompt, model version, temperature, and input evidence;
- don't ask the judge to make up a 1–10 score that looks precise but has no basis.

Writing the rubric as a DAG helps split the judgment apart, but the DAG itself does not automatically make the metric correct. What matters most is still whether each node corresponds to a real, verifiable standard.

Few-shot, reference-based, pairwise, and probability weighting are often discussed as if they were one thing, but they are different dimensions. For the fuller breakdown, see:

→ [LLM-as-a-Judge: how do criterion, reference, and scoring actually combine?](llm-as-a-judge/README.en.md)
