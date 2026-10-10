# Interview preparation

<span id="code-exercises-turn-understanding-into-practice"></span>

[中文](README.md) · **English**

Recognizing a concept, explaining it, and implementing it are different skills. This section separates preparation into three routes: ML / LLM fundamentals, ML coding, and Python with algorithm patterns. Pick the skill you need rather than treating everything as a checklist.

The material uses public knowledge and original exercises, not confidential company questions. It is also useful for checking your understanding outside interview season.

## Choose what to practice {#prep-routes}

<nav class="study-route prep-route" aria-label="Interview preparation routes">
<a href="basics/README.en.md"><small>01 / EXPLAIN</small><strong>ML / LLM fundamentals</strong><span>Probability, optimization, mechanisms, and tradeoffs</span></a>
<a href="../learn/ml-exercises/README.en.md"><small>02 / IMPLEMENT</small><strong>ML coding</strong><span>PyTorch, training loops, attention, and losses</span></a>
<a href="leetcode.en.md"><small>03 / SOLVE</small><strong>Python & LeetCode</strong><span>Containers, complexity, and reusable patterns</span></a>
</nav>

| Where you get stuck | Start here | What to check |
| --- | --- | --- |
| Familiar terms, difficult follow-up questions | [Fundamentals review](basics/README.en.md): explain one concept with a small example | Assumptions, and whether the conclusion survives a changed condition |
| Correct formula, wrong tensor shapes | [ML coding](../learn/ml-exercises/README.en.md): start with a small implementation | Outputs, gradients, masks, and numerical stability |
| Solutions make sense, but finding one is hard | [Algorithm patterns](leetcode.en.md): practice methods rather than memorizing answers | Why a method applies, and its time and space costs |

## ML questions & implementations

Practice these together, but distinguish the gaps. Explaining why attention scales a dot product belongs in [fundamentals review](basics/README.en.md). Implementing its causal mask and checking the softmax dimension belongs in [ML coding](../learn/ml-exercises/README.en.md). Return to [Foundations](../learn/README.en.md) for an unfamiliar step.

## Python & algorithms

If syntax slows you down, start with [Useful Python](python.en.md) and run the examples. For algorithms, work from complexity, hash maps, pointers, binary search, and DFS / BFS toward backtracking and DP. The emphasis is on transferable Easy / Medium methods, not completing a list of Hard problems.

If a copy changes the original, or a function reuses data from an earlier call, try [names, copies, and function calls](python-objects.en.md). Following which objects are shared is more useful than memorizing a separate gotcha for every symptom.

For [DFS / BFS](algorithms/traversal.en.md), trace a small graph with both a stack and a queue. Check when visited nodes are marked and which traversal guarantees shortest paths in an unweighted graph instead of just memorizing the templates.

## System design

System design lives in [Engineering practice](../practice/README.en.md), alongside project walkthroughs. For interview preparation, go directly to the [feed, RAG, and memory exercises](../learn/system-design/README.en.md): clarify requirements, compare alternatives, and explain failure handling. A diagram or a list of frameworks alone is not a design.

## How to review

1. Explain or implement it yourself, then read the relevant passage when you get stuck.
2. Check a small input. Write down assumptions, edge cases, and complexity.
3. Change one condition. Does the method still work with padding, or with data that no longer fits in memory?

For a first encounter with a topic, the sequential explanations in [Foundations](../learn/README.en.md) are often easier than a question list. Personal experiences, mindset, and preparation habits remain in [Career](../career/README.en.md).
