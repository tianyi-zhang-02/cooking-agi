# Industry Practice

[中文](README.md) · **English**

Study notes explain how a method works. This section follows what happens when we put it to use: how a problem was noticed, why a particular change was chosen, and how we checked whether it helped.

There isn't much here yet, and I'll add to it over time. Everything here needs public code, references, or reproducible experiments behind it—not internal company projects.

## What a practice note covers

1. **What was the problem?** Describe the task, constraints, and observations. Separate verified facts from hypotheses.
2. **Why this approach?** Compare the options and explain both the choice and its trade-offs.
3. **How was it implemented?** Use public code to follow the data flow, interfaces, and important details.
4. **How was it checked?** Record reproduction steps, tests, and measurement conditions, including what hasn't been verified.

A note doesn't need a big performance improvement to be worth sharing. A well-understood bug, an experiment that ruled out a hypothesis, or an approach that wasn't adopted can help the next person facing a similar problem.

## When you need the foundations

- [Post-training](../05-post-training/README.en.md): understand the training objective before following its implementation.
- [Evaluation](../07-evaluation/README.en.md): design experiments and understand what a metric can and cannot tell you.
- [Systems and multimodal](../06-systems/README.en.md): follow a request through its components and observe the intermediate steps.

Small teaching experiments stay in [Study notes](../00-foundations/README.en.md). This section focuses on working through concrete problems. The two link to each other rather than duplicating articles.

## Share your own experience

Debugging stories, experiments, and implementations from public projects are welcome. Explain the context, process, and evidence; it doesn't need to be a complete solution. See the [contributing guide](../CONTRIBUTING.md#english) to get involved.

Please don't submit internal company materials, private data, or other sensitive information. Adding this section does not make any existing private notes public.
