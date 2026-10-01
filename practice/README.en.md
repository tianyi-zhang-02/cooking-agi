# Industry practice

[中文](README.md) · **English**

Understanding a model is different from understanding the system around it. Here we follow requests through public projects: where the data comes from, what each component does, what can go wrong, and how to check a change.

## Start with recommendation systems

Why do those particular posts appear when you refresh a feed? We use Twitter's open-source code as a starting point, then look at how recommendation systems have changed. This is not a file-by-file translation or a recipe for recreating a production service.

| Episode | What we'll work through |
| --- | --- |
| [01 · How a post reaches your feed](recommender-systems/01-feed-pipeline.en.md) | Candidate sources, retrieval, ranking, and filtering |
| [02 · Find candidates before comparing them](recommender-systems/02-candidate-retrieval.en.md) | What social graphs, interest communities, and dual encoders retrieve |
| [03 · The best item isn't the best list](recommender-systems/03-ranking-and-diversity.en.md) | Multiple objectives, repetition, and an interactive diversity experiment |
| [04 · What has changed in modern recommenders](recommender-systems/04-modern-recsys.en.md) | Longer histories, Transformers, multimodal inputs, and generation |
| [05 · The score went up. Now what?](recommender-systems/05-evaluation-lab.en.md) | Candidate pools, cohort evaluation, and a Python example where an unchanged model loses points |

[Read the series introduction →](recommender-systems/README.en.md)

For the reasoning behind those choices, continue with:

- [06 · Why two towers](recommender-systems/06-why-two-towers.en.md): trading representational flexibility for reusable computation, and when not to.
- [07 · Component choices](recommender-systems/07-component-choices.en.md): alternatives for encoders, indexes, rankers, feature updates, and merging.
- [08 · Serving lifecycle](recommender-systems/08-serving-lifecycle.en.md): interfaces, timeouts, versions, freshness, rollout, and rollback, with a runnable reference.

## Beyond the architecture diagram

Each note tries to answer four questions:

1. **What's the problem?** State the task and constraints; separate observations from hypotheses.
2. **Why this choice?** Compare alternatives and what each gives up.
3. **How do the pieces connect?** Follow the data through public code, interfaces, and boundaries.
4. **How would we verify it?** Distinguish teaching examples from actual measurements.

A well-understood bug or an experiment that rules out a hypothesis is worth recording. No clear gain doesn't mean nothing was learned.

## Want to try a design yourself?

The [system-design exercises](../learn/system-design/README.en.md) cover a community feed, a RAG knowledge base, and an assistant with memory. There you make choices under stated constraints; here you read existing public implementations. Use one to question the other.

For the underlying ideas, return to [Study notes](../learn/README.en.md).

## Join us!

If you've debugged, tested, or implemented something in a public project, we'd love to read about it. Explain the context, process, and evidence; it doesn't need to be a universal solution. See the [contribution guide](../CONTRIBUTING.md).

This section uses public materials, synthetic data, and reproducible experiments—not internal company projects or private datasets.
