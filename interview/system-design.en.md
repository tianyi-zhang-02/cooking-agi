# System design: reading and trade-offs

[中文](system-design.md) · **English**

> Reading time: ~2 min · Last reviewed: 2026-10

> **Read this first**: interview formats and emphases change fast, so this is time-sensitive; it lists public resources and my own understanding only, with no company's questions or process.

I'm still learning system design too. This page keeps the public resources and ideas I've found useful. For hands-on practice, start with the [system-design exercises](../learn/system-design/README.en.md): a feed, a RAG knowledge base, or an assistant with memory.

## There is no fixed formula

In my experience, there is no system-design template you can memorize and apply unchanged. In an interview, you need to explain two things: **the trade-offs behind each choice** and **how the parts work together as a complete system**.

So start from the basics:

1. What problem are we solving? What does success look like?
2. What data do we have? What data do we not have?
3. What are the constraints — scale, latency, cost, freshness?
4. For each step, what does this choice give up?

## An example: recommendation

Say you are building recommendations. Start from what you have:

- **If follows, friendships, or interactions are available**, use those relationships to find candidates. A graph need not be a social graph: [PinSage](https://arxiv.org/abs/1806.01973) uses Pinterest's item–board relationships. Keep these different data structures distinct.
- **And if there is no relationship data?** Fall back to behavior itself: item–item co-occurrence, two-tower retrieval, content features (titles, categories, multimodal embeddings). In cold start, content features and a popularity fallback often beat a complicated model.

Each subsequent layer involves trade-offs: how fast retrieval has to be, how large a model the ranking stage can afford, how fresh the features must be, whether an offline gain will show up online, and whether the complexity is worth maintaining. Explaining these choices is more useful than reciting an architecture diagram.

## Public resources

General system design:

- [System Design Primer](https://github.com/donnemartin/system-design-primer): free and broad, with a useful overview of system design
- [ByteByteGo](https://bytebytego.com): Alex Xu's *System Design Interview* series, heavily illustrated and easy to read
- [Designing Data-Intensive Applications](https://www.dataintensive.net): the one worth real time if you want to understand the underlying trade-offs. As of October 2026 the latest is the [second edition, published in March 2026](https://martin.kleppmann.com/2026/03/24/designing-data-intensive-applications-2e.html) (Martin Kleppmann and Chris Riccomini, O'Reilly); the book's website still describes the first edition

ML and recommendation:

- [Designing Machine Learning Systems](https://huyenchip.com/books/) (Chip Huyen)
- [Evidently's ML system design case collection](https://www.evidentlyai.com/ml-system-design): hundreds of public write-ups of real systems
- [Eugene Yan: system design for discovery](https://eugeneyan.com/writing/system-design-for-discovery/): the layered design of search and recommender systems
- [Twitter's open-sourced recommendation algorithm](https://github.com/twitter/the-algorithm): public components of a recommendation pipeline, not a complete deployable production service

## Related notes here

- [Search: how a model finds what it does not currently know](../04-search/README.en.md)
- [Modern AI systems: the model is only one part](../06-systems/README.en.md)
- [Agents: how they are used in different settings](../10-agents/scenarios.en.md)
- [Evaluation: why should we believe the system improved](../07-evaluation/README.en.md)
- [Kubernetes and GPU clusters](../12-infra/kubernetes.en.md): why distributed training needs gang scheduling, and why scaling inference is more than adding replicas (draft, pending review)
