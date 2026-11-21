# Production ML: from a model to live results

[中文](README.md) · **English**

A recommendation model can score well offline and still fail in a product. The candidate set may miss what a user wants, features may arrive an hour later than they did in training, or the user may leave before the request finishes. This series uses a recommendation feed to connect models, data, services, and experiments.

Production ML means **keeping a machine-learning system useful on real requests, with a way to inspect and improve its results**. Training is one part; data updates, releases, monitoring, and recovery matter too. All examples use invented small systems, not any company's internal implementation.

## Choose a starting point {#route}

| What you want to understand | Suggested order | Try this afterward |
| --- | --- | --- |
| You know models but not the whole product | [One recommendation request](ranking-system.en.md) → [Data and experiments](data-and-experiments.en.md) | Draw the request path and feedback path, naming what each stage hands off |
| You have built models and want to check overlooked details | [Time consistency](data-and-experiments.en.md#time) → [Release versions](ranking-system.en.md#release) → [Experiment slices](data-and-experiments.en.md#experiment) | Propose a cause of an offline gain becoming an online loss, then a check that distinguishes it |
| You want implementation or system-design practice | [Request budget](ranking-system.en.md#latency) → [Small experiment](data-and-experiments.en.md#exercise) → [Capacity and failures](../15-ai-infrastructure-and-distributed-systems/queues-and-failures.en.md) | Add timeouts, fallbacks, and rollback rather than drawing only the successful path |

For vector similarity, start with [dual-encoder retrieval](../04-search/dual-encoder.en.md). To connect probability, loss, and training, use [linear models to neural networks](../00-foundations/from-linear-to-neural.en.md). You do not need to study every language-model architecture first.

## Three questions for a first pass {#core}

| Question | A tempting mistake | How we work through it |
| --- | --- | --- |
| How do retrieval and ranking work together? | A stronger ranker can recover items that were never retrieved | Track the same candidate funnel and count what each stage loses |
| Which metric establishes a product improvement? | Higher offline AUC or overall click-through rate is enough to launch | Separate model quality, online behavior, and service quality, then examine user slices |
| Do training and serving have the same information? | Reading the same feature table prevents skew | Compare event time, availability time, transformation versions, and actual serving logs |

For example, an action occurs at 09:58 but reaches the online feature store at 10:04. Using it to train a prediction made at 10:00 gives training information that serving did not have. A better ranker does not fix that mismatch. The [data chapter](data-and-experiments.en.md#time) works through the timestamps.

## Continue into the existing notes {#deeper}

This series connects the system. Detailed derivations and implementations of individual components remain in their existing chapters.

| Component | Read next | Question to bring |
| --- | --- | --- |
| Candidate generation | [Retrieval](../practice/recommender-systems/02-candidate-retrieval.en.md) · [Why two towers](../practice/recommender-systems/06-why-two-towers.en.md) | What can be precomputed? How does candidate count affect cost and coverage? |
| Ranking and reranking | [Ranking and diversity](../practice/recommender-systems/03-ranking-and-diversity.en.md) | What does the score predict, and which constraints still apply after scoring? |
| Features and training | [Data to model](../practice/ml-system/data-to-model.en.md) · [What data teaches](../01-data-and-feedback/data-as-supervision.en.md) | When does each feature or label become available? |
| Evaluation | [Evaluation lab](../practice/recommender-systems/05-evaluation-lab.en.md) · [Comparisons and slices](../07-evaluation/ablation-and-slices.en.md) | What changed in the comparison? Who is poorly represented by the aggregate? |
| Releases and feedback | [Serving lifecycle](../practice/recommender-systems/08-serving-lifecycle.en.md) · [Serving and feedback](../practice/ml-system/serving-and-feedback.en.md) | Can features, indexes, and configuration roll back with the model? |

## Who owns each part? {#ownership}

Data, modeling, infrastructure, and product work often belong to different people. Understanding the whole path does not mean implementing everything yourself. It means knowing what happens when an input is missing, who rejects incompatible versions, who investigates an unhealthy metric, and what evidence confirms a fix.

The [AI Infra route](../15-ai-infrastructure-and-distributed-systems/README.en.md) continues below the same product: why requests queue, why replicas do not absorb traffic immediately, and how training and serving share resources. The [systems learning routes](../practice/learning-routes.en.md) connect all five directions.
