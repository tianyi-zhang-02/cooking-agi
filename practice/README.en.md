# Engineering practice

<span id="industry-practice"></span>

[中文](README.md) · **English**

These pages start from concrete projects: how data, models, services, and evaluation fit together into one system, and why it is designed that way.

- **Your first complete project:** start with [this site and a reading list](web-lab/README.en.md). Follow a Markdown edit into a published page, then connect a browser, API, and database. No GPU or frontend framework is required.
- **Understanding ML systems:** read the recommendation, RAG, or post-training project and follow the data through each module. The recommender series draws on public code; the other two start from small teaching implementations.
- **Preparing for a system-design interview:** go straight to [design practice](#system-design). Propose a design yourself, then compare it with the projects' implementations.

The [project route](../learn/choose-your-path.en.md#projects) suggests other reading sequences. Engineering practice covers design, implementation, and debugging; read the explanations first and build when you have time.

<span id="beyond-the-architecture-diagram"></span>

## Pick a question

| What do you want to understand? | Start here | What to look for |
| --- | --- | --- |
| You know models but not the surrounding product | [An ML system end to end](ml-system/README.en.md) | Three timelines, data and service contracts, acceptance, and ownership |
| You've written code but haven't connected a page, server, and automated checks | [Build something small](web-lab/README.en.md) | Static hosting, CI/CD, requests, persistence, and useful errors |
| You have requirements but no starting design | [System-design exercises](../learn/system-design/README.en.md) | Goals and constraints, then components, cost, and failure handling |
| Useful content wasn't recommended | [Inside recommender systems](recommender-systems/README.en.md) | Missing retrieval, filtering, or a low rank |
| Documents were retrieved, but the answer is wrong | [A RAG knowledge assistant](rag/README.en.md) | Whether evidence was current, reached the model, and was used correctly |
| Loss fell, but task performance didn't improve | [Post-training experiments](post-training/README.en.md) | A mismatched training objective or gaps in evaluation |

Recommendation, RAG, and post-training all involve data, cost, and evaluation, but the checks differ. No click doesn't necessarily mean dislike in recommendation; missing evidence prevents a knowledge assistant from confidently supporting an answer. Keep those differences in mind as you read.

## System design: make the choices yourself {#system-design}

For the overall map, start with [data to users](ml-system/README.en.md): content updates, model training, and live requests are separate paths. Follow [data to model](ml-system/data-to-model.en.md) through examples, frameworks, checkpoints, and indexes, then [serving and feedback](ml-system/serving-and-feedback.en.md). You need not implement every layer, but should understand what each hands over.

Choose a [feed](../learn/system-design/feed.en.md), [RAG assistant](../learn/system-design/rag.en.md), or [long-term memory system](../learn/system-design/memory.en.md). Write down requirements, data scale, and latency budgets. Sketch a minimal design, then change a condition: more frequent updates, a permission change, or a service timeout. What needs to change in your design?

Then read the projects below and see how implementations handle these issues. The exercises have no single standard answer, and a public project's choices depend on its context; neither should be copied unquestioningly.

<span id="start-with-recommendation-systems"></span>

## Recommendation: follow one feed refresh

Begin with [00 · Architecture](recommender-systems/00-architecture.en.md) to separate requests, content updates, and training. Then follow [01 · Requests](recommender-systems/01-feed-pipeline.en.md) → [02 · Retrieval](recommender-systems/02-candidate-retrieval.en.md) → [03 · Ranking](recommender-systems/03-ranking-and-diversity.en.md) → [05 · Evaluation](recommender-systems/05-evaluation-lab.en.md).

Once the flow makes sense, ask why it uses [two towers](recommender-systems/06-why-two-towers.en.md), how to [choose components](recommender-systems/07-component-choices.en.md), and how to [handle serving failures](recommender-systems/08-serving-lifecycle.en.md). Then read [newer public architectures](recommender-systems/04-modern-recsys.en.md) and identify which stage a new model changes.

The worked examples show how to allocate slots between retrieval sources, handle missing labels, and explain why an aggregate score can rise while every group's performance stays unchanged.

## RAG: from finding documents to answering correctly

An event's registration deadline changes from Friday to Wednesday. How does the assistant pick up the change? [Data and retrieval](rag/data-and-retrieval.en.md) handles revisions and access, combines retrieval results, then selects passages within a context limit.

[Evidence and evaluation](rag/evidence-and-evaluation.en.md) checks the answer itself. A real citation can accompany a misread date. We change dates, remove supporting sentences, and introduce conflicting documents to explain how to test each failure.

## Post-training: check what the model is learning

Keep the same assistant: the evidence is correct, but citations are often missing. [Data and objectives](post-training/data-and-objectives.en.md) starts with demonstrations, then checks which tokens contribute to loss, how much weight longer answers receive, and whether related examples have leaked into the test set.

The route also covers the engineering decisions between data and results: [SFT, LoRA and framework selection](post-training/training-plan.en.md), [batch construction](post-training/data-pipeline.en.md), [multi-GPU choices](post-training/distributed-training.en.md), and [recovery](post-training/checkpoint-and-resume.en.md). Each explains when a tool is needed, its cost, and how to check that it has not changed the intended objective.

[Experiments and release](post-training/experiments-and-release.en.md) compares old and new models on six questions. One more correct answer overall can hide an extra unsupported answer. Inspect the changes before deciding to revise, scale the experiment, or keep the old model.

<span id="want-to-try-a-design-yourself"></span>

## Run the small examples

Once a model is trained, other people need to use it reliably. [A request through the service](../06-systems/llm-serving.en.md) connects inference mechanics to system design: queueing, long-prompt chunks, cancellation, and the choice between replicas and model partitioning. Use that flow to plan a representative load test rather than starting with a supposedly fastest framework.

From the repository root, without a GPU or API key:

```bash
python3 practice/rag/code/evidence_pipeline.py
python3 practice/post-training/code/experiment_checks.py
python3 practice/post-training/code/training_contracts.py
```

These programs check the rules and calculations in the small examples; they are not reports of full-model training or deployment. For underlying ideas, return to [Foundations](../learn/README.en.md). For individual functions, ML questions, or algorithms, use [Interview preparation](../interview/README.en.md).

## Join us!

Share where you got stuck, what you tried, and how you found the cause. An experiment that didn't work as expected can still help someone else. See the [contribution guide](../CONTRIBUTING.md) for how to join in.

This section uses public code, papers, and synthetic data. Public-project behavior, original teaching designs, and actual measurements are distinguished; internal company projects and private datasets stay out.
