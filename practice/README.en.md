# Engineering practice

<span id="industry-practice"></span>

[中文](README.md) · **English**

Once you can call a model, other questions tend to follow. Where does its data come from? Which computations can run ahead of time? When the result is wrong, did the model fail, or did something go wrong earlier?

This section covers three tasks: recommendation, a RAG knowledge assistant, and post-training experiments. The recommender series draws on public code; the other two use small original projects. Follow the data through a system, then change one condition and see what breaks. Read the explanations first if you prefer; the code is there when you're ready.

**Engineering practice includes more than system design.** You can start from requirements and propose a design, or follow an existing project through data, components, training, serving, and evaluation. The first develops judgment; the second checks that judgment against an implementation.

<span id="beyond-the-architecture-diagram"></span>

## Pick a question

| What went wrong? | Start here | What to look for |
| --- | --- | --- |
| You have requirements but no starting design | [System-design exercises](../learn/system-design/README.en.md) | Goals and constraints, then components, cost, and failure handling |
| Useful content wasn't recommended | [Inside recommender systems](recommender-systems/README.en.md) | Missing retrieval, filtering, or a low rank |
| Documents were retrieved, but the answer is wrong | [A RAG knowledge assistant](rag/README.en.md) | Whether evidence was current, reached the model, and was used correctly |
| Loss fell, but task performance didn't improve | [Post-training experiments](post-training/README.en.md) | A mismatched training objective or gaps in evaluation |

All three involve data, cost, and evaluation, but the checks differ. No click doesn't necessarily mean dislike in recommendation; missing evidence prevents a knowledge assistant from confidently supporting an answer. Keep those differences in mind as you read.

## System design: make the choices yourself {#system-design}

Choose a [feed](../learn/system-design/feed.en.md), [RAG assistant](../learn/system-design/rag.en.md), or [long-term memory system](../learn/system-design/memory.en.md). Write down requirements, data scale, and latency budgets. Sketch a minimal design, then change a condition: more frequent updates, a permission change, or a service timeout. What needs to change in your design?

Then read the projects below and see how implementations handle these issues. The exercises have no single standard answer, and a public project's choices depend on its context; neither should be copied unquestioningly.

<span id="start-with-recommendation-systems"></span>

## Recommendation: follow one feed refresh

Begin with [00 · Architecture](recommender-systems/00-architecture.en.md) to separate requests, content updates, and training. Then follow [01 · Requests](recommender-systems/01-feed-pipeline.en.md) → [02 · Retrieval](recommender-systems/02-candidate-retrieval.en.md) → [03 · Ranking](recommender-systems/03-ranking-and-diversity.en.md) → [05 · Evaluation](recommender-systems/05-evaluation-lab.en.md).

Once the flow makes sense, ask why it uses [two towers](recommender-systems/06-why-two-towers.en.md), how to [choose components](recommender-systems/07-component-choices.en.md), and how to [handle serving failures](recommender-systems/08-serving-lifecycle.en.md). Then read [newer public architectures](recommender-systems/04-modern-recsys.en.md) and identify which stage a new model changes.

The examples go beyond boxes: allocating slots between sources, handling missing labels, and explaining why an aggregate can rise while every group stays unchanged.

## RAG: from finding documents to answering correctly

An event's registration deadline changes from Friday to Wednesday. How does the assistant pick up the change? [Data and retrieval](rag/data-and-retrieval.en.md) handles revisions and access, combines retrieval results, then selects passages within a context limit.

[Evidence and evaluation](rag/evidence-and-evaluation.en.md) checks the answer itself. A real citation can accompany a misread date. We change dates, remove supporting sentences, and introduce conflicting documents to explain how to test each failure.

## Post-training: check what the model is learning

Keep the same assistant: the evidence is correct, but citations are often missing. [Data and objectives](post-training/data-and-objectives.en.md) starts with demonstrations, then checks which tokens contribute to loss, how much weight longer answers receive, and whether related examples have leaked into the test set.

The route also covers the engineering decisions between data and results: [SFT, LoRA and framework selection](post-training/training-plan.en.md), [batch construction](post-training/data-pipeline.en.md), [multi-GPU choices](post-training/distributed-training.en.md), and [recovery](post-training/checkpoint-and-resume.en.md). Each explains when a tool is needed, its cost, and how to check that it has not changed the intended objective.

[Experiments and release](post-training/experiments-and-release.en.md) compares old and new models on six questions. One more correct answer overall can hide an extra unsupported answer. Inspect the changes before deciding to revise, scale the experiment, or keep the old model.

<span id="want-to-try-a-design-yourself"></span>

## Run the small examples

From the repository root, without a GPU or API key:

```bash
python3 practice/rag/code/evidence_pipeline.py
python3 practice/post-training/code/experiment_checks.py
python3 practice/post-training/code/training_contracts.py
```

These programs verify small examples and contracts; they are not full-model training or deployment reports. For underlying ideas, return to [Foundations](../learn/README.en.md). For individual functions, ML questions, or algorithms, use [Interview preparation](../interview/README.en.md).

## Join us!

Share where you got stuck, what you tried, and how you found the cause. An experiment that didn't work as expected can still help someone else. See the [contribution guide](../CONTRIBUTING.md) for how to join in.

This section uses public code, papers, and synthetic data. Public-project behavior, original teaching designs, and actual measurements are distinguished; internal company projects and private datasets stay out.
