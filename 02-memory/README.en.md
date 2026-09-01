# Representation and Memory: How Should a Model Remember a Person?

[中文](README.md) · **English**

## Start here: memory is not complete history

Memory is not storing every historical event. It is preserving information that remains useful, has clear provenance, can be retrieved at the right moment, and can be revised by new evidence.

## A simple example

A user who avoided coffee six months ago may be actively learning pour-over today. A useful system keeps both the historical pattern and the current intent, together with time, evidence, and uncertainty, rather than freezing one permanent preference.

## Representation versus memory

- **Representation** asks how information is encoded for use now.
- **Memory** asks what persists, how it is retrieved, and when it is revised or forgotten.

Useful layers include working memory, episodic memory, semantic memory, preference and goal state, and procedural knowledge about how to collaborate with the user.

## Four core operations

- **Write:** decide what deserves persistence.
- **Read:** retrieve the state relevant to the current task.
- **Update:** integrate new evidence without erasing useful history.
- **Forget:** remove or downweight stale, wrong, sensitive, or irrelevant information.

One vector may be too lossy when a user has multiple intents. Multi-vector representations, separate profile/history/session encoders, and task-conditioned routing are possible tools. The goal is not more vectors; it is preserving distinctions that matter for downstream behavior.

## Evaluation

Measure whether memory helps the current task, whether incorrect memory contaminates future behavior, whether correction works, whether long-term use improves experience, and whether state has inspectable evidence.
