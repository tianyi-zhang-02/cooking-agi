# Notes from practice

[中文](README.md) · **English**

> Reading time: ~3 min · Type: index · Last reviewed: 2026-08

The other chapters explain how a field works. These notes are **what I actually ran into in a real system, and how I ended up thinking about it**.

Different genre, so it gets its own section: a teaching chapter can start from the cleanest example, a practice note has to start from the dirtiest — the data is skewed, the labels are guesses, and offline went up while online didn't.

## The desensitisation boundary

These notes come from real work, so the boundary has to be a **written rule** rather than a judgment call each time. I missed one once; hence the checklist.

**Never written here:**

- Company, product, team or colleague names;
- Names of internal systems, services, tables, jobs or repositories;
- Data volumes, field names, model configs, hyperparameters, measured results;
- Project timelines, or anything that pins down a specific organisational decision.

**Also never written, and the easiest to miss — internal vocabulary.**

Everyone remembers to strip the numbers. Nobody strips the words. Calling users by a particular house term, or naming a tier with a particular house name, **is itself a fingerprint** — and a sharper one than any number: a peer reads one paragraph and knows the company, often the team. A disclaimer saying "no fields or configs correspond to any company" does not help, because the leak is in the vocabulary, not the data.

So: **every proprietary term gets replaced by the generic one.** Users are users; the middle tier is the near-real-time tier.

**Only written here:** problems, trade-offs, and decision criteria that anyone building a comparable system would hit.

**Self-check:** show it to someone who builds similar systems elsewhere and ask "can you tell which company this is?" If they can, it isn't clean yet.

## The notes

### [Offline went up, online didn't](offline-online-skew.en.md)

Selective observation plus a closed feedback loop. Why log-based evaluation systematically overrates a model that will make the bias worse, and why **no change to the loss function fixes it** — only exposure that the model doesn't control, with the propensity logged.

### [What counts as a positive, and as a negative](positive-negative-design.en.md)

"No click" is not "not interested". Four sample classes with genuinely different semantics, plus a contradiction present in most implementations and rarely admitted: the same document that says unobserved items aren't negatives goes on to train with in-batch negatives, which treat them as exactly that.

### [From noisy feedback to a servable retrieval system](noise-to-signal-retrieval.en.md)

A full system design note — signal construction, selective visual understanding, contrastive post-training, release gates. Much longer than the two above; it's the one that connects the whole chain.

## The shared question

```text
How far is the label I have from the thing I want to optimise?
Is that gap random, or was it manufactured by the previous policy?
Is my evaluation set drawn from the same contaminated distribution?
```

The first two decide what the model can learn. The third decides whether you can even detect that the first two went wrong.

## Where to read next

- [Data & feedback](../01-data-and-feedback/README.en.md) — the general form of these problems
- [Evaluation](../07-evaluation/README.en.md) — why one number isn't enough
- [Post-training](../05-post-training/README.en.md) — training algorithms only start to matter once the signal is clean
