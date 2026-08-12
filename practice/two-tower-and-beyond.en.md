# Two towers, and why the user side becomes several

[中文](two-tower-and-beyond.md) · **English**

> Reading time: ~6 min · Type: practice note · Last reviewed: 2026-08

## In one sentence

The two-tower architecture isn't a modelling preference that happens to work — it is the only shape **precomputation** allows. Once that's clear, it explains why the user side can split into several towers and the content side cannot.

## The constraint produces the architecture

First-stage retrieval faces a corpus around $10^8$ items. No request can afford $10^8$ forward passes, so the scoring function **must factorise**:

$$s(u,c) = \big\langle f_\theta(u),\; g_\phi(c) \big\rangle$$

$g_\phi(c)$ doesn't depend on the user, so it is computed offline and written into a vector index. Online you run $f_\theta(u)$ once and hand the rest to ANN search.

**That factorisation is the entire definition of a two-tower model.** The two towers aren't there for symmetry; they're there because the split has to fall between user and content.

The cost comes from the same place: **$u$ and $c$ never interact until that final inner product.** The model cannot express "paragraph three of this candidate answers exactly what this user asked about last week" — it can only compress each side into a vector and take a dot product.

That line is hard:

| Architecture | Interaction | Candidates it can score | Stage |
| --- | --- | --- | --- |
| Two-tower | only the final inner product | $10^8$ (via ANN) | retrieval |
| Late interaction (e.g. ColBERT) | token-level, deferred to query time | $10^5$–$10^6$ | pre-ranking |
| Cross-encoder | full | $10^2$–$10^3$ | ranking |

More interaction, fewer candidates. This is a **structural** capability/cost trade-off, not something tuning gets around. Two towers are used in retrieval not because they're accurate but because nothing else runs at that scale — and the job there isn't to rank precisely, it's to **not miss**.

## The problem with one user vector

A single user tower output compresses every interest into one point.

If someone follows both machine-learning papers and home cooking, the mean vector lands between the two. **ANN takes top-K around that mean, and what comes back may belong to neither interest.**

This isn't a capacity problem — swapping sub-1B for 8B doesn't fix it. It's a problem of **representational form**: a mean doesn't preserve modes. And it has an ugly property — **the more varied a user's interests, the more they get diluted**, which is exactly the population that most needs personalisation.

$$\mathbf{z}_u = \frac{1}{|H_u|}\sum_{i \in H_u} \mathbf{e}_i \quad\Longrightarrow\quad \text{minority interests are divided away by } |H_u|$$

## So the user side expands and the content side doesn't

The content side stays a single tower: **a piece of content has one identity.** It is about what it is about; there is no "multiple intents" to represent.

The user side expands into several vectors:

- **profile view** — stable but generally weaker long-term evidence;
- **history view** — stronger but narrower recent evidence;
- **latent intent views** — keep minority interests from being averaged away.

The asymmetry is principled rather than decorative. **It follows from the two sides being different kinds of object: content is an item, a user is a distribution.**

## Serving with several vectors

The one thing that must not happen: **averaging them before querying.** That walks straight back to a single point and discards the whole exercise.

Instead, allocate a query budget per view under a fixed total $B$:

$$K_r = \operatorname{round}\!\big(B \cdot \operatorname{softmax}(g(u))_r\big), \qquad \mathcal{C}(u) = \bigcup_r \operatorname{ANN}(\mathbf{z}_{u,r},\, K_r)$$

Each view queries the **same** candidate index; results are unioned, deduplicated, and passed to the ranker along with which view produced them, the similarity, and the router weight.

Note $\sum_r K_r = B$ is fixed. **Multi-tower isn't "retrieve more", it's "cover more interests within the same number of candidates".** Online candidate volume is unchanged; complementarity is what grows. That's also what makes it shippable — otherwise ranking cost grows with it.

## When multi-tower isn't worth it

One criterion: **do the views contribute relevant candidates the other views cannot?**

If view A and view B overlap 80%, you ran the same query twice, paid double the ANN cost, and bought a 20% margin. A single tower with longer context is usually the better deal there.

So what needs monitoring isn't "we have more towers", it's:

- **routing entropy** — if every user's softmax collapses onto one view, the others are decoration;
- **view utilisation** — a view that never earns budget never learned anything of its own;
- **unique-target contribution** — the share of relevant candidates only that view produced, the one metric that maps directly to value.

All three must be **constrained during training**, not observed after launch. Nothing in a retrieval loss rewards keeping views diverse; the cheapest solution is for every view to converge to the same vector.

## Two training traps

**Sampling bias in in-batch negatives.** Two-tower training almost always uses them, and their sampling distribution is proportional to item frequency, so it needs a logQ correction. See [what counts as a positive, and as a negative](positive-negative-design.en.md).

**The index goes stale relative to the encoder.** Retrain the content tower and every vector in the index must be re-encoded. During a staged rollout the index holds a mix of old and new vectors — **the space is inconsistent and similarities are not comparable**. It is the easiest train/serve parity break to miss, because it doesn't live in the code, it lives in the release process.

## Where to read next

- [What counts as a positive, and as a negative](positive-negative-design.en.md)
- [Offline went up, online didn't](offline-online-skew.en.md) — why retrieval is especially hard to evaluate offline
- [From noisy feedback to a servable retrieval system](noise-to-signal-retrieval.en.md)

## Papers to start with

- [DSSM](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/) — 2013, where two towers start
- [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/pub45530/) — the canonical retrieval/ranking split
- [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/pub48840/) — the logQ correction
- [MIND](https://arxiv.org/abs/1904.08030), [ComiRec](https://arxiv.org/abs/2005.09347) — multi-interest user representation
- [PinnerSage](https://arxiv.org/abs/2007.03634) — clustering user actions into multiple embeddings in production
- [ColBERT](https://arxiv.org/abs/2004.12832) — late interaction, between the two extremes
