# Mechanistic interpretability: opening the model to see how it computes

[中文](README.md) · **English**

> Reading time: ~7 min · Last reviewed: 2026-10

You may have seen the “Golden Gate Bridge” experiment: amplifying an internal feature made Claude repeatedly mention the bridge, even claiming to be it. Other work examined how Claude 3.5 Haiku chooses rhyming words when writing poetry. How do researchers reach those conclusions?

Mechanistic interpretability looks for features and circuits inside a model that could explain a specific behaviour, then tests the explanation by intervening on internal states. Alongside asking what the model answered, it asks which computations contributed to that answer.

As of October 2026, these tools can test local explanations for some models and prompts. They cannot yet fully explain a frontier model. Throughout the series, we distinguish what each experiment demonstrates from what remains uncertain.

## What it asks: the step beyond behavioural evaluation

Behavioural evaluation asks "on these inputs, is the output right?". Interpretability asks "which intermediate quantity inside the model produced this output, and if I change it, does the output change as predicted?".

An example. Two models both complete "Fact: the capital of the state containing Dallas is" with Austin. One may first work out Texas internally and then go from Texas to Austin; the other may simply have memorised the pairing of Dallas and Austin. Accuracy is identical, and a behavioural evaluation cannot tell them apart. In Claude 3.5 Haiku, Anthropic found both: a two-step path through "Texas" features, plus a shortcut edge from Dallas straight to "say Austin"; suppressing the "Dallas" features weakened the "Texas" features and what lies downstream of them ([Lindsey et al., March 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)). With a new set of cities, or when the model believes it is being tested, this kind of difference can start to matter.

Two key terms:

- **Feature**: the unit inside a model that represents some concept. The most common working assumption is that a feature corresponds to a direction in activation space (the linear representation hypothesis; see note 1).
- **Circuit**: a computational path connecting features through specific components (attention heads, MLPs), which can be tested by intervention.

| | Behavioural evaluation | Mechanistic interpretability |
| --- | --- | --- |
| What it asks | Is the output right on these inputs? | Which internal quantity caused this output? |
| Evidence | Accuracy, success rate, preferences | Probe readouts, patching effects, attribution graphs plus interventions |
| Strengths | Cheap, scales, maps directly onto what users care about | Separates "same score, different mechanism"; may reveal states that never show up in the output |
| Weaknesses | Does not know why the model is right; a model that knows it is being tested may behave differently | Slow, expensive, narrow coverage; often explains an approximate replacement model |

The world-models series note [Do they actually understand the world?](../world-models/do-they-understand.en.md) already covers using probes to check whether a model holds world state internally (Othello-GPT and others). This series does not repeat that; it covers the tools themselves.

## The main tools

| Tool | What it does | Evidence it gives | Main pitfall | Note |
| --- | --- | --- | --- | --- |
| Linear probe | Trains a linear classifier on frozen activations | The information can be read out | The probe may have learned the task itself; readable does not mean used | Note 1 |
| Activation steering | Adds or subtracts a direction in the activations | This direction can change behaviour | Being able to change behaviour does not make it "the" concept; strength needs tuning and capabilities can suffer | Note 1 |
| Sparse autoencoders, transcoders | Write an activation as a sum of a few dictionary directions | Many candidate features, found without labels | Reconstruction is incomplete; labels can be wrong; often lose to simple baselines downstream | Note 2 |
| Activation patching (causal tracing) | Swaps an intermediate value from one run into another | Whether a component is necessary or sufficient for a behaviour | Conclusions shift with the metric and the corruption method | Note 3 |
| Path patching, automated circuit discovery | Patches only along specific edges | Which paths a behaviour flows through | With nonlinearities, path effects do not add up to the total | Note 3 |
| Attribution graphs | Draw the feature-to-feature graph for one prompt using a replacement model | Automatically generated mechanism hypotheses | Explain the replacement model; the original version does not explain why attention looks where it does | Note 3 |
| Natural-language readouts | Have a model translate activations into text | Hypotheses you can read directly | Can invent details; cannot say which part of the activation an explanation rests on | Notes 2 and 3 |

## A timeline

- **2020-03** [Zoom In](https://distill.pub/2020/circuits/zoom-in/) (Olah et al., Distill): proposes describing networks in terms of "features" and "circuits", in vision models.
- **2021-12** [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Anthropic): writes a Transformer as layers reading from and writing to the residual stream, and finds induction heads in two-layer attention-only models.
- **2022-02** [ROME](https://arxiv.org/abs/2202.05262): uses causal tracing to find that factual recall in GPT-2 XL concentrates in mid-layer MLPs at the last subject token.
- **2022-03** [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html): early in training there is a phase change in which induction heads form at the same time as in-context learning improves; the authors mark the evidence as causal for small attention-only models and mainly correlational for large ones.
- **2022-09** [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html): superposition, "more features than dimensions", observed directly in small ReLU networks.
- **2022-11** [The IOI circuit](https://arxiv.org/abs/2211.00593): a circuit of 26 attention heads in 7 roles inside GPT-2 small.
- **2023-09 to 10** Sparse autoencoders (SAEs) arrive: [Cunningham et al.](https://arxiv.org/abs/2309.08600); Anthropic's [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) decomposes the 512 neurons of a one-layer model into up to 131,072 features.
- **2024-05** [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html): SAEs with 1M, 4M and 34M features on the middle layer of Claude 3 Sonnet; "Golden Gate Claude" is online for 24 hours as a research demo ([Anthropic](https://www.anthropic.com/news/golden-gate-claude)).
- **2024-06 to 08** OpenAI's [TopK SAE](https://arxiv.org/abs/2406.04093) trains 16 million latents on GPT-4; Google DeepMind releases [Gemma Scope](https://arxiv.org/abs/2408.05147), more than 400 open SAEs.
- **2025-03** Anthropic publishes [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html), attribution graphs for Claude 3.5 Haiku. The same month, the Google DeepMind interpretability team reports that SAEs lost to linear probes on a downstream task and that it is deprioritising fundamental SAE research ([Alignment Forum, 2025-03-26](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)).
- **2025-05 to 07** Anthropic [open-sources its attribution-graph tools](https://www.anthropic.com/research/open-source-circuit-tracing) (with Gemma-2-2b and Llama-3.2-1b as example models, integrated with Neuronpedia), then adds [QK attributions](https://transformer-circuits.pub/2025/attention-qk/index.html) to explain "why attention looks here".
- **2025-09 to 12** The [Claude Sonnet 4.5 system card](https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf) inhibits directions related to "evaluation awareness" during pre-deployment evaluation; OpenAI publishes [weight-sparse Transformers](https://arxiv.org/abs/2511.13653) (later at ICML 2026); the Google DeepMind team sets out a [pragmatic vision for interpretability](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) (12-01) and releases [Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/) (12-19).
- **2026-01 to 07** Anthropic publishes, in turn, the [assistant axis](https://www.anthropic.com/research/assistant-axis) (01-19), [emotion concepts](https://transformer-circuits.pub/2026/emotions/index.html) (04-02), [natural language autoencoders (NLAs)](https://transformer-circuits.pub/2026/nla/index.html) (05-07) and [verbalizable representations forming a global workspace](https://transformer-circuits.pub/2026/workspace/index.html) (07-06). The last borrows global workspace theory from neuroscience to organise its evidence; the authors state they take no position on subjective experience.

## As of October 2026

- **Interpretability has entered labs' pre-deployment audits, but the evidence is mostly lab-reported case studies.** The Sonnet 4.5 system card (Anthropic, September 2025) reports that after inhibiting several directions related to "this is a test", on 100 synthetic honeypot prompts the model said out loud that it was being tested less often, and misaligned actions rose from zero to at most 8–9%. The NLA paper (Anthropic, 2026-05-07) says that in the pre-deployment audit of Claude Opus 4.6, NLAs helped surface cases where the model believed it was being tested without saying so.
- **SAEs have lost standing and simple baselines have gained it.** The Google DeepMind team deprioritised fundamental SAE research in March 2025 and in December 2025 argued for trying simple methods first (prompting, steering, probing) and adding complexity only when they fall short; [AxBench](https://arxiv.org/abs/2501.17148) (January 2025) found that on steering, prompting beats every representation-based method. In 2026 there are still preprints defending SAEs ([Jørgensen & Hansen, May 2026](https://arxiv.org/abs/2605.31183)).
- **There are more open tools.** Gemma Scope 2 covers every Gemma 3 size from 270M to 27B; Google DeepMind self-reports storing about 110 PB of data and training over 1 trillion parameters. circuit-tracer plus Neuronpedia let outside researchers draw attribution graphs on open models.
- **The new kind of readout is "translate it into text".** NLA-style methods explain activations in natural language. They are easy to read, but the authors themselves list limitations: confabulation, no mechanistic grounding, and high training and inference cost.

## What has settled and what is still moving

Each cell corresponds to a dated public record; the "what to watch" column describes which result would change the assessment, not a prediction that it will happen.

| Fairly settled | Still moving | Signals to watch |
| --- | --- | --- |
| Superposition really occurs in toy models; SAEs find many readable, steerable directions in large models (Elhage et al., September 2022; Templeton et al., May 2024) | Whether these directions are features the model "really uses": with the same data and only the random seed changed, a 131K-wide SAE shares only about 30% of its features ([Paulo & Belrose, January 2025](https://arxiv.org/abs/2501.16615)) | Features stable across seeds and architectures; a clear lead when randomly initialised models are used as the baseline |
| On a small model a circuit can be verified end to end: the IOI circuit recovers 87% of the full model's logit difference and was checked for completeness and minimality (November 2022) | Automated circuit discovery on large models: Anthropic says attribution graphs give a satisfying explanation on only about a quarter of the prompts it tried (March 2025) | That fraction rising in later reports; other labs reproducing it on their own models |
| Simple linear directions can read out, and change, some safety-relevant behaviour: a refusal direction holds across 13 open models ([Arditi et al., June 2024](https://arxiv.org/abs/2406.11717)); deception probes reach AUROC 0.96–0.999 ([Apollo, February 2025](https://arxiv.org/abs/2502.03407)) | Whether such probes are robust on real deployment distributions: Apollo itself says they are not yet sufficient as a robust defence | Detection results on problems that were not deliberately implanted, reproduced by third parties |
| On downstream tasks with strong baselines, SAEs often do not come out ahead ([Kantamneni et al., February 2025](https://arxiv.org/abs/2502.16681); Google DeepMind, March 2025; AxBench, January 2025) | Whether improved SAEs reverse this: [SAEBench](https://arxiv.org/abs/2503.09532) (March 2025) finds proxy metrics disconnected from practical usefulness | New dictionary methods consistently beating difference-in-means and linear probes on proxy tasks fixed in advance |
| Interpretability tools have found concrete problems in pre-deployment audits, such as a batch of mismatched training data for Opus 4.6 (self-reported by Anthropic, May 2026) | Whether problems can be found reliably without knowing the answer: in an auditing game, 3 of 4 teams found the hidden objective and the API-only team did not ([Marks et al., March 2025](https://arxiv.org/abs/2503.10965)) | Amodei's April 2025 goal is for interpretability to "reliably detect most model problems" by 2027 ([essay](https://www.darioamodei.com/post/the-urgency-of-interpretability)); public results from third-party white-box audits |

"Fairly settled" means there is a first-hand result in the original paper's setting; it does not mean it holds for other models or tasks.

## How this relates to other notes on this site

- For the residual stream see [Residual connections](../../00-foundations/core/residual-connections.en.md), for attention heads see [Multi-head attention](../../00-foundations/core/multi-head-attention.en.md), and for MLPs see [FFN and gates](../../00-foundations/core/ffn-and-gates.en.md). Most interpretability claims are claims about "which component wrote what into the residual stream".
- For probing world models, see [Do they actually understand the world?](../world-models/do-they-understand.en.md).
- The safety part of note 3 uses reward hacking, defined in [Verifiable rewards](../../05-post-training/verifiable-rewards.en.md); for the behavioural side see [Evaluation](../../07-evaluation/README.en.md).

## Where to start in the primary sources

- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html): why neurons end up meaning several things.
- [Interpretability in the Wild (IOI)](https://arxiv.org/abs/2211.00593): what checks a circuit claim needs.
- [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255): practical advice on patching experiments.
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html): SAEs on a production model, including the authors' own limitations section.
- [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html): the method behind attribution graphs and its limitations section.
- [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496) (Sharkey et al., January 2025): a list of open problems written by authors from many institutions.
- [A Pragmatic Vision for Interpretability](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) (December 2025): a different view, starting from concrete tasks and simple baselines.

## How to read this series

1. [Features, superposition, and linear probes](features-and-superposition.en.md): the linear representation hypothesis, what probe accuracy shows, a hand-worked example of 3 features squeezed into 2 dimensions, and activation steering
2. [Sparse autoencoders](sparse-autoencoders.en.md): the reconstruction-plus-sparsity objective, two values of λ worked by hand, dead latents and feature splitting, automated labelling, scaling numbers, and the 2025–2026 critiques
3. [Circuits, attribution, and causal interventions](circuits.en.md): activation and path patching, a 3-node example by hand, the IOI circuit, attribution graphs, and how this connects to safety evaluation
4. [Review questions](review.en.md): interview questions and self-check
