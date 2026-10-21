# Interpretability: sparse autoencoders

[中文](sparse-autoencoders.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

"34 million features extracted from Claude 3 Sonnet", "16 million latents trained on GPT-4": what do these numbers mean? And why did Google DeepMind's interpretability team announce in March 2025 that it was deprioritising fundamental SAE research?

A sparse autoencoder (SAE) writes an activation as a weighted sum of a few dictionary directions. Without labels, it finds many directions that admit readable semantic descriptions and can steer the model, making it useful for exploration and debugging. But accurate reconstruction and fluent labels do not guarantee that these are features the model actually uses. On downstream tasks with strong baselines, SAEs often perform worse than linear probes, difference-in-means and prompting.

## The objective: reconstruction plus sparsity

[The previous note](features-and-superposition.en.md) argued that if features are directions in superposition, you have to go and find them. An SAE is an approximation to dictionary learning: a very wide autoencoder with one hidden layer that must reconstruct its input while keeping most of the hidden layer at zero. This is the form Anthropic uses in [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html):

$$
f(x)=\mathrm{ReLU}(W_{\rm enc}x+b_{\rm enc}),\qquad \hat x=W_{\rm dec}f(x)+b_{\rm dec}
$$

$$
L=\mathbb E_x\Big[\|x-\hat x\|_2^2+\lambda\sum_i f_i(x)\,\|W_{\rm dec}^{(i)}\|_2\Big]
$$

$x\in\mathbb R^d$ is an activation somewhere in the model, $f(x)\in\mathbb R^F$ is the sparse code, and $F\gg d$ (overcomplete). Column $i$ of $W_{\rm dec}$, written $W_{\rm dec}^{(i)}$, is the direction of feature $i$. The first term asks for accurate reconstruction, the second is an L1 penalty, and $\lambda$ sets the trade-off between them. Multiplying by the norm of the decoder column stops the model from dodging the penalty by shrinking $f_i$ and enlarging the decoder vector.

| Variant | How sparsity is controlled | What it fixes | Cost or problem | Used in |
| --- | --- | --- | --- | --- |
| ReLU + L1 | Penalise the sum of activations, tune $\lambda$ | Simple | Shrinkage: non-zero activations are systematically underestimated | [Bricken et al. 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html), Templeton et al. 2024 |
| TopK | Keep only the $k$ largest per token; the loss is just reconstruction | Sets sparsity directly, with no L1 shrinkage | The features learned are less stable across random seeds | [Gao et al. 2024](https://arxiv.org/abs/2406.04093) (OpenAI) |
| JumpReLU + L0 | Learn a threshold per latent and penalise the number of non-zeros directly | More accurate reconstruction at the same sparsity | Not differentiable at the threshold, so it needs straight-through estimators | [Rajamanoharan et al. 2024](https://arxiv.org/abs/2407.14435), Gemma Scope |
| Transcoder | Predict an MLP's output from the MLP's input | Can replace the MLP directly, which helps with drawing circuits | Only an approximation of the MLP; the mechanism may differ | [Dunefsky et al. 2024](https://arxiv.org/abs/2406.11944); see [note 3](circuits.en.md) |

## By hand: one 2-D activation, three dictionary directions (a teaching example)

Take $d=2$ and a dictionary of three unit directions: $d_1=(1,0)$, $d_2=(0,1)$, $d_3=(0.707,0.707)$. The activation is $x=(1,1)$. To see the objective itself, we solve directly for the best non-negative coefficients rather than going through an encoder:

$$
\min_{f\ge 0}\ \|x-Df\|_2^2+\lambda\,(f_1+f_2+f_3)
$$

With $\lambda=0$ there are infinitely many solutions: $(1,1,0)$, $(0,0,\sqrt2)$ and any mixture of the two all reconstruct $x$ exactly. Reconstruction alone cannot pick out which of them is "the" feature.

Adding L1 changes that. For an exact reconstruction, using only $d_3$ costs $\sqrt2\approx1.414$ in L1, while $d_1+d_2$ costs 2. Compare the optimal coefficients of the two options at two values of $\lambda$: with $d_3$ alone, $a=\sqrt2-\lambda/2$; with only $d_1,d_2$, $b=1-\lambda/2$ (take the derivative of the loss with respect to the coefficient and set it to zero).

| $\lambda$ | Option | Coefficient | Reconstruction | Squared error | $\lambda\times$ L1 | Total loss |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | $d_3$ only | $a=1.364$ | (0.965, 0.965) | 0.0025 | 0.136 | **0.139** |
| 0.1 | $d_1+d_2$ | $b=0.95$ | (0.95, 0.95) | 0.005 | 0.190 | 0.195 |
| 1.0 | $d_3$ only | $a=0.914$ | (0.646, 0.646) | 0.25 | 0.914 | **1.164** |
| 1.0 | $d_1+d_2$ | $b=0.5$ | (0.5, 0.5) | 0.5 | 1.0 | 1.5 |

At both values of $\lambda$, using only $d_3$ wins (the KKT conditions confirm it is the global optimum). Three things can be read off the table:

- **L1 prefers few directions that point "straight at" the data.** If the data were really two independent features that often co-occur, an SAE might learn one merged direction instead of the two atomic features.
- **Shrinkage.** Without the penalty, the coefficient on $d_3$ should be $\sqrt2\approx1.414$. At $\lambda=0.1$ it loses 0.05; at $\lambda=1.0$ only 0.914 is left. The squared reconstruction error is 0.25, or 12.5% of $\|x\|^2=2$, giving $1-\|x-\hat x\|^2/\|x\|^2=87.5\%$. This is the complement of relative squared error, not a claim that the reconstructed vector retains 87.5% of the energy. For $\lambda\ge2\sqrt2\approx2.83$ the optimum is all zeros and nothing fires.
- **TopK makes a different trade.** With $k=1$ it picks the single direction that best explains $x$, which is $d_3$, with a coefficient equal to the projection $\sqrt2$: exact reconstruction and no shrinkage. The price is that how many latents fire per token is fixed in advance.

A real SAE does not optimise the coefficients separately for each input. It computes them directly with a single-layer encoder, whose result can match the optimum under the same objective but may be worse.

<!-- paper-figure:jumprelu-toy -->

## Dead latents, feature splitting, and absorption

**Dead latents.** In the example above, $d_2$ is never chosen when $x$ lies on the diagonal. If no input in the whole training set makes a ReLU latent fire, it gets no gradient from the reconstruction term and never comes back. Anthropic counts a feature as dead if it has not fired over $10^7$ tokens: the 1M, 4M and 34M SAEs had roughly 2%, 35% and 65% dead features, and the 34M one had only about 12 million alive ([Templeton et al., May 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)). OpenAI reports up to 90% dead latents without mitigations; initialising the encoder as the transpose of the decoder and adding an auxiliary loss in which dead latents fit the reconstruction residual (AuxK) left only 7% dead in a 16-million-wide SAE ([Gao et al., June 2024](https://arxiv.org/abs/2406.04093)).

**Feature splitting.** As the dictionary grows, one feature splits into several finer ones. Bricken et al. saw a base64 feature in a one-layer model split into three; Templeton et al. saw a "San Francisco" feature in the 1M SAE become 2 features in the 4M SAE and 11 in the 34M SAE. Dictionaries of different sizes act like different resolutions.

**Absorption.** [Chanin et al. (September 2024)](https://arxiv.org/abs/2409.14507) found that when features form a hierarchy, a seemingly monosemantic parent feature fails to fire where it should, because its role is "absorbed" by child features; they attribute this to optimising for sparsity and find that changing dictionary size or sparsity does not fix it.

**Change the seed and the features change.** [Paulo & Belrose (January 2025)](https://arxiv.org/abs/2501.16615) trained 131K-wide SAEs on an MLP of Llama 3 8B, changing only the random seed, and found only about 30% of features shared; TopK SAEs depend on the seed more than L1 ones. The authors conclude that the features an SAE finds should be viewed as "a pragmatically useful decomposition of activation space, rather than an exhaustive and universal list …" of the features the model uses.

## How features get labels: automated interpretation and its limits

Millions of features cannot be inspected by hand, so models are used to label models. The pipeline: collect the contexts where a feature fires most strongly, have an LLM write a one-sentence explanation, then score the explanation. [Bills et al. (OpenAI, May 2023)](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) used GPT-4 to explain GPT-2 XL neurons and scored by simulation: another model predicts activations from the explanation alone, and the score is the Pearson correlation between predicted and true activations. [Paulo et al. (October 2024)](https://arxiv.org/abs/2410.13928) added several cheaper scoring methods, including intervention scoring.

| Scoring method | What it asks | What it cannot see |
| --- | --- | --- |
| Simulation | Can the explanation alone predict activation values? | What role the feature plays in the output |
| Detection, fuzzing | Can the explanation alone separate contexts (or tokens) where it fires from those where it does not? | Whether several different features are described by the same sentence |
| Intervention scoring | Can the explanation alone predict how the output changes when the feature is changed? | Whether the model actually uses it that way on natural inputs |

The limits come in three layers. First, a label says "when it fires", not "what it does": [Huang et al. (September 2023)](https://arxiv.org/abs/2309.10312) checked the explanations from Bills et al. and found that even the most confident ones had high error rates and almost no causal efficacy. Second, you need random baselines: [Heap et al. (January 2025)](https://arxiv.org/abs/2501.17727) found that SAEs trained on randomly initialised Pythia models get auto-interpretability scores and reconstruction metrics similar to those of trained models in many settings. Third, labels collide: a single-author preprint ([McCann, May 2026](https://arxiv.org/abs/2605.12874)) reanalysed 722 human-annotated features (Gemma 2 2B and Pythia 70M) and found that each label is shared by 3.07 features on average, 82.1% of features share their label with at least one other, and the single label "plural nouns" covers 101 features; the author proves that detection-style scoring is completely insensitive to such collisions.

That is why Anthropic's 2024 work did not stop at activation examples but also clamped features high and low to check that behaviour matched the label (the Golden Gate Bridge in the previous note is one case).

## Scaling: a few public numbers

| Work | Model and location | Dictionary size | Training data | Reported numbers |
| --- | --- | --- | --- | --- |
| Towards Monosemanticity (Anthropic, October 2023) | MLP of a one-layer Transformer, 512 neurons | 512 to 131,072 (1× to 256×) | 8 billion activation samples | Human and automated evaluations both find most features interpretable |
| Scaling Monosemanticity (Anthropic, May 2024) | Middle-layer residual stream of Claude 3 Sonnet | 1M, 4M, 34M | Not disclosed | Fewer than 300 features active per token on average; at least 65% of variance explained; about 2%, 35% and 65% dead features |
| TopK SAE (OpenAI, June 2024) | GPT-4 residual stream, about 5/6 of the way through | 16M | 40 billion tokens | 7% dead latents; with the SAE substituted into GPT-4, language-modelling loss matches a model trained with 10% of GPT-4's pretraining compute |
| [Gemma Scope](https://arxiv.org/abs/2408.05147) (Google DeepMind, August 2024) | All layers and sublayers of Gemma 2 2B and 9B, selected layers of 27B | 400+ SAEs, over 30 million features in total | 4 to 16 billion tokens each | Compute of over 20% of GPT-3's training compute; about 20 PiB of activations stored |
| [Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/) (Google DeepMind, 2025-12-19) | Gemma 3 from 270M to 27B, every layer | SAEs, transcoders, skip transcoders, cross-layer transcoders | Not given in the blog post | Self-reported: about 110 PB of data stored, over 1 trillion total parameters trained |

How to read these numbers: "variance explained" and "loss after substituting it into the model" are different things. OpenAI points out that measuring loss recovered against a baseline of zeroing the activations lets even poor SAEs score highly (their 16M SAE scores 98.2% that way), so they use "equivalent pretraining compute" instead. The 10% compares language-modelling loss after SAE substitution with the corresponding pretraining-compute level; it is not a parameter-count ratio and does not by itself determine an equivalent model's size. Anthropic itself writes that it is probably orders of magnitude short of "finding all the features", and that finding them all might take more compute than training the model.

## The 2025–2026 critiques: on downstream tasks

| Work | Task | Baselines compared | Result |
| --- | --- | --- | --- |
| [Kantamneni et al.](https://arxiv.org/abs/2502.16681) (February 2025) | Sparse probing under data scarcity, class imbalance, label noise and covariate shift | Linear probes and others | SAEs occasionally win on individual datasets, but adding them to an ensemble does not consistently beat ensembles of baselines alone |
| [Google DeepMind team](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks) (2025-03-26) | Detecting harmful user intent, tested out of distribution (Gemma 2 9B IT, layer 20) | Linear probe on the residual stream, OOD AUROC 0.999 | k-latent SAE probes are distinctly worse out of distribution; the team deprioritises fundamental SAE research |
| [AxBench](https://arxiv.org/abs/2501.17148) (January 2025) | Steering and concept detection (Gemma 2 2B and 9B) | Prompting, finetuning, difference-in-means | Steering: prompting is best, then finetuning; detection: difference-in-means is best; SAEs are not competitive on either |
| [SAEBench](https://arxiv.org/abs/2503.09532) (March 2025) | 8 metrics, over 200 SAEs | 8 architectures compared with each other | Gains on proxy metrics do not reliably translate into practical performance |
| [Anthropic](https://transformer-circuits.pub/2024/features-as-classifiers/index.html) (October 2024) | Classifying bioweapon-related prompts | Linear classifier on raw activations | Feature classifiers are sometimes comparable or slightly better; raw activations are a strong baseline; the benefit of features is spotting spurious correlations in the training data |

SAEs have also proved useful, mostly in situations where you do not know what to look for. In Anthropic's auditing game, one team allowed to use SAEs found the hidden objective in 90 minutes, one person following a "reward model bias" feature; but the paper also notes that this solution could be reproduced with a simpler embedding-based semantic search ([Marks et al., March 2025](https://arxiv.org/abs/2503.10965)). OpenAI used SAEs to compare models before and after finetuning and found a "toxic persona" feature that most strongly controls emergent misalignment and can predict whether a model will show it ([Wang et al., June 2025](https://arxiv.org/abs/2506.19823)). The Google DeepMind team also says SAEs help with exploratory debugging and finding spurious correlations in datasets, and are worth keeping in the toolbox.

In 2026 both views are heard: a preprint reports that with a supervised pipeline to select and label features, SAEs come close to the LoRA reference on AxBench ([Jørgensen & Hansen, May 2026](https://arxiv.org/abs/2605.31183)), while the Google DeepMind team's December 2025 [pragmatic agenda](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) argues for trying prompting, steering and probing first and adding complexity only when they fall short. The practical conclusion: when you already know which concept you are after and have labels, try a linear probe or difference-in-means first; SAEs have the advantage when you do not know what to look for and need hypotheses to surface without supervision.

## Where it is heading: transcoders, cross-layer dictionaries, and natural-language readouts

An SAE only reconstructs the activation at one place; it does not say how computation flows from one layer to the next. A [transcoder](https://arxiv.org/abs/2406.11944) instead predicts an MLP's output from its input; its authors report that on models of 120M, 410M and 1.4B parameters it is at least on par with SAEs on sparsity, faithfulness and interpretability. Anthropic's [crosscoder](https://transformer-circuits.pub/2024/crosscoders/index.html) (October 2024) lets one feature read and write across several layers and is also used to compare two models. Cross-layer transcoders are the basis of the attribution graphs in [note 3](circuits.en.md).

<!-- paper-figure:transcoder-vs-sae -->

Another route drops the fixed dictionary and has a model translate activations into text directly. Anthropic's [NLA](https://transformer-circuits.pub/2026/nla/index.html) (2026-05-07) uses two modules initialised from the target model, one that writes an activation as an explanation and one that reconstructs the activation from the explanation, trained jointly with RL to reconstruct the residual stream. The authors report it was useful in the pre-deployment audit of Claude Opus 4.6 and list its limits: it invents details that are not in the context, it cannot say which part of the activation an explanation rests on, the two modules are very expressive, and it is expensive to train and to use.

## What to watch on this line next

1. **Can it beat strong baselines?** As of October 2026, the results from Kantamneni et al., Google DeepMind and AxBench all have SAEs not coming out ahead. A result that would change the assessment: a new dictionary method that consistently beats linear probes and difference-in-means on proxy tasks fixed in advance.
2. **Can the labels be trusted?** The random-model baseline of Heap et al. (January 2025) and the collision problem of McCann (May 2026) both show that current scoring overestimates. A result that would change the assessment: major SAE releases reporting random-model baselines by default and using metrics that penalise collisions.
3. **Are the features stable?** Changing only the seed leaves about 30% of features shared (January 2025). A result that would change the assessment: a training method with much higher overlap across seeds and architectures, together with evidence that the shared features are used inside the model.
4. **Dictionaries or text?** The NLA paper reports that on its own automated auditing benchmark, agents equipped with NLAs beat SAE-equipped baselines (May 2026, self-reported by Anthropic). A result that would change the assessment: third parties reproducing the comparison on other models and benchmarks, and quantifying how often the explanations confabulate.

Next: [Circuits, attribution, and causal interventions](circuits.en.md).
