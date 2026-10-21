# Mechanistic interpretability: review questions

[中文](review.md) · **English**

> Reading time: ~5 min · Last reviewed: 2026-10

This page ties the series together with interview questions and a self-check; every answer points to the note where you can look it up again.

## Interview questions

<details class="interview" markdown="1">
<summary>How does mechanistic interpretability differ from behavioural evaluation, and when is it worth using?</summary>

Behavioural evaluation asks "is the output right on these inputs?"; interpretability asks "which internal quantity produced this output, and does changing it change the output as predicted?". Two models can both answer "capital of the state containing Dallas" correctly, one by first thinking of Texas and then reaching Austin, the other by having memorised the pairing; Claude 3.5 Haiku has both routes.

It is worth using when the same score with different mechanisms affects out-of-distribution behaviour, when a model may change its behaviour because it knows it is being tested (the Sonnet 4.5 system card), and when you need hypotheses without labels. The cost is that it is slow, expensive and narrow, and often explains a replacement model. See [the guide](README.en.md).

</details>

<details class="interview" markdown="1">
<summary>A linear probe reads some property from a layer with 95% accuracy. What does that show, and how do you make the conclusion solid?</summary>

It only shows that the information can be read linearly from that layer. It does not show that the model's output uses it, nor rule out that the accuracy comes from the probe's own capacity to fit. Hewitt & Liang's control tasks give each word type a random but fixed label: for part-of-speech tagging on ELMo's first layer, a linear probe gets 97.2 against 71.2 on the control task, a selectivity of 26.0, while an MLP probe gets 97.3 against 92.8, a selectivity of only 4.5.

To make it solid: report selectivity; use random directions or randomly initialised models as baselines (Anthropic's sleeper agent probe reached AUROC 99.3%, and the best of 1,000 random directions still reached 95.3%); intervene along the probe direction and check that the output changes as expected; and keep in mind that subspace patching can create illusions (Makelov et al.). See [Features and probes](features-and-superposition.en.md).

</details>

<details class="interview" markdown="1">
<summary>What is superposition, and why does it pay more the sparser the features are?</summary>

Superposition is a network representing more features than it has dimensions, by letting feature directions be not quite orthogonal and tolerating some interference. The toy model of Elhage et al. is $x'=\mathrm{ReLU}(W^\top Wx+b)$: with dense features it learns only the $m$ most important ones; as features get sparser superposition appears, and it does so as a phase change.

By hand: put 3 features 120° apart in 2 dimensions, so each pair interferes at −0.5. With one feature active, the negative interference is cut off by the ReLU and reconstruction is exact; with two active, each loses half; with all three active they cancel completely. If each feature is active with probability $p$, the expected loss is $1.5p^2+1.5p^3$, against $p$ for dropping one feature: at $p=0.1$ that is 0.0165 against 0.1, and the break-even point is about 0.457. Without the ReLU, superposition no longer pays. See [Features and probes](features-and-superposition.en.md).

</details>

<details class="interview" markdown="1">
<summary>What is the SAE objective? How do L1, TopK and JumpReLU handle sparsity?</summary>

$f=\mathrm{ReLU}(W_{\rm enc}x+b)$, $\hat x=W_{\rm dec}f+b'$, and the loss is reconstruction error plus $\lambda\sum_i f_i\|W_{\rm dec}^{(i)}\|$; multiplying by the decoder column norm stops the model from dodging the penalty by shrinking $f_i$ and enlarging the decoder vector. L1 causes shrinkage: non-zero activations are systematically underestimated. TopK (OpenAI) keeps only the $k$ largest per token, setting sparsity directly and leaving only the reconstruction term; JumpReLU (Google DeepMind) learns a threshold per latent and penalises L0 directly, trained with straight-through estimators.

By hand: $x=(1,1)$, with dictionary directions $(1,0)$, $(0,1)$ and $(0.707,0.707)$. L1 prefers using only the diagonal direction (a cost of $\sqrt2$ against 2); at $\lambda=1$ the coefficient is 0.914 and only 87.5% is reconstructed; for $\lambda\ge2\sqrt2$ nothing fires. TopK with $k=1$ gives a coefficient of $\sqrt2$ with no shrinkage. See [Sparse autoencoders](sparse-autoencoders.en.md).

</details>

<details class="interview" markdown="1">
<summary>What are dead latents, feature splitting and absorption? What happens if you change the random seed?</summary>

Dead latents never fire during training, so they get no gradient. Anthropic's 1M, 4M and 34M SAEs had roughly 2%, 35% and 65% dead; OpenAI, initialising the encoder as the transpose of the decoder and adding the AuxK auxiliary loss, had only 7% dead at 16 million wide (up to 90% without mitigations). Feature splitting: as the dictionary grows, one feature splits into finer ones, such as "San Francisco" going from 1 to 2 to 11 features. Absorption: when features form a hierarchy, a parent feature fails to fire where it should because child features absorb it, and changing size or sparsity does not fix it.

Changing the seed: 131K-wide SAEs on an MLP of Llama 3 8B share only about 30% of their features, and TopK depends more on the seed. So an SAE gives a pragmatically useful decomposition, not a complete list of the model's features. See [Sparse autoencoders](sparse-autoencoders.en.md).

</details>

<details class="interview" markdown="1">
<summary>Why did the Google DeepMind team deprioritise fundamental SAE research in 2025, and what are SAEs still good for?</summary>

Their 2025-03-26 progress update: on detecting harmful intent with Gemma 2 9B IT, tested out of distribution, a linear probe on the residual stream reached OOD AUROC 0.999 and SAE probes were distinctly worse. Around the same time, Kantamneni et al. could not consistently beat baselines in four difficult sparse-probing settings; AxBench found prompting best for steering and difference-in-means best for detection, with SAEs competitive on neither; SAEBench found proxy metrics disconnected from practical usefulness. In December 2025 the team went further and argued for trying prompting, steering and probing first.

SAEs still suit exploration when you do not know what to look for: in the auditing game, one person found the hidden objective in 90 minutes through an SAE feature (the paper notes semantic search could reproduce it); OpenAI compared models before and after finetuning with SAEs and found a "toxic persona" feature; and they help find spurious correlations in training data. When you know the concept and have labels, start with a probe or difference-in-means. See [Sparse autoencoders](sparse-autoencoders.en.md).

</details>

<details class="interview" markdown="1">
<summary>How do denoising and noising differ in activation patching? Why does IOI need a "completeness" criterion?</summary>

Denoising puts clean values into the corrupted run and asks "is this enough to rescue the behaviour?"; noising puts corrupted values into the clean run and asks "can the model do without it?". With backup components the two disagree: in the 3-node example, denoising B restores 86% but noising B loses only 43%, because C steps in when B is absent. Path effects are not additive either; the two paths sum to 143%.

The IOI circuit recovers 87% of the full model's logit difference (faithfulness), but faithfulness cannot see backups: when the name movers are removed, the backup name movers take over. Completeness requires that removing the same set of components from the circuit and from the full model has similar effects; in the example, removing B leaves the model at 2.0 and the circuit at 0, exposing that C was left out. See [Circuits](circuits.en.md).

</details>

<details class="interview" markdown="1">
<summary>How is an attribution graph built? When you hear "interpretability can detect deception or reward hacking", what do you ask?</summary>

First train a cross-layer transcoder to replace every MLP; for a single prompt, freeze the attention patterns and normalisation and add error nodes so that the local replacement model matches the original's outputs; edge weights are $A_{s\to t}=a_s w_{s\to t}$; prune, group, and go back to the original model to validate with interventions. The authors list the limits themselves: it does not explain why attention looks where it does (QK attributions came later), error nodes can hide the key computation, it looks at single prompts, and the replacement model may not share the original's mechanism; it is satisfying on about a quarter of prompts.

What to ask: was the problem implanted, or did it occur naturally? Who reported it, and has a third party reproduced it? What was it compared with (random directions, chain-of-thought monitoring, behavioural evaluation)? What is the false-positive rate? For example, Apollo's deception probes reach AUROC 0.96–0.999, yet the authors say this is not enough for a robust defence; inhibiting evaluation awareness in Sonnet 4.5 raised misaligned actions to at most 8–9%, self-reported by the lab and varying by dataset. See [Circuits](circuits.en.md).

</details>

## Self-check

<div class="taste-check">
  <strong>If you really understand it, you should be able to explain:</strong>
  <ol>
    <li>Three features 120° apart in 2 dimensions, zero bias, with a ReLU: what is the reconstruction when features 1 and 2 are both active? At $p=0.1$, what are the expected losses of superposition and of "dropping one feature"?</li>
    <li>A linear probe at 97.2 against 71.2 on the control task, an MLP probe at 97.3 against 92.8: what is each selectivity? Why do near-identical accuracies lead to different conclusions?</li>
    <li>Dictionary $(1,0)$, $(0,1)$, $(0.707,0.707)$ and activation $(1,1)$: at $\lambda=1$, what are the L1 SAE's optimal coefficients and reconstruction? At what $\lambda$ does nothing fire? What does TopK with $k=1$ do?</li>
    <li>In the 3-node example, why does denoising B restore 86% while noising B loses only 43%? Why do the two path effects add up to 143%?</li>
    <li>An attribution graph says "the model thinks of Texas before saying Austin": what further evidence do you need before believing this is the original model's mechanism and not just the replacement model's?</li>
  </ol>
</div>

## References

- [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/): where the language of features and circuits comes from
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): the residual stream as something layers read from and write to
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html): induction heads and a phase change in training
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html): toy models of superposition
- [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368): control tasks and selectivity
- [The Linear Representation Hypothesis and the Geometry of LLMs](https://arxiv.org/abs/2311.03658): two formalisations of linear representation
- [Not All Language Model Features Are One-Dimensionally Linear](https://arxiv.org/abs/2405.14860): circular multi-dimensional features (a counterexample)
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717): difference-in-means and directional ablation
- [Inference-Time Intervention](https://arxiv.org/abs/2306.03341): activation steering on TruthfulQA, and its cost
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) and [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html): Anthropic's two SAE papers
- [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093): TopK SAEs and 16 million latents on GPT-4
- [Gemma Scope](https://arxiv.org/abs/2408.05147) and the [Gemma Scope 2 blog post](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/): open SAEs and transcoders
- [A is for Absorption](https://arxiv.org/abs/2409.14507) and [SAEs Trained on the Same Data Learn Different Features](https://arxiv.org/abs/2501.16615): absorption and seed dependence
- [Are Sparse Autoencoders Useful?](https://arxiv.org/abs/2502.16681), [AxBench](https://arxiv.org/abs/2501.17148), [SAEBench](https://arxiv.org/abs/2503.09532): comparisons on downstream tasks
- [Negative Results for SAEs On Downstream Tasks](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks): the Google DeepMind team's progress update
- [Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers](https://arxiv.org/abs/2501.17727): automated interpretation needs random baselines
- [Locating and Editing Factual Associations in GPT (ROME)](https://arxiv.org/abs/2202.05262): causal tracing
- [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255): practical advice on patching experiments
- [Interpretability in the Wild (IOI)](https://arxiv.org/abs/2211.00593): a circuit, with faithfulness, completeness and minimality
- [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html): the method and case studies of attribution graphs
- [Weight-sparse transformers have interpretable circuits](https://arxiv.org/abs/2511.13653): making circuits small through training
- [Auditing language models for hidden objectives](https://arxiv.org/abs/2503.10965): the blind auditing game
- [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/abs/2502.03407): deception probes
- [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html): translating activations into text, and its use in the Opus 4.6 audit
- [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496): a list of open problems
