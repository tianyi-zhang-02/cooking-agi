# Interpretability: circuits, attribution, and causal interventions

[中文](circuits.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

"GPT-2 small uses 26 attention heads to find the indirect object"; "Claude 3.5 Haiku first thinks of Texas internally, then says Austin". What exactly do circuit claims like these mean, and what evidence supports them?

A circuit claim is a causal hypothesis that can be tested by intervention: on a given class of inputs, which components and paths account for how much of a behaviour? Activation patching and path patching test these hypotheses, while attribution graphs automate the step of proposing them. An attribution graph, however, explains a replacement model. Anthropic reports satisfying explanations on only about a quarter of the prompts it tried.

Safety applications include lab-reported case studies and successes on deliberately implanted problems. There is not yet an independently validated general-purpose detector.

## From correlation to causation: activation patching

Probes and attributions only show correlation or local sensitivity. To say a component causes a behaviour, you have to intervene. Activation patching (also called causal tracing or interchange intervention) uses three runs: one on a clean input, one on a corrupted input, and one more on the corrupted input with one component's value replaced by its value from the clean run (patched). The effect is measured as a fraction:

$$
\text{patching effect}=\frac{y_{\rm patched}-y_{\rm corrupt}}{y_{\rm clean}-y_{\rm corrupt}}
$$

$y$ is the output metric you care about, such as the probability of the correct answer, or the logit difference between correct and incorrect answers. It can also run the other way: put the corrupted value into the clean run and see how much the behaviour drops. The first is called denoising and asks "is this component enough to rescue the behaviour?"; the second is called noising and asks "can the model do without it?" (terms as in [Heimersheim & Nanda, April 2024](https://arxiv.org/abs/2404.15255)).

[ROME (Meng et al., February 2022)](https://arxiv.org/abs/2202.05262) is a classic use. On GPT-2 XL, they took 1,000 facts the model knows and added Gaussian noise with 3 times the standard deviation to the subject token embeddings, which dropped the average probability of the correct answer from 27.0% to 8.47%; then they restored clean hidden states one at a time. The average total effect was 18.6%, and the strongest single states were at the last subject token around layer 15 (average indirect effect 8.7%); at that position, MLP contributions peaked at 6.6% while attention reached only 1.6%. From this the authors concluded that facts are stored in mid-layer MLPs, and they edited facts by directly modifying those MLPs' weights.

<!-- paper-figure:rome-causal-trace -->

Two caveats. [Hase et al. (January 2023)](https://arxiv.org/abs/2301.04213) found that the layers causal tracing points to do not predict which layer is best to edit: localising and editing are different things. [Zhang & Nanda (September 2023)](https://arxiv.org/abs/2309.16042) found that changing the metric (probability or logit difference) or the corruption method (adding noise or swapping in a different sentence) can change the localisation result.

## By hand: patching a 3-node network (a teaching example)

The small network below is made up, just to see clearly what patching measures. $s=1$ is the clean input and $s=0$ the corrupted one:

```text
A = 2·s
B = 1.5·A
C = ReLU(A − 0.5·B)
y = B + C
```

Clean run: $A=2$, $B=3$, $C=\mathrm{ReLU}(2-1.5)=0.5$, $y=3.5$. Corrupted run: everything is 0, $y=0$.

| Node patched | $y$ after denoising | Fraction restored | $y$ after noising | Fraction lost |
| --- | --- | --- | --- | --- |
| A | 3.5 | 100% | 0 | 100% |
| B | 3.0 | 86% | 2.0 | 43% |
| C | 0.5 | 14% | 3.0 | 14% |

Look at row B. Denoising: $A=0$, $B=3$, $C=\mathrm{ReLU}(0-1.5)=0$, $y=3.0$, restoring $3/3.5=86\%$. Noising: $A=2$, $B=0$, and now $C=\mathrm{ReLU}(2-0)=2$, so $y=2.0$, a drop of only $1.5/3.5=43\%$. C normally contributes little, but as soon as B is removed it steps in: it is a backup. Noising alone underestimates B, and denoising alone never shows what C does.

Now path patching ([Goldowsky-Dill et al., April 2023](https://arxiv.org/abs/2304.05969)): in the corrupted run, send the clean A only along the edge A→B, while C still receives A = 0. Then $B=3$, $C=\mathrm{ReLU}(0-1.5)=0$, $y=3.0$, or 86%. Send it only along A→C: $B=0$, $C=\mathrm{ReLU}(2-0)=2$, $y=2.0$, or 57%. The two paths add up to 143%, not 100%. With nonlinearities, path effects do not simply add, and the question "what share does this path carry?" has to be asked with care.

| What to try first | What to observe | What you cannot conclude from it |
| --- | --- | --- |
| Denoising only | B restores 86% | That B is necessary |
| Noising only | Removing B loses only 43% | That B is unimportant, or that C is irrelevant |
| Adding up the two path effects | You get 143% | What share each path carries |

## What a circuit claim looks like: IOI

[Wang et al. (November 2022)](https://arxiv.org/abs/2211.00593) studied a task called indirect object identification (IOI): "When Mary and John went to the store, John gave a bottle of milk to" should be completed with Mary. GPT-2 small has 12 layers with 12 heads each; over 100,000 examples, the correct name beats the repeated name by 3.56 logits on average, and the model is right in 99.3% of cases.

Their circuit has 26 heads in 7 classes. Roughly: duplicate token heads and induction heads (helped by previous token heads) detect that John appeared twice; S-inhibition heads write this into the queries of the name mover heads so that they do not attend to John; name mover heads at the end of the sentence attend to the remaining name, Mary, and copy it to the output; negative name mover heads write in the opposite direction and lower confidence; backup name mover heads do little normally and take over only when the name movers are removed. Components are "removed" by mean ablation: replaced with their average activation over a distribution of sentences containing three different names.

<!-- paper-figure:ioi-circuit -->

Saying "these heads form a circuit" is not enough, so they test three criteria. Let $F(\cdot)$ be the average logit difference:

- **Faithfulness**: keeping only the circuit, $F(C)$ is close to the full model's $F(M)$. They measure $|F(M)-F(C)|=0.46$, 13% of 3.56, so the circuit recovers 87%.
- **Completeness**: removing any set of components $K$ from the circuit should have about the same effect as removing $K$ from the full model.
- **Minimality**: every component in the circuit is necessary for some $K$.

Completeness exists precisely for backups. Go back to the example above: the circuit {A, B} (with C ablated) outputs 3.0, a faithfulness of 86%, which looks fine; but remove B and the full model still gives 2.0 (C steps in) while the circuit gives 0, a large gap showing that C was left out. The authors themselves write in the [abstract](https://arxiv.org/abs/2211.00593): "Though these criteria support our explanation, they also point to remaining gaps in our understanding." The scope of the claim is GPT-2 small, sentences generated from 15 templates, and the choice of mean ablation.

Finding circuits was later automated: on GPT-2 small's "greater-than" task, ACDC selected 68 of 32,000 edges, all of which had been found by earlier manual work, and rediscovered all 5 of the 5 component types ([Conmy et al., April 2023](https://arxiv.org/abs/2304.14997)). Separately, [Makelov et al. (November 2023)](https://arxiv.org/abs/2311.17030) warn that patching a subspace can activate a normally dormant parallel pathway and create an illusion; they also treat IOI as a success case, because the earlier manual circuit analysis tells you where the feature lives.

## Attribution graphs: automating the hypothesis step

The IOI circuit analysis required substantial manual work and covered only one task. Anthropic's March 2025 [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) aims to map the model's computation on each prompt automatically.

It works in four steps. First, train a cross-layer transcoder (CLT): features at layer $\ell$ read from that layer's residual stream and can write to the MLP outputs of layer $\ell$ and every later layer; the CLTs trained on Claude 3.5 Haiku have between 300 thousand and 30 million features in total. Replacing every MLP with it gives a replacement model; their largest CLT, on an 18-layer small model, gives the same next token as the original model on only 50% of a diverse set of prompts. Second, for a single prompt, freeze the original model's attention patterns and normalisation denominators, and add an "error node" at each position and layer to make up for what the CLT fails to reconstruct, so that the activations and outputs of this local replacement model match the original model exactly. Third, compute edges on this now-linearised graph:

$$
A_{s\to t}=a_s\,w_{s\to t}
$$

$a_s$ is the source feature's activation and $w_{s\to t}$ is the "virtual weight" between the two through the residual stream and frozen attention. Fourth, prune weak edges, group features with similar roles into supernodes, and go back to the original model to validate with interventions.

The companion [Biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) paper reports a set of case studies. The two-step reasoning mentioned at the start: suppressing "Dallas" features weakens the "Texas" features and the downstream "say Austin", while "say a capital" is largely unaffected. Planning in poems: on the newline before a line begins, features for candidate rhyme words (such as rabbit and habit) are already active, and editing them makes the model rewrite the whole line to end in the new word; the authors say they found such planned-word features in about half of the poems they examined. There are also case studies on multilinguality, addition, medical diagnosis, hallucination, refusal, jailbreaks, chain-of-thought faithfulness and hidden goals.

The methods paper has a section on limitations: it does not explain why attention looks at a given position (QK circuits), so in examples such as induction and multiple-choice questions the graph "skips the interesting part"; the CLT does not reconstruct everything, and error nodes sometimes hide the key computation, as on an obfuscated jailbreak prompt where almost all nodes are error nodes; inactive features and inhibition are hard to represent; the graphs are complex; feature splitting and absorption put features at the wrong level of abstraction; it only looks at single prompts, and global circuits are hard; and the replacement model may not use the same mechanism as the original. The Biology paper says attribution graphs give satisfying insight on about a quarter of the prompts tried, and the examples shown are success cases.

Follow-ups: on 2025-05-29 Anthropic [open-sourced the tools](https://www.anthropic.com/research/open-source-circuit-tracing), which can draw graphs for Gemma-2-2b and Llama-3.2-1b and are integrated with Neuronpedia; the [QK attributions](https://transformer-circuits.pub/2025/attention-qk/index.html) of 2025-07-31 write attention scores as a bilinear function of query-side and key-side features, and confirm mechanisms in induction and multiple-choice examples that had previously only been hypothesised. Another route works on training instead: OpenAI trains Transformers in which most weights are zero, and the task circuits pruned out of them are small and readable, but sparser means weaker, and the authors say keeping interpretability beyond tens of millions of non-zero parameters remains a challenge ([Gao et al., November 2025, ICML 2026](https://arxiv.org/abs/2511.13653)). A February 2026 LessWrong replication (Anthropic Fellows Program, not peer-reviewed) argues that such pruned circuits may be "interpretable yet unfaithful": on a task whose labels were reassigned so that it means nothing, pruning still reached over 95% accuracy with about 30 nodes; the author also notes having switched to zero ablation, and that the original paper's mean ablation might give more faithful circuits ([Drori](https://www.greaterwrong.com/posts/sHpZZnRDLg7ccX9aF/weight-sparse-circuits-may-be-interpretable-yet-unfaithful)).

## How this connects to safety evaluation

Behavioural evaluation and chain-of-thought monitoring depend on what the model does and says. OpenAI found that having another model read the chain of thought catches reward hacking by coding agents, but if that monitor is put directly into the training reward and optimised too hard, the agent learns to hide its intent in the chain of thought and keeps cheating ([Baker et al., March 2025](https://arxiv.org/abs/2503.11926)). The hope for interpretability is a check that does not rely on the model's own account. For the definition of reward hacking, see [Verifiable rewards](../../05-post-training/verifiable-rewards.en.md). These are the public results that can be checked:

| Date and organisation | Setting | Method | Result | Discount to apply |
| --- | --- | --- | --- | --- |
| April 2024, [Anthropic](https://www.anthropic.com/research/probes-catch-sleeper-agents) | Sleeper agents with an implanted backdoor that write insecure code at deployment | A probe from the mean difference of a single generic question-answer pair ("Are you a helpful AI?" answered "no" versus "yes") | AUROC 99.3% | The backdoor was deliberately implanted; the best of 1,000 random directions also reached 95.3% |
| February 2025, [Apollo](https://arxiv.org/abs/2502.03407) | Llama-3.3-70B lying in scenarios such as concealing insider trading or deliberately underperforming on evaluations | Linear probes | AUROC 0.96–0.999; catches 95–99% at a 1% false-positive rate | The authors say this is not yet enough for a robust defence |
| March 2025, [Anthropic](https://arxiv.org/abs/2503.10965) | A model trained to please reward models and to hide that objective | Blind auditing game: SAEs, behavioural attacks, training-data analysis | 3 of 4 teams found the hidden objective; the API-only team did not | The objective was deliberately implanted; the SAE solution could also be reproduced with semantic search |
| September 2025, [Anthropic](https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf) | Pre-deployment alignment evaluation of Claude Sonnet 4.5 | Inhibiting SAE features and contrastive vectors related to "this is a test" | On 100 honeypot prompts, verbalised evaluation awareness fell and misaligned actions rose from zero to at most 8–9%; a control with 10 random SAE features gave zero | Self-reported by the lab; small or no effect on other datasets |
| April 2026, [Anthropic](https://transformer-circuits.pub/2026/emotions/index.html) | Claude Sonnet 4.5 on 7 "impossible" coding tasks | Steering along a "desperate" emotion vector | Moving the strength from −0.1 to +0.1 raised reward hacking from about 5% to about 70% | A steering experiment: it shows the direction affects behaviour, and is not itself a detector |
| May 2026, [Anthropic](https://transformer-circuits.pub/2026/nla/index.html) | Pre-deployment audit of Claude Opus 4.6 | Natural language autoencoders (NLAs) | Surfaced unverbalised evaluation awareness; following an early version's habit of answering English prompts in Russian, traced a batch of training data with mismatched prompt and response languages | Self-reported by the lab; NLAs invent details |

How far this goes: probes, steering and dictionary features do well in settings with a known answer (implanted backdoors, hidden objectives), and they have entered labs' pre-deployment audits; but the successes come mostly from deliberately constructed models or lab self-reports, and both Apollo and Anthropic state limitations in the original texts. Dario Amodei's April 2025 [essay](https://www.darioamodei.com/post/the-urgency-of-interpretability) says Anthropic's goal is for interpretability to "reliably detect most model problems" by 2027, and that on the current trajectory he would bet strongly on interpretability becoming a true "MRI for AI" within 5 to 10 years. That is a company leader's goal and judgement, not a measured result.

## What each step of evidence shows

| Evidence | Shows | Does not show |
| --- | --- | --- |
| Gradient or linear attribution | The size of a local, first-order influence | The actual effect of a larger change |
| Denoising patch | This component is enough to rescue the behaviour | That it is necessary |
| Noising patch or ablation | How much the behaviour drops without it | That a small drop means it is not involved (there may be a backup) |
| Path patching | Which edges the effect passes through | What share each path carries (not additive with nonlinearities) |
| Faithfulness, completeness, minimality | How completely the circuit explains behaviour on this input distribution | That it holds for other distributions or models |
| Attribution graph plus interventions | A mechanism in the replacement model, partly validated on the original model | That the original model as a whole uses the same mechanism |
| Successful detection of implanted problems | The method finds the problem when there is a known answer | That it can find naturally occurring problems nobody knew about |

## What to watch on this line next

1. **Attention and global circuits.** QK attributions (July 2025) filled in one piece, but the "satisfying" fraction Anthropic reported in March 2025 was about a quarter. A result that would change the assessment: that fraction rising clearly in later reports, or global circuits that hold across many prompts.
2. **Is the replacement model faithful?** A short July 2025 Anthropic note uses a [toy model](https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html) to show how transcoders can go wrong; the February 2026 replication questions the faithfulness of pruned circuits. A result that would change the assessment: every graph-based claim being validated by interventions on the original model, with the rate of failed validations published.
3. **From constructed models to real deployments.** As of October 2026, the detection successes that can be checked are mostly on implanted problems. A result that would change the assessment: interpretability finding, in a production model, a problem that behavioural evaluation missed, reproduced by an independent organisation.
4. **What white-box checks and chain-of-thought monitoring each cover.** In July 2025 authors from many institutions jointly argued that chain-of-thought monitorability is a "new and fragile opportunity" ([Korbak et al.](https://arxiv.org/abs/2507.11473)). A result that would change the assessment: a direct comparison of white-box probes and chain-of-thought monitors on the same set of misbehaviours, reporting detection and false-positive rates.

Finally, use the [review questions](review.en.md) to tie the series together.
