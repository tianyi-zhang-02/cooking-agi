# Interpretability: features, superposition, and linear probes

[中文](features-and-superposition.md) · **English**

> Reading time: ~11 min · Last reviewed: 2026-10

A linear probe reads "is this sentence true or false?" from one layer's activations with 95% accuracy. Can you say the model "represents truth" at that layer? A second phenomenon: the same neuron fires on academic citations and on Korean text. Both rest on the same set of ideas. **"A concept corresponds to a direction in activation space" is a useful working assumption that does not always hold; probe accuracy only shows that information can be read out, and it takes control tasks and interventions before you can talk about the model using it; when there are more features than dimensions and the features are sparse, a network superposes them on the same neurons, so looking at neurons one by one is not enough and you have to look for directions instead.**

## Is a feature a direction? The linear representation hypothesis

At a given layer and token position, the model's state is a vector $h$ in the residual stream (see [Residual connections](../../00-foundations/core/residual-connections.en.md)). The linear representation hypothesis says:

$$
h(x)\approx b+\sum_i a_i(x)\,d_i
$$

$d_i$ is the direction (a unit vector) of feature $i$, $a_i(x)\ge 0$ is how strongly that feature is present on input $x$, and $b$ is a bias. A concept corresponds to a direction, and its strength to how far you move along it.

The evidence for this goes back a long way. In word vectors, "king − man + woman" lands near "queen" ([Mikolov et al., 2013](https://aclanthology.org/N13-1090/)). [Park et al. (November 2023)](https://arxiv.org/abs/2311.03658) split "linear representation" into two meanings, one in the output space where a direction reads the concept out (matching probes) and one in the input space where adding a direction changes the concept (matching steering), and prove that under a particular inner product the two can be unified. Anthropic's 2024 SAE work takes it as its starting point ([Templeton et al.](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)).

But it is a working assumption, not a theorem. [Engels et al. (May 2024)](https://arxiv.org/abs/2405.14860) found days of the week and months of the year arranged on a circle in GPT-2 and Mistral 7B, an irreducible two-dimensional feature, and used interventions on Mistral 7B and Llama 3 8B to show that the model uses exactly this circle for modular arithmetic such as "which day is it a few days after Monday". The [Toy Models paper](https://transformer-circuits.pub/2022/toy_model/index.html) also lists several competing definitions of "feature", and its authors say plainly that they do not want to settle on one too early.

<!-- paper-figure:circular-features -->

## Probes: what accuracy does and does not show

A linear probe ([Alain & Bengio, 2016](https://arxiv.org/abs/1610.01644)) freezes the model and trains only a logistic regression on the activations:

$$
p(y=1\mid h)=\sigma(w^\top h+c)
$$

$w,c$ are the probe's parameters and $\sigma$ is the sigmoid. High accuracy means the information in label $y$ can be read linearly from $h$. It does not show two things: whether the model's output uses this information (the pushed-cup example in [Do they actually understand the world?](../world-models/do-they-understand.en.md) is readable but unused), and whether the readout comes from the representation itself or from the probe's own capacity to fit.

For the second, [Hewitt & Liang (2019)](https://arxiv.org/abs/1909.03368) use control tasks: give each word type a random but fixed label. Such labels have nothing to do with linguistic structure, so a probe can learn them only by memorising which word goes with which label. Selectivity is then defined as accuracy on the real task minus accuracy on the control task.

<!-- paper-figure:control-task-probes -->

### Computing selectivity once

Here are three rows from Table 1 of the paper, for the first layer of ELMo, part-of-speech tagging, development set:

| Probe | POS accuracy | Control task accuracy | Selectivity |
| --- | --- | --- | --- |
| Linear, default hyperparameters | 97.2 | 71.2 | 97.2 − 71.2 = 26.0 |
| One-hidden-layer MLP, default hyperparameters | 97.3 | 92.8 | 97.3 − 92.8 = 4.5 |
| One-hidden-layer MLP, regularisation tuned with control tasks | 97.2 | 80.6 | 97.2 − 80.6 = 16.6 |

The MLP probe is only 0.1 points better than the linear one, yet it also reaches 92.8 on random labels, so most of its accuracy comes from its own memorisation. The authors also found that dropout does not improve MLP selectivity while other regularisation does, and that probes on the first layer are slightly more accurate than on the second, while the second layer is far more selective. "Which layer represents part of speech better" depends on which of the two you measure.

| What to try first | What to observe | What you cannot conclude from it |
| --- | --- | --- |
| Add a control task | Selectivity, not just accuracy | That the model's output uses this information |
| Compare a linear and an MLP probe | How much accuracy differs, and how much selectivity differs | That a linear failure means the information is absent |
| Do the same thing on a randomly initialised model, or with random directions | How far the result is above that baseline | That beating the baseline makes it the model's "true" feature |
| Intervene along the probe direction | Whether the output changes as expected | That this direction is the model's only mechanism |

## Superposition: why neurons mean several things

If each concept had its own neuron, reading a model would be easy. In practice many neurons are polysemantic: in a one-layer model Anthropic studied, one neuron responds to academic citations, English dialogue, HTTP requests and Korean text all at once ([Bricken et al., October 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html)). The explanation offered by [Elhage et al. (September 2022)](https://transformer-circuits.pub/2022/toy_model/index.html) is superposition: a network that wants to represent more features than it has dimensions lets feature directions be not quite orthogonal and tolerates a little interference.

Their toy model: each coordinate of the input $x\in\mathbb R^n$ is a feature, which is 0 with probability $S$ and otherwise uniform on $[0,1]$. It is compressed to $m<n$ dimensions and reconstructed:

$$
h=Wx,\qquad x'=\mathrm{ReLU}(W^\top W x+b),\qquad L=\sum_i I_i\,(x_i-x'_i)^2
$$

Column $W_i$ of $W\in\mathbb R^{m\times n}$ is the direction of feature $i$ in the hidden space, and $I_i$ is its importance. The paper reports that with dense features the model learns only the $m$ most important ones, much like PCA; as features become sparser, superposition appears, and it does so as a phase change; features arrange themselves into geometric structures such as digons, triangles and pentagons. The authors also caution that these are small ReLU networks, and it is unclear how far the results carry over to real networks.

### By hand: 3 sparse features squeezed into 2 dimensions (a teaching example)

Take $n=3$ and $m=2$, with the three feature directions 120° apart: $W_1=(1,0)$, $W_2=(-\tfrac12,\tfrac{\sqrt3}{2})$, $W_3=(-\tfrac12,-\tfrac{\sqrt3}{2})$, and bias $b=0$. To keep the arithmetic simple, an active feature takes the value 1 (the paper uses a uniform distribution), and all three features are equally important. The inner product of any two directions is $\cos120^\circ=-0.5$; that is the interference.

| Which features are active | $h=Wx$ | $W^\top Wx$ (before ReLU) | Reconstruction after ReLU | Squared error |
| --- | --- | --- | --- | --- |
| Only 1 | (1, 0) | (1, −0.5, −0.5) | (1, 0, 0) | 0 |
| 1 and 2 | (0.5, 0.866) | (0.5, 0.5, −1) | (0.5, 0.5, 0) | 0.25 + 0.25 = 0.5 |
| 1, 2 and 3 | (0, 0) | (0, 0, 0) | (0, 0, 0) | 3 |

With one active feature, all the interference is negative, ReLU cuts it off, and the reconstruction is exact. With two active, each loses half. With all three active, the three directions cancel exactly.

Let each feature be active independently with probability $p$. Compare the expected loss of superposition with the alternative of "give the first two features one dimension each and simply drop the third":

$$
\mathbb E[L_{\text{superpose}}]=3p^2(1-p)\times0.5+p^3\times3=1.5p^2+1.5p^3,\qquad \mathbb E[L_{\text{drop}}]=p
$$

- $p=0.1$: superposition gives $0.015+0.0015=0.0165$ against 0.1 for dropping, about 6 times better.
- $p=0.5$: superposition gives $0.375+0.1875=0.5625$ against 0.5, so dropping is better.
- The break-even point: $1.5p^2+1.5p-1=0$, so $p\approx0.457$.

Now remove the ReLU and redo it. With one active feature the reconstruction is $(1,-0.5,-0.5)$, an error of 0.5; at $p=0.1$ the expected loss is 0.165, worse than dropping. Negative interference can only be filtered out by the nonlinearity, which matches the paper's finding that a linear model learns only the top $m$ features. The paper also defines a per-feature "dimensionality"; here it is $1/(1+0.25+0.25)=2/3$, exactly the value of the triangle line in the paper.

The example fixes the geometry instead of training it and assumes the features are independent; the paper also finds that anti-correlated features prefer to sit in opposite directions. It is meant to show one thing only: **the sparser the features, the more superposition pays**. For someone reading a model, this means that looking neuron by neuron shows a jumble of meanings, and what you need to find is directions. [The next note, on sparse autoencoders](sparse-autoencoders.en.md), is about exactly that.

## Activation steering: pushing along a direction

If concepts correspond to directions, adding or subtracting along a direction should change behaviour. The most common way to get a direction is difference-in-means: average the activations over two contrasting sets of inputs and subtract. You can then add it, or erase it entirely:

$$
r=\mu_{+}-\mu_{-},\qquad h'=h+\alpha r,\qquad h'=h-\hat r\hat r^\top h
$$

$\mu_{+},\mu_{-}$ are the mean activations of the two sets, $\alpha$ is the strength, and $\hat r=r/\|r\|$. The second equation is activation addition and the third is directional ablation, which sets the component of $h$ along $\hat r$ to zero.

### Worked example: from a difference of means to erasing a direction (a teaching example)

Let the mean activation for harmful instructions be $\mu_{+}=(4,5)$ and for harmless ones $\mu_{-}=(1,1)$. Then $r=(3,4)$, $\|r\|=5$, and $\hat r=(0.6,0.8)$. A new input arrives with $h=(2,1)$: its projection is $\hat r^\top h=1.2+0.8=2.0$, and after ablation $h'=(2,1)-2.0\times(0.6,0.8)=(0.8,-0.6)$; check that $\hat r^\top h'=0.48-0.48=0$. In the other direction, adding $r$ to the harmless input $(1,1)$ lands exactly on the harmful mean $(4,5)$.

[Arditi et al. (June 2024, NeurIPS 2024)](https://arxiv.org/abs/2406.11717) did exactly this on real models. In 13 open chat models (up to 72B, including Qwen, Yi, Gemma, Llama-2 and Llama-3) they used the difference of means between harmful and harmless instructions to find a "refusal direction". Erasing it at every layer and position clearly lowered the refusal rate on the 100 harmful instructions of JailbreakBench and produced unsafe completions; adding it at the layer it was extracted from, on 100 harmless instructions, made the models refuse even harmless requests. They also turned it into a rank-one weight edit as a white-box jailbreak, and point out that this is a safety problem for releasing open-weight models.

Other public examples: ActAdd builds a steering vector from the activation difference of a single pair of prompts (such as "Love" and "Hate") and uses it for sentiment shift and detoxification on models including LLaMA-3 and OPT ([Turner et al., August 2023](https://arxiv.org/abs/2308.10248)); ITI raises Alpaca's "true and informative" rate on TruthfulQA from 32.5% to 65.1%, at the cost of cross-entropy rising from 2.56 to 2.92 ([Li et al., June 2023](https://arxiv.org/abs/2306.03341)); Anthropic clamped the "Golden Gate Bridge" SAE feature to 10 times its maximum activation, and Claude 3 Sonnet began to describe itself as the bridge ([Templeton et al., May 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)).

Three things to watch. The strength has to be tuned by hand, and too much of it hurts capability; the rise in ITI's cross-entropy is that cost. "It moves the behaviour" is not the same as "this is the concept the model uses": [Makelov et al. (November 2023)](https://arxiv.org/abs/2311.17030) construct, and observe in IOI and in factual recall, cases where patching a subspace works by activating a parallel pathway that is normally dormant, giving the illusion that the feature was changed. Finally, compare against prompting: [AxBench (January 2025)](https://arxiv.org/abs/2501.17148) found on Gemma 2 that, for steering, simply writing a prompt beats every representation-based method.

## What each kind of evidence shows

| Evidence | Shows | Does not show | Example |
| --- | --- | --- | --- |
| Probe accuracy | The information can be read linearly from this layer | That the model uses it; whether the readout relies on the probe's own capacity | ELMo POS probe at 97.2% |
| Selectivity with a control task | The readout comes mainly from the representation, not the probe's memory | That the model uses it | Linear 26.0 versus MLP 4.5 |
| Random-direction or random-model baselines | The result is not just "any direction would do" | That the direction found is the only one | Anthropic's sleeper agent probe reached AUROC 99.3%, and the best of 1,000 random directions still reached 95.3% ([April 2024](https://www.anthropic.com/research/probes-catch-sleeper-agents)) |
| Adding, subtracting or erasing a direction | This direction can change a behaviour | That it is the model's natural, unique mechanism | The refusal direction; the subspace-patching illusion |
| Superposition in a toy model | The mechanism appears under the given conditions | How many features large models have, or their geometry | The 120° triangle example |

## What to watch on this line next

1. **Are all features one-dimensional directions?** Engels et al. (May 2024) found circular two-dimensional features; Anthropic's October 2025 paper [When Models Manipulate Manifolds](https://transformer-circuits.pub/2025/linebreaks/index.html) describes geometric structure in a counting task. A result that would change the assessment: a method that finds multi-dimensional features automatically and shows by intervention that the model really uses them.
2. **Can probes serve as deployment-time monitors?** Apollo's deception probes reach AUROC 0.96–0.999, and the authors themselves say this is not yet enough for a robust defence (February 2025). A result that would change the assessment: detection of naturally occurring (not implanted) problems at a false-positive rate fixed in advance, reproduced by a third party.
3. **Can steering beat prompting?** On AxBench (January 2025), prompting is ahead. A result that would change the assessment: steering consistently beating prompting on unseen concepts while also reporting the capability cost.
4. **How does superposition theory map onto real models?** A September 2026 survey ([Shi et al.](https://arxiv.org/abs/2609.06862)) stresses that accurate reconstruction does not establish that what was found is the model's feature, nor that it is used. A result that would change the assessment: checking, on models whose features are known (for example constructed or weight-sparse models), whether feature-finding methods recover those features.

Next: [Sparse autoencoders](sparse-autoencoders.en.md).
