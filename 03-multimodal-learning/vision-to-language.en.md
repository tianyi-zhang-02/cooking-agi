# How images enter a language model

[中文](vision-to-language.md) · **English**

> Reading time: about 7 minutes · Level: intermediate · Last reviewed: 2026-10

CLIP can score an image against a caption, but it does not continue with a written answer. A visual language model needs usable visual inputs and a training objective that teaches it to answer.

We follow **ViT + projector + decoder-only LLM**, not a universal VLM architecture. Read [CLIP](clip.en.md) and [Decoder-only](../00-foundations/core/decoder-only.en.md) first if needed.

## An image need not become one vector

[ViT](https://arxiv.org/abs/2010.11929) divides an image into patches, flattens and projects them, adds position information, and applies a Transformer.

The shape arithmetic is worked in [ViT](vit.en.md): a $224\times224$ RGB image cut into $16\times16$ patches gives 196 patches of 768 raw values each. What matters here is that the 196 projected vectors can remain a sequence rather than immediately becoming one global vector. Original ViT classification adds a class token, so include one extra position when counting that sequence.

```mermaid
flowchart TD
    A["Image"] --> B["Patches + positions"]
    B --> C["Vision encoder: P visual features"]
    C --> D["Projector: vision width → LLM width"]
    D --> E["Visual vectors + question embeddings"]
    E --> F["Decoder-only LLM"]
    F --> G["Generate answer tokens"]
```

Visual inputs are often continuous features, not discrete vocabulary IDs. A projector changes feature width; **an ordinary tokenwise linear layer or MLP does not automatically reduce token count**. Length compression needs pooling, a resampler, or another explicit mechanism.

## Matching dimensions is not matching meaning

Mapping $d_v$ to $d_{LM}$ makes the matrices compatible. It does not, by itself, teach the LLM what the visual features mean.

The original [LLaVA training recipe](https://arxiv.org/abs/2304.08485) gives a concrete example:

| Stage | Updated parameters | Purpose |
| --- | --- | --- |
| Image–text alignment | Projector; vision encoder and LLM frozen | Make visual features usable as language-model conditioning |
| Visual instruction tuning | Projector and LLM; vision encoder frozen | Learn to answer using the image and question |

This is one paper's recipe, not a rule that all VLMs freeze vision. Check each alternative against its data, trainable modules, and compute constraints.

## Which positions receive loss

In this answer-generation teaching setup, the input contains the image, question, and true answer prefix. The objective predicts the next answer token. Visual and question positions do not directly contribute to answer-only loss, but they remain valid context: they influence answers and can carry gradients to trainable modules.

Padding is different. It fills out the batch shape and should be excluded from both the loss and the context attended to by valid answer positions. **The loss mask chooses which predictions are scored; the attention mask chooses which inputs can be read. Neither replaces the other.** Properly masked padding input positions should not provide information or gradient paths to the answer. This is a statement about those input positions, not a claim that shared embedding / output parameters never receive gradients from elsewhere. [PyTorch padding-mask documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html)

$$L=-\frac{1}{M}\sum_{t\in\mathcal A}\log p_\theta(a_t\mid I,q,a_{<t}),\qquad M=|\mathcal A|.$$

$\mathcal A$ contains scored answer positions. This differs from the contrastive objective of matching images with captions. To check shifting and masks, revisit [one language-model training step](../00-foundations/deep-dives/training-step.en.md).

## Three design trade-offs worth asking about

- **Detail versus length:** with fixed-size patches and one image scale, doubling both image dimensions gives four times as many patches. The dense attention score matrix over those tokens has sixteen times as many entries (worked in [ViT](vit.en.md)). End-to-end cost also depends on text length, implementation, and compression; this does not imply a sixteen-fold runtime increase.
- **Freezing versus adaptation:** training fewer modules can reduce their gradient and optimizer-state costs, but forward computation and some activation costs remain. Measure the effect on domain adaptation rather than assuming it.
- **Descriptions versus evidence:** a fluent caption may omit small text, spatial relations, or details needed by the question. OCR or higher resolution should address a specific information gap, not be added automatically.

## Test more than a plausible answer

Make paired synthetic images with checkable answers. For “Is the red square left of the circle?”, swap only the shapes' positions; the answer should change. Hold the prompt and decoding settings fixed and compare. What removing the image, changing one piece of evidence, masking the key region, or an image–text conflict can and cannot show is listed once in [the multimodal overview's comparison table](README.en.md#modality-checks). Spatial questions add two more: supply a mismatched image and see whether the model blindly follows the question's suggestion; ask about an attribute that cannot be seen and see whether it admits the information is missing rather than giving a longer explanation.

These are proposed teaching experiments, not measured claims about a model. Track small text, counting, and spatial relations separately rather than relying only on aggregate accuracy.

Connect these checks to [controlled comparisons and slices](../07-evaluation/ablation-and-slices.en.md) before deciding whether more training is necessary. Robot VLA models use the same vision encoder, connector and LLM, but output actions instead of text; see [Robot learning: VLA](../13-embodied/robot-learning/vla.en.md).
