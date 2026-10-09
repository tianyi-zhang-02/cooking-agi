# ViT: how does an image become a Transformer input?

[中文](vit.md) · **English**

> Reading time: about 10 minutes · Last reviewed: 2026-10

In a street photo, you notice cars, people, and traffic lights. A model initially receives only pixel values. ViT does not begin by finding those objects. It divides the image into patches and turns each patch into a vector. Attention then lets positions exchange information.

Keep patching separate from understanding. After this note, [CLIP](clip.en.md) shows why an image encoder's architecture and its training objective are different choices.

## First count the tokens

Assume a $224\times224$ RGB image, $16\times16$ non-overlapping patches, and no padding:

$$N=\frac{224}{16}\frac{224}{16}=196,\qquad P^2C=16^2\times3=768.$$

196 counts patches; 768 counts raw values per patch. They are different dimensions. The table uses the channels-first layout common in PyTorch; other frameworks may put channels last.

| Step | Batched shape | What happens |
| --- | --- | --- |
| Input | $B\times3\times224\times224$ | Height, width, and color channels remain separate |
| Patch and flatten | $B\times196\times768$ | Each position contains one patch's pixels |
| Shared linear projection | $B\times196\times D$ | Every patch uses the same weights |
| Add CLS and positions | $B\times197\times D$ | Add a summary position and location information |
| Transformer encoder | $B\times197\times D$ | Update representations without shortening the sequence |

**Projection need not compress.** $D$ is the hidden width: 768 preserves the dimension, while 1024 increases it. The projection learns useful pixel combinations; it does not look up an image in a vocabulary of word IDs.

## Walk through a 4 × 4 image

This is an invented single-channel example, not a trained-model result. Flatten each $2\times2$ patch by rows, then order patches from top left to bottom right.

```text
 1  2 |  3  4       top left     → [1, 2, 5, 6]
 5  6 |  7  8       top right    → [3, 4, 7, 8]
------+------       bottom left  → [9, 10, 13, 14]
 9 10 | 11 12       bottom right → [11, 12, 15, 16]
13 14 | 15 16
```

Choose two projections by hand: the patch mean, and the right-column mean minus the left-column mean. The top-left patch becomes `[3.5, 1.0]`. A real model learns the weights; we do not have to specify a left–right difference detector.

```python
def patchify_gray(image, patch_size):
    if not isinstance(patch_size, int) or patch_size <= 0 or not image or not image[0]:
        raise ValueError("Expected a nonempty image and a positive integer patch size")
    height, width = len(image), len(image[0])
    if any(len(row) != width for row in image):
        raise ValueError("Rows must have equal widths")
    if height % patch_size or width % patch_size:
        raise ValueError("This example does not pad or crop partial patches")
    return [
        [image[top + row][left + column]
         for row in range(patch_size) for column in range(patch_size)]
        for top in range(0, height, patch_size)
        for left in range(0, width, patch_size)
    ]

image = [[1 + 4 * row + column for column in range(4)] for row in range(4)]
patches = patchify_gray(image, 2)
projection_rows = [[0.25, 0.25, 0.25, 0.25], [-0.5, 0.5, -0.5, 0.5]]
embeddings = [[sum(value * weight for value, weight in zip(patch, weights))
               for weights in projection_rows] for patch in patches]
assert embeddings == [[3.5, 1.0], [5.5, 1.0], [11.5, 1.0], [13.5, 1.0]]
```

The example exposes a limitation: different patches can collapse to the same two features. `[1, 2, 5, 6]` and `[2, 3, 4, 5]` both produce `[3.5, 1.0]`. Later attention cannot recover that lost distinction from these two numbers alone.

A convolution with kernel size and stride equal to patch size commonly implements patch projection. Without overlap, this corresponds to applying the same linear map to every patch. Follow `embedding`, reshape, `cls`, and `Transformer` in the [official ViT implementation](https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py) to trace the shapes.

## CLS is not a label, and positions are not decoration

CLS starts as a learned vector, not the answer “this image is a car.” Attention can let it summarize the patches; a classification head then predicts a class. Global average pooling is another possible readout, so CLS is not mandatory in every vision model.

Position information addresses a different issue: rearranging the same patches can change the image. Original ViT adds learned position vectors to content vectors, rather than appending coordinate columns. Attention still considers all patches; position embeddings do not turn it into a local convolution. [Original paper, §3](https://arxiv.org/abs/2010.11929)

## What one block does

Ignoring dropout and writing the full sequence as $X$, a Pre-LN block is:

$$U=X+\operatorname{Attention}(\operatorname{LN}(X)),\qquad
Y=U+\operatorname{MLP}(\operatorname{LN}(U)).$$

Attention exchanges information across positions, the MLP transforms features at each position, and residual connections preserve an existing path. The top-left output can now depend on a distant road sign, not just top-left pixels. For Q/K/V mechanics, continue with [multi-head attention](../00-foundations/core/multi-head-attention.en.md).

This is still an encoder: patches communicate bidirectionally rather than hiding future tokens as in text generation. Calling an encoder layer a block does not make it a different architecture category.

## Resolution and patch size change different things

Hold hidden width and layer count fixed. Count patch tokens only, excluding CLS:

| Image / patch | Tokens | Entries in one full attention-score matrix |
| --- | --- | --- |
| $224\times224$ / $16\times16$ | 196 | 38,416 |
| $448\times448$ / $16\times16$ | 784 | 614,656 |
| $448\times448$ / $32\times32$ | 196 | 38,416 |

Row 2 retains finer spatial sampling but uses 4 times as many tokens and 16 times as many score entries. Row 3 restores the token count by packing more pixels into each patch. **Equal counts do not imply equal ability to recognize small text, edges, or spatial relationships.**

This is not a whole-network memory or runtime prediction. MLPs, projections, batch size, and whether the kernel materializes scores all matter. Higher resolution also requires handling positions: original ViT interpolates the patch grid in two dimensions during fine-tuning, handling CLS separately. Compatible shapes do not guarantee unchanged accuracy at a new resolution.

## Fine-tuning the backbone is not a linear probe

| Setup | What changes | What it mainly measures |
| --- | --- | --- |
| Linear probe | A linear head; backbone frozen | Linear separability of existing features |
| Fine-tuning | Backbone and task head, with any freezing specified | Adaptation to a new task |
| Feature extraction | Nothing; no new head trained | Representations for retrieval or another model |

Original ViT transfer does not universally freeze the backbone: it replaces the head with a zero-initialized linear classifier and fine-tunes. The paper also reports few-shot evaluation on frozen representations. Check the protocol before comparing those results. [Original paper, §3.2 and §4](https://arxiv.org/abs/2010.11929)

ViT explains how images are encoded, not what that encoding must accomplish. [CLIP](clip.en.md) learns from image–text pairs; [vision to language](vision-to-language.en.md) connects visual features to generation. Similar structure does not make the objectives or learned capabilities equivalent.
