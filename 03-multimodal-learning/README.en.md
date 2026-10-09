# Multimodal learning: from matching images to answering questions

[中文](README.md) · **English**

> Last reviewed: 2026-10

A station timetable has a caption saying only “Weekend service has changed.” To answer when the last train leaves, the caption is not enough. But finding the right image and reading the correct time are different tasks too.

These notes start with turning images into vectors, then follow image–text matching, visual inputs to language models, and finally training and evaluation. Separating the tasks makes the architectures easier to understand.

<span id="alignment"></span>
<span id="multimodality-is-not-just-attaching-one-more-encoder"></span>
<span id="start-here-multimodality-adds-evidence"></span>

## Separate three tasks

| Task | Output | Main route |
| --- | --- | --- |
| Find images from text, or text from an image | Vectors and matching scores | ViT → CLIP → retrieval |
| Describe an image, read text, or answer a question | Text conditioned on visual and textual inputs | Vision encoder → connector → LLM |
| Generate or edit an image from text | Pixels or image latents | Image generation, flow matching, and editing |

CLIP aligns matching images and text; it is not a chat model that answers arbitrary questions. A visual question-answering model produces answers, but its representations are not automatically suitable for large-scale ANN retrieval. Images appear in both inputs, while outputs and objectives differ. The [CLIP paper](https://arxiv.org/abs/2103.00020) and [LLaVA visual instruction tuning](https://arxiv.org/abs/2304.08485) illustrate the first two tasks.

<span id="start-with-a-short-route"></span>

## Read from the foundations

| Order | Read | Question to resolve |
| --- | --- | --- |
| 1 · Vectors | [Vectors and similarity](../00-foundations/core/embeddings-and-similarity.en.md) | Why cosine is not a probability and what temperature changes |
| 2 · Image inputs | [ViT](vit.en.md) | Patches, projections, positions, and CLS; how resolution affects computation |
| 3 · Matching | [CLIP](clip.en.md) | How two sets of vectors learn to match and why other pairs in a batch may be false negatives |
| 4 · Answers | [Images entering an LLM](vision-to-language.en.md) | How visual features enter generation and why answers may still rely on text alone |
| 5 · Connectors | [BLIP and Q-Former](blip-and-q-former.en.md) | Separate visual compression, training objectives, and instruction-aware features |
| 6 · Training | [VLM fine-tuning](vlm-finetuning.en.md) | Processors, answer masks, frozen parameters, and independent generation tests |

For unfamiliar training mechanics, revisit [one parameter update](../00-foundations/deep-dives/training-step.en.md). For tensors and gradients, use the [small PyTorch experiments](../00-foundations/pytorch/README.en.md).

<span id="an-image-can-supply-what-the-text-cannot-show"></span>
<span id="perception"></span>
<span id="reasoning"></span>

## One image, three different checks

Keep the timetable example. Suppose the last departure shown is **22:40**, and the question asks when the last train leaves. This is an original teaching scenario, not a model evaluation result.

<figure class="worked-update">
<ol>
<li><small>01 / INPUT</small><strong>The caption omits the time</strong><span>The digits and their row and column relationships in the image supply the evidence.</span></li>
<li><small>02 / REPRESENTATION</small><strong>Is the resized text readable?</strong><span>Inspect the image the processor actually supplies. A clear original can still become an unreadable input.</span></li>
<li><small>03 / OUTPUT</small><strong>Does the answer follow the evidence?</strong><span>Change only the pictured time to 23:10. Keep the question fixed and compare generations.</span></li>
</ol>
<figcaption>Check input information loss separately from answer errors. These checks can start during data preparation, before training.</figcaption>
</figure>

For **retrieval**, check whether this image ranks ahead of irrelevant candidates. For **reading the time**, check digits, direction, and table alignment. For **answering the question**, also check that the model has not confused the first departure with the last. One aggregate score rarely explains all three.

OCR extracts characters but may lose layout relationships. A VLM can use visual context but may misread small text. Not every pipeline needs a separate OCR service: establish a direct-image baseline, then compare OCR, cropping, and higher resolution on the same examples.

<span id="common-training-approaches"></span>
<span id="continue-reading"></span>
<span id="reference-papers"></span>

## Continue into architectures, budgets, and new modalities

| Question | Read | Keep fixed when comparing |
| --- | --- | --- |
| Is extra tiling worth it for small text? | [LLaVA and DeepSeek-VL](vlm-designs.en.md) | Image and question; record tile count and input length |
| How are image positions and video time encoded? | [Qwen-VL](qwen-vl.en.md) | Version, grid, and visual sequence; do not transfer configurations between versions |
| Why can streaming speech still stall? | [Omni](omni-streaming.en.md) | First-chunk waiting, audio frames, interruption, and tool state |
| Does representing a document as an image save resources? | [OCR and context compression](ocr-compression.en.md) | Input/output cost and preservation of characters and reading order |
| How does generating an image differ from understanding it? | [Qwen-Image](image-generation.en.md) | Generation objective, edited region, and text preservation |
| Did fine-tuning improve formatting or visual understanding? | [VLM training and evaluation](vlm-finetuning.en.md) | Unseen sources and layouts, missing-image and edited-image comparisons |

Contrastive learning, caption generation, instruction tuning, and preference optimization supply different supervision. They can be combined in multi-task training, but task sampling and loss weights still need separate checks. More tasks do not automatically produce better representations.

<span id="how-to-evaluate"></span>
<span id="multimodality-and-personal-agi"></span>
<span id="value-judgment"></span>

## Check whether the new modality helps

A correct “22:40” is not enough. Retain the original and create a few comparisons whose expected behavior can be checked:

| Comparison | What you hope to observe | Interpretation limit |
| --- | --- | --- |
| Remove the image | The answer is no longer directly determined | A drop can reflect an unusual input; it does not alone establish correct visual understanding |
| Change the time, keeping layout fixed | The answer follows the new digits | This tests local evidence use, not all visual reasoning |
| Hide the time field | Acknowledge missing evidence instead of inventing a time | The data must actually include unanswerable cases |
| Text says 22:40; image says 23:10 | Identify the conflict and follow the task's evidence rules | Neither modality is always more trustworthy |
| Change the source or layout | Locate the right field despite the change | A random split may put the same template in both training and test sets |

Return to the task: retrieval should recover useful candidates; question answering should produce correct, grounded answers. Inspect cost and failures in both. Break results down by content type, clarity, language, and relevant user groups to check whether gains are confined to one slice. See [ablations and slices](../07-evaluation/ablation-and-slices.en.md).

For personalization, images, viewing behavior, and speech still provide limited evidence. Finishing a video does not establish preference; uploading an image does not permit permanent inference of sensitive attributes. Consider storage, correction, and deletion alongside [data and feedback](../01-data-and-feedback/README.en.md) and [memory management](../02-memory/README.en.md).

Foundational mechanisms use public, inspectable versions; later versions are discussed separately in each article. Reported results, worked arithmetic, runnable examples, and unexecuted training plans are explicitly distinguished. To connect representations to a system, continue with [search and retrieval](../04-search/README.en.md).
