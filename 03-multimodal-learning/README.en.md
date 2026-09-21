# Multimodal Learning: Why Is Text Alone Not Enough?

[中文](README.md) · **English**

## Start here: multimodality adds evidence

Multimodal learning lets a model understand text, images, video, audio, and behavior together, because a great deal of meaning is never fully written down in text.

## An image can supply what the text cannot show

The text of a post says only: “Finally finished today.”

The attached image might be:

- a marathon finish line;
- a graduation ceremony;
- a screenshot of a project going live;
- or a LEGO set that was just completed.

From the text alone, the model knows that “something was finished.” Only after seeing the image can it understand what was finished, which topic the content belongs to, and why it is valuable to a certain kind of reader.

## Multimodality is not just attaching one more encoder

The real question is not “can it read images.” It is what the model should learn from the image, and how that should affect downstream behavior.

It can be split into four layers:

### Perception

Recognize objects, text, actions, scenes, and change over time.

### Alignment

Connect the signals in different modalities that describe the same thing, for example a chart in the image and the conclusion in the body text.

### Reasoning

Combine several modalities to answer questions, detect contradictions, or understand causal relationships.

### Value judgment

Judge whether the content is useful for the current user and task, rather than only describing what is in the picture.

This last layer is the hardest, because “content value” usually has no clean label.

## Common training approaches

- **Contrastive learning**: pull matching image–text pairs closer and push mismatched ones apart.
- **Caption / generation objective**: generate a description from an image, or an answer from multimodal context.
- **Instruction tuning**: use visual question answering and multimodal tasks to teach the model to follow instructions.
- **Multi-task learning**: jointly train several objectives such as understanding, classification, retrieval, and generation.
- **Preference / reward learning**: learn which multimodal outputs are more helpful or better match the user's goal.

Each method teaches a different capability. Being able to write captions does not mean being able to judge content quality; being able to do visual question answering does not mean understanding the value of content to a specific user.

## Multimodality and Personal AGI

User state does not come only from text either. Image choices, viewing behavior, tone of voice, and tool operations may all carry information.

But these signals are easy to over-interpret. Watching a video to the end does not necessarily mean liking it; uploading a photo does not give the system permission to infer sensitive attributes forever.

Multimodal personalization therefore has to consider, together:

- strength of evidence;
- time and context;
- uncertainty;
- privacy and user control;
- consistency with text and explicit feedback.

## How to evaluate

Beyond benchmark accuracy, also check:

- whether the model really uses the visual information rather than guessing from text priors;
- how behavior changes when one modality is masked;
- whether the model notices when image and text conflict;
- whether the added modality improves the end task, not only an intermediate metric;
- whether different content types and user slices benefit fairly.

## Continue reading

- [Data and feedback](../01-data-and-feedback/README.en.md)
- [Representation and memory](../02-memory/README.en.md)
- [Search](../04-search/README.en.md)
- [Evaluation](../07-evaluation/README.en.md)

## Reference papers

- [CLIP](https://arxiv.org/abs/2103.00020)
- [Flamingo](https://arxiv.org/abs/2204.14198)
