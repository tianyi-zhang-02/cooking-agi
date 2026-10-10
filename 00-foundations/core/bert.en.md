# BERT: how an encoder learns to read

[中文](bert.md) · **English**

The word “cold” means something different in “cold water” and “a cold welcome.” The surrounding words help us tell the difference. BERT uses context too: it computes a representation for each token in this particular input, rather than stopping at a fixed word lookup.

More concretely, **BERT produces one vector for each input token**. Each vector belongs to a position, but the encoder lets it incorporate information from elsewhere in the input. These vectors are not answers yet. A task head, trained with the appropriate labels, turns them into review categories, token labels, or answer positions.

How does it learn those representations? One pretraining task hides some tokens and asks the model to recover them from the remaining text. Start with `the tea is cold`: hide `tea`, keep it as the answer, and follow the input through the model.

<div class="bert-story" data-bert-story data-lang="en" id="bert-walkthrough" markdown="1">

**Input → embeddings → bidirectional encoder → task output**

For MLM, `[CLS] the [MASK] is cold [SEP]` enters the encoder. Every position gets a contextual vector; the output at the selected position is trained to predict `tea`. Both `the` and `is cold` can provide context. For classification, supply the complete sentence and fine-tune with a classification head reading `[CLS]` instead. The encoder is reused, not the MLM vocabulary predictions.

The static diagram includes both original pretraining tasks: MLM and NSP. Enable JavaScript for the step-by-step view; it illustrates the architecture without running a model.

![Original BERT pretraining: corrupted tokens enter a shared bidirectional encoder. MLM predicts original tokens at selected positions; NSP uses CLS to classify segment adjacency.](../assets/bert-flow.en.svg)

</div>

You do not need every implementation detail on the first read:

- **To understand the model**, follow [inputs and vectors](#input), [context mixing](#context-calculation), and the [review classifier](#classify-example). The [answer-span example](#answer-span) shows how the same encoder can support a different task.
- **To implement or debug training**, focus on [the three masks](#three-masks), [loss](#loss), and the [Python checks](#small-lab). Derivations and code are available to expand as needed.

This page covers the original BERT introduced in 2018 and identifies later changes separately.

## 1. Why hide a word first? {#context}

If the complete sentence is visible and we ask what is at position 2, the model can copy `tea`. It has little reason to learn how the sentence works. Replacing `tea` with `[MASK]` removes that shortcut: `the` provides a grammatical clue, and `is cold` provides more context. Predicting the missing token now requires using the surrounding text.

`the [MASK] is cold` does not uniquely determine `tea`: `coffee` is plausible too. Training still uses the original word as its target, and cross-entropy rewards assigning probability to it. We are learning a distribution from many examples, not solving blanks with a guaranteed single answer.

This also explains where the labels come from: **the original text supplies the answers**. Save `tea` before changing the input, and we have a training example without asking anyone to annotate it. Building supervision from the data itself is usually called self-supervised learning.

A downstream task still needs its own labels. Recovering a missing word does not tell the model which reviews count as positive or negative. The idea is to learn useful representations from a large amount of text, then adapt them with task data rather than starting from random weights each time. [BERT §3](https://arxiv.org/html/1810.04805v2#S3)

“Bidirectional” means attention can use both left and right context at every layer. It does not mean running a left-to-right pass and a right-to-left pass separately, then joining their outputs. Whether we should allow that access depends on the task:

| What are we doing? | Available information | Required behavior |
| --- | --- | --- |
| Classify a supplied passage | The entire input is available | Combine its content into a judgment |
| Extract an answer | Both question and passage are available | Locate an answer span |
| Continue writing | Future output does not exist yet | Generate from the available prefix |

In the first two tasks, the full input is already available. In the third, the continuation has not been generated yet. That distinction is more useful than asking whether encoders or decoders are better. The original BERT paper mainly studies the first kind of text-understanding task.

<details markdown="1">
<summary>How does this differ from LSTM or decoder-only models?</summary>

These names describe different things. LSTM is a recurrent cell. BERT combines an encoder architecture with a pretraining recipe. Decoder-only names an architecture, commonly trained with an autoregressive objective.

| What are we comparing? | LSTM / BiLSTM | Original BERT | Common autoregressive decoder-only |
| --- | --- | --- | --- |
| Information exchange | Sequential state updates; BiLSTM uses two directions | Bidirectional self-attention | Causally masked self-attention |
| Parallel computation | Examples can run together, but ordinary recurrence depends on the preceding state in each direction | Positions within a layer run together; layers remain dependent | Given target text permits parallel training; ordinary generation remains token by token |
| Training objective | The cell itself specifies no loss | MLM + NSP | Next-token cross-entropy |
| Right-context access | BiLSTM can; unidirectional LSTM cannot | Right context in the corrupted input is visible | Future output is unavailable during generation |

For classification of “The service was slow, but the food was excellent,” both BiLSTM and BERT can use the second clause. Attention is not the only way to read both directions. Their computation paths differ: ordinary LSTM passes information along recurrent states, while full self-attention permits direct interaction between distant positions within one layer, at the cost of pairwise computation that grows with length. Easier parallelization does not guarantee lower latency on every input or device. [Transformer §4](https://arxiv.org/html/1706.03762v7#S4)

Now try continuing “The service was slow, but”: the second clause does not exist yet. Classifying a complete input and generating from a prefix are different tasks. An LSTM can learn next-token prediction; an encoder can learn classification. **Do not mistake a training objective for an intrinsic property of a basic component.** See [RNN / LSTM](recurrent-models.en.md) for the recurrent computation.

Pretraining and transfer were already active research directions. In 2018, [ULMFiT](https://aclanthology.org/P18-1031/) pretrained an LSTM language model, then adapted it to a target corpus and classification task. [ELMo](https://aclanthology.org/N18-1202/) supplied contextual features from bidirectional LSTM language models. That background helps explain BERT's choices: deep bidirectional attention, pretraining objectives suited to it, and a shared encoder design that could be adapted to many tasks.

Reading both directions also does not mean reading arbitrarily long documents. Original BERT supports inputs of up to 512 tokens, and full attention has quadratic computation in sequence length. If a crucial fact at token 600 is truncated away, bidirectional attention cannot recover it. Long documents still need an appropriate chunking, retrieval, or validated long-context approach. [BERT Appendix A.2](https://arxiv.org/html/1810.04805v2#A1.SS2)

</details>

## 2. Follow one input through the model {#input}

### Tokenize the text and mark the boundaries

Original BERT uses WordPiece to split text into tokens from its vocabulary, then maps them to integer IDs. One word can become several subwords, so ten words do not necessarily occupy ten input positions. See [Tokenization](tokenization.en.md) for how that split works.

One text segment becomes `[CLS] A [SEP]`; a pair becomes `[CLS] A [SEP] B [SEP]`. The first token, `[CLS]`, provides a position whose representation can be used for classification. `[SEP]` marks the end of a segment. Both have IDs and embeddings and pass through the model alongside ordinary tokens.

### Three embeddings provide three kinds of information

The model needs to know which token this is, where it appears, and which segment it belongs to:

| Vector | What does it identify? | In our example |
| --- | --- | --- |
| Token embedding | The token at this position | Look up `tea`, or `[MASK]` after replacing it |
| Position embedding | Its position in the sequence | `[CLS]` starts at 0; `tea` is at position 2 |
| Segment embedding | Which text segment contains it | A for one segment; A and B for a pair |

All three vectors have the same width. Add them element by element, then apply LayerNorm and dropout before the encoder. With three illustrative dimensions, `[1, 0, 2] + [0, 1, 1] + [1, 1, 0] = [2, 2, 3]`: the result still has three dimensions. These are made-up numbers showing addition, not learned weights; BERT-Base uses width 768. Original BERT learns absolute position embeddings rather than using RoPE. [Paper §3](https://arxiv.org/html/1810.04805v2#S3), [embedding implementation](https://github.com/google-research/bert/blob/master/modeling.py)

For “Rain is forecast tomorrow” paired with “Bring an umbrella,” the first segment, including `[CLS]` and its `[SEP]`, uses segment A. The second segment and its final `[SEP]` use B. Positions increase across the whole input rather than restarting at B. The segments can read each other through self-attention; there is no additional decoder cross-attention sublayer.

### The shape stays the same; the representations change

The same token ID always looks up the same row in the embedding table. After that, the encoder recomputes its representation using the current sentence. Change `the tea is cold` to `the weather is cold`: the lookup for `cold` is unchanged, but the context it receives is different, so its final vector can differ. **The lookup is the starting point; the contextual hidden state is what the task uses.**

| Stage | Toy shapes: batch 2, padded length 7, hidden size 8 |
| --- | --- |
| Token IDs | `[2, 7]`, integer indices |
| Sum of embeddings | `[2, 7, 8]`; adding vectors does not concatenate them into 24 dimensions |
| Each encoder layer | Still `[2, 7, 8]`; exchange information, then apply position-wise FFNs |
| Final hidden states | An 8-dimensional representation at each position, not class probabilities |
| Task output layer | Vocabulary, class, or answer-position scores, depending on the task |

The table uses small dimensions to make the shapes easy to follow. BERT-Base actually returns `[batch, length, 768]`; it does not automatically reduce the whole sentence to one vector. Getting a sentence representation requires choosing how to read or pool those positions. Revisit [Vanilla Transformer](vanilla-transformer.en.md) for residual, LayerNorm, and FFN operations.

### How does context actually change a vector? {#context-calculation}

Look at one attention head at one output position. It assigns weights to the visible positions, then adds their value vectors using those weights. A weight tells us how much this particular attention calculation draws from a position—not how important that word is in general.

For a small calculation, use three positions with two-dimensional values and assume we have already computed the softmax weights:

<figure class="worked-update" lang="en" id="context-mixture">
  <figcaption>One weighted sum · These are illustrative values, not measured BERT weights.</figcaption>
  <ol>
    <li><small>Position A · weight 0.2</small><strong>[1, 0] × 0.2</strong><span>Contributes [0.2, 0]</span></li>
    <li><small>Position B · weight 0.3</small><strong>[0, 2] × 0.3</strong><span>Contributes [0, 0.6]</span></li>
    <li><small>Position C · weight 0.5</small><strong>[2, 1] × 0.5</strong><span>Contributes [1, 0.5]</span></li>
  </ol>
</figure>

Adding these contributions gives `[1.2, 1.1]` at the current query position. Keep the weights fixed and change only A's value from `[1, 0]` to `[3, 0]`: the result becomes `[1.6, 1.1]`. **Changing another position changes what this position receives.** In a real model, different context can also change the queries, keys, and attention weights, not just the values.

Where do the weights come from? The current position's query is compared with each key, then the scores are scaled and passed through softmax. Multiple heads do this separately; their outputs are concatenated and projected. Our two-dimensional result is just one head's output, not the final layer output: residual connections, normalization, and the FFN still have their part to play. See [Transformer §3.2](https://arxiv.org/html/1706.03762v7#S3.SS2).

<details markdown="1">
<summary>Inside one layer: attention, residuals, and the FFN</summary>

Original BERT uses post-LN. For layer input $H$, bidirectional multi-head attention is followed by a position-wise FFN. Write $D$ for dropout and $\operatorname{LN}$ for LayerNorm so we can show each residual addition separately:

$$
\begin{aligned}
A&=D(\operatorname{MHA}(H)),\\
U&=\operatorname{LN}(H+A).
\end{aligned}
$$

$$
\begin{aligned}
F&=D(\operatorname{FFN}(U)),\\
H'&=\operatorname{LN}(U+F).
\end{aligned}
$$

MHA here includes the output projection after concatenating heads; the FFN is dense → GELU → dense. Each sublayer has its own residual addition and LayerNorm, not one shared addition at the end of the block. Base stacks 12 layers and Large 24. Token, segment, and position embeddings are added before entering the stack, not again in every block. Compare [`transformer_model` in the original implementation](https://github.com/google-research/bert/blob/master/modeling.py). A modern decoder diagram with pre-norm, RMSNorm, and SwiGLU does not describe this block.

We can also check the weighted sum without downloading a model. Set the already-scaled attention scores to `log(2)`, `log(3)`, and `log(5)`; their softmax gives `0.2`, `0.3`, and `0.5`:

```python
import math

scores = [math.log(2), math.log(3), math.log(5)]
exp_scores = [math.exp(score - max(scores)) for score in scores]
weights = [score / sum(exp_scores) for score in exp_scores]
values = [[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]]

def mix(current_values):
    return [sum(weight * value for weight, value in zip(weights, column))
            for column in zip(*current_values)]

before = mix(values)
values[0] = [3.0, 0.0]
after = mix(values)
assert all(math.isclose(actual, expected)
           for actual, expected in zip(before, [1.2, 1.1]))
assert all(math.isclose(actual, expected)
           for actual, expected in zip(after, [1.6, 1.1]))
print([round(value, 2) for value in before])
print([round(value, 2) for value in after])
```

This checks a single head's weighted sum, without Q/K/V projections, dropout, or training.

</details>

## 3. Three masks, three different responsibilities {#three-masks}

Code often contains `[MASK]`, an `attention_mask`, and ignored labels together. They answer different questions: **what changed in the input, what attention may read, and which outputs contribute loss.** Keeping those questions separate makes the data much easier to debug.

### Input corruption changes what the model sees

Masked language modeling first selects prediction positions, then corrupts their inputs. The paper uses a prediction rate of roughly 15%; of the selected positions, 80% receive `[MASK]`, 10% a random token, and 10% stay unchanged. The target always remains the original token. Structural special tokens and padding are excluded as ordinary targets. [Original data construction](https://github.com/google-research/bert/blob/master/create_pretraining_data.py)

Suppose a batch already has 150 selected targets. About 120 take the `[MASK]` branch, 15 take random replacement, and 15 stay unchanged. These are expected counts, not fixed quotas; a random replacement may also happen to equal the original token.

<details markdown="1">
<summary>One detail to check in the original 15% calculation</summary>

The code calculates the prediction count from unpadded `len(tokens)`, including `[CLS]` and `[SEP]`. It rounds the count, requests at least one, and caps it at `max_predictions_per_seq`, then samples candidates that exclude structural tokens. This is not independent 15% selection of each ordinary position. For eight ordinary tokens plus `[CLS]` and `[SEP]`, length is 10: with a sufficient cap, `round(10 × 0.15)` requests two targets, not a mandatory 1.2.

The repository later added optional whole-word masking, grouping `##` subwords for selection. That option is not the recipe of every original checkpoint. Fix the version, sampling rules, and random state when reproducing a pipeline.

</details>

An unchanged selected position still contributes loss. The model is not told which unchanged inputs were selected. Random and unchanged inputs reduce the mismatch between seeing `[MASK]` during pretraining and usually not seeing it downstream; they do not eliminate the mismatch.

The choice of positions changes which clues the model can use. Baidu's 2019 [ERNIE §3.2](https://arxiv.org/html/1904.09223v1#S3.SS2) tried masking whole phrases and entities. In “New York City is in the United States,” hiding just “York” leaves “New” and “City” as clues. Hiding the whole name requires more context from outside the entity. This was an early ERNIE data-construction method, separate from the architecture of later chat models sharing the name.

### Attention masking controls readable positions

A bidirectional encoder can read both sides of a token. Only valid query rows are shown; `1` permits reading the key:

| Query \ Key | `tea` | `[MASK]` | `again` | `[PAD]` |
| --- | --- | --- | --- | --- |
| `tea` | 1 | 1 | 1 | 0 |
| `[MASK]` | 1 | 1 | 1 | 0 |
| `again` | 1 | 1 | 1 | 0 |

`[MASK]` participates as an input token; it is not the negative-infinity attention mask. Padding keys are excluded. A padding query may still produce a nonzero hidden state, depending on implementation; that does not make it eligible for loss or pooling. Packing independent samples also requires blocking cross-sample attention: a separator alone does not isolate them.

Padding and constructing a padding mask are separate operations: one adds placeholder tokens; the other identifies them for attention. Sequences need not always be padded to the model's maximum length; padding to the longest example in the batch is common. A causal mask instead restricts which earlier or later positions are visible. See the [three attention sites](vanilla-transformer.en.md#the-three-attention-sites-ask-different-questions).

### Loss masking selects outputs to score

Use a hand-made vocabulary: `PAD=0, CLS=1, SEP=2, MASK=3, the=4, tea=5, is=6, cold=7, hot=8`. It is not an actual WordPiece vocabulary. Select zero-based positions 2 and 4, leave `tea` unchanged, and replace `cold` with `[MASK]`:

| Field | Values |
| --- | --- |
| Original token IDs | `[1, 4, 5, 6, 7, 2, 0]` |
| Model input | `[1, 4, 5, 6, 3, 2, 0]` |
| Attention mask | `[1, 1, 1, 1, 1, 1, 0]` |
| Labels | `[-100, -100, 5, -100, 7, -100, -100]` |

Here `-100` is our ignore marker, a common PyTorch cross-entropy convention rather than a vocabulary item or the original TensorFlow data format. There is one `[MASK]` but two supervised positions. **Do not replace the loss mask with `input_ids == mask_token_id`.** We selected two positions deliberately to explain the fields, not to simulate 15% sampling.

## 4. Where is the loss computed? {#loss}

In the previous example, training scores two predictions: `tea` at one position and `cold` at another. The other positions provide context without contributing their own MLM loss terms.

Let $M$ contain the batch's selected positions, $x_i$ be the original target, and $\tilde x$ the corrupted input. A common aggregation is

$$
L_{\mathrm{MLM}}=-\frac{1}{|M|}\sum_{i\in M}\log p_\theta(x_i\mid\tilde x).
$$

Each selected position gets a score for every vocabulary entry. Cross-entropy checks the probability assigned to the original token. Labels line up with their output positions: position 2 recovers the original token at position 2, rather than predicting position 3 as in next-token training.

The original MLM head gathers selected hidden states, applies dense + activation + LayerNorm, then projects through the shared input-embedding matrix and adds a vocabulary bias. For a first pass, the important change is **one position's representation becomes a set of vocabulary scores**. [Original implementation](https://github.com/google-research/bert/blob/master/run_pretraining.py)

The TensorFlow implementation stores supervision as `masked_lm_positions`, `masked_lm_ids`, and `masked_lm_weights`. Padded prediction slots have zero weight; its denominator is the weight sum plus `1e-5`. The calculation and teaching code below use an exact valid-target mean rather than reproducing that epsilon.

Suppose the two correct-token probabilities are 0.8 and 0.25:

$$
\begin{aligned}
L&=\frac{-\log0.8-\log0.25}{2}\\
 &\approx0.804719.
\end{aligned}
$$

Dividing by all seven positions would give about 0.229920. The predictions did not improve; the denominator simply grew. There are two valid targets here, and adding `[PAD]` tokens should not change the answer. Our teaching code raises an error for zero selected targets to catch data problems early.

### What happens in one training step? {#training-step}

Here is how the data and loss fit into an update:

1. **Save the answers, then change the inputs.** Keep the original tokens, select prediction positions, and build corrupted inputs with their labels.
2. **Encode the whole sequence.** Compute hidden states at all valid positions, then use the MLM head to score the vocabulary at selected positions.
3. **Compute loss against the original tokens.** Backpropagate the valid-target cross-entropy and update the embeddings, encoder, and MLM head. Original BERT also adds the NSP loss described next.
4. **Use the updated weights on the next batch.** Repeated parameter updates are what gradually change the model's predictions.

Selecting 15% of positions does not mean doing only 15% of the computation. Those positions need the surrounding tokens as context, so the encoder still processes the sequence. Gathering selected states saves part of the vocabulary-output work, not the rest of the encoder computation. [Original training implementation](https://github.com/google-research/bert/blob/master/run_pretraining.py)

The same sentence can also supply different questions: hide `tea` this time, `cold` another time. The original implementation generates and saves several masking variants in preprocessing; RoBERTa generates masks when sequences are fed to the model. The distinction is when and how masks change, not whether one method is random and the other is not. [RoBERTa §4.1](https://arxiv.org/html/1907.11692v1#S4.SS1)

<details markdown="1">
<summary>Try it: change the input or add padding. What happens to loss?</summary>

<div class="encoder-lab" data-encoder-lab="mlm" data-lang="en" id="mlm-lab" markdown="1">

**Changed input. Same target.**

For the selected position `cold`, the three corruption branches look like this. The target is always `cold`, including when the input is unchanged:

| Branch | Input | Target |
| --- | --- | --- |
| Mask | `[MASK]` | `cold` |
| Random replacement (one possible draw) | `hot` | `cold` |
| Unchanged | `cold` | `cold` |

With hand-set correct-token probabilities of 0.80 for `tea` and 0.25 for `cold`, mean loss is 0.805 across the two targets. Adding padding leaves it unchanged. Enable JavaScript to try the replacements and vary the probability; the calculation does not run a model.

</div>

</details>

<details markdown="1">
<summary>Can positions without direct loss still learn?</summary>

Yes. Unselected positions can supply context to selected positions, and their representations can receive gradients through attention. Ignoring an output score does not disconnect that position from the whole computation.

For vocabulary logit $z_{i,v}$ at a selected position, this mean loss gives

$$
\frac{\partial L}{\partial z_{i,v}}
=\frac{p_{i,v}-\mathbf 1[v=x_i]}{|M|}.
$$

Unselected output logits do not enter this loss directly. Shared parameters and contextual hidden states can still receive gradients. Separate “this output was not scored” from “this computation did not participate in learning.”

</details>

## 5. NSP: did these segments occur together? {#nsp}

Original BERT also predicts whether one segment follows another. Half the pairs are contiguous and half are randomly paired; the output is binary, not the next sentence's tokens. [Paper §3.1](https://arxiv.org/html/1810.04805v2#S3.SS1)

Here a “sentence” can be a contiguous span containing multiple sentences. The NSP head reads the pooled `[CLS]` representation. The original implementation adds MLM loss and NSP loss before backpropagation.

Suppose a passage reads, “It started raining. I brought the laundry inside.” Taking those adjacent spans gives a positive pair. Replacing the second span with a randomly sampled statement about matrix multiplication gives a negative pair. The label records adjacency in the corpus, not whether we think the text sounds plausible: a random pair can happen to fit quite well.

RoBERTa in 2019 revisited masking, input organization, NSP, and training scale. It uses dynamic masking and drops NSP in a revised input recipe. Its ablations also change how text is assembled; attributing every improvement to deleting one loss would miss that distinction. [RoBERTa §4](https://arxiv.org/html/1907.11692v1#S4)

For a later encoder, ask separately about its architecture, data construction, and training objectives. Looking like BERT does not necessarily mean it was trained with NSP.

## 6. Turn pretraining into a task model {#finetuning}

Suppose we have downloaded a pretrained checkpoint and want to classify reviews as positive or negative. Now we provide complete reviews without hiding words. The training target changes from “which token was here?” to “which class does this review belong to?”

### Start with a review classifier {#classify-example}

For “The tea was cold, and I waited ages,” we want the negative class. The encoder still produces a vector at every position, and the classifier reads `[CLS]` to produce class scores. `[CLS]` is not a predefined average of the sentence: it receives information through attention and learns what to retain through training. The original classification code also applies a dense layer and tanh to pool that state. [Classifier implementation](https://github.com/google-research/bert/blob/master/run_classifier.py), [pooler implementation](https://github.com/google-research/bert/blob/master/modeling.py)

Let $h_{\mathrm{CLS}}$ be the pooled representation and $W,b$ the classifier parameters:

$$
\begin{aligned}
z&=W h_{\mathrm{CLS}}+b,\\
p&=\operatorname{softmax}(z).
\end{aligned}
$$

BERT-Base provides 768 features, so a two-class $W$ has shape `[2, 768]` and produces two logits. Suppose they are `[1, 3]` in the order `[positive, negative]`. Writing the negative-class probability as $p_-$:

$$
\begin{aligned}
p_-&=\frac{e^3}{e^1+e^3}\\
   &\approx0.880797,\\
L&=-\log p_-\approx0.126928.
\end{aligned}
$$

These are assumed scores for a calculation, not model measurements. Here softmax compares **two classes**; in MLM it compares vocabulary entries. We can reuse the encoder while changing what its outputs mean through the head and training labels.

### For question answering, read the token positions {#answer-span}

The passage says, “Meeting moved to Friday afternoon.” The question is “When is the meeting now?” The answer is already there. Rather than writing a new sentence, the model needs to locate **where “Friday afternoon” begins and ends**. This is extractive question answering.

BERT reads the question and passage together, so the passage representations depend on the question too. The QA head assigns two scores at each position: how plausible it is as the **start** of the answer and how plausible it is as the **end**. Start and end each have their own softmax over positions, not over the vocabulary.

For illustration, split the passage into six slots: `Meeting / moved / to / Friday / afternoon / .`, numbered from 0. These are hand-made slots, not actual WordPiece output. Suppose we have these scores:

<figure class="worked-update worked-update--pairs" lang="en" id="answer-boundaries">
  <figcaption>Extracting “Friday afternoon” · Scores are hand-set for this example.</figcaption>
  <ol>
    <li><small>Start · position 3</small><strong>Friday · score 4</strong><span>Begin the answer here. Start scores compare possible beginning positions.</span></li>
    <li><small>End · position 4</small><strong>afternoon · score 5</strong><span>Stop here, including this slot. End scores are computed separately.</span></li>
  </ol>
</figure>

This span scores `4 + 5 = 9`. With the full score arrays below, it is the highest-scoring valid span, so we extract positions 3 through 4: “Friday afternoon.” During training, annotated start and end positions supervise these two predictions. [Original paper §4.2](https://arxiv.org/html/1810.04805v2#S4.SS2)

**Taking two independent maxima is not enough.** The best start might be at position 4 while the best end is at position 3—a backwards span. We also need to exclude the question, special tokens, and padding, and may want an answer-length limit. The small program below only handles this selection step; it does not run BERT.

<details markdown="1">
<summary>Select a span in Python: keep its endpoints ordered and its positions valid</summary>

```python
import math

def best_answer_span(start_scores, end_scores, allowed, max_length):
    length = len(start_scores)
    if length != len(end_scores) or length != len(allowed):
        raise ValueError("Scores and mask must have the same length")
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if any(type(value) is not bool for value in allowed):
        raise ValueError("allowed must contain booleans")
    if not all(math.isfinite(score) for score in [*start_scores, *end_scores]):
        raise ValueError("This example expects finite scores")
    best = None
    for start in range(length):
        for end in range(start, min(length, start + max_length)):
            if not allowed[end]:
                break
            score = start_scores[start] + end_scores[end]
            if best is None or score > best[0]:
                best = (score, start, end)
    return best

start_scores = [0, 0, 0, 4, 1, -2]
end_scores = [0, 0, 0, 1, 5, -2]
answer = best_answer_span(start_scores, end_scores, [True] * 6, 3)
assert answer == (9, 3, 4)
print(answer)
```

`allowed` marks positions that can belong to an answer; `max_length` limits its length. The loop considers contiguous spans with ordered endpoints and stops extending a span at an invalid position. Ties keep the first candidate encountered. It neither computes probabilities nor checks factual correctness. For input length $L$ and maximum answer length $A$, enumeration costs $O(L\min(L,A))$. This teaching input is short enough to keep the implementation simple.

With a real tokenizer, do not reconstruct the answer by blindly joining token strings. Keep the character offsets and slice the original passage so that spaces, punctuation, and subwords survive correctly. The [Transformers QA example](https://huggingface.co/docs/transformers/tasks/question_answering) uses offset mappings and sequence IDs to handle the alignment.

</details>

Now ask, “Which room is the meeting in?” The passage does not say. Yet as long as some span is allowed, our selection code will still return one. **A highest-scoring span is not proof that the passage answers the question.** Unanswerable questions need appropriate training labels and a no-answer decision. Original BERT's SQuAD 2.0 setup uses `[CLS]` for a null answer and chooses a decision threshold on development data. [Paper §4.3](https://arxiv.org/html/1810.04805v2#S4.SS3)

A missing answer and a truncated answer are different problems: one calls for abstention, while the other calls for checking document windows and input length. And if the task asks for a newly written explanation rather than a passage excerpt, that is a generative QA requirement—not something this span-selection head does.

The same distinction carries across the other task heads:

| Task | Read from | Outputs and supervision |
| --- | --- | --- |
| Text classification | Aggregate `[CLS]` representation | Class logits and a label per example |
| Named entity recognition | Relevant token representations | Token labels, with word/subword alignment handled first |
| Extractive QA | Candidate answer positions | Start/end logits; exclude question and padding positions from candidate answers |
| Pair relevance | Jointly encoded query and document | A pair score with relevance supervision |
| Single-vector retrieval | Separately encoded and pooled inputs | Appropriate representation-learning objectives and retrieval evaluation |

### Should the encoder be trained too? {#adaptation}

Freezing the encoder and training only the classifier is a useful first experiment. There is no need to retain encoder activations for backpropagation. With fixed inputs and dropout disabled, features can even be computed in advance. But a small classifier has limited ways to recover distinctions that the frozen representation did not preserve.

Full fine-tuning lets the encoder adapt to the task as well. It uses more training memory and computation and can overfit small datasets. The original paper fine-tunes end to end and selects learning rates on development data; it also reports sensitivity to classifier initialization and data order on small datasets. [BERT §3.2 and §4.1](https://arxiv.org/html/1810.04805v2#S3.SS2)

Compare the two approaches on the same data split. Record task scores, training cost, and variation across seeds rather than assuming that unfreezing everything must win and keeping only the best run.

### Why do retrieval systems distinguish cross-encoders and dual encoders? {#retrieval-use}

Encode a query and document together and they can exchange information at every layer. That is useful for judging a particular pair, but the document states depend on this query: they cannot be computed once and reused for every query. With 100,000 candidate documents, this approach evaluates 100,000 pairs per request. Batching helps throughput without removing the pair evaluations.

A dual encoder processes queries and documents separately, allowing document vectors to be computed and indexed in advance. That makes retrieval cheaper, but gives up direct token-level interaction during encoding. A common combination retrieves a small candidate set with vectors, then reranks it with a cross-encoder. [Sentence-BERT (2019)](https://aclanthology.org/D19-1410/) explains why useful cosine-similarity embeddings need suitable training. For the system design, continue to [dual-encoder retrieval](../../04-search/dual-encoder.en.md) and [BGE-M3 / Qwen3 Embedding](../../04-search/embedding-models.en.md).

## 7. Run the field and loss checks {#small-lab}

The [teaching script](../code/mlm_contracts.py) uses only Python's standard library, fixes the replacements, downloads no model, and performs no pretraining:

```bash
python3 00-foundations/code/mlm_contracts.py
```

```text
Input IDs: [1, 4, 5, 6, 3, 2, 0]
Labels: [-100, -100, 5, -100, 7, -100, -100]
Attention: [1, 1, 1, 1, 1, 1, 0]
Selected-token loss: 0.804719
```

`corruption_action` uses one uniform draw to illustrate the 80/10/10 branch after selection. Original code uses conditional draws: the probabilities match, but a shared seed need not give identical outcomes. `build_example` accepts explicit replacements while preserving targets. `masked_cross_entropy` averages only supervised positions. This is not a random collator and includes no attention, learned head, or backpropagation. It retains the unchanged-but-selected case so you can compare against the incorrect rule of counting `[MASK]` tokens.

Change one thing at a time:

| Change | Expected result | What it checks |
| --- | --- | --- |
| Append padding with labels set to `-100` | Loss stays the same | The denominator counts valid targets only |
| Make ignored-position logits very large | Loss stays the same | Unsupervised outputs do not enter the calculation |
| Raise a correct target logit, holding the others fixed | Loss decreases | Target IDs, dimensions, and loss direction agree |

Passing these checks establishes the relationship between data and loss. With a real model, also check gradients and parameter updates, then evaluate on held-out task data.

## 8. What changes when real data arrive? {#checks}

Start with an experiment small enough to inspect. Try fitting a few dozen training examples: if that fails, check labels, masks, gradients, and the optimizer before scaling up. Once it works, move to an independent validation set. The small-set experiment is a debugging check, not evidence of generalization.

| Observation | Inspect first |
| --- | --- |
| Suspiciously low loss | Target leakage, total-length denominators, scoring only unchanged positions |
| Large differences across batches | Valid-target counts, lengths, duplicates, and sampling consistency |
| Good MLM loss, poor classification | Labels, distribution, task head, and fine-tuning; pretraining loss is not the task metric |
| Strong offline results, poor deployment | Near-duplicate leakage and truncation that removes key information |

Task metrics also need context. Suppose validation contains 95 positive reviews and five negative ones. Predicting “positive” every time gives **95% accuracy but 0% recall on negative reviews**. If the purpose is to find complaints, that classifier is not much help. Inspect per-class precision and recall alongside actual errors. Whether to split by user, source, or time depends on what kind of generalization the application needs.

### A wrong prediction does not always call for a different model {#inspect-a-mistake}

Suppose “The packaging looks lovely, but the tea has gone bad” is classified as positive. First print **the tokens the model actually received**. If truncation kept only “The packaging looks lovely,” it never saw the complaint. Fixing the input comes before tuning the learning rate.

If the full sentence is present, check the class IDs. Training with `0` meaning negative but displaying it as positive can make a working classifier look broken. Once the input and label mapping agree, investigate whether the model relies too heavily on phrases such as “looks lovely.”

Two short checks could be “The tea has gone bad” and “The tea has not gone bad.” They help test sensitivity to negation, but passing both does not establish that the model handles all negation, contrast, or sarcasm. Use small examples to locate a problem and independent test data to assess how widely it occurs.

Save the tokenizer, special tokens, label mapping, and truncation length with the checkpoint. Producing valid outputs does not prove those settings agree. Fix examples and random seeds when debugging masking; use independent validation data for model comparison, record the sampling recipe, and inspect examples across lengths, classes, and sources.

The whole path is now connected: **build training targets from text → learn contextual representations → attach a task head → adapt and evaluate with task data.** Continue to [Decoder-only](decoder-only.en.md) to see what changes when the goal is to keep writing from a prefix.

Checked 2026-10-10. Numerical examples are illustrative. Code checks cover a local attention calculation, data, loss, and answer-span selection; they do not reproduce BERT pretraining or published benchmark scores.
