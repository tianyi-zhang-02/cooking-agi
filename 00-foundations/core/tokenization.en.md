# Tokenization: text to IDs

[中文](tokenization.md) · **English**

> Reading time: ~12 min · Level: core · Last reviewed: 2026-08

The same sentence can become different ID sequences under different tokenizers. If a toy tokenizer splits `playing` into `play` and `ing`, the model receives two IDs, not one English word. That choice affects sequence length, compute, and how unfamiliar words are represented. We will follow the text through tokenization and the embedding lookup.

## The model receives token IDs {#the-model-receives-token-ids}

We say all the time that a model “reads” a sentence, but it has never actually seen text. A tokenizer first splits the string into tokens from a finite vocabulary and then maps them to integer IDs; from start to finish, those numbers are all the model really receives.

## Step 1: how text is split into tokens {#step-1-how-text-is-split-into-tokens}

```text
"unbelievable!" → ["un", "believ", "able", "!"] → [431, 9821, 612, 5]
```

The model uses an embedding table to look each ID up as a vector:

$$x_t = E[\text{token\_id}_t], \qquad E \in \mathbb{R}^{|V| \times d}$$

$|V|$ is the vocabulary size and $d$ is the model dimension. The tokenizer decides how long the sequence is; the embedding decides which continuous vector each discrete symbol starts from.

## Step 2: why not just split on words {#step-2-why-not-just-split-on-words}

The intuition is reasonable, but a real-world word list has no end: new names, spelling variants, code, emoji, and other languages cannot all be listed. Making every character a token can produce longer sequences; characters absent from the base alphabet can still be unknown inputs.

A subword tokenizer is the compromise between the two: frequent fragments stay whole, and rare words are split into smaller units.

| Granularity | Advantage | Cost |
| --- | --- | --- |
| word | short sequences, intuitive semantics | vocabulary explosion, severe unknown-word problem |
| character | combines known characters without listing every word | unseen characters still need handling; sequences can be longer |
| byte | a complete byte alphabet can represent any UTF-8 text | a base unit is not necessarily a character; sequences can be longer |
| subword | vocabulary size and sequence length reasonably balanced | segmentation depends on the corpus; boundaries do not always match human intuition |

## Step 3: how BPE merges common fragments {#step-3-how-bpe-merges-common-fragments}

Byte Pair Encoding knows nothing about word roots and nothing about grammar. It just keeps asking: **which two adjacent symbols appear together most often?** Then it glues them together.

```text
l o w </w>
l o w e r </w>

most frequent pair: (l, o)  →  lo
next round:         (lo, w) →  low
```

Training learns an **ordered list of merge rules**; encoding applies them in the same order. BPE is not looking up English word roots; it is compressing the string patterns that recur in the current corpus.

For two worked merges and the differences between BPE, WordPiece, Unigram, and byte fallback, continue to [tokenizer algorithms](../deep-dives/tokenizer-algorithms.en.md). This page first completes the input pipeline.

<details markdown="1">
<summary><b>deeper</b>: why token boundaries affect model behavior</summary>

If the same concept is 1 token in one language and 5 tokens in another, the latter takes up more context, more attention compute, and more prediction steps. How digits, whitespace, and code are split also changes the difficulty of arithmetic and copying tasks.

So the tokenizer is not inconsequential preprocessing; it defines the basic unit of the model's prediction problem.

</details>

## Four objects that are easy to confuse {#four-objects-that-are-easy-to-confuse}

1. **Vocabulary**: the static mapping between tokens and IDs.
2. **Merge rules / model**: how raw symbols are combined into tokens.
3. **Normalizer / pre-tokenizer**: how case, Unicode, whitespace, and the like are handled before text reaches the model.
4. **Special tokens**: BOS, EOS, PAD, and the role boundaries in chat templates.

`decode(encode(text))` should normally recover the text, but normalization can make the round trip not byte-for-byte reversible. PAD exists only to align sequences within a batch and should not be treated as content by the model; EOS is the real signal that generation should stop.

## How a conversation becomes a string of IDs the model can read {#how-a-conversation-becomes-a-string-of-ids-the-model-can-read}

### 1. Chat templates serialize role structure {#1-chat-templates-serialize-role-structure}

An application may represent a conversation as

```json
[
  {"role": "system", "content": "You are a helpful assistant"},
  {"role": "user", "content": "How are you?"}
]
```

The Transformer does not receive that dictionary directly. A chat template first
serializes the messages into the format seen during training. A ChatML-style rendering
might be

```text
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
```

At inference time the final assistant marker has no answer after it; generation begins
there. Other checkpoints may use `[INST]...[/INST]`, header tokens, or entirely
different formats. The **chat template must match the checkpoint and tokenizer** rather
than being copied across model families.

### 2. Special tokens are still vocabulary symbols {#2-special-tokens-are-still-vocabulary-symbols}

To the model, system, user, assistant, and message boundaries eventually become token
IDs. If `<|im_start|>` is registered as a special token, it normally maps to one ID;
otherwise it may be split into ordinary pieces. Whether a role name occupies one token
or several also depends on the vocabulary.

The architecture contains no hard-coded rule that says “system has highest priority.”
The model learns the usual effect of these symbols from repeated combinations of role
markers, message content, and target responses in its training data.

Do not copy numeric IDs from a screenshot. Special-token strings, IDs, stop sets, and
embeddings are part of a specific tokenizer/checkpoint protocol and must be loaded again
when the model changes.

### 3. The full boundary: messages → template → IDs → embeddings {#3-the-full-boundary-messages-template-ids-embeddings}

The complete input pipeline is

$$
\boxed{
\text{structured messages}
\xrightarrow{\text{chat template}}
\text{serialized token text}
\xrightarrow{\text{tokenizer}}
\text{token IDs}
\xrightarrow{\text{embedding table}}
\text{vectors}
}
$$

Libraries often wrap template rendering and tokenization in one function, but their
conceptual responsibilities remain distinct. The template determines the ordering of
roles and message boundaries; the tokenizer selects vocabulary units; the embedding
table turns integers into model-width vectors.

A chat template does not alter the attention equation. It gives an ordinary token
sequence a learnable grammatical structure.

## Connecting the output shapes to the neural network {#connecting-the-output-shapes-to-the-neural-network}

Suppose a batch holds 4 texts, the padded length is 12, and the model dimension is 768:

```text
token_ids       (B, T)    = (4, 12)
attention_mask  (B, T)    = (4, 12)
embeddings      (B, T, d) = (4, 12, 768)
```

The tokenizer ends at `(B, T)`; the neural network begins at `(B, T, d)`.

## Experiment: compare different tokenizations {#experiment-compare-different-tokenizations}

Run [`../code/tokenizer_from_scratch.py`](../code/tokenizer_from_scratch.py). It trains a miniature BPE using only the Python standard library, with no extra dependencies. Change the training corpus and you can watch directly why the same sentence gets split into completely different tokens under another corpus.

## Self-check {#self-check}

<div class="taste-check">
  <strong>Before closing the page, try answering without jargon:</strong>
  <ol>
    <li>Why does a new name not simply break a subword tokenizer?</li>
    <li>Why can the same sentence cost a different amount of inference when the tokenizer changes?</li>
    <li>At which step does <code>(B, T)</code> become <code>(B, T, d)</code>?</li>
  </ol>
</div>

## Where to read next {#where-to-read-next}

Text has finally become vectors, but the positions still know nothing about each other. The next page is [RNN and LSTM](recurrent-models.en.md): if you can only read from left to right, where exactly should the past be stored?

## Quick learning: what a tokenizer does at the model boundary {#quick-learning-what-a-tokenizer-does-at-the-model-boundary}

<details class="interview" markdown="1">
<summary>The standard text-to-IDs answer and one key misconception</summary>

**Quick memory**: a tokenizer segments text into vocabulary pieces and maps them to integers; a chat template first serializes the role structure, and special tokens are just special IDs in the vocabulary that carry boundary semantics.

**Interview answer**

> The complete path is: messages pass through a chat template and become text with role boundaries, the tokenizer turns that text into token IDs, and embedding lookup finally turns the IDs into continuous vectors. The Transformer never sees strings directly, and system, user, and assistant are not hard-coded anywhere in the architecture.

<details markdown="1">
<summary><b>Deep dive</b>: why can tokenizers and chat templates not be swapped freely across models?</summary>

Whether a special string is a single token, which ID it maps to, and how the start and end of an assistant turn are written are all part of the model's training distribution. A template that does not match the vocabulary splits structural markers into pieces or maps them to the wrong IDs; even when the tensor shapes are fine, the protocol the model sees has already changed.

</details>
</details>
