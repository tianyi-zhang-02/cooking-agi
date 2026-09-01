# Tokenization: text to IDs

[中文](tokenization.md) · **English**

> Reading time: ~7 min · Level: core · Last reviewed: 2026-08

<div class="lesson-recipe">
  <div><span>What we are making</span><strong>A finite symbol system for open-ended strings</strong></div>
  <div><span>Prerequisites</span><strong>raw text · vocabulary · merge rules</strong></div>
  <div><span>Output</span><strong>token IDs · attention mask · embeddings</strong></div>
  <div><span>Most common mistake</span><strong>Treating tokenization as neutral preprocessing</strong></div>
</div>

## Quick learning: what does a tokenizer do at the model boundary?

<details class="interview" markdown="1">
<summary>The standard text-to-IDs answer and one critical misconception</summary>

**Quick memory**: a tokenizer segments text into vocabulary pieces and maps them to integers. A chat template first serializes role structure; special tokens are vocabulary IDs carrying boundary semantics.

**Interview answer**

> The complete path is messages to a chat template, then token IDs, then continuous vectors through embedding lookup. A Transformer never sees raw strings, and system, user, and assistant roles are not hard-coded in its architecture.

<details markdown="1">
<summary><b>Deep dive</b>: why can tokenizers and chat templates not be swapped arbitrarily?</summary>

Whether a boundary string is one token, which ID it receives, and how assistant turns begin and end are all part of the model's training distribution. A mismatched template may split markers or map them incorrectly; tensor shapes still work while the model-level protocol is broken.

</details>
</details>

## Models receive token IDs, not text

A tokenizer splits a string into tokens from a finite vocabulary and maps them to integer IDs. The model never sees “text”; it only sees those IDs.

```text
"unbelievable!" → ["un", "believ", "able", "!"] → [431, 9821, 612, 5]
```

An embedding table turns each ID into a vector:

$$x_t=E[\text{token\_id}_t], \qquad E\in\mathbb{R}^{|V|\times d}$$

## Why not tokenize by whole words

A word vocabulary is open-ended: names, spelling variants, code, emoji, and languages never stop arriving. Characters or bytes avoid unknown inputs but produce long sequences. Subwords keep frequent fragments whole and split rare strings into smaller units.

| Unit | Advantage | Cost |
| --- | --- | --- |
| word | short and intuitive | exploding vocabulary, unknown words |
| character / byte | almost no unknown input | longer sequences |
| subword | balanced vocabulary and length | corpus-dependent, unintuitive boundaries |

## The BPE operation

Byte Pair Encoding repeatedly merges the most frequent adjacent symbol pair in the training corpus. Training produces an **ordered list of merge rules**; encoding applies those rules in order. BPE is compression over recurring string patterns, not linguistic morphology.

## Four objects that are easy to confuse

1. **Vocabulary:** the static token-to-ID map.
2. **Merge model:** how primitive symbols become tokens.
3. **Normalizer / pre-tokenizer:** Unicode, case, and whitespace handling.
4. **Special tokens:** BOS, EOS, PAD, and role boundaries.

For a batch of 4 sequences padded to length 12 with model width 768:

```text
token_ids       (B, T)    = (4, 12)
attention_mask  (B, T)    = (4, 12)
embeddings      (B, T, d) = (4, 12, 768)
```

<details markdown="1">
<summary><b>Deep dive</b>: token boundaries change behavior</summary>

If the same concept takes one token in one language and five in another, the latter consumes more context, more attention compute, and more prediction steps. Tokenization defines the unit of the modeling problem; it is not neutral preprocessing.

</details>

Run [`../code/tokenizer_from_scratch.py`](../code/tokenizer_from_scratch.py), a standard-library-only miniature BPE trainer.

## Self-check

<div class="taste-check">
  <strong>Before moving on, explain:</strong>
  <ol>
    <li>Why does a new name not immediately break a subword tokenizer?</li>
    <li>Why can changing the tokenizer change inference cost?</li>
    <li>Where does <code>(B, T)</code> become <code>(B, T, d)</code>?</li>
  </ol>
</div>

Then continue to [RNN and LSTM](recurrent-models.en.md).
