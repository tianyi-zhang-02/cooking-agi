# Tokenizer algorithms: what does BPE actually learn?

[中文](tokenizer-algorithms.md) · **English**

> Last reviewed: 2026-10 · Prerequisite: [Text to IDs](../core/tokenization.en.md)

A model can usually accept an unfamiliar name without reporting “word not found.” The tokenizer has not understood the name; it has broken a new string into existing pieces. Whether those pieces are useful or excessively long is a separate question.

We will work through BPE, compare WordPiece and Unigram, and check why a small tokenizer change can break a model. All vocabularies and numbers below are teaching examples, not outputs from a particular commercial model.

## Separate three decisions

| Name you encounter | Question it answers | What it does not imply |
| --- | --- | --- |
| Characters and bytes | What are the starting symbols? | Every final token is one starting symbol |
| BPE, WordPiece, Unigram | How do we learn a vocabulary and segment text? | A neural-network architecture |
| SentencePiece, Tokenizers, and similar tools | How is the processing implemented and saved? | One tool name identifies exactly one algorithm |

Byte-level BPE starts from bytes but can merge several into one token. SentencePiece supports both BPE and Unigram; a `.model` extension alone does not identify which one. The [SentencePiece documentation](https://github.com/google/sentencepiece) describes supported algorithms and normalization.

## Work through two BPE merges

Ignore spaces, word-end markers, and punctuation for now. Do not merge across strings:

| String | Corpus frequency | Initial segmentation |
| --- | ---: | --- |
| `rain` | 3 | `r a i n` |
| `rail` | 2 | `r a i l` |

Count **occurrences**, not distinct strings. Both `(r, a)` and `(a, i)` occur 5 times; `(i, n)` occurs 3 times and `(i, l)` twice. Ties need a deterministic rule. For this example, choose `(r, a)` first.

| Round | Merge | `rain` becomes | `rail` becomes |
| --- | --- | --- | --- |
| 0 | None | `r a i n` | `r a i l` |
| 1 | `r + a → ra` | `ra i n` | `ra i l` |
| 2 | `ra + i → rai` | `rai n` | `rai l` |

Recount adjacent symbols after each merge; the original character-pair counts are no longer enough. Applying the learned rules to the new string `rair` gives `rai r`. We do not need a vocabulary entry for the entire new word.

One implementation detail matters: **merging does not delete the base alphabet.** Keep `r`, `a`, and intermediate pieces such as `ra` in the vocabulary. Otherwise a new combination may tokenize successfully but have no matching IDs. Try the site's [TinyBPE](../code/tokenizer_from_scratch.py):

```python
from tokenizer_from_scratch import TinyBPE

tokenizer = TinyBPE()
tokenizer.train(["rain rain rain rail rail"], num_merges=4)
pieces = tokenizer.tokenize("rair")
ids = tokenizer.encode("rair")
assert tokenizer.token_to_id["<unk>"] not in ids
assert tokenizer.decode_tokens(pieces) == "rair"
print(pieces, ids)
```

Run from `00-foundations/code/`. This implementation adds a word-start marker `▁`, so its exact merge order need not match the table. It also collapses repeated whitespace: it is not a production, byte-reversible tokenizer. See [Sennrich et al.](https://arxiv.org/abs/1508.07909) for the original subword BPE method.

## How can a byte-level tokenizer have long tokens?

```text
Character 中
    ↓ UTF-8 encoding
E4 B8 AD                  3 bytes
    ↓ Learned merge rules
[E4] [B8] [AD]            Possibly still separate
[E4 B8 AD]                Possibly merged into 1 token
```

These are two possible vocabularies, not a random choice on every call. An ordinary deterministic encoder with fixed configuration should give a fixed result.

With all 256 base bytes retained and no lossy preprocessing, any UTF-8 string can be represented without a new vocabulary entry for each character. An individual token may contain bytes that do not form a valid character by themselves. Reassemble the complete byte sequence before decoding it.

Character-based BPE has no automatic guarantee of this coverage. A character missing from its alphabet may still require `<unk>` or byte fallback. **Representing input is not the same as understanding it**: rare text can be fragmented into pieces the model handles poorly.

## WordPiece does not replay merges at encoding time

WordPiece commonly uses longest matching: take the longest valid vocabulary piece from the current position, then process the remainder. With a hypothetical vocabulary:

```text
Available pieces: play, ##ing, ##er
playing → play + ##ing
player  → play + ##er
```

Here `##` marks a continuation, not literal hash characters in the input. A BERT-style implementation may replace an entire word with `[UNK]` when it cannot fully segment it; check the implementation.

Keep WordPiece **training** separate from **encoding**. Tutorials often use `freq(a,b)/(freq(a)freq(b))` to explain merge candidates, but it is not a universal specification for every WordPiece trainer. The [Hugging Face teaching implementation](https://huggingface.co/learn/llm-course/chapter6/6) explicitly calls its training procedure a reconstruction from published material, since the original implementation was not fully released. Longest matching at encoding time, versus BPE's ranked merges, is the more useful distinction to remember.

## Unigram scores complete segmentations

Unigram assigns probabilities to pieces. In a simplified model, a segmentation's probability is the product of its piece probabilities, or the sum of their log probabilities.

Suppose the vocabulary has probabilities `a: 0.3`, `b: 0.2`, and `ab: 0.5`. For `ab`:

| Segmentation | Score |
| --- | ---: |
| `ab` | 0.5 |
| `a b` | 0.3 × 0.2 = 0.06 |

The best segmentation is `ab`. But marginalizing over the valid segmentations for this training-likelihood calculation gives `0.5 + 0.06 = 0.56`, not just the maximum `0.5`. **Finding the best segmentation** and **summing over latent segmentations** are different operations.

Dynamic programming finds the best segmentation of a longer string without enumerating every combination. Training typically starts from a larger candidate vocabulary, estimates probabilities, and prunes less useful pieces—the opposite direction from adding BPE merges. The [Unigram / Subword Regularization paper](https://arxiv.org/abs/1804.10959) also considers sampling segmentations during training rather than depending on one fixed boundary pattern.

## Compare the mechanisms, not just the names

| Method | Main stored objects | Deterministic encoding | Common misunderstanding |
| --- | --- | --- | --- |
| BPE | Vocabulary and ranked merges | Apply merges to base symbols | Frequent pieces need not be morphemes |
| WordPiece | Vocabulary and continuation conventions | Greedy longest valid match | Training scores are not the encoding procedure |
| Unigram | Pieces and their scores | Highest-scoring segmentation via dynamic programming | One string can have several valid segmentations |

Newer is not automatically better. Language coverage, corpus, vocabulary size, and boundary handling all matter. Compare token counts, unknown inputs, reversibility, and downstream behavior on the same text—not vocabulary size alone.

## Why not minimize token count at any cost?

Suppose tokenizer A produces 200 tokens and B produces 300 for the same text. A naive full-attention score matrix per head grows from 40,000 to 90,000 entries, a factor of 2.25. **End-to-end inference need not become 2.25 times slower**: projections, FFNs, batching, and the actual attention kernel also matter.

A larger vocabulary has a cost too. At embedding width 4096, adding 32,000 entries adds 131,072,000 parameters. An untied output head may add the same amount again. Saving prediction steps must be weighed against a larger table.

Per-token perplexity is not directly comparable across different tokenizers either: the prediction unit changed. Fix the tokenizer, or explicitly normalize to a common unit such as bytes over identical text, reporting preprocessing and normalization.

## Checks before connecting a tokenizer to a model

- **Reversibility:** test Chinese, emoji, repeated spaces, and newlines. Know what normalization changes.
- **ID alignment:** equal vocabulary sizes do not imply equal ID assignments. Row 42 of the old embedding does not automatically acquire the meaning of new ID 42.
- **Special tokens:** if the template already adds BOS/EOS, does encoding add them again?
- **Training labels:** which tokens contribute to loss? Padding, causal, and loss masks have different jobs.
- **Budgets:** measure length distributions on your own text rather than applying an English character heuristic to every language.

Continue to [one training step](training-step.en.md) to connect IDs to labels and loss, or [KV cache and inference cost](kv-cache-and-inference.en.md) to work out the memory cost of longer sequences.
