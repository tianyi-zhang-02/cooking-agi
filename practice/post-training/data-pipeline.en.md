# From one demonstration to a training batch

[中文](data-pipeline.md) · **English**

> Original teaching project · Checked: 2026-10-09. Synthetic examples; the validator checks structure and arithmetic, not semantic correctness.

A `messages` column can be enough to start SFT. But if the model later keeps citing outdated evidence, conversation text alone is a poor audit trail.

Retain both **traceable source records** and **the samples actually consumed by the trainer**. The former supports review and rebuilding; the latter exposes template and loss behavior.

## Keep provenance in the source record

This is one JSONL record, expanded for readability. The project handles single-turn text questions only. Multi-turn conversations, tool calls, and images need an extended schema and validator.

```json
{
  "sample_id": "demo-001",
  "group_id": "registration-deadline",
  "split": "train",
  "question": "When does registration close?",
  "evidence": [
    {"id": "policy-7", "revision": "v2", "text": "Registration closes on Wednesday."}
  ],
  "answer": "Registration closes on Wednesday [policy-7].",
  "citations": ["policy-7"],
  "answerable": true,
  "label_source": "synthetic-demonstration",
  "review_status": "approved"
}
```

`revision` identifies the evidence used; `label_source` separates human and generated demonstrations; `group_id` keeps paraphrases together. `approved` records a workflow decision, not proof of correctness. Review whether the cited evidence actually supports the answer.

For unanswerable examples, explicitly set `answerable: false`, provide an appropriate abstention demonstration, and leave citations empty. Neither a blank answer nor a randomly chosen citation is a substitute.

## Split before producing training views

Deduplicate, establish groups, and split train/dev/test first. Apply paraphrasing, templates, and tokenization separately afterward. Randomly splitting rows after generating near-identical paraphrases makes evaluation artificially easy.

Question grouping does not solve every leakage problem. New-document generalization requires document or document-family separation; future-data evaluation needs a temporal boundary. Choose according to the claim being tested. Content hashes catch exact duplicates, not all semantic near-duplicates.

| Artifact | Retained information | What it lets us inspect |
| --- | --- | --- |
| Reviewed records | Question, evidence revision, answer, provenance, group | Why a demonstration entered training |
| Training view | `messages` with a separate sample-ID mapping | How instructions and evidence were assembled |
| Tokenized cache | Tokens, target mask, lengths, preprocessing fingerprint | Why particular positions receive supervision |
| Run manifest | Snapshot, template, tokenizer revision, configuration | Whether dependency changes altered the experiment |

The training view can contain a `system` instruction, a `user` message with question and evidence, and an `assistant` demonstration. Keep provenance in a sidecar rather than blindly copying it all into context. Delimit evidence and treat it as untrusted material that cannot override system instructions.

Templates are more than arbitrary role names. The [Transformers chat-template guide](https://huggingface.co/docs/transformers/chat_templating) explains model-specific control tokens. Training uses complete demonstrations rather than appending another generation prefix; when formatting to a string before tokenizing, avoid duplicating special tokens.

## Inspect labels, not only the JSON

Consider toy tokens `[11, 12, 21, 22, 2]`: two context tokens, two answer tokens, and end token `2`. The target mask is `[0, 0, 1, 1, 1]`.

```text
input_ids      11    12    21    22     2
labels       -100  -100    21    22     2
attention       1     1     1     1     1
```

A common causal-LM loss interface aligns `logits[:, :-1]` with `labels[:, 1:]` internally. The output at position 1 therefore predicts token 21; it is not predicting itself after seeing the answer. A custom loss must establish who performs this shift, rather than applying it twice.

Excluding user tokens as targets does not hide the context. The attention mask controls visibility or valid positions; `labels=-100` is this example's ignored-loss convention. See [Data and objectives](data-and-objectives.en.md) for the arithmetic.

One subtle case: pad and EOS can share a token ID. Masking every occurrence of `pad_id` then removes real end-token supervision too. Our teaching collator uses **actual lengths** to identify newly padded positions and preserves genuine EOS targets.

## Length budgets and packing: what are we saving?

Suppose input uses 1,600 tokens and the answer uses 500, with a 2,048-token limit. At least 52 tokens must go. Keeping the first 2,048 blindly cuts the answer tail—possibly its citation and end marker. Decide which evidence or history can be shortened, rebuild the sample, and recheck support. Reject or separately bucket examples that cannot retain valid targets; do not truncate silently.

| Choice | Benefit | Cost and checks |
| --- | --- | --- |
| Per-batch padding | Clear boundaries and easy debugging | Length imbalance wastes computation; length bucketing changes sample order |
| Pack multiple examples | Less padding and more useful token computation | Verify attention isolation, position IDs, and labels at example boundaries |
| Token-budget batches | More consistent computational load | Example counts vary; revisit denominators and accumulation |

Two sequences of lengths 3 and 5 require 10 padded slots, with 8 real tokens: 80% utilization. Packing can use 8 slots, but independent-example semantics still matter. The second example must not attend to the first, and the first example's final output must not be trained to predict the next example's first token.

Resetting position IDs alone does not guarantee isolation in every attention backend. TRL packing depends on strategy and backend; verify the pinned combination. The code below **implements inspectable padding, not an efficient packing kernel**.

## Run it, then break the inputs

The [standard-library implementation](code/training_contracts.py) includes record validation and a teaching collator. From the repository root:

```bash
python3 practice/post-training/code/training_contracts.py
python3 -m unittest discover -s site/tests -p 'test_training_contracts.py'
```

Try an unknown citation ID, an all-zero target mask, or a pad ID equal to EOS. The first two should fail. The last should preserve genuine EOS targets and ignore only added padding. Tests also reject groups spanning multiple dataset splits.

These checks cannot detect a misread date in an existing document. That still requires content review, counterexamples, and fixed evaluation. Schema validity is not a data-quality verdict.

Next, [calculate the loss](data-and-objectives.en.md), then ask [whether multi-GPU training preserves it](distributed-training.en.md).
