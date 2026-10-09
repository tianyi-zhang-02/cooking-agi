# BLEU, ROUGE, and edit distance: what does the score compare?

[中文](text-metrics.md) · **English**

> Last reviewed: 2026-10-08 · Prerequisite: [The evaluation stack](evaluation-stack.en.md)

The reference says “the valve is open”; the model says “the valve is not open.” Almost every word matches, but the meaning is reversed. Text metrics remain useful when we know what they count: overlapping words, order, edits, or the probability assigned to a fixed text.

## Choose the comparison first

| Metric | What it compares | Useful for | Does not directly establish |
| --- | --- | --- | --- |
| BLEU | Candidate n-grams matched in references | Corpus-level translation comparisons under fixed protocols | Factual or semantic correctness |
| ROUGE-N | Reference n-grams covered | Clues about summary coverage | Whether the content was negated or misunderstood |
| ROUGE-L | Longest common subsequence | Overlap and relative order | Complete semantic relationships |
| Edit distance / CER / WER | Substitutions, deletions, insertions | OCR and speech transcription | Whether a paraphrase is acceptable |
| Perplexity | Probability of fixed evaluation text | Language-model fit | Usefulness of freely generated answers |

The examples below use already-tokenized short sentences. `split()` is an English teaching convention, not a Chinese evaluation tokenizer.

## BLEU: repetition and short answers should not win

BLEU uses clipped counts: an n-gram cannot match more often than it occurs in the reference. Three candidate occurrences of `red` match at most one if the reference has one. With multiple references, each n-gram uses the largest count in an individual reference, not a concatenation of all references. [Original paper, §2](https://aclanthology.org/P02-1040.pdf)

For one reference, candidate length $c$, reference length $r$, and clipped precision $p_n$:

$$
\operatorname{BLEU}_N=BP\exp\left(\frac1N\sum_{n=1}^N\log p_n\right),
\qquad BP=\exp\left(\min(0,1-r/c)\right).
$$

An empty candidate is handled separately as zero. The brevity penalty discourages returning only an easy-to-match fragment.

Take reference `the bus leaves at ten` and candidate `the bus leaves`. Every candidate unigram and bigram matches, but the candidate contains only 3 words against 5. Teaching BLEU-2 is $e^{1-5/3}\approx0.5134$, not 1.

Now use `the bus leaves at nine`. Lengths match, unigram precision is 4/5, and bigram precision is 3/4. BLEU-2 is $\sqrt{0.8\times0.75}\approx0.7746$. The time is wrong yet the score is higher. If departure time is the task, exact field accuracy is the direct measure.

### A runnable small version

```python
from collections import Counter
import math

def ngram_counts(tokens, order):
    if type(order) is not int or order < 1:
        raise ValueError("order must be a positive integer")
    return Counter(tuple(tokens[start:start + order])
                   for start in range(len(tokens) - order + 1))

def bleu_example(candidate, reference, max_order=2):
    if type(max_order) is not int or max_order < 1:
        raise ValueError("max_order must be a positive integer")
    if not reference:
        raise ValueError("reference must not be empty")
    if not candidate:
        return 0.0
    precisions = []
    for order in range(1, max_order + 1):
        proposed = ngram_counts(candidate, order)
        expected = ngram_counts(reference, order)
        total = sum(proposed.values())
        matches = sum((proposed & expected).values())
        if total == 0 or matches == 0:
            return 0.0
        precisions.append(matches / total)
    penalty = math.exp(min(0.0, 1 - len(reference) / len(candidate)))
    return penalty * math.exp(sum(map(math.log, precisions)) / max_order)

reference = "the bus leaves at ten".split()
assert math.isclose(bleu_example("the bus leaves".split(), reference), math.exp(-2 / 3))
assert math.isclose(bleu_example("the bus leaves at nine".split(), reference), math.sqrt(0.6))
assert bleu_example(["red", "red", "red"], ["red", "blue"], 1) == 1 / 3
```

This implements a single-reference, single-sentence, equally weighted, unsmoothed teaching variant. Any missing order returns zero. Production evaluation must specify order, smoothing, multiple references, and short-sentence handling. Corpus BLEU aggregates counts before scoring; it is not the arithmetic mean of sentence BLEU. [SacreBLEU](https://github.com/mjpost/sacrebleu) records a signature covering tokenization, casing, smoothing, and related settings.

## ROUGE: word coverage is not meaning coverage

Single-reference ROUGE-N recall divides matching n-grams by reference n-grams. Precision and F1 can also be reported; “ROUGE improved” does not identify which variant was used. [Original paper](https://aclanthology.org/W04-1013.pdf)

For reference `the valve is open` and candidate `the valve is not open`, ROUGE-1 recall is 1, precision is 4/5, and F1 is 8/9. One negation reverses the conclusion while word coverage remains high.

ROUGE-L uses the longest common subsequence (LCS). Matches need not be adjacent but must retain relative order. `we ship today` and `today we ship` have LCS length 2, not 3. This adds order beyond a bag of words, without understanding negation. Sentence-level ROUGE-L and summary-level `rougeLsum` also require separate treatment.

### Compute subsequences and edits together

Both algorithms below use prefix dynamic programming with $O(mn)$ time and $O(n)$ rolling-array memory, where $n$ is the second sequence's length. LCS maximizes retained order; edit distance minimizes modification cost.

```python
def lcs_length(left, right):
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for column, right_token in enumerate(right, start=1):
            current.append(previous[column - 1] + 1 if left_token == right_token
                           else max(previous[column], current[-1]))
        previous = current
    return previous[-1]

def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for row, left_token in enumerate(left, start=1):
        current = [row]
        for column, right_token in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (left_token != right_token)))
        previous = current
    return previous[-1]

assert lcs_length("we ship today".split(), "today we ship".split()) == 2
assert edit_distance("we ship today".split(), "today we ship".split()) == 2
assert edit_distance("1280", "1230") == 1
assert lcs_length([], ["word"]) == 0
assert edit_distance([], ["one", "two"]) == 2
```

Substitution, insertion, and deletion each cost 1 here. Transcribing `1280` as `1230` makes one character error, giving CER 1/4, while exact amount-field accuracy is zero. They answer different questions.

WER normally divides by reference words; CER by reference characters. Many insertions can produce an error rate above 100%. An empty reference needs an explicit policy, not a silently added epsilon. For combining marks or emoji, specify whether “character” means a Unicode code point or a user-perceived grapheme.

## Perplexity evaluates a fixed text, not the model's own essay

For a fixed test sequence, average the negative log-probabilities of valid target tokens and exponentiate:

$$
\operatorname{PPL}=\exp\left(-\frac1T\sum_{t=1}^T\log p(x_t\mid x_{<t})\right).
$$

If two targets receive probabilities 0.5 and 0.25, PPL is $\sqrt8\approx2.828$. This is not accuracy of 1/2.828, nor a literal claim of 2.828 equally likely choices at each position.

Fix data, tokenizer, context window, and valid targets. Different tokenizers change the counting unit, so their values are not directly comparable. Sliding windows provide more context, but overlapping targets must not be counted repeatedly. Aggregate batch losses by valid token count, not an unweighted average across unequal lengths. [Implementation guidance](https://huggingface.co/docs/transformers/perplexity)

## Are these metrics still useful in 2026?

Yes, but not as universal model-quality scores. New models do not invalidate the definitions; tasks, evaluation data, and required evidence change.

| Actual task | Checks worth retaining | What else is needed |
| --- | --- | --- |
| OCR or speech transcription | CER / WER | Key fields, missing lines, speakers, timestamps |
| Translation | Protocol-controlled BLEU / chrF | Meaning, named entities, human review |
| Summarization | ROUGE and length | Factual consistency, omissions, source support |
| RAG answers | Exact match / field checks | Retrieved evidence, attribution, unanswerability |
| Agents | Final text as an auxiliary signal | Actual state changes, execution, side effects |

Semantic metrics such as [BERTScore](https://arxiv.org/abs/1904.09675) compare contextual representations; [COMET](https://aclanthology.org/2020.emnlp-main.213/) learns translation-quality assessment. They capture more than lexical overlap but still depend on the model, training data, and domain. They do not automatically replace fact checking. LLM judges also require [calibration](llm-as-a-judge/calibration.en.md).

Keep dataset version, tokenization and normalization, reference count, implementation version, aggregation, and failure examples together. Small differences need paired uncertainty estimates rather than conclusions based on an extra decimal place. Continue to [metric robustness](metric-robustness.en.md).
