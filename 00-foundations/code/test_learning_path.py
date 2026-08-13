"""Fast shape and invariant checks for the learning-path examples."""

from __future__ import annotations

import numpy as np
import torch

from sequence_numpy import lstm_forward, rnn_forward, scaled_dot_product_attention
from sequence_torch import DelayCopyModel, Seq2SeqModel
from tokenizer_from_scratch import TinyBPE


def test_tokenizer_round_trip() -> None:
    corpus = ["tokens become ids", "tokens become vectors", "vectors become states"]
    tokenizer = TinyBPE()
    tokenizer.train(corpus, num_merges=12)
    text = "tokens become vectors"
    assert tokenizer.decode_tokens(tokenizer.tokenize(text)) == text
    assert len(tokenizer.encode(text)) > 0


def test_numpy_shapes_and_mask() -> None:
    rng = np.random.default_rng(3)
    inputs = rng.normal(size=(6, 4))
    rnn = rnn_forward(inputs, rng.normal(size=(5, 4)), rng.normal(size=(5, 5)), np.zeros(5))
    lstm, cell = lstm_forward(inputs, rng.normal(size=(20, 9)), np.zeros(20))
    output, weights = scaled_dot_product_attention(rnn, rnn, rnn, causal=True)
    assert rnn.shape == (6, 5)
    assert lstm.shape == cell.shape == (6, 5)
    assert output.shape == (6, 5)
    assert np.allclose(weights.sum(-1), 1.0)
    assert np.allclose(np.triu(weights, 1), 0.0)


def test_torch_shapes() -> None:
    tokens = torch.randint(0, 11, (3, 8))
    for kind in ("rnn", "lstm"):
        delay_model = DelayCopyModel(kind, vocab_size=11, width=16)
        assert delay_model(tokens).shape == (3, 8, 11)

        seq2seq = Seq2SeqModel(kind, vocab_size=11, width=16)
        assert seq2seq(tokens, tokens).shape == (3, 8, 11)
        assert seq2seq.generate(tokens, bos_id=1).shape == (3, 8)


if __name__ == "__main__":
    test_tokenizer_round_trip()
    test_numpy_shapes_and_mask()
    test_torch_shapes()
    print("learning-path checks passed")
