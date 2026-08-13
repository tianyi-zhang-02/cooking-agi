"""RNN, LSTM, and attention forward passes without PyTorch or autograd."""

from __future__ import annotations

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = value - value.max(axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=axis, keepdims=True)


def rnn_forward(inputs: np.ndarray, input_weight: np.ndarray,
                hidden_weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """inputs: (T, input_dim), returns all hidden states: (T, hidden_dim)."""
    hidden = np.zeros(hidden_weight.shape[0])
    states = []
    for current_input in inputs:
        hidden = np.tanh(input_weight @ current_input + hidden_weight @ hidden + bias)
        states.append(hidden.copy())
    return np.stack(states)


def lstm_forward(inputs: np.ndarray, weight: np.ndarray,
                 bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-matrix LSTM. weight: (4 * hidden_dim, input_dim + hidden_dim)."""
    hidden_dim = weight.shape[0] // 4
    hidden = np.zeros(hidden_dim)
    cell = np.zeros(hidden_dim)
    hidden_states, cell_states = [], []

    for current_input in inputs:
        gates = weight @ np.concatenate([current_input, hidden]) + bias
        forget, write, candidate, expose = np.split(gates, 4)
        forget = sigmoid(forget)
        write = sigmoid(write)
        candidate = np.tanh(candidate)
        expose = sigmoid(expose)
        cell = forget * cell + write * candidate
        hidden = expose * np.tanh(cell)
        hidden_states.append(hidden.copy())
        cell_states.append(cell.copy())
    return np.stack(hidden_states), np.stack(cell_states)


def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray,
                                 value: np.ndarray, causal: bool = False
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """query/key/value: (T, d). Returns output and attention weights."""
    scores = query @ key.T / np.sqrt(query.shape[-1])
    if causal:
        future = np.triu(np.ones_like(scores, dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)
    weights = softmax(scores)
    return weights @ value, weights


def demo() -> None:
    rng = np.random.default_rng(7)
    sequence_length, input_dim, hidden_dim = 5, 3, 4
    inputs = rng.normal(size=(sequence_length, input_dim))

    rnn_states = rnn_forward(
        inputs,
        rng.normal(scale=0.3, size=(hidden_dim, input_dim)),
        rng.normal(scale=0.3, size=(hidden_dim, hidden_dim)),
        np.zeros(hidden_dim),
    )
    lstm_states, cell_states = lstm_forward(
        inputs,
        rng.normal(scale=0.3, size=(4 * hidden_dim, input_dim + hidden_dim)),
        np.zeros(4 * hidden_dim),
    )
    attention_output, attention_weights = scaled_dot_product_attention(
        rnn_states, rnn_states, rnn_states, causal=True
    )

    print("inputs:          ", inputs.shape)
    print("RNN states:      ", rnn_states.shape)
    print("LSTM h / c:      ", lstm_states.shape, cell_states.shape)
    print("attention output:", attention_output.shape)
    print("causal weights:\n", np.round(attention_weights, 3))

    assert np.allclose(attention_weights.sum(axis=-1), 1.0)
    assert np.allclose(np.triu(attention_weights, k=1), 0.0)


if __name__ == "__main__":
    demo()
