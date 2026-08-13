"""Train manual RNN/LSTM cells on delay-copy or seq2seq reversal tasks."""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.nn import functional as F


class RNNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.hidden_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, current_input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.input_projection(current_input) + self.hidden_projection(hidden))


class LSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, current_input: torch.Tensor,
                state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        forget, write, candidate, expose = self.gates(
            torch.cat([current_input, hidden], dim=-1)
        ).chunk(4, dim=-1)
        cell = torch.sigmoid(forget) * cell + torch.sigmoid(write) * torch.tanh(candidate)
        hidden = torch.sigmoid(expose) * torch.tanh(cell)
        return hidden, cell


class RecurrentBackbone(nn.Module):
    def __init__(self, kind: str, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.kind = kind
        self.hidden_dim = hidden_dim
        self.cell = RNNCell(input_dim, hidden_dim) if kind == "rnn" else LSTMCell(input_dim, hidden_dim)

    def zero_state(self, batch_size: int, device: torch.device):
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hidden if self.kind == "rnn" else (hidden, torch.zeros_like(hidden))

    @staticmethod
    def hidden(state):
        return state if isinstance(state, torch.Tensor) else state[0]


class DelayCopyModel(nn.Module):
    def __init__(self, kind: str, vocab_size: int, width: int = 48) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.backbone = RecurrentBackbone(kind, width, width)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        state = self.backbone.zero_state(tokens.shape[0], tokens.device)
        logits = []
        for current_input in self.embedding(tokens).unbind(dim=1):
            state = self.backbone.cell(current_input, state)
            logits.append(self.output(self.backbone.hidden(state)))
        return torch.stack(logits, dim=1)


class Seq2SeqModel(nn.Module):
    def __init__(self, kind: str, vocab_size: int, width: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.encoder = RecurrentBackbone(kind, width, width)
        self.decoder = RecurrentBackbone(kind, width, width)
        self.output = nn.Linear(width, vocab_size)

    def encode(self, source: torch.Tensor):
        state = self.encoder.zero_state(source.shape[0], source.device)
        for current_input in self.embedding(source).unbind(dim=1):
            state = self.encoder.cell(current_input, state)
        return state

    def forward(self, source: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        state = self.encode(source)
        logits = []
        for current_input in self.embedding(decoder_input).unbind(dim=1):
            state = self.decoder.cell(current_input, state)
            logits.append(self.output(self.decoder.hidden(state)))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(self, source: torch.Tensor, bos_id: int) -> torch.Tensor:
        state = self.encode(source)
        current = torch.full((source.shape[0],), bos_id, device=source.device)
        output = []
        for _ in range(source.shape[1]):
            state = self.decoder.cell(self.embedding(current), state)
            current = self.output(self.decoder.hidden(state)).argmax(dim=-1)
            output.append(current)
        return torch.stack(output, dim=1)


def train_delay(kind: str, steps: int, delay: int = 8) -> None:
    vocab_size, sequence_length, batch_size = 12, 28, 64
    model = DelayCopyModel(kind, vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(1, steps + 1):
        tokens = torch.randint(0, vocab_size, (batch_size, sequence_length))
        targets = tokens[:, :-delay]
        logits = model(tokens)[:, delay:]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            accuracy = (logits.argmax(-1) == targets).float().mean().item()
            print(f"step={step:4d} loss={loss.item():.4f} accuracy={accuracy:.3f}")


def train_reverse(kind: str, steps: int) -> None:
    vocab_size, bos_id, length, batch_size = 14, 1, 7, 64
    model = Seq2SeqModel(kind, vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3)
    for step in range(1, steps + 1):
        source = torch.randint(2, vocab_size, (batch_size, length))
        target = source.flip(dims=(1,))
        decoder_input = torch.cat([
            torch.full((batch_size, 1), bos_id), target[:, :-1]
        ], dim=1)
        logits = model(source, decoder_input)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            generated = model.generate(source[:16], bos_id)
            accuracy = (generated == target[:16]).float().mean().item()
            print(f"step={step:4d} loss={loss.item():.4f} greedy_accuracy={accuracy:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("rnn", "lstm"), default="lstm")
    parser.add_argument("--task", choices=("delay", "reverse"), default="reverse")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    torch.manual_seed(7)
    if args.task == "delay":
        train_delay(args.model, args.steps)
    else:
        train_reverse(args.model, args.steps)


if __name__ == "__main__":
    main()
