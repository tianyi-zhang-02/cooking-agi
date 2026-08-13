"""A tiny, readable BPE tokenizer using only the Python standard library."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


def merge_pair(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            merged.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged


@dataclass
class TinyBPE:
    merge_rules: list[tuple[str, str]] = field(default_factory=list)
    token_to_id: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _words(text: str) -> list[list[str]]:
        return [["▁", *word] for word in text.strip().split()]

    def train(self, corpus: list[str], num_merges: int = 24) -> None:
        sequences = [word for text in corpus for word in self._words(text)]
        self.merge_rules.clear()

        for _ in range(num_merges):
            counts = Counter(
                (sequence[index], sequence[index + 1])
                for sequence in sequences
                for index in range(len(sequence) - 1)
            )
            if not counts:
                break
            best_pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < 2:
                break
            self.merge_rules.append(best_pair)
            sequences = [merge_pair(sequence, best_pair) for sequence in sequences]

        vocabulary = {"<unk>"}
        for sequence in sequences:
            vocabulary.update(sequence)
        self.token_to_id = {token: index for index, token in enumerate(sorted(vocabulary))}

    def tokenize(self, text: str) -> list[str]:
        output: list[str] = []
        for sequence in self._words(text):
            for pair in self.merge_rules:
                sequence = merge_pair(sequence, pair)
            output.extend(sequence)
        return output

    def encode(self, text: str) -> list[int]:
        unknown_id = self.token_to_id["<unk>"]
        return [self.token_to_id.get(token, unknown_id) for token in self.tokenize(text)]

    def decode_tokens(self, tokens: list[str]) -> str:
        return "".join(tokens).replace("▁", " ").strip()


def demo() -> None:
    corpus = [
        "small models learn patterns",
        "small tokenizers learn repeated pieces",
        "large models still start with tokens",
        "tokens turn text into model inputs",
    ]
    tokenizer = TinyBPE()
    tokenizer.train(corpus, num_merges=20)

    text = "small models learn tokens"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    restored = tokenizer.decode_tokens(tokens)

    print("merge rules:", tokenizer.merge_rules[:10])
    print("text:       ", text)
    print("tokens:     ", tokens)
    print("ids:        ", ids)
    print("decoded:    ", restored)
    assert restored == text


if __name__ == "__main__":
    demo()
