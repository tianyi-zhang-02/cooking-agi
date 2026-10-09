from collections import Counter
import math


def bm25_term(frequency, length, average_length, inverse_frequency, k1, length_weight):
    if frequency == 0:
        return 0.0
    length_factor = 1 - length_weight + length_weight * length / average_length
    return inverse_frequency * frequency * (k1 + 1) / (frequency + k1 * length_factor)


def _tokens(values):
    if isinstance(values, (str, bytes)):
        raise TypeError("Pass token sequences, not raw text")
    tokens = tuple(values)
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError("Tokens must be non-empty strings")
    return tokens


class LexicalIndex:
    def __init__(self, documents):
        if isinstance(documents, (str, bytes)):
            raise TypeError("Pass a sequence of tokenized documents")
        self.counts = [Counter(_tokens(document)) for document in documents]
        self.lengths = [sum(counts.values()) for counts in self.counts]
        self.document_count = len(self.counts)
        self.average_length = (
            sum(self.lengths) / self.document_count if self.document_count else 0.0
        )
        self.document_frequency = Counter()
        for counts in self.counts:
            self.document_frequency.update(counts.keys())

    def tfidf(self, query):
        terms = tuple(dict.fromkeys(_tokens(query)))
        weights = {
            term: math.log(self.document_count / self.document_frequency[term])
            for term in terms if self.document_frequency[term]
        }
        return [
            math.fsum(counts[term] * weight for term, weight in weights.items())
            for counts in self.counts
        ]

    def bm25(self, query, k1=1.2, length_weight=0.75):
        for name, value in (("k1", k1), ("length_weight", length_weight)):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if k1 < 0 or not 0 <= length_weight <= 1:
            raise ValueError("Require k1 >= 0 and 0 <= b <= 1")
        terms = tuple(dict.fromkeys(_tokens(query)))
        weights = {
            term: math.log1p(
                (self.document_count - self.document_frequency[term] + 0.5)
                / (self.document_frequency[term] + 0.5)
            )
            for term in terms if self.document_frequency[term]
        }
        return [
            math.fsum(
                bm25_term(counts[term], length, self.average_length, weight, k1, length_weight)
                for term, weight in weights.items()
            )
            for counts, length in zip(self.counts, self.lengths)
        ]


if __name__ == "__main__":
    documents = [
        "cache cache error".split(),
        "cache error".split(),
        "cache guide setup notes".split(),
        "release notes".split(),
    ]
    index = LexicalIndex(documents)
    query = "cache error".split()
    print("TF-IDF:", [round(score, 4) for score in index.tfidf(query)])
    print("BM25:", [round(score, 4) for score in index.bm25(query)])
