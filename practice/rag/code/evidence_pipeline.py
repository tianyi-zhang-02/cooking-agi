from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    revision: int
    text: str
    visibility: frozenset[str]


def unique_chunks(chunks):
    identifiers = [chunk.chunk_id for chunk in chunks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("chunk IDs must be unique")


def visible_current(chunks, groups, current_revisions):
    unique_chunks(chunks)
    return [
        chunk for chunk in chunks
        if chunk.revision == current_revisions.get(chunk.document_id)
        and chunk.visibility.intersection(groups)
    ]


def terms(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(query, chunks, groups, current_revisions, limit=3):
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    eligible = visible_current(chunks, groups, current_revisions)
    query_terms = terms(query)
    scored = [
        (len(query_terms.intersection(terms(chunk.text))), chunk)
        for chunk in eligible
    ]
    scored.sort(key=lambda entry: (-entry[0], entry[1].chunk_id))
    return [chunk for score, chunk in scored if score > 0][:limit]


def reciprocal_rank_fusion(rankings, constant=10):
    if not math.isfinite(constant) or constant <= 0:
        raise ValueError("constant must be finite and positive")
    scores = {}
    for ranking in rankings:
        deduplicated = list(dict.fromkeys(ranking))
        for rank, identifier in enumerate(deduplicated, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda entry: (-entry[1], entry[0]))


def pack_evidence(chunks, budget):
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    unique_chunks(chunks)
    selected = []
    used = 0
    for chunk in chunks:
        cost = len(chunk.text.split())
        if cost == 0:
            raise ValueError("evidence must not be empty")
        if used + cost <= budget:
            selected.append(chunk)
            used += cost
    return selected, used


def unknown_citations(citations, provided_chunks):
    allowed = {chunk.chunk_id for chunk in provided_chunks}
    return sorted(set(citations) - allowed)


def demo():
    chunks = [
        Chunk("rules-v1", "rules", 1, "Registration deadline Friday",
              frozenset({"members"})),
        Chunk("rules-v2", "rules", 2, "Registration deadline Wednesday",
              frozenset({"members"})),
        Chunk("budget-v1", "budget", 1, "Registration deadline budget plan",
              frozenset({"organizers"})),
    ]
    results = retrieve("Registration deadline", chunks, {"members"},
                       {"rules": 2, "budget": 1}, limit=1)
    print("retrieved:", [chunk.chunk_id for chunk in results])
    print("RRF:", reciprocal_rank_fusion([["A", "B"], ["C", "A"]]))
    passages = [
        Chunk("first", "demo", 1, "one two three four five six", frozenset({"members"})),
        Chunk("second", "demo", 1, "one two three four five six", frozenset({"members"})),
        Chunk("third", "demo", 1, "one two three four", frozenset({"members"})),
    ]
    packed, used = pack_evidence(passages, 10)
    print("packed:", [chunk.chunk_id for chunk in packed], "used:", used)
    print("unknown citations:", unknown_citations(["rules-v1"], results))


if __name__ == "__main__":
    demo()
