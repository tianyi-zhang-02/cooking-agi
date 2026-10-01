from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EmbeddingSpec:
    space_version: str
    dimensions: int
    metric: str


def validate_spec(spec: EmbeddingSpec) -> None:
    if not isinstance(spec.space_version, str) or not spec.space_version.strip():
        raise ValueError("A representation-space version is required")
    if not isinstance(spec.dimensions, int) or isinstance(spec.dimensions, bool) or spec.dimensions < 1:
        raise ValueError("Dimensions must be a positive integer")
    if spec.metric not in {"inner_product", "cosine"}:
        raise ValueError("Unsupported score metric")


def prepare_vector(vector: Sequence[float], spec: EmbeddingSpec) -> tuple[float, ...]:
    if len(vector) != spec.dimensions:
        raise ValueError("Vector dimension does not match the manifest")
    values = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Vector values must be finite")
    if spec.metric == "cosine":
        norm = math.hypot(*values)
        if norm == 0 or not math.isfinite(norm):
            raise ValueError("Cosine needs a finite, nonzero norm")
        values = tuple(value / norm for value in values)
    return values


def exact_top_k(
    query: Sequence[float],
    query_spec: EmbeddingSpec,
    items: Mapping[str, Sequence[float]],
    index_spec: EmbeddingSpec,
    limit: int,
    excluded_ids: Set[str] = frozenset(),
) -> list[tuple[str, float]]:
    validate_spec(query_spec)
    validate_spec(index_spec)
    if query_spec != index_spec:
        raise ValueError("Query and index manifests are incompatible")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    prepared_query = prepare_vector(query, query_spec)
    scored = []
    for item_id, vector in items.items():
        if item_id in excluded_ids:
            continue
        prepared_item = prepare_vector(vector, index_spec)
        score = sum(left * right for left, right in zip(prepared_query, prepared_item))
        if not math.isfinite(score):
            raise ValueError("Computed score must be finite")
        scored.append((item_id, score))
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]


def main() -> None:
    spec = EmbeddingSpec("teaching-space-v1", 2, "cosine")
    items = {"A": (0.8, 0.6), "B": (1.0, 1.0), "C": (0.0, 1.0)}
    query = (1.0, 0.0)
    print("Exact top-2:", [item_id for item_id, score in exact_top_k(query, spec, items, spec, 2)])
    print("Exclude A:", [item_id for item_id, score in exact_top_k(
        query, spec, items, spec, 2, {"A"})])
    incompatible = EmbeddingSpec("teaching-space-v2", 2, "cosine")
    try:
        exact_top_k(query, incompatible, items, spec, 2)
    except ValueError as error:
        print("Rejected:", error)
    vector_bytes = 1_000_000 * 256 * 4
    print(f"Vector-only storage: {vector_bytes / 1_000_000_000:.3f} GB / {vector_bytes / 2**30:.3f} GiB")


if __name__ == "__main__":
    main()
