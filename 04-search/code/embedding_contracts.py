import math


def checked_vector(values):
    vector = [float(value) for value in values]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Expected a nonempty finite vector")
    return vector


def checked_matrix(rows):
    matrix = [checked_vector(row) for row in rows]
    if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("Expected a nonempty rectangular matrix")
    return matrix


def last_valid_pool(hidden_states, attention_mask):
    states = checked_matrix(hidden_states)
    mask = list(attention_mask)
    if len(mask) != len(states) or any(value not in (0, 1) for value in mask):
        raise ValueError("Expected one binary mask value per position")
    positions = [index for index, valid in enumerate(mask) if valid]
    if not positions:
        raise ValueError("Cannot pool an all-padding sequence")
    return states[positions[-1]].copy()


def normalized_prefix(values, dimensions=None):
    vector = checked_vector(values)
    if dimensions is not None:
        if type(dimensions) is not int or not 1 <= dimensions <= len(vector):
            raise ValueError("Prefix dimension must be an integer in range")
        vector = vector[:dimensions]
    scale = max(abs(value) for value in vector)
    if scale == 0:
        raise ValueError("A zero vector has no cosine direction")
    scaled = [value / scale for value in vector]
    norm = math.hypot(*scaled)
    return [value / norm for value in scaled]


def dot(left, right):
    first, second = checked_vector(left), checked_vector(right)
    if len(first) != len(second):
        raise ValueError("Vector dimensions must agree")
    value = math.fsum(first_value * second_value
                      for first_value, second_value in zip(first, second))
    if not math.isfinite(value):
        raise ValueError("Dot product is not finite")
    return value


def mean_maxsim(query_tokens, document_tokens):
    queries = [normalized_prefix(row) for row in checked_matrix(query_tokens)]
    documents = [normalized_prefix(row) for row in checked_matrix(document_tokens)]
    if len(queries[0]) != len(documents[0]):
        raise ValueError("Query and document dimensions must agree")
    maxima = (max(dot(query, document) for document in documents) for query in queries)
    return math.fsum(maxima) / len(queries)


def main():
    states = [[9, 9], [9, 9], [1, 0], [2, 0], [3, 4]]
    print("Last valid:", last_valid_pool(states, [0, 0, 1, 1, 1]))
    old_documents = [[1, 0], [0, 1]]
    new_documents = [[0, 1], [-1, 0]]
    new_query = [0, 1]
    mixed = [dot(new_query, document) for document in old_documents]
    rebuilt = [dot(new_query, document) for document in new_documents]
    print(f"Mixed spaces: {mixed}; rebuilt: {rebuilt}")
    print("Mean MaxSim:", mean_maxsim([[1, 0], [0, 1]], [[0.8, 0.6], [0, 1]]))


if __name__ == "__main__":
    main()
