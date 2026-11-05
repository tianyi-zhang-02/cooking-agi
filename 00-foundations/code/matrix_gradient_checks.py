"""Check a real linear layer's hand-derived gradients using finite differences.

Teaching only: no GPU performance or convergence claim. Python stdlib only.
"""

import copy
import json
import math


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def matmul(left, right):
    assert len(left[0]) == len(right)
    return [[sum(left_value * right_value for left_value, right_value in zip(row, column))
             for column in zip(*right)] for row in left]


def loss(inputs, weights, targets):
    predictions = matmul(inputs, weights)
    return 0.5 * sum((prediction - target) ** 2
                     for row, wanted in zip(predictions, targets)
                     for prediction, target in zip(row, wanted))


def check_case(inputs, targets, expected_weight_gradient, expected_after, initial_weights=None):
    weights = initial_weights if initial_weights is not None else [[1.0, 0.0], [0.0, 1.0]]
    predictions = matmul(inputs, weights)
    upstream = [[prediction - target for prediction, target in zip(row, wanted)]
                for row, wanted in zip(predictions, targets)]
    weight_gradient = matmul(transpose(inputs), upstream)
    input_gradient = matmul(upstream, transpose(weights))
    assert weight_gradient == expected_weight_gradient
    max_error = 0.0
    epsilon = 1e-6
    for which, values, analytical in [('weights', weights, weight_gradient),
                                      ('inputs', inputs, input_gradient)]:
        for row_index, row in enumerate(values):
            for column_index in range(len(row)):
                plus, minus = copy.deepcopy(values), copy.deepcopy(values)
                plus[row_index][column_index] += epsilon
                minus[row_index][column_index] -= epsilon
                if which == 'weights':
                    numerical = (loss(inputs, plus, targets) - loss(inputs, minus, targets)) / (2 * epsilon)
                else:
                    numerical = (loss(plus, weights, targets) - loss(minus, weights, targets)) / (2 * epsilon)
                expected = analytical[row_index][column_index]
                max_error = max(max_error, abs(numerical - expected))
                assert math.isclose(numerical, expected, abs_tol=1e-8, rel_tol=1e-8)
    updated = [[value - 0.1 * gradient for value, gradient in zip(row, gradients)]
               for row, gradients in zip(weights, weight_gradient)]
    after = loss(inputs, updated, targets)
    assert math.isclose(after, expected_after, abs_tol=1e-12)
    return {'input': inputs, 'weights': weights, 'target': targets, 'upstream_gradient': upstream,
            'dW': weight_gradient, 'dX': input_gradient, 'loss_before': loss(inputs, weights, targets),
            'loss_after': after, 'max_central_difference_error': max_error}


if __name__ == '__main__':
    cases = [
        check_case([[1.0, 2.0]], [[0.0, 1.0]], [[1.0, 1.0], [2.0, 2.0]], 0.25),
        check_case([[1.0, 2.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]],
                   [[1.0, 1.0], [1.0, 3.0]], 1.14),
        check_case([[1.0, -1.0, 2.0], [0.0, 3.0, -2.0]],
                   [[0.0, 1.0], [2.0, -1.0]],
                   [[-3.0, 0.75], [19.5, -4.5], [-17.0, 4.0]],
                   11.1128125,
                   initial_weights=[[0.5, -1.0], [2.0, 0.25], [-0.75, 1.5]]),
    ]
    result = {'status': 'passed', 'method': 'stdlib matrix arithmetic + central differences for every X/W entry',
              'epsilon': 1e-6, 'loss_definition': '0.5 * sum((X @ W - T)^2)', 'cases': cases}
    print(json.dumps(result, indent=2))
