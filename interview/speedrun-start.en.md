# Starting out: make one small example clear

[中文](speedrun-start.md) · **English** · [Back to Speedrun](speedrun.en.md)

> Last reviewed: 2026-10 · Suggested focused learning time, split across as many sessions as you need. Not a promise of instant proficiency.

When you are new to ML, terms such as parameters, loss, gradients, and training can arrive all at once. One small model lets you see how they fit together. Start with Python inputs and outputs, calculate a prediction and loss, then update a parameter. You can connect these steps to larger models and interview exercises later; there is no need to begin by implementing a Transformer.

Skip familiar steps. If one example takes the evening, do not chase the schedule. Aim to understand a little more and know where to resume. Readers focused on SWE can use just the Python, algorithm, and code-reading parts.

<nav class="study-route speedrun-route" aria-label="Stages of the starter route">
<a href="#first-two"><small>TRY 2 HOURS FIRST</small><strong>A function and a prediction</strong><span>Inputs, outputs, and how to notice a wrong result</span></a>
<a href="#next-four"><small>ADD 4 HOURS</small><strong>How error changes a parameter</strong><span>Some algorithms, one update, data, and evaluation</span></a>
<a href="#next-six"><small>ADD 6 HOURS</small><strong>Connect models and systems</strong><span>One manageable path, not an entire ML project at once</span></a>
</nav>

## Start from something you can connect to {#starting-point}

- **Functions, lists, and dictionaries are unfamiliar:** run a small example from [Useful Python](python.en.md), change its input, and follow the program.
- **You know Python but not ML:** start with the prediction below. No PyTorch installation or GPU is needed yet.
- **You can train models but struggle in interviews:** use the [review route](speedrun.en.md) and let implementation and debugging exercises show what you still need.

## Try two hours: what does the program do, and what does a model add? {#first-two}

| Suggested time | What to do | What to understand for now |
| --- | --- | --- |
| 15 minutes | Look at your role or learning goal and choose a starting point | Is the gap Python or model concepts? |
| 25 minutes | Write a frequency-counting function using a dictionary | Input, loop, return value, and why empty input matters |
| 30 minutes | Work through [prediction and loss](#prediction) | A parameter is not an output; see how loss is computed |
| 25 minutes | Try a small list / dictionary algorithm exercise | Start with a simple solution and explain which work can be avoided |
| 15 minutes | Describe a small project or course assignment | Input, approach, result, and what you personally wrote |
| 10 minutes | Leave a place to resume | Even one loop you do not yet understand |

Suggested total: 120 minutes, not necessarily in one sitting. Start algorithms with the counting example in [hash maps](algorithms/hash-and-prefix.en.md): understand the approach, then practice it on new inputs.

### One prediction and loss calculation is enough to begin {#prediction}

Take inputs `[1,2]` and targets `[2,4]`. The model has one parameter, `weight`, and predicts `weight × input`. With `weight = 1`, the predictions are `[1,2]`.

<div class="compact-table" markdown="1">

| Input | Target | Prediction | Squared error |
| --- | --- | --- | --- |
| 1 | 2 | 1 | 1 |
| 2 | 4 | 2 | 4 |

</div>

Average the squared errors: loss is `(1+4)/2 = 2.5`. Squaring is a choice of objective here, not what every task does. Classification commonly uses a different objective, such as cross-entropy.

```python
def predict(values, weight):
    return [weight * value for value in values]


def mean_squared_error(predictions, targets):
    if len(predictions) != len(targets) or not targets:
        raise ValueError("Need equally sized, nonempty inputs")
    errors = [(prediction - target) ** 2
              for prediction, target in zip(predictions, targets)]
    return sum(errors) / len(errors)


inputs = [1.0, 2.0]
targets = [2.0, 4.0]
print(mean_squared_error(predict(inputs, 1.0), targets))
```

Change the parameter to 1.5; calculate before running. Predictions become `[1.5,3]` and loss is `(0.25+1)/2 = 0.625`. This is a small example of fitting training data better. **It has not established better performance on unseen data.**

After running it, point to where the inputs, parameter, predictions, targets, and loss appear in the code. The update example below uses each of them again.

## Add four hours: why change the parameter in this direction? {#next-four}

| Additional time | What to do | What to understand for now |
| --- | --- | --- |
| 60 minutes | Pick two of counting, binary search, or BFS and trace small inputs | The conditions that make a method valid, not just its code |
| 60 minutes | Connect the gradient example below to [linear models](../00-foundations/from-linear-to-neural.en.md) | How a gradient describes a local direction of change |
| 45 minutes | [Tensors and shapes](../00-foundations/pytorch/README.en.md): predict dimensions, then run | Which axis represents the batch, features, and outputs |
| 45 minutes | Separate training and validation data, then change examples and inspect loss | Training results are not automatically results on new data |
| 30 minutes | Redo the function that caused trouble earlier | What is clear now and what still needs another example |

Suggested cumulative budget: six hours. If tensors are new, use that block just to understand two-dimensional arrays. You do not need to squeeze a training loop into it too.

### Begin with the gradient of one parameter {#update}

Before differentiating, try a small change: move `weight` from 1 to 1.01. Loss falls from 2.5 to 2.45025. At this point, increasing the parameter a little reduces the error. A gradient describes that local change; with one parameter, it is the slope of the loss curve.

For the same examples, the squared-error gradient with respect to `weight` is the mean of `2 × (prediction−target) × input`. At `weight=1`, the terms are −2 and −8, averaging to −5.

The update is `new parameter = old parameter − learning rate × gradient`. A learning rate of 0.1 gives `1 − 0.1 × (−5) = 1.5`, the value we just tried; loss falls to 0.625.

```python
weight = 1.0
gradient = sum(2 * (weight * value - target) * value
               for value, target in zip(inputs, targets)) / len(inputs)
weight -= 0.1 * gradient
print(weight, mean_squared_error(predict(inputs, weight), targets))
```

This does not mean every gradient step reduces loss. A learning rate of 1 jumps to `weight=6` and raises loss to 40. Understanding this one change gives you something concrete to use when reading about learning rates, optimizers, and larger networks.

## Add six hours: connect the example to larger problems {#next-six}

| Additional time | What to do | A place to stop for now |
| --- | --- | --- |
| 90 minutes | Try a small MLP forward pass, or start with matrix multiplication | Explain each layer's shapes; backward can be another session |
| 60 minutes | Read a little [tokenization](../00-foundations/core/tokenization.en.md) and the [attention walkthrough](../00-foundations/transformer-lab.en.md) | Explain text inputs and weighted mixing without implementing a whole network |
| 60 minutes | [Read code](code-comprehension.en.md#cache), tracing two calls | Find a reproducible bug and explain a fix |
| 60 minutes | Practice two related algorithm exercises or a variation on an old one | Reconsider the method when conditions change |
| 45 minutes | Follow one prediction through [an ML system](../practice/ml-system/README.en.md) | Distinguish data, training, deployment, and evaluation without memorizing every component |
| 45 minutes | Explain a familiar idea to a friend or record yourself practicing | Find one unclear step to revisit next time |

Suggested cumulative budget: twelve hours. This is not a claim of interview readiness. It is a start on connecting code, objectives, models, and systems. Move to the [review route](speedrun.en.md#core) when useful, or keep working with one example.

## How do separate sessions add up? {#keep-going}

If this session covered one gradient step, next time loop the update ten times and print the parameter and loss each time. Guess the result before running it, and stop to check any value that differs from your guess. That makes problems easier to see than switching to a bigger model.

With an interview close by, choose one or two items for the confirmed round and be honest about work you have not done. A short timeline is a reason to reduce scope, not add pressure to learning.
