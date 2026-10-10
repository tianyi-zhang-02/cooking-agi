# Python: names, copies, and function calls

[中文](python-objects.md) · **English**

> Reading time: ~15 min · Last reviewed: 2026-10

“I changed the copy. Why did the original change too?” Draw the names and the objects they point to, then ask whether the operation changes an object or rebinds a name. That usually explains more than memorizing a list of rules.

This follows [Useful Python](python.en.md). Start with sections 1–3 for shared data, 4–6 for function calls, or the last two for iteration and organizing state. Each code block runs independently using only the standard library. Examples are checked on Python 3.12, with API details checked against Python 3.14 documentation; no result depends on a particular memory address.

## 1. Two names do not imply two copies {#names-and-objects}

`backup = labels` does not copy a list. Both names initially refer to the same object. `append` changes that object, so either name reveals the change. Later, `backup = []` binds just that name to another list.

```python
labels = ["draft"]
backup = labels
backup.append("checked")
assert labels == ["draft", "checked"]
assert backup is labels

backup = []
assert labels == ["draft", "checked"]
assert backup == []
assert backup is not labels
```

Distinguish **mutation from rebinding** rather than saying that changing a variable changes all its aliases. `is` compares identity; `==` performs an equality comparison. Separate lists can be equal without being the same object. Do not extrapolate from small integers or strings that happen to be reused.

An immutable container may hold mutable objects. In `bundle = (["draft"],)`, the tuple's slot cannot be replaced, but the list can still receive an `append`. This tuple is also unhashable because it contains an unhashable list. `id()` represents identity; its interpretation as a memory address is specific to CPython. [Object model](https://docs.python.org/3.14/reference/datamodel.html#objects-values-and-types)

## 2. Which layer does a shallow copy separate? {#shallow-copy}

Take a note record with a title and a list of tags. Copy only the outer dictionary:

```python
record = {"title": "Attention", "tags": ["draft"]}
edited = record.copy()
edited["title"] = "Attention, revised"
edited["tags"].append("checked")

assert edited is not record
assert edited["tags"] is record["tags"]
assert record == {"title": "Attention", "tags": ["draft", "checked"]}
assert edited["title"] == "Attention, revised"
```

<figure class="worked-update" lang="en" id="copy-boundary">
<figcaption>The titles differ, but the tags change together. Follow the write.</figcaption>
<ol>
<li><strong>After copying</strong><span>record → original dictionary<br>edited → new dictionary<br>Both tags entries → one shared list</span></li>
<li><strong>Replace the title</strong><span>The edited dictionary refers to a new string. The original dictionary's title is untouched.</span></li>
<li><strong>Append a tag</strong><span>The shared list changes. Reading tags through either dictionary now includes checked.</span></li>
</ol>
</figure>

If tags are the only thing you need to change, copying the entire object graph may be unnecessary. Copy the path you intend to modify:

```python
record = {"title": "Attention", "tags": ["draft"]}
edited = {**record, "tags": [*record["tags"], "checked"]}
assert record["tags"] == ["draft"]
assert edited["tags"] == ["draft", "checked"]
assert edited["tags"] is not record["tags"]
```

The tags here are immutable strings. Mutable tag objects would require another decision about sharing. A shallow copy is not useless for nested data: it **separates the outer container and retains inner references**. For an ordinary built-in list, `values[:]`, `list(values)`, and `values.copy()` make shallow copies. Other sliceable types need not follow list semantics. [Sequence operations](https://docs.python.org/3.14/library/stdtypes.html#common-sequence-operations)

## 3. Deep copying does not break every shared reference {#deep-copy}

A deep copy recursively copies data, but “nothing is shared afterward” is too broad. These records originally share a tag list. The copied records still share a list **with each other**, though not with the originals.

<details markdown="1">
<summary>Check a shared reference and a cycle</summary>

```python
from copy import deepcopy

shared_tags = ["draft"]
records = [{"tags": shared_tags}, {"tags": shared_tags}]
cloned = deepcopy(records)
assert cloned[0]["tags"] is cloned[1]["tags"]
assert cloned[0]["tags"] is not shared_tags
cloned[0]["tags"].append("checked")
assert cloned[1]["tags"] == ["draft", "checked"]
assert shared_tags == ["draft"]

cycle = []
cycle.append(cycle)
copied_cycle = deepcopy(cycle)
assert copied_cycle is not cycle
assert copied_cycle[0] is copied_cycle
```

</details>

`deepcopy` records previously copied objects in a memo, handling repeated references and cycles. Functions can be reused, classes can customize copying, and files or sockets cannot simply be cloned this way. [Copy documentation](https://docs.python.org/3.14/library/copy.html)

Ask who will write the data and which layers must be independent. Copying a few tags is inexpensive; cloning a large cache may not be. Shared configuration can be intentional. Tensor storage has its own rules too; do not infer them from lists. See [PyTorch: tensors and storage](../00-foundations/pytorch/tensors-and-storage.en.md).

## 4. What survives a function call? {#function-binding}

A parameter name is bound to the supplied object. The function can mutate that object, but rebinding its parameter does not reassign the caller's variable.

```python
def append_then_replace(items):
    items.append("checked")
    items = ["replacement"]
    return items

original = ["draft"]
returned = append_then_replace(original)
assert original == ["draft", "checked"]
assert returned == ["replacement"]
assert returned is not original
```

The `append` reaches the original list. The later assignment changes only the local binding of `items`. To replace the caller's binding too, explicitly assign the return value: `original = append_then_replace(original)`.

For ordinary lists, `+=` extends in place while `+` creates a list. Look at the type's operation, not just whether the statement contains an equals sign. [Argument binding](https://docs.python.org/3.14/faq/programming.html#how-do-i-write-a-function-with-output-parameters-call-by-reference)

## 5. `*args` and `**kwargs`: collect here, unpack there {#arguments}

The special syntax is `*` and `**`, not the conventional names `args` and `kwargs`. Here, scores are collected into a tuple, extra settings into a dictionary, and `scale` is keyword-only.

```python
def summarize(*scores, scale=1, **metadata):
    return sum(scores) * scale, scores, metadata

values = [2, 5]
options = {"scale": 2, "source": "toy"}
total, received, metadata = summarize(*values, **options)
assert total == 14
assert received == (2, 5)
assert metadata == {"source": "toy"}
```

In the definition, `*scores` **collects** positional arguments; in the call, `*values` **unpacks** an iterable. Keys supplied by `**options` must be strings. Passing `scale` explicitly as well as through the mapping raises `TypeError`; it does not select the last value.

Accepting arbitrary keywords is convenient for forwarding configuration, but may also hide misspelled names. Prefer explicit parameters for a fixed interface, and decide which keys a forwarding layer permits. [Arguments and unpacking](https://docs.python.org/3.14/tutorial/controlflow.html#arbitrary-argument-lists)

## 6. Defaults and closures: when is the value chosen? {#defaults-and-closures}

Default arguments are evaluated when the definition executes, not recreated on each call. If each omitted argument should produce a new list, create it inside the function:

```python
def add_tag(tag, tags=None):
    if tags is None:
        tags = []
    tags.append(tag)
    return tags

first = add_tag("draft")
second = add_tag("checked")
assert first == ["draft"] and second == ["checked"]
assert first is not second

existing = []
assert add_tag("ready", existing) is existing
assert existing == ["ready"]
```

This does not make the function side-effect-free. An explicitly supplied list is still modified. The `None` convention prevents accidental sharing when the caller **omits the argument**.

A closure can instead read an outer variable at call time. Creating functions during a loop does not automatically capture a separate value each time:

```python
readers = [lambda: step for step in range(3)]
fixed_readers = [lambda saved=step: saved for step in range(3)]
assert [reader() for reader in readers] == [2, 2, 2]
assert [reader() for reader in fixed_readers] == [0, 1, 2]
```

The second version uses definition-time defaults to retain each integer. It retains a reference, not a deep copy: capture a list and later mutations remain visible. The same closure behavior applies to ordinary `def` functions. [Default arguments](https://docs.python.org/3.14/tutorial/controlflow.html#default-argument-values) · [Late binding](https://docs.python.org/3.14/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result)

## 7. `any`, `all`, and iterators: stopping also consumes input {#iteration-and-truth}

`any` stops at the first truthy value; `all` stops at the first falsey value. Both short-circuit. With empty input, `any` is false and `all` is true: no witness, but no counterexample either. Checking that input is nonempty **and** every item passes requires more than `all`. [Built-in functions](https://docs.python.org/3.14/library/functions.html#all)

```python
events = []

def passing_scores(scores):
    for score in scores:
        events.append(score)
        yield score >= 60

checks = passing_scores([80, 40, 90])
assert any(checks) is True
assert events == [80]
assert list(checks) == [False, True]
assert events == [80, 40, 90]
assert list(checks) == []
assert any([]) is False and all([]) is True
```

The first `any` consumes only the 80. Iteration then resumes with 40 and 90, not from the beginning. A list can provide fresh iterators; an exhausted **iterator object** does not rewind itself. A `for` loop also obtains an iterator and requests items until `StopIteration`. [Iterators and generators](https://docs.python.org/3.14/tutorial/classes.html#iterators)

A generator avoids storing every output, but still holds execution state and may retain its entire input list. Wrapping an already-loaded file in a generator cannot recover that input memory. Keep file iteration within the file's `with` block so it does not try to read an already-closed handle.

## 8. Functions or classes? Decide who owns the state {#functions-or-classes}

Python supports both styles. A function is straightforward for a one-off average. If observations arrive in batches and statistics must persist between calls, a class can keep the state beside the operations that update it.

<details markdown="1">
<summary>The same scores, organized two ways</summary>

```python
def average(scores):
    count, total = 0, 0.0
    for score in scores:
        count += 1
        total += score
    if count == 0:
        raise ValueError("At least one score is required")
    return total / count

class RunningMean:
    def __init__(self):
        self.count = 0
        self.total = 0.0

    def add(self, score):
        self.count += 1
        self.total += score

    def value(self):
        if self.count == 0:
            raise ValueError("At least one score is required")
        return self.total / self.count

meter = RunningMean()
for score in [2, 4, 9]:
    meter.add(score)
assert meter.value() == average([2, 4, 9]) == 5.0
other = RunningMean()
assert other.count == 0
```

</details>

Here, each instance owns its `self.count`. A mutable list stored as a class attribute could instead be shared accidentally. Using a class does not automatically make code easier to maintain, and using functions does not automatically make it faster. Both examples use a linear scan and constant auxiliary space; performance depends on the implementation. They assume ordinary finite numbers, without handling missing values, concurrent updates, or high-precision accumulation. [Class and instance variables](https://docs.python.org/3.14/tutorial/classes.html#class-and-instance-variables)

Try three changes: use dictionaries as tags, replace `any(checks)` with `all(checks)`, and create two `RunningMean` instances. Predict which objects and states change before running the code. Then return to [complexity and containers](algorithms/complexity-and-tools.en.md) to connect behavior with cost.
