# From batches to a training loop: running is not learning

**English** · [中文](training-loop.md)

> Reading time: about 18 minutes · Prerequisites: autograd and cross-entropy · Last reviewed: 2026-10

A `Linear` layer producing two scores is not yet a trained classifier. A falling loss does not show that it handles unseen data. Beyond forward and backward, a training loop must handle data splits, parameter registration, metric aggregation, and what gets restored after an interruption.

We will follow those steps in a tiny binary-classification experiment. It uses synthetic data and downloads nothing; numerical checks use CPU / PyTorch 2.8.0. This is a test of training logic, **not a model-quality or performance benchmark**.

## 1. Give each component a clear job

| Component | Its job here | Common misunderstanding |
| --- | --- | --- |
| `torch.Tensor` / `torch.tensor` | The former is a type; the latter is a common data-construction function | Treating the names as entirely interchangeable |
| `nn.Module` | Registers parameters, submodules, and buffers; defines forward computation | Assuming every module has weights and a bias |
| `nn.functional` | Provides stateless operations, including losses, linear operations, and activations | Assuming it contains only activation functions |
| `Dataset` | Defines how to retrieve a sample | Confusing a sample with a batch |
| `DataLoader` | Samples, batches, and organizes loading according to its configuration | Assuming arbitrary input automatically becomes model-ready tensors |
| `Optimizer` | Updates parameters using gradients and internal state | Assuming `backward` already updated parameters |

We use `TensorDataset` with features in the first tensor and labels in the second; entries at the same position belong together. `shuffle=True` shuffles sample indices, not features and labels independently. Default collation tries to assemble each field into a batch. Variable-length text still needs padding, masks, or a custom `collate_fn`.

For `Linear(2, 2)`, input is `[batch, 2]`, weights are `[2, 2]`, and output is `[batch, 2]`. In general, `Linear(in_features, out_features)` stores weights as `[out_features, in_features]`, computes $XW^\top+b$, and preserves leading input dimensions.

## 2. Separate data before learning from it

We generate two-dimensional points and assign classes with a fixed rule. Training and validation use different random seeds but the same synthetic distribution. In real tasks, also check leakage across users, time, and document sources; splitting an array alone does not establish independence.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def make_dataset(count, seed):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(count, 2, generator=generator)
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).long()
    return TensorDataset(features, labels)

torch.manual_seed(17)
train_data = make_dataset(80, seed=5)
validation_data = make_dataset(47, seed=6)
train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True,
    generator=torch.Generator().manual_seed(7),
)
validation_loader = DataLoader(validation_data, batch_size=16, shuffle=False)
model = nn.Linear(2, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.15, momentum=0.9)

features, labels = next(iter(validation_loader))
assert features.shape == (16, 2)
assert labels.shape == (16,) and labels.dtype == torch.long
assert model(features).shape == (16, 2)
```

The task defines the number of classes—two here—not the number that happens to appear in one batch. For this single-label task, `CrossEntropyLoss` takes logits without a preceding softmax and class indices of dtype `long`. Do not apply softmax first or casually reshape labels to `[batch, 1]`.

For a multi-label task, several classes may be true simultaneously. A common choice is `BCEWithLogitsLoss` with floating-point targets of matching shape. That is not merely renaming CE: target semantics, shapes, and reduction must change together.

Validation deliberately contains 47 samples, leaving 15 in the final batch. This small inconvenience helps reveal incorrect metric averaging.

## 3. What happens during one update

```text
Retrieve features and labels
  → clear previous gradients
  → model(features): produce logits
  → loss: average over the current batch
  → backward: calculate and accumulate gradients
  → step: update parameters
  → accumulate this epoch's loss total and sample count
```

```python
def train_epoch(model, loader, optimizer):
    model.train()
    loss_total = 0.0
    sample_count = 0
    for features, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        count = labels.numel()
        loss_total += loss.detach().item() * count
        sample_count += count
    if sample_count == 0:
        raise ValueError("Training loader is empty")
    return loss_total / sample_count

def evaluate(model, loader):
    previous_mode = model.training
    model.eval()
    loss_total = 0.0
    correct_count = 0
    sample_count = 0
    try:
        with torch.no_grad():
            for features, labels in loader:
                logits = model(features)
                loss_total += nn.functional.cross_entropy(
                    logits, labels, reduction="sum"
                ).item()
                correct_count += (logits.argmax(dim=-1) == labels).sum().item()
                sample_count += labels.numel()
    finally:
        model.train(previous_mode)
    if sample_count == 0:
        raise ValueError("Evaluation loader is empty")
    return {"loss": loss_total / sample_count, "accuracy": correct_count / sample_count}

initial_metrics = evaluate(model, validation_loader)
history = []
for epoch in range(25):
    train_loss = train_epoch(model, train_loader, optimizer)
    validation_metrics = evaluate(model, validation_loader)
    history.append({"epoch": epoch + 1, "train_loss": train_loss, **validation_metrics})

assert history[-1]["loss"] < initial_metrics["loss"]
assert 0.0 <= history[-1]["accuracy"] <= 1.0
```

Use `model(features)` rather than calling `model.forward(features)` directly, so Module call machinery such as hooks runs normally. Our tiny model has neither Dropout nor BatchNorm, but we still distinguish `train` from `eval` before a larger replacement makes the omission matter.

Training loss is calculated **before** each batch update, so it describes a model changing throughout the epoch. Validation loss is measured on the fixed model at the epoch’s end. They are not identical measurements; their relative heights alone do not settle a diagnosis.

Evaluation temporarily switches modes and restores the previous state. This simplified implementation assumes the model originally has a uniform training/evaluation mode; mixed submodule modes need individual handling. It also rejects an empty dataset rather than hiding it behind division by zero or a misleading zero metric.

## 4. Check the denominator before interpreting the metric

Suppose two batches contain 16 samples and one sample, with mean losses of 1 and 9. Averaging the means gives 5. The actual sample mean is:

$$
\frac{16\times1+1\times9}{17}=\frac{25}{17}\approx1.47.
$$

The training code’s `loss * count` reconstructs each batch’s loss total before dividing by the total sample count. Evaluation directly accumulates losses with `reduction="sum"`.

That conversion relies on this example’s conditions: no class weights, no ignored labels, and one loss per sample. With `ignore_index`, class weights, or variable-length token masks, the denominator may be a valid-token count or weight sum rather than `labels.numel()`. Nor should evaluation use `drop_last=True` merely to make batches tidy.

| Observation | Check first |
| --- | --- |
| Training loss does not move | Whether the optimizer owns the parameters; gradients; learning rate; targets and labels |
| Training improves while validation worsens | Splits, overfitting, and distribution differences—not immediately model size |
| Results vary substantially across runs | Data order, initialization, randomness, sample size, and split design |
| Accuracy looks implausibly high | Label leakage or related samples crossing the train/validation boundary |
| Loss falls but accuracy stays constant | Probabilities can improve without crossing an argmax decision boundary |

Fixed seeds help reproduce this experiment but do not promise bitwise equality across PyTorch versions, hardware, and all operators. Validation guides tuning; real experiments also need a test set that is not repeatedly inspected. These 47 synthetic points support no claim about a real task.

## 5. Parameters, buffers, and ordinary attributes

Assigning a tensor to an object does not automatically register a parameter. Use `nn.Parameter` or existing layers for trainable weights. State that should move or save with the model but does not receive gradient updates can be a buffer.

```python
class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 4), nn.Linear(4, 2)])
        self.register_buffer("input_scale", torch.ones(2))

    def forward(self, features):
        hidden = torch.relu(self.layers[0](features / self.input_scale))
        return self.layers[1](hidden)

registered_model = TinyClassifier()
assert "layers.0.weight" in dict(registered_model.named_parameters())
assert "input_scale" not in dict(registered_model.named_parameters())
assert "input_scale" in registered_model.state_dict()
assert registered_model(torch.ones(3, 2)).shape == (3, 2)
```

A plain Python list does not automatically register child layers like `ModuleList` does. Forward may work while `model.parameters()` misses their weights. A buffer is persistent by default and appears in `state_dict`; an ordinary tensor attribute does not receive the same automatic management.

`ReLU` and `Dropout` can be Modules without trainable parameters. Conversely, functional calls can construct differentiable computations. What matters is who owns and registers the participating weights, and how training mode is supplied.

## 6. Saving and logging: what exactly was restored?

The next example saves and loads in memory without creating a file. `deepcopy` makes an independent snapshot so continued training does not mutate the state you thought you had saved in memory. `weights_only=True` is an explicit loading choice here, not permission to load arbitrary untrusted files.

```python
import copy
import io

snapshot = copy.deepcopy({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": len(history),
})
buffer = io.BytesIO()
torch.save(snapshot, buffer)
buffer.seek(0)
loaded = torch.load(buffer, weights_only=True)
restored_model = nn.Linear(2, 2)
restored_model.load_state_dict(loaded["model"])
restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.15, momentum=0.9)
restored_optimizer.load_state_dict(loaded["optimizer"])
with torch.no_grad():
    torch.testing.assert_close(model(features), restored_model(features))
assert loaded["epoch"] == 25
```

The assertion only checks that the restored model produces the same output for the same input. It does not establish bitwise continuation from an arbitrary training position. Resuming the same trajectory also involves schedulers, AMP scalers, RNG states, data-sampling position, and distributed state. `state_dict` does not store the complete architecture definition or every runtime mode; restoration needs a compatible model.

Logs should at least separate training and validation, with an explicit x-axis: epochs, optimizer steps, or processed tokens. Do not label the last batch’s loss as an epoch mean, or keep accumulating accuracy across epochs while calling it “this epoch’s accuracy.”

The optional TensorBoard function below requires the separate `tensorboard` package; earlier experiments do not. Defining this function is not evidence that its logging interface has been inspected in a browser.

```python
def write_tensorboard(history, log_dir):
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(log_dir=log_dir) as writer:
        for entry in history:
            writer.add_scalar("loss/train", entry["train_loss"], entry["epoch"])
            writer.add_scalar("loss/validation", entry["loss"], entry["epoch"])
            writer.add_scalar("accuracy/validation", entry["accuracy"], entry["epoch"])
```

A TensorBoard graph depends on capture method and example inputs; it is not proof of all dynamic program branches. Use curves to raise questions, then check code and data. Visualization does not replace validation.

## Sources and next steps

We deliberately avoid old torchtext data-loading tutorials: torchtext development has stopped, with 0.18 its final stable release. Learning PyTorch does not require installing every domain library, and an old screenshot is not a dependency specification.

- [Linear](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Linear.html) and [Module](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html): dimensions, registration, state, and calling conventions.
- [Data utilities](https://docs.pytorch.org/docs/2.8/data.html): Dataset, DataLoader, collation, and multiprocessing; iterable-dataset sharding and `drop_last` need additional care.
- [CrossEntropyLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html): class indices, probability targets, weights, and reduction.
- [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) and [TensorBoard](https://docs.pytorch.org/docs/2.8/tensorboard.html): checkpoints and logging.
- [Reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html) and [torchtext status](https://docs.pytorch.org/text/stable/index.html): seed guarantees and dependency maintenance.

Next, replace this tiny model with a [from-scratch module](../code/README.en.md), or follow [one language-model training update](../deep-dives/training-step.en.md). Being able to explain each batch shape, loss denominator, and parameter-update boundary makes scaling up much safer.
