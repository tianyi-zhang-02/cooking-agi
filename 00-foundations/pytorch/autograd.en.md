# Autograd: computed gradients are not always stored gradients

**English** · [中文](autograd.md)

> Reading time: about 16 minutes · Prerequisites: the chain rule and tensor operations · Last reviewed: 2026-10

You call `loss.backward()` and inspect an intermediate tensor’s `.grad`. It is `None`. That does not necessarily mean the graph is broken: **computing a gradient along the path** and **saving it in that tensor’s `.grad` field** are different things.

The small calculations below separate these ideas. They need only a CPU; numerical checks use PyTorch 2.8.0. We cover ordinary real-valued differentiation, not complex-gradient conventions.

## 1. Follow one chain first

Take a parameter $w=2$, an intermediate value $h=w^2$, and a loss $L=3h$:

$$
\begin{aligned}
\frac{\partial L}{\partial h}&=3,\\
\frac{\partial h}{\partial w}&=2w=4,\\
\frac{\partial L}{\partial w}&=3\times4=12.
\end{aligned}
$$

The forward values are `2 → 4 → 12`. The backward pass propagates derivatives, not the forward values in reverse.

```text
Forward:  w = 2  ── square ──> h = 4 ── times 3 ──> L = 12
Backward: dL/dw = 12 <── ×4 ── dL/dh = 3 <── ×3 ── dL/dL = 1
```

```python
import torch

weight = torch.tensor(2.0, requires_grad=True)
hidden = weight.square()
hidden.retain_grad()
loss = 3 * hidden
loss.backward()

assert weight.is_leaf and weight.grad_fn is None
assert not hidden.is_leaf and hidden.grad_fn is not None
assert weight.grad.item() == 12.0
assert hidden.grad.item() == 3.0
assert weight.item() == 2.0
```

`weight` is a leaf tensor that we created; `hidden` is a non-leaf produced by an operation. By default, backward accumulates `.grad` on leaves that require gradients, while intermediate results normally do not retain it. Calling `retain_grad()` lets us inspect `hidden.grad` afterward. Without it, gradients still flow through `hidden` to `weight`, but reading `hidden.grad` usually gives `None` and a warning.

The final assertion matters: **backward did not update the parameter**. It calculated a gradient; `optimizer.step()` applies the update rule. Nor is `grad_fn` a switch indicating whether gradients are possible: a leaf can have `grad_fn=None` and still receive gradients.

## 2. A vector output needs an objective

Calling `backward()` without an argument usually assumes a real output with one element and starts from derivative 1. Given a vector, PyTorch does not know whether you want its sum, its mean, or one particular component.

For $y=(w^2,3w)$, the component derivatives are $2w$ and 3. Supplying weights $v=(1,2)$ requests a vector–Jacobian product, or VJP:

$$
\begin{aligned}
v^\top J&=\frac{\partial (y_1+2y_2)}{\partial w}\\
&=2w+6.
\end{aligned}
$$

At $w=2$, the result is 10—not 7, and not the entire Jacobian.

```python
parameter = torch.tensor(2.0, requires_grad=True)
outputs = torch.stack((parameter.square(), 3 * parameter))
seed = torch.tensor([1.0, 2.0])
derivative, = torch.autograd.grad(outputs, parameter, grad_outputs=seed)

assert derivative.item() == 10.0
assert parameter.grad is None
```

`autograd.grad` returns derivatives rather than normally writing them into the inputs’ `.grad` fields. `outputs.backward(seed)` instead accumulates results into leaves. These interfaces compute related derivatives but serve different purposes.

In ordinary training, we more often use `mean()` or `sum()` to obtain a scalar loss. That choice changes gradient scale. The previous chapter’s distinction between averaging over samples and averaging over valid tokens directly affects the update.

## 3. Why did the second gradient get larger?

By default, successive backward calls **add** gradients rather than replacing them. This enables accumulation across small batches, but also makes forgetting to clear gradients an easy mistake.

```python
parameter = torch.tensor(2.0, requires_grad=True)
parameter.square().backward()
first_gradient = parameter.grad.item()
parameter.square().backward()
accumulated_gradient = parameter.grad.item()
parameter.grad = None
parameter.square().backward()

assert first_gradient == 4.0
assert accumulated_gradient == 8.0
assert parameter.grad.item() == 4.0
```

Each call recomputes `parameter.square()`, creating a fresh graph. Gradients from those graphs accumulate in the same `.grad` field. This is not repeated backward through a graph whose saved intermediates have already been freed.

A normal update usually starts with `optimizer.zero_grad(set_to_none=True)`. For deliberate accumulation, scale losses according to the objective over the whole effective batch. If microbatches have different sizes, averaging their means equally may be wrong.

`None` is also different from zero. It can mean no gradient was written for that parameter this time; a zero tensor is a gradient with value zero. An optimizer may skip a parameter whose `grad is None` while still applying momentum or weight decay to one with a zero gradient. Keep those cases separate when debugging.

## 4. detach, no_grad, and eval control different things

| Operation | What it changes | What it does not guarantee |
| --- | --- | --- |
| `clone()` | Copies tensor storage | Does not automatically cut gradient flow |
| `detach()` | Returns a tensor disconnected from the original graph | Does not copy storage; mutations can still affect both tensors |
| `with torch.no_grad():` | Usually disables backward recording inside the block | Does not switch Dropout / BatchNorm behavior |
| `model.eval()` | Changes modules affected by training mode | Does not disable autograd |
| `torch.inference_mode()` | Removes additional autograd overhead for inference | Is stricter than `no_grad`; its tensors cannot be freely reused in computations that need backward recording |

```python
source = torch.tensor([2.0, 3.0], requires_grad=True)
copied = source.clone()
detached = source.detach()
independent = source.detach().clone()
assert copied.requires_grad
assert not detached.requires_grad
assert detached.data_ptr() == source.data_ptr()
assert independent.data_ptr() != source.data_ptr()

copied.sum().backward()
assert torch.equal(source.grad, torch.ones(2))

model = torch.nn.Linear(2, 1)
model.eval()
assert model(torch.ones(1, 2)).requires_grad
with torch.no_grad():
    prediction = model(torch.ones(1, 2))
assert not prediction.requires_grad
```

For a snapshot that neither participates in training nor follows later mutations of the original data, use `detach().clone()`. Logging often uses `loss.item()`, but that returns a Python number and cannot be used for backward. On a GPU it may also introduce synchronization.

One exception to remember: some factory functions explicitly accept `requires_grad` and are not affected by `no_grad` in the same way. It does not mean “no tensor requiring gradients can possibly be created in this block.”

## 5. Graph lifetime and in-place changes

An ordinary eager forward pass builds a graph from the operations actually executed; the next pass may follow different branches. Backward usually releases saved intermediates that are no longer needed. Calling backward again on the original loss may fail because those intermediates are gone. Not every simple graph necessarily fails, so a successful second call does not prove that the complete graph was retained.

Three similar names solve different problems:

| Argument / method | Use it when |
| --- | --- |
| `retain_grad()` | You need a non-leaf tensor’s `.grad` for inspection |
| `retain_graph=True` | You genuinely need to reuse the same forward graph |
| `create_graph=True` | You need to differentiate a derivative, such as a second derivative |

The default is to run another forward pass on the next step, not to add `retain_graph=True` everywhere. Unnecessarily retaining graphs increases memory use.

Now consider an in-place change. Differentiating $w^2$ requires the value of $w$ used in the forward pass. What if you overwrite it before backward? PyTorch’s version checks catch many such inconsistencies. Do not bypass them with `.data`. Custom parameter updates belong after backward, inside `no_grad`; usually an optimizer should handle them.

```python
parameter = torch.tensor(2.0, requires_grad=True)
objective = parameter.square()
objective.backward()
with torch.no_grad():
    parameter.add_(parameter.grad, alpha=-0.1)
parameter.grad = None
assert torch.isclose(parameter, torch.tensor(1.6))

coordinate = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
first, = torch.autograd.grad(coordinate.pow(3), coordinate, create_graph=True)
second, = torch.autograd.grad(first, coordinate)
assert first.item() == 12.0
assert second.item() == 12.0
```

`torch.compile` may capture and compile parts of execution. That does not replace eager-autograd reasoning with “the entire program always has one static graph.” Get gradient and state handling right first, then consider compilation boundaries.

## 6. Check a derivative instead of trusting its appearance

For a small smooth function, compare autograd against a central finite difference:

$$
f'(x)\approx\frac{f(x+\varepsilon)-f(x-\varepsilon)}{2\varepsilon}.
$$

Large $\varepsilon$ introduces truncation error; very small $\varepsilon$ suffers from floating-point roundoff. Below, double precision checks $f(x)=x^3+2x$, followed by `gradcheck` on the same function. Do not apply this test uncritically at a nondifferentiable point, such as ReLU at zero.

```python
def smooth_objective(value):
    return value.pow(3) + 2 * value

coordinate = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
analytic, = torch.autograd.grad(smooth_objective(coordinate), coordinate)
step = 1e-6
center = coordinate.detach()
numerical = (smooth_objective(center + step) - smooth_objective(center - step)) / (2 * step)
assert torch.allclose(analytic, numerical, atol=1e-7, rtol=1e-7)
assert torch.autograd.gradcheck(smooth_objective, (coordinate,))
```

When training fails, check in order: whether the loss is still a tensor connected to the graph; whether the parameter participates in that graph; whether its gradient was not retained, not produced, or genuinely zero; and whether gradients were cleared. Only then move on to learning rates and optimizers. Successful backward proves that derivatives were calculated for this graph—not that its labels, masks, or objective were correct.

## Sources and next steps

These are teaching experiments, not a complete trainer. Continue to the [training loop](training-loop.en.md) to connect data, parameters, validation, and logging.

- [Autograd mechanics](https://docs.pytorch.org/docs/2.8/notes/autograd.html): leaves, gradient modes, graph lifetime, and in-place changes.
- [Tensor.backward](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.backward.html) and [autograd.grad](https://docs.pytorch.org/docs/2.8/generated/torch.autograd.grad.html): gradient seeds, accumulation, and returned derivatives.
- [retain_grad](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.retain_grad.html) and [detach](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.detach.html): retaining an intermediate gradient is not copying data or keeping an entire graph.
- [Gradcheck mechanics](https://docs.pytorch.org/docs/2.8/notes/gradcheck.html): numerical gradient checking and its limitations.
