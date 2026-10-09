# From SGD to AdamW: What Changes in One Update?

[中文](optimizers.md) · **English**

This chapter starts with coordinate-wise updates. For matrix optimizers, continue with [Muon](muon.en.md). It is an optional follow-up, not a prerequisite for learning the basics.

> Reading time: about 15 minutes · Level: foundations to advanced · Reviewed: 2026-10

A gradient supplies a local direction; an optimizer decides how to use it. It may remember recent directions, adapt the step to each coordinate's history, or both. Separating those choices makes the names easier to understand.

These equations cover common dense real-valued minimization settings. $t$ counts optimizer updates, not epochs, and $g_t=\nabla L_t(\theta_{t-1})$. Squares, divisions, and square roots are coordinate-wise.

## SGD: Start with the Learning Rate

The basic update is $\theta_t=\theta_{t-1}-\eta g_t$. For $L(w)=\frac12w^2$,

$$w_t=(1-\eta)w_{t-1}.$$

Starting at 1, $\eta=0.5$ gives `1 → 0.5 → 0.25`; $\eta=1.5$ gives `1 → -0.5 → 0.25`, oscillating but converging; $\eta=2.1$ gives `1 → -1.1 → 1.21`, diverging. The stability interval $0<\eta<2$ belongs to this scalar quadratic, not every neural network.

Stochastic / minibatch gradients add sampling noise. “SGD gets stuck in local optima” is not a complete explanation, and Momentum does not guarantee escape.

## Momentum and Nesterov: Remember the Direction?

One common convention is

$$v_t=\mu v_{t-1}+g_t,\qquad \theta_t=\theta_{t-1}-\eta v_t.$$

Consistent directions accumulate while alternating components partly cancel. Some derivations multiply $g_t$ by $1-\mu$ instead; that changes the scale, so do not transfer learning rates silently. Implementations may also treat the first momentum step specially.

A conceptual Nesterov update evaluates the gradient at the look-ahead point $\theta_{t-1}-\eta\mu v_{t-1}$. Reparameterized implementations need not physically move weights to a temporary location. This improves optimization under particular assumptions; it does not predict the true next gradient. [PyTorch SGD](https://docs.pytorch.org/docs/main/generated/torch.optim.SGD.html) documents implementation differences.

## Adaptive Steps: Why Store Squared Gradients?

If two coordinates have gradients 1 and 100, a common learning rate gives the second a 100-fold larger update. That may reflect scale rather than a desirable step. Adaptive methods use historical squared gradients to rescale coordinates.

| Method | State | Main tradeoff |
| --- | --- | --- |
| AdaGrad | $s_t=s_{t-1}+g_t^2$ | Accumulator never decreases; frequently updated coordinates may take increasingly small steps |
| RMSProp | $s_t=\rho s_{t-1}+(1-\rho)g_t^2$ | Tracks recent scale; not a full Hessian |
| AdaDelta | Moving averages of squared gradients and squared updates | Uses update-scale / gradient-scale ratios, not just an EMA replacement for AdaGrad |
| Adam | First moment $m_t$ and second moment $v_t$ | Smooths direction and rescales; includes bias correction and state memory |

A common AdaGrad denominator is $\sqrt{s_t}+\epsilon$. A coordinate with a smaller accumulator can receive a larger effective step, but update frequency and accumulated squared magnitude are not identical. See the [AdaGrad paper](https://www.jmlr.org/papers/v12/duchi11a.html).

AdaDelta also maintains $u_t=\rho u_{t-1}+(1-\rho)\Delta_t^2$ and forms

$$\Delta_t=-\frac{\sqrt{u_{t-1}+\epsilon}}{\sqrt{s_t+\epsilon}}g_t.$$

This differs from basic RMSProp, which only stores the gradient-square EMA. Libraries may still expose a learning-rate multiplier. Reducing manual learning-rate requirements in the original derivation does not mean every implementation lacks that parameter. See [AdaDelta](https://arxiv.org/abs/1212.5701).

## Adam: Why Correct the First Steps?

Starting with $m_0=v_0=0$,

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.$$

Zero initialization makes early moving averages small, motivating

$$\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t},\quad
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

For $g_1=2,\beta_1=0.9,\beta_2=0.999$, raw states are $m_1=0.2,v_1=0.004$, corrected to 2 and 4. Ignoring epsilon, the update magnitude is about $\eta$. Later steps need not have that length: changing directions, state history, and epsilon matter.

Adam stands for **adaptive moment estimation**, not AdaDelta plus Momentum. A second moment is not a second derivative; Adam is not Newton's method. The [Adam paper](https://arxiv.org/abs/1412.6980) defines the algorithm and corrections. NAdam incorporates a Nesterov idea, but no optimizer is a universal final form.

## AdamW: Why Separate Weight Decay?

For plain SGD without Momentum, adding squared-L2 gradient $\lambda\theta$ gives

$$\theta'=(1-\eta\lambda)\theta-\eta g.$$

That equals direct weight shrinkage. But feeding $g+\lambda\theta$ into Adam changes both moment histories and adaptive scaling, generally breaking the equivalence.

AdamW separates shrinkage from the adaptive gradient step:

$$\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

With fresh state, $\theta=2$, zero data gradient, $\eta=0.1$, and $\lambda=0.2$, AdamW shrinks to 1.96. Feeding an L2 gradient of 0.4 into Adam gives approximately 1.9 on the first step. These are different updates. See [Decoupled Weight Decay](https://arxiv.org/abs/1711.05101).

```python
import math

def adamw_scalar(parameter, gradient, state, learning_rate=0.1,
                 decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8):
    first, second, steps = state
    values = (parameter, gradient, first, second, learning_rate, decay, beta1, beta2, epsilon)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("finite values required")
    if type(steps) is not int or steps < 0 or second < 0:
        raise ValueError("invalid optimizer state")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("betas must be in [0, 1)")
    if learning_rate < 0 or decay < 0 or epsilon <= 0:
        raise ValueError("invalid optimizer settings")
    steps += 1
    first = beta1 * first + (1 - beta1) * gradient
    second = beta2 * second + (1 - beta2) * gradient ** 2
    first_corrected = first / (1 - beta1 ** steps)
    second_corrected = second / (1 - beta2 ** steps)
    updated = parameter * (1 - learning_rate * decay)
    updated -= learning_rate * first_corrected / (math.sqrt(second_corrected) + epsilon)
    return updated, (first, second, steps)
```

This scalar teaching implementation excludes sparse gradients, AMSGrad, mixed precision, and distributed execution. A zero gradient is not `grad=None`: a framework may skip a parameter entirely in the latter case. Check the actual [AdamW implementation](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html).

## Connect It to the Training Loop

| Decision | Check |
| --- | --- |
| Learning rate / scheduler | Does warmup or decay count successful optimizer updates or microbatches? |
| Accumulation | Are unequal sample / valid-token counts normalized correctly? |
| Clipping | Under AMP, are gradients unscaled before clipping? Which parameter group is clipped? |
| Resume | Are moments, step, scheduler, and random states restored alongside weights? |
| Freezing | Does the optimizer contain the intended parameters? Is stale state still occupying memory? |

AdamW is a useful baseline, not a parameter-free answer. Give competing optimizers reasonable learning-rate search budgets, then compare validation behavior, time to target, and state memory. One shared learning rate often tests which method suits that rate rather than which method suits the task.

Return to [one training step](training-step.en.md): `backward()` computes and accumulates gradients; `step()` uses optimizer state to update weights. They are different operations.
