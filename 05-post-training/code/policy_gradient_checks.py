import math

import torch


def centered_advantages(rewards, epsilon=1e-8):
    if rewards.ndim != 2 or rewards.shape[1] < 2 or epsilon <= 0:
        raise ValueError("Expected [groups, responses] with at least two responses")
    if not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be finite")
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    deviation = rewards.std(dim=-1, correction=0, keepdim=True)
    return (centered / (deviation + epsilon)).detach()


def clipped_terms(log_probs, old_log_probs, advantages, epsilon=0.2):
    if not 0 < epsilon < 1:
        raise ValueError("Expected 0 < epsilon < 1")
    if log_probs.shape != old_log_probs.shape or log_probs.shape != advantages.shape:
        raise ValueError("All inputs must have the same shape")
    ratios = (log_probs - old_log_probs.detach()).exp()
    fixed_advantages = advantages.detach()
    direct = ratios * fixed_advantages
    bounded = ratios.clamp(1 - epsilon, 1 + epsilon) * fixed_advantages
    return -torch.minimum(direct, bounded)


def masked_policy_loss(log_probs, old_log_probs, advantages, valid, reduction="sequence"):
    if log_probs.ndim != 2 or log_probs.shape != old_log_probs.shape:
        raise ValueError("Expected matching [responses, time] log-probabilities")
    if valid.shape != log_probs.shape or valid.dtype != torch.bool:
        raise ValueError("Expected a boolean mask matching log-probabilities")
    if advantages.shape != (log_probs.shape[0],):
        raise ValueError("Expected one fixed advantage per response")
    lengths = valid.sum(dim=-1)
    if not torch.all(lengths > 0):
        raise ValueError("Every response needs a valid token")
    if not torch.isfinite(log_probs[valid]).all() or not torch.isfinite(old_log_probs[valid]).all():
        raise ValueError("Valid log-probabilities must be finite")
    if not torch.isfinite(advantages).all():
        raise ValueError("Advantages must be finite")
    safe_current = log_probs.masked_fill(~valid, 0)
    safe_old = old_log_probs.masked_fill(~valid, 0)
    expanded = advantages[:, None].expand_as(log_probs)
    terms = clipped_terms(safe_current, safe_old, expanded).masked_fill(~valid, 0)
    if reduction == "sequence":
        return (terms.sum(dim=-1) / lengths).mean()
    if reduction == "token":
        return terms.sum() / lengths.sum()
    raise ValueError("Expected sequence or token reduction")


def zero_loss_example():
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    logits = torch.stack((theta, torch.zeros_like(theta)))
    current = logits.log_softmax(dim=-1)
    old = current.detach().clone()
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
    loss = clipped_terms(current, old, advantages).mean()
    gradient = torch.autograd.grad(loss, theta)[0]
    next_probability = torch.sigmoid(theta.detach() - 0.2 * gradient)
    return loss.item(), gradient.item(), next_probability.item()


def clipping_example():
    current = torch.tensor([0.7, 0.3, 0.3, 0.7], dtype=torch.float64).log().requires_grad_()
    old = torch.full_like(current, math.log(0.5))
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
    losses = clipped_terms(current, old, advantages)
    gradients = torch.autograd.grad(losses.sum(), current)[0]
    return losses.detach(), gradients


def regularizer_example():
    theta = torch.tensor(math.log(7 / 3), dtype=torch.float64, requires_grad=True)
    log_probs = torch.stack((theta, torch.zeros_like(theta))).log_softmax(dim=-1)
    probabilities = log_probs.exp()
    old_selected = torch.tensor(math.log(0.5), dtype=torch.float64)
    policy = clipped_terms(log_probs[0], old_selected, torch.ones_like(old_selected))
    reference = torch.full_like(log_probs, math.log(0.5))
    exact_kl = (probabilities * (log_probs - reference)).sum()
    entropy = -(probabilities * log_probs).sum()
    total = policy + 0.1 * exact_kl - 0.01 * entropy
    policy_gradient = torch.autograd.grad(policy, theta, retain_graph=True)[0]
    total_gradient = torch.autograd.grad(total, theta)[0]
    return policy_gradient.item(), total_gradient.item()


def shared_parameter_example():
    theta = torch.tensor(math.log(7 / 3), dtype=torch.float64, requires_grad=True)
    selected = torch.nn.functional.logsigmoid(theta).repeat(2)
    old = torch.full_like(selected, math.log(0.5))
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
    losses = clipped_terms(selected, old, advantages)
    first_gradient = torch.autograd.grad(losses[0], theta, retain_graph=True)[0]
    batch_gradient = torch.autograd.grad(losses.mean(), theta)[0]
    next_probability = torch.sigmoid(theta.detach() - 0.1 * batch_gradient)
    return first_gradient.item(), batch_gradient.item(), next_probability.item()


def main():
    loss, gradient, probability = zero_loss_example()
    print(f"zero loss: {loss:.6f}; gradient: {gradient:.6f}; next p: {probability:.6f}")
    losses, gradients = clipping_example()
    print("clipped losses:", losses.tolist())
    print("log-prob gradients:", gradients.tolist())
    print("policy and regularized gradients:", regularizer_example())
    print("shared-parameter update:", shared_parameter_example())


if __name__ == "__main__":
    main()
