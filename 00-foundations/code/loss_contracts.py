"""Small CPU examples of loss, masking, and reduction contracts; not training kernels."""

import torch


def _finite_float(values, name):
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain finite floating-point values")


def _hard_terms(logits, targets):
    if logits.ndim < 2 or logits.shape[-1] < 2:
        raise ValueError("logits must have shape (..., classes), with at least two classes")
    if targets.shape != logits.shape[:-1] or targets.dtype != torch.long:
        raise ValueError("targets must be long integers with shape logits.shape[:-1]")
    if targets.device != logits.device:
        raise ValueError("logits and targets must share a device")
    valid = targets != -100
    if not valid.any():
        raise ValueError("there are no supervised targets")
    selected_logits = logits[valid]
    selected_targets = targets[valid]
    _finite_float(selected_logits, "supervised logits")
    if ((selected_targets < 0) | (selected_targets >= logits.shape[-1])).any():
        raise ValueError("a supervised target is outside the class range")
    shifted = selected_logits - selected_logits.max(dim=-1, keepdim=True).values
    log_partition = shifted.exp().sum(dim=-1).log()
    target_scores = shifted.gather(-1, selected_targets.unsqueeze(-1)).squeeze(-1)
    return log_partition - target_scores


def hard_cross_entropy(logits, targets):
    """Mean over valid targets, with the class axis last and -100 ignored."""
    return _hard_terms(logits, targets).mean()


def soft_cross_entropy(logits, target_probabilities):
    """Unweighted mean CE for probability targets; no ignore-index convention."""
    if logits.ndim < 2 or logits.shape[-1] < 2 or logits.numel() == 0:
        raise ValueError("expected nonempty logits with a final class axis")
    if target_probabilities.shape != logits.shape or target_probabilities.device != logits.device:
        raise ValueError("soft targets must match logits shape and device")
    _finite_float(logits, "logits")
    _finite_float(target_probabilities, "soft targets")
    totals = target_probabilities.sum(dim=-1)
    if (target_probabilities < 0).any() or not torch.allclose(totals, torch.ones_like(totals)):
        raise ValueError("soft targets must be nonnegative and sum to one")
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_probabilities = shifted - shifted.exp().sum(dim=-1, keepdim=True).log()
    return -(target_probabilities * log_probabilities).sum(dim=-1).mean()


def binary_cross_entropy_with_logits(logits, targets):
    """Stable, unweighted binary CE; mean over all elements."""
    if logits.shape != targets.shape or logits.device != targets.device or logits.numel() == 0:
        raise ValueError("binary targets must match nonempty logits shape and device")
    _finite_float(logits, "logits")
    _finite_float(targets, "targets")
    if ((targets < 0) | (targets > 1)).any():
        raise ValueError("binary targets must lie in [0, 1]")
    zeros = torch.zeros_like(logits)
    positive_loss = torch.logaddexp(zeros, -logits)
    negative_loss = torch.logaddexp(zeros, logits)
    return (targets * positive_loss + (1 - targets) * negative_loss).mean()


def causal_sft_loss(logits, labels, attention_mask):
    """Shift once, then average over valid next-token targets in an unpacked batch."""
    if logits.ndim != 3 or logits.shape[1] < 2:
        raise ValueError("logits must be [batch, sequence >= 2, vocabulary]")
    if labels.shape != logits.shape[:2] or attention_mask.shape != labels.shape:
        raise ValueError("labels and attention_mask must match [batch, sequence]")
    if labels.dtype != torch.long or attention_mask.dtype != torch.bool:
        raise ValueError("labels must be long and attention_mask must be boolean")
    if labels.device != logits.device or attention_mask.device != logits.device:
        raise ValueError("inputs must share a device")
    valid_pairs = attention_mask[:, :-1] & attention_mask[:, 1:]
    shifted_labels = labels[:, 1:].masked_fill(~valid_pairs, -100)
    return hard_cross_entropy(logits[:, :-1], shifted_labels)


def dft_token_loss(logits, targets):
    """Token-level DFT with a detached probability weight and a token-count mean."""
    negative_log_probabilities = _hard_terms(logits, targets)
    weights = (-negative_log_probabilities).exp().detach()
    return (weights * negative_log_probabilities).mean()
