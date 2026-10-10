"""CPU teaching examples: sparse dispatch, grouped selection, and balance losses."""

import torch
from torch import nn
from torch.nn import functional as functional


def selected_gates(logits, top_k, normalization="selected"):
    if logits.ndim != 2 or not 1 <= top_k <= logits.shape[-1]:
        raise ValueError("Expected [tokens, experts] logits and a valid top_k")
    if normalization not in ("selected", "full"):
        raise ValueError("normalization must be selected or full")
    values, indices = logits.topk(top_k, dim=-1)
    if normalization == "selected":
        weights = values.softmax(dim=-1)
    else:
        weights = logits.softmax(dim=-1).gather(-1, indices)
    return indices, weights


def sparse_moe(hidden, logits, experts, top_k, normalization="selected"):
    if hidden.ndim != 2 or logits.shape != (hidden.shape[0], len(experts)):
        raise ValueError("Expected [tokens, width] inputs and one logit per expert")
    indices, weights = selected_gates(logits, top_k, normalization)
    output = torch.zeros_like(hidden)
    for expert_id, expert in enumerate(experts):
        token_ids, slots = torch.where(indices == expert_id)
        if token_ids.numel() == 0:
            continue
        expert_inputs = hidden.index_select(0, token_ids)
        expert_outputs = expert(expert_inputs)
        weighted = expert_outputs * weights[token_ids, slots, None]
        output = output.index_add(0, token_ids, weighted)
    return output


def group_limited_topk(scores, group_count, groups_to_keep, top_k, group_top_k=1):
    if scores.ndim != 2 or group_count < 1 or scores.shape[-1] % group_count:
        raise ValueError("Experts must divide evenly into nonempty groups")
    group_width = scores.shape[-1] // group_count
    if not 1 <= groups_to_keep <= group_count or not 1 <= group_top_k <= group_width:
        raise ValueError("Invalid number of groups or group scoring width")
    if not 1 <= top_k <= groups_to_keep * group_width:
        raise ValueError("Selected groups must contain at least top_k experts")
    grouped = scores.reshape(scores.shape[0], group_count, group_width)
    group_scores = grouped.topk(group_top_k, dim=-1).values.sum(dim=-1)
    group_ids = group_scores.topk(groups_to_keep, dim=-1).indices
    allowed_groups = torch.zeros_like(group_scores, dtype=torch.bool)
    allowed_groups.scatter_(1, group_ids, True)
    allowed = allowed_groups.repeat_interleave(group_width, dim=-1)
    return scores.masked_fill(~allowed, -torch.inf).topk(top_k, dim=-1).indices


def balance_statistics(logits, valid, top_k, scope="sequence", scoring="softmax"):
    if logits.ndim != 3 or valid.shape != logits.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("Expected [batch, time, experts] logits and a boolean token mask")
    if not 1 <= top_k <= logits.shape[-1]:
        raise ValueError("Invalid top_k")
    if scope not in ("sequence", "batch") or scoring not in ("softmax", "sigmoid"):
        raise ValueError("Unknown aggregation scope or scoring function")
    if logits.shape[0] == 0 or torch.any(valid.sum(dim=1) == 0):
        raise ValueError("Every sequence must contain at least one valid token")
    clean_logits = logits.masked_fill(~valid[..., None], 0)
    if scoring == "softmax":
        probabilities = clean_logits.softmax(dim=-1)
    else:
        probabilities = functional.logsigmoid(clean_logits).softmax(dim=-1)
    indices = clean_logits.topk(top_k, dim=-1).indices
    assignments = functional.one_hot(indices, logits.shape[-1]).sum(dim=-2)
    mask = valid[..., None].to(logits.dtype)
    axes = (1,) if scope == "sequence" else (0, 1)
    token_count = mask.sum(dim=axes)
    fractions = (assignments * mask).sum(dim=axes) / (top_k * token_count)
    mean_probabilities = (probabilities * mask).sum(dim=axes) / token_count
    return fractions, mean_probabilities


def balance_loss(logits, valid, top_k, scope="sequence", scoring="softmax", alpha=1.0):
    fractions, probabilities = balance_statistics(logits, valid, top_k, scope, scoring)
    return alpha * logits.shape[-1] * (fractions * probabilities).sum(dim=-1).mean()


def main():
    torch.manual_seed(7)
    hidden = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    router = nn.Linear(4, 3, bias=False, dtype=torch.float64)
    experts = nn.ModuleList([
        nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Linear(6, 4)).double()
        for expert_id in range(3)
    ])
    output = sparse_moe(hidden, router(hidden), experts, top_k=2)
    output.square().mean().backward()
    print("output shape:", tuple(output.shape))
    print("router receives gradient:", router.weight.grad.abs().sum().item() > 0)
    logits = torch.tensor([[[0.9, 0.1]] * 4, [[0.1, 0.9]] * 4]).double().log()
    valid = torch.ones(2, 4, dtype=torch.bool)
    for scope in ("sequence", "batch"):
        print(scope, "balance loss:", round(balance_loss(logits, valid, 1, scope).item(), 4))


if __name__ == "__main__":
    main()
