"""BatchNorm, LayerNorm and RMSNorm on the same activations.

Training-mode BatchNorm couples samples through its reduction axes. With one
value per channel PyTorch raises an error, whereas evaluation with running
statistics still works. Temporal BatchNorm can train with one sequence but
couples its timesteps; these are different constraints.

Run: python norm_compare.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)
B, T, D = 4, 5, 8                      # batch, sequence, features


def show(name, x, axis_note):
    print(f"  {name:12s} {axis_note}")
    print(f"               mean {x.mean():+.4f}   std {x.std():.4f}")


# --------------------------------------------------------------------------- #
x = torch.randn(B, D) * 3 + 1
print(f"input {tuple(x.shape)}  (rows = tokens, cols = features)\n")

print("what each one normalises:")
bn_manual = (x - x.mean(0, keepdim=True)) / (x.var(0, unbiased=False, keepdim=True) + 1e-5).sqrt()
ln_manual = (x - x.mean(1, keepdim=True)) / (x.var(1, unbiased=False, keepdim=True) + 1e-5).sqrt()
rms_manual = x / x.pow(2).mean(1, keepdim=True).add(1e-5).sqrt()

show("BatchNorm", bn_manual, "over dim 0 (the batch) -> one mean per FEATURE")
show("LayerNorm", ln_manual, "over dim 1 (features)  -> one mean per TOKEN")
show("RMSNorm", rms_manual, "over dim 1, no mean removed")

# the library agrees
bn, ln = nn.BatchNorm1d(D), nn.LayerNorm(D)
bn.train()
print(f"\n  matches nn.BatchNorm1d : {(bn(x) - bn_manual).abs().max():.2e}")
print(f"  matches nn.LayerNorm   : {(ln(x) - ln_manual).abs().max():.2e}")

# after LayerNorm every ROW has mean 0; after BatchNorm every COLUMN does
print(f"\n  LayerNorm: per-row means  {ln_manual.mean(1).abs().max():.2e}  (all ~0)")
print(f"  BatchNorm: per-col means  {bn_manual.mean(0).abs().max():.2e}  (all ~0)")
print(f"  RMSNorm:   per-row means  {rms_manual.mean(1).abs().max():.4f}  "
      f"(NOT 0 -- it never subtracted one)")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 68)
print("why this matters: does the output depend on the rest of the batch?\n")

row = x[:1]
bn.eval()                                    # use running stats, as at inference
ln.eval()

alone_bn, alone_ln = bn(row), ln(row)
with_others_bn, with_others_ln = bn(x)[:1], ln(x)[:1]
print(f"  BatchNorm(eval)  row alone vs in a batch : "
      f"{(alone_bn - with_others_bn).abs().max():.2e}   (running stats, so stable)")
print(f"  LayerNorm        row alone vs in a batch : "
      f"{(alone_ln - with_others_ln).abs().max():.2e}")

bn.train()                                   # training mode uses batch stats
train_alone = bn(x[:2])[:1]                  # a different batch composition
train_batch = bn(x)[:1]
print(f"  BatchNorm(train) same row, two batches   : "
      f"{(train_alone - train_batch).abs().max():.4f}   <- output CHANGED")

# --------------------------------------------------------------------------- #
print("\nbatch size 1, training mode:")
try:
    bn(x[:1])
    print("  BatchNorm ran (unexpected)")
except ValueError as e:
    print(f"  BatchNorm -> ValueError: {e}")
    print("  PyTorch refuses outright rather than returning something wrong:")
    print("  with one sample the batch variance is 0 and x - mean is 0, so the")
    print("  normalized output would contain only the learned bias.")
    print("  This is a training-mode [1, C] restriction, not an inference rule.")
print(f"  LayerNorm -> {ln(x[:1]).detach().numpy().round(3)[0][:5]} ...  (unbothered)")

sequence = torch.randn(1, D, T)
temporal_bn = nn.BatchNorm1d(D)
temporal_output = temporal_bn(sequence)
print(f"\n  BatchNorm(train) [1, C, L] -> {tuple(temporal_output.shape)}")
print("  There are L values per channel, but statistics now mix timesteps.")
bn.eval()
print(f"  BatchNorm(eval) [1, C] -> {tuple(bn(x[:1]).shape)} (running statistics)")
