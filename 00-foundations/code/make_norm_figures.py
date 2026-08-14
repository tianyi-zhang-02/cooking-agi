"""Figures for the normalisation and residual notes.

  norm-axes.svg        which numbers does each norm average over? (schematic)
  residual-gradient.svg what actually happens to the gradient with depth? (measured)

The second one is a real run, not a drawing: a 40-layer MLP, forward and
backward, with and without residual connections, plotting the gradient norm
reaching each layer. The exponential decay is measured.

Run: python make_norm_figures.py
"""

import math
import os

import torch
import torch.nn as nn

from svgkit import arrow, box, esc, svg, text, write

OUT = os.path.join(os.path.dirname(__file__), "..", "assets")


# --------------------------------------------------------------------------- #
# 1. which axis gets normalised  (schematic -- this one is a definition)
# --------------------------------------------------------------------------- #
def norm_axes():
    W, H = 720, 320
    CELL, ROWS, COLS = 20, 5, 6
    b = [text(24, 24, "What each norm averages over", "ttl"),
         text(24, 41, "one row = one token's feature vector · "
                      "one column = the same feature across the batch", "sub")]

    def panel(x, title, sub, hot):
        b.append(text(x, 78, title, "lbl"))
        b.append(text(x, 93, sub, "lbl-s"))
        for r in range(ROWS):
            for c in range(COLS):
                cls = "box-1" if hot(r, c) else "box-q"
                b.append(f'<rect class="{cls}" x="{x + c * CELL}" y="{110 + r * CELL}" '
                         f'width="{CELL - 2}" height="{CELL - 2}" rx="2"/>')
        b.append(f'<rect class="frame" x="{x - 3}" y="{107}" '
                 f'width="{COLS * CELL + 4}" height="{ROWS * CELL + 4}" rx="3"/>')

    P = 218
    panel(40, "BatchNorm", "one feature, across the batch",
          lambda r, c: c == 2)
    panel(40 + P, "LayerNorm", "one token, across its features",
          lambda r, c: r == 2)
    panel(40 + 2 * P, "RMSNorm", "same axis, but no mean removed",
          lambda r, c: r == 2)

    # axis labels on the first panel only
    b.append(text(34, 110 + ROWS * CELL / 2, "tokens", "lbl-s", "end"))
    b.append(text(40 + COLS * CELL / 2, 110 + ROWS * CELL + 16,
                  "features →", "lbl-s", "middle"))

    b.append(text(24, 252,
                  "BatchNorm's statistics depend on the other examples in the batch. "
                  "LayerNorm's depend only on the token itself —", "sub"))
    b.append(text(24, 268,
                  "which is why variable-length sequences, batch size 1, and "
                  "autoregressive decoding all leave BatchNorm without a usable", "sub"))
    b.append(text(24, 284,
                  "estimate, and LayerNorm completely unbothered.", "sub"))
    b.append(text(24, 306,
                  "RMSNorm keeps LayerNorm's axis and drops the mean subtraction: "
                  "one fewer reduction, no measurable loss.", "sub"))
    return svg(W, H, "\n".join(b))


# --------------------------------------------------------------------------- #
# 2. what depth does to the gradient  (measured)
# --------------------------------------------------------------------------- #
def measure_gradients(depth=40, width=64, residual=False, seed=0, gain=0.8):
    """Forward+backward a deep MLP, return the gradient norm arriving at each layer.

    gain=0.8 puts the init 20% below the critical value for tanh. At exactly
    critical (gain=1.0) a plain stack sits on a knife's edge and neither vanishes
    nor explodes -- which would make a misleadingly flat picture. Real training
    is never exactly at the critical point, and the whole practical value of the
    residual connection is that it stops caring where you are.
    """
    torch.manual_seed(seed)
    layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
    for lin in layers:
        nn.init.normal_(lin.weight, 0.0, gain / math.sqrt(width))
        nn.init.zeros_(lin.bias)

    x = torch.randn(64, width)
    acts = []
    h = x
    for lin in layers:
        h = h.detach().requires_grad_(True) if False else h
        h.retain_grad() if h.requires_grad else None
        pre = torch.tanh(lin(h))
        h = h + pre if residual else pre
        h.retain_grad()
        acts.append(h)
    loss = h.pow(2).mean()
    loss.backward()
    return [a.grad.norm().item() for a in acts]


def residual_gradient():
    plain = measure_gradients(residual=False)
    res = measure_gradients(residual=True)
    depth = len(plain)

    W, H = 700, 350
    L, T = 66, 76
    PW, PH = W - L - 150, H - T - 92
    lo = min(min(v for v in plain if v > 0), min(res)) * 0.5
    hi = max(max(plain), max(res)) * 2
    lx = lambda i: L + i / (depth - 1) * PW
    ly = lambda v: T + PH - (math.log10(max(v, lo)) - math.log10(lo)) / \
        (math.log10(hi) - math.log10(lo)) * PH

    b = [text(24, 24, "What depth does to the gradient", "ttl"),
         text(24, 41, f"{depth}-layer MLP, tanh, init 20% below critical — "
                      "identical weights and input, the only difference is the "
                      "residual add", "sub")]

    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        y = ly(10.0 ** e)
        if T <= y <= T + PH:
            b.append(f'<line class="divider" x1="{L}" y1="{y:.1f}" x2="{L + PW}" '
                     f'y2="{y:.1f}" stroke-dasharray="2 4" opacity=".6"/>')
            b.append(text(L - 8, y + 3.5, f"1e{e}", "lbl-s", "end"))

    for series, cls, name in ((plain, "arrow-0", "plain stack"),
                              (res, "arrow-1", "with residual")):
        pts = " ".join(f"{lx(i):.1f},{ly(v):.1f}" for i, v in enumerate(series))
        b.append(f'<polyline class="{cls}" points="{pts}" fill="none" stroke-width="2"/>')

    b.append(f'<rect class="frame" x="{L}" y="{T}" width="{PW}" height="{PH}"/>')
    for i in (0, depth // 2, depth - 1):
        b.append(text(lx(i), T + PH + 17, str(i + 1), "lbl-s", "middle"))
    b.append(text(L + PW / 2, T + PH + 34, "layer (1 = closest to the input)",
                  "lbl-s", "middle"))
    b.append(text(L, T - 9, "‖grad‖ reaching this layer", "lbl-s"))

    b.append(f'<line x1="{L + PW + 16}" y1="{ly(res[0])}" x2="{L + PW + 34}" '
             f'y2="{ly(res[0])}" class="arrow-1" stroke-width="2"/>')
    b.append(text(L + PW + 40, ly(res[0]) + 4, "with residual", "lbl-s"))
    b.append(f'<line x1="{L + PW + 16}" y1="{ly(plain[0])}" x2="{L + PW + 34}" '
             f'y2="{ly(plain[0])}" class="arrow-0" stroke-width="2"/>')
    b.append(text(L + PW + 40, ly(plain[0]) + 4, "plain stack", "lbl-s"))

    decay = plain[-1] / max(plain[0], 1e-30)   # layer 40 -> layer 1
    ratio = res[0] / max(plain[0], 1e-30)
    b.append(text(24, H - 26,
                  f"Walking back from layer {depth} to layer 1 the plain stack loses "
                  f"{decay:.0e}× of its gradient; the residual one is flat.", "sub"))
    b.append(text(24, H - 10,
                  f"At layer 1 the two differ by {ratio:.0e}×. Same weights, same "
                  "data — the identity path is all of it.", "sub"))
    return svg(W, H, "\n".join(b)), plain, res


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("drawing normalisation + residual figures...")
    write(OUT, "norm-axes.svg", norm_axes())
    fig, plain, res = residual_gradient()
    write(OUT, "residual-gradient.svg", fig)
    print(f"\n  measured gradient norm at layer 1:")
    print(f"    plain stack    {plain[0]:.3e}")
    print(f"    with residual  {res[0]:.3e}")
    print(f"    ratio          {res[0] / max(plain[0], 1e-30):.2e}")
    print("done.")
