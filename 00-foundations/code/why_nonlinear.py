"""From logistic regression to a neural net, on the problem that forced the jump.

XOR: four blobs, diagonal pairs share a label. No straight line separates them.
(This is the example Minsky & Papert used in 1969 to show what a perceptron
cannot do -- it stalled the field for over a decade.)

Three models, trained identically:
  A  logistic regression                  Linear(2,1)
  B  a "deep" net with NO activation      Linear(2,8) -> Linear(8,1)
  C  the same net with one ReLU           Linear(2,8) -> ReLU -> Linear(8,1)

B is the whole lesson. B and C are the SAME architecture with the SAME 33
parameters; the only difference is one ReLU. B is exactly as weak as the
3-parameter logistic regression, because W2(W1 x) = (W2 W1) x is still one
linear map -- depth without a nonlinearity buys nothing at all.

Run: python why_nonlinear.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
def xor_data(n_per_blob=100):
    """4 gaussian blobs at (+-1.5, +-1.5); label = 1 iff the signs agree."""
    centers = torch.tensor([[1.5, 1.5], [-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5]])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    X = torch.cat([c + 0.45 * torch.randn(n_per_blob, 2) for c in centers])
    y = torch.cat([l.repeat(n_per_blob) for l in labels])
    return X, y


def train(model, X, y, steps=3000, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()  # sigmoid + cross-entropy, fused
    for _ in range(steps):
        loss = lossf(model(X).squeeze(-1), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = ((model(X).squeeze(-1) > 0).float() == y).float().mean().item()
    return loss.item(), acc


# --------------------------------------------------------------------------- #
def render(score_fn, pts, y, title, lim=None, w=57, h=21):
    """ASCII map: shade the plane by what `score_fn` predicts, overlay the points.

    '#' = predicted class 1 region, '.' = predicted class 0, 'O'/'x' = data.
    Used twice -- once in the INPUT space, once in the hidden layer's space.
    """
    if lim is None:  # fit the box to the data
        lo, hi = pts.min(0).values - 0.15, pts.max(0).values + 0.15
    else:
        lo = torch.tensor([-lim, -lim])
        hi = torch.tensor([lim, lim])
    span = (hi - lo).clamp(min=1e-6)

    xs = torch.linspace(lo[0], hi[0], w)
    ys = torch.linspace(hi[1], lo[1], h)  # top row = max
    grid = torch.stack([xs.repeat(h), ys.repeat_interleave(w)], dim=1)
    with torch.no_grad():
        pred = (score_fn(grid).squeeze(-1) > 0).reshape(h, w)
    canvas = [["#" if pred[i, j] else "." for j in range(w)] for i in range(h)]

    for (px, py), lbl in zip(pts.tolist(), y.tolist()):
        j = round((px - lo[0].item()) / span[0].item() * (w - 1))
        i = round((hi[1].item() - py) / span[1].item() * (h - 1))
        if 0 <= i < h and 0 <= j < w:
            canvas[i][j] = "O" if lbl > 0.5 else "x"

    print(f"\n  {title}")
    for row in canvas:
        print("    " + "".join(row))


# --------------------------------------------------------------------------- #
X, y = xor_data()
print("XOR: 'O' is class 1 (top-right + bottom-left), 'x' is class 0.")
print("No straight line puts all the O's on one side.\n")

# ---- A: logistic regression ------------------------------------------------
A = nn.Linear(2, 1)
loss_a, acc_a = train(A, X, y)
print(f"A  logistic regression      {sum(p.numel() for p in A.parameters()):3d} params"
      f"   loss {loss_a:.4f}   acc {acc_a:.1%}")

# ---- B: deep, but linear ---------------------------------------------------
B = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 1))
loss_b, acc_b = train(B, X, y)
print(f"B  Linear->Linear, no act.  {sum(p.numel() for p in B.parameters()):3d} params"
      f"   loss {loss_b:.4f}   acc {acc_b:.1%}   <- 11x the params of A, same power")

# ---- C: one tanh away from B -----------------------------------------------
C = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
loss_c, acc_c = train(C, X, y)
print(f"C  Linear->ReLU->Linear     {sum(p.numel() for p in C.parameters()):3d} params"
      f"   loss {loss_c:.4f}   acc {acc_c:.1%}   <- same params as B, one ReLU apart")

# ---- why B collapses -------------------------------------------------------
W1, W2 = B[0].weight, B[1].weight
print(f"\nB's two weight matrices are {tuple(W2.shape)} @ {tuple(W1.shape)}, and their")
print(f"product is {tuple((W2 @ W1).shape)}: {(W2 @ W1).detach().flatten().tolist()}")
print("That single 1x2 row is the ONLY thing B can express -- one hyperplane,")
print("identical in form to A. Composing linear maps gives a linear map.")

render(A, X, y, "A: logistic regression -- boundary is a straight line", lim=3.0)
render(B, X, y, "B: deep linear net -- still a straight line", lim=3.0)
render(C, X, y, "C: same net, one ReLU added -- the boundary can bend", lim=3.0)

# ---- what the hidden layer actually did ------------------------------------
# A separate 2-unit net: 2 hidden units is simply what fits on a 2D page.
torch.manual_seed(1)
V = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
loss_v, acc_v = train(V, X, y, steps=1500)
print(f"\n(for the picture below: a 2-unit version, acc {acc_v:.1%})")
with torch.no_grad():
    hidden = V[1](V[0](X))
# the output layer is Linear(2,1) -- it draws ONE straight line in hidden space
render(V[2], hidden, y,
       "C's hidden layer space — same points after tanh warp, with the output\n"
       "  layer's boundary (one straight line) drawn on top:")
print("""
    The hidden layer warped the plane until the classes fell on opposite
    sides of ONE straight line. The output layer is then just logistic
    regression again -- on coordinates that were learned rather than given.

    That is all a neural net is: (learned change of coordinates) -> (linear classifier).
    Stack more layers = warp repeatedly. Every architecture -- CNN, transformer --
    is a different prior on what kind of warping is allowed.

    In the transformer we built: `lm_head` / `generator` is the logistic
    regression (softmax over the vocabulary). Every attention block below it
    exists to bend the space until the next token is linearly readable.""")
