"""Generate the figures for `from-linear-to-neural.md` from real trained models.

Nothing here is hand-drawn. Every boundary, contour and point comes out of an
actual training run, so the pictures cannot drift away from the claims in the
text. Run this and the .svg files in ../assets/ are rebuilt.

Output SVGs are theme-aware: each carries its own <style> with a
prefers-color-scheme block, so they read correctly on GitHub in light and dark
mode and on the site.

Run: python make_figures.py
"""

import math
import os

import torch
import torch.nn as nn

torch.manual_seed(0)
OUT = os.path.join(os.path.dirname(__file__), "..", "assets")

STYLE = """
  :root {
    --ink: #1c1e22; --dim: #8a857e; --rule: #d9d5ce; --warp: #9aa0a6;
    --c0: #2f5d7c; --c1: #c8501e;
    --f0: rgba(47,93,124,0.09); --f1: rgba(200,80,30,0.12);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ink: #e6e4e0; --dim: #7d766d; --rule: #35322d; --warp: #5d5952;
      --c0: #74a9d8; --c1: #e8794f;
      --f0: rgba(116,169,216,0.12); --f1: rgba(232,121,79,0.15);
    }
  }
  text { font-family: Charter, "Bitstream Charter", Cambria, Georgia, serif; fill: var(--ink); }
  .ttl  { font-size: 13px; font-weight: 700; }
  .sub  { font-size: 10.5px; fill: var(--dim); font-style: italic; }
  .tick { font-size: 9px; fill: var(--dim); }
  .frame { fill: none; stroke: var(--rule); stroke-width: 1; }
  .contour { fill: none; stroke: var(--ink); stroke-width: 1.9;
             stroke-linecap: round; stroke-linejoin: round; }
  .warp { fill: none; stroke: var(--warp); stroke-width: 0.55; opacity: 0.55; }
  .r0 { fill: var(--f0); } .r1 { fill: var(--f1); }
  .p0 { fill: var(--c0); } .p1 { fill: var(--c1); }
"""


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img">\n<style>{STYLE}</style>\n{body}\n</svg>\n')


def write(name, content):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote {os.path.basename(path):26s} {len(content) / 1024:5.1f} KB")


# --------------------------------------------------------------------------- #
# contour extraction + segment joining
# --------------------------------------------------------------------------- #
def cell_segments(F, i, j):
    """Crossings of the 0-level set on one grid cell's four edges."""
    pts = []
    for r1, c1, r2, c2 in ((i, j, i, j + 1), (i, j + 1, i + 1, j + 1),
                           (i + 1, j + 1, i + 1, j), (i + 1, j, i, j)):
        a, b = F[r1][c1], F[r2][c2]
        if (a > 0) != (b > 0):
            t = a / (a - b)
            pts.append((c1 + t * (c2 - c1), r1 + t * (r2 - r1)))
    if len(pts) == 2:
        return [(pts[0], pts[1])]
    if len(pts) == 4:  # saddle -- either pairing is defensible at this resolution
        return [(pts[0], pts[1]), (pts[2], pts[3])]
    return []


def join(segments, tol=1e-6):
    """Chain loose segments into polylines so the SVG stays small and smooth."""
    key = lambda p: (round(p[0] / tol), round(p[1] / tol))
    ends = {}
    for s in segments:
        for a, b in ((s[0], s[1]), (s[1], s[0])):
            ends.setdefault(key(a), []).append((a, b))
    used, chains = set(), []
    for seg in segments:
        if id(seg) in used:
            continue
        used.add(id(seg))
        chain = [seg[0], seg[1]]
        for _ in range(2):  # extend forward, then reverse and extend again
            while True:
                nxt = None
                for a, b in ends.get(key(chain[-1]), []):
                    cand = [s for s in segments if id(s) not in used
                            and (key(s[0]) == key(chain[-1]) or key(s[1]) == key(chain[-1]))]
                    if cand:
                        nxt = cand[0]
                    break
                if nxt is None:
                    break
                used.add(id(nxt))
                chain.append(nxt[1] if key(nxt[0]) == key(chain[-1]) else nxt[0])
            chain.reverse()
        chains.append(chain)
    return chains


def region_runs(F):
    """Run-length encode the positive region row by row -> (row, col_start, col_end)."""
    runs, n = [], len(F)
    for i in range(n):
        j = 0
        while j < n:
            if F[i][j] > 0:
                k = j
                while k < n and F[i][k] > 0:
                    k += 1
                runs.append((i, j, k))
                j = k
            else:
                j += 1
    return runs


# --------------------------------------------------------------------------- #
def panel(score_fn, pts, labels, x, y, size, title, subtitle, box, n=60, warp=None):
    """One square plot: shaded decision region, contour, optional warped grid, points."""
    x0, x1, y0, y1 = box
    sx = lambda px: x + (px - x0) / (x1 - x0) * size
    sy = lambda py: y + (y1 - py) / (y1 - y0) * size

    gxs = [x0 + (x1 - x0) * c / (n - 1) for c in range(n)]
    gys = [y1 - (y1 - y0) * r / (n - 1) for r in range(n)]
    grid = torch.tensor([[gx, gy] for gy in gys for gx in gxs])
    with torch.no_grad():
        flat = score_fn(grid).squeeze(-1).tolist()
    F = [flat[r * n:(r + 1) * n] for r in range(n)]

    cx = lambda c: x + c / (n - 1) * size
    cy = lambda r: y + r / (n - 1) * size

    out = [f'<text class="ttl" x="{x}" y="{y - 21}">{title}</text>',
           f'<text class="sub" x="{x}" y="{y - 8}">{subtitle}</text>',
           f'<g clip-path="url(#clip{x}_{y})">',
           f'<rect class="r0" x="{x}" y="{y}" width="{size}" height="{size}"/>']

    cell = size / (n - 1)
    for r, c0, c1 in region_runs(F):
        out.append(f'<rect class="r1" x="{cx(c0) - cell / 2:.1f}" y="{cy(r) - cell / 2:.1f}" '
                   f'width="{(c1 - c0) * cell:.1f}" height="{cell:.1f}"/>')

    if warp:
        for line in warp:
            pts_s = " ".join(f"{sx(px):.1f},{sy(py):.1f}" for px, py in line)
            out.append(f'<polyline class="warp" points="{pts_s}"/>')

    segs = [s for i in range(n - 1) for j in range(n - 1) for s in cell_segments(F, i, j)]
    for chain in join(segs):
        pts_s = " ".join(f"{cx(c):.1f},{cy(r):.1f}" for c, r in chain)
        out.append(f'<polyline class="contour" points="{pts_s}"/>')

    for (px, py), lbl in zip(pts.tolist(), labels.tolist()):
        out.append(f'<circle class="{"p1" if lbl > 0.5 else "p0"}" '
                   f'cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="2"/>')

    out.append("</g>")
    out.append(f'<rect class="frame" x="{x}" y="{y}" width="{size}" height="{size}"/>')
    out.insert(0, f'<defs><clipPath id="clip{x}_{y}"><rect x="{x}" y="{y}" '
                  f'width="{size}" height="{size}"/></clipPath></defs>')
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def xor_data(n_per_blob=90):
    centers = torch.tensor([[1.5, 1.5], [-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5]])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    X = torch.cat([c + 0.42 * torch.randn(n_per_blob, 2) for c in centers])
    y = torch.cat([l.repeat(n_per_blob) for l in labels])
    return X, y


def train(model, X, y, steps=3000, lr=0.05, wd=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        loss = lossf(model(X).squeeze(-1), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = ((model(X).squeeze(-1) > 0).float() == y).float().mean().item()
    return acc


def train_until(model, X, y, target=0.985, max_steps=6000, lr=0.05):
    """Stop as soon as the problem is solved -- a fully converged net saturates its
    activations and its hidden space collapses onto the edges, which plots badly."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for step in range(max_steps):
        loss = lossf(model(X).squeeze(-1), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = ((model(X).squeeze(-1) > 0).float() == y).float().mean().item()
        if acc >= target:
            return acc, step
    return acc, max_steps


print("training...")
X, y = xor_data()

# A, B and C are the experiment: B and C are the SAME architecture with the same
# parameter count, differing by exactly one ReLU.
A = nn.Linear(2, 1)
B = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 1))
C = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
acc_a, acc_b, acc_c = train(A, X, y), train(B, X, y), train(C, X, y)
n_par = lambda m: sum(p.numel() for p in m.parameters())
print(f"  A logistic regression        {n_par(A):3d} params  {acc_a:.1%}")
print(f"  B Linear -> Linear           {n_par(B):3d} params  {acc_b:.1%}")
print(f"  C Linear -> ReLU -> Linear   {n_par(C):3d} params  {acc_c:.1%}")

# A separate 2-unit net for the hidden-space picture: 2 hidden units is simply
# what fits on a 2D page. ReLU is unbounded, so the warped grid stays legible
# instead of being crushed onto the edges of tanh's square.
for seed in range(24):
    torch.manual_seed(seed)
    V = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    acc_v, steps_v = train_until(V, X, y)
    if acc_v >= 0.98:
        break
with torch.no_grad():
    H = V[1](V[0](X))
print(f"  V 2-unit ReLU (for the plot)   9 params  {acc_v:.1%}  "
      f"seed {seed}, {steps_v} steps\n")

# --------------------------------------------------------------------------- #
# figure 1 -- three decision boundaries
# --------------------------------------------------------------------------- #
S, GAP, PAD, TOP = 190, 34, 16, 46
BOX = (-3.0, 3.0, -3.0, 3.0)
body = [panel(A, X, y, PAD, TOP, S, "A &#183; logistic regression",
              f"3 params &#183; {acc_a:.0%} accuracy", BOX),
        panel(B, X, y, PAD + S + GAP, TOP, S, "B &#183; Linear &#8594; Linear",
              f"33 params, no activation &#183; {acc_b:.0%}", BOX),
        panel(C, X, y, PAD + 2 * (S + GAP), TOP, S,
              "C &#183; Linear &#8594; ReLU &#8594; Linear",
              f"33 params &#183; {acc_c:.0%} accuracy", BOX)]
body.append(f'<text class="sub" x="{PAD}" y="{TOP + S + 22}">'
            'black curve = decision boundary &#183; shading = predicted class &#183; '
            'B and C are the same architecture with the same 33 parameters, '
            'differing by one ReLU</text>')
write("decision-boundaries.svg", svg(PAD * 2 + 3 * S + 2 * GAP, TOP + S + 34, "\n".join(body)))

# --------------------------------------------------------------------------- #
# figure 2 -- the warp, shown with a grid pushed through the hidden layer
# --------------------------------------------------------------------------- #
NL, NS = 13, 45  # grid lines per axis, samples per line
lin = lambda a, b, k: [a + (b - a) * i / (k - 1) for i in range(k)]
in_grid = ([[(gx, gy) for gy in lin(-3, 3, NS)] for gx in lin(-3, 3, NL)] +
           [[(gx, gy) for gx in lin(-3, 3, NS)] for gy in lin(-3, 3, NL)])
with torch.no_grad():
    warped = [[tuple(v) for v in V[1](V[0](torch.tensor(line))).tolist()] for line in in_grid]

m = 0.06
hbox = (H[:, 0].min().item() - m, H[:, 0].max().item() + m,
        H[:, 1].min().item() - m, H[:, 1].max().item() + m)
span = max(hbox[1] - hbox[0], hbox[3] - hbox[2])
hbox = (hbox[0] - 0.04 * span, hbox[0] + 1.04 * span, hbox[2] - 0.04 * span, hbox[2] + 1.04 * span)

AW = 96
body = [panel(V, X, y, PAD, TOP, S, "input space", "no line separates these", BOX, warp=in_grid),
        panel(V[2], H, y, PAD + S + AW, TOP, S,
              "hidden space &#183; h = ReLU(W&#8321;x + b&#8321;)",
              "one straight line now suffices", hbox, warp=warped)]
ax = PAD + S + 16
body += ['<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
         'markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--dim)"/>'
         '</marker></defs>',
         f'<path d="M{ax} {TOP + S / 2} L{ax + 40} {TOP + S / 2}" stroke="var(--dim)" '
         f'stroke-width="1.4" fill="none" marker-end="url(#a)"/>',
         f'<text class="sub" x="{ax + 2}" y="{TOP + S / 2 - 9}">warp</text>',
         f'<text class="sub" x="{PAD}" y="{TOP + S + 21}">'
         'the grey grid is the same regular grid in both panels. ReLU folds it along a crease,'
         '</text>',
         f'<text class="sub" x="{PAD}" y="{TOP + S + 35}">'
         'and the two classes end up on opposite sides of one straight line.</text>']
write("hidden-space.svg", svg(PAD * 2 + 2 * S + AW + 30, TOP + S + 48, "\n".join(body)))

# --------------------------------------------------------------------------- #
# figure 3 -- sigmoid and its derivative
# --------------------------------------------------------------------------- #
W, Hh, L, T = 520, 212, 54, 24
PW, PH = W - L - 26, Hh - T - 36
sig = lambda z: 1 / (1 + math.exp(-z))
zs = [-8 + 16 * i / 240 for i in range(241)]
px = lambda z: L + (z + 8) / 16 * PW
py = lambda v: T + (1 - v) * PH

body = [f'<line x1="{L}" y1="{py(0)}" x2="{L + PW}" y2="{py(0)}" stroke="var(--rule)"/>',
        f'<line x1="{px(0)}" y1="{T - 6}" x2="{px(0)}" y2="{T + PH + 6}" stroke="var(--rule)"/>']
for v, lab in ((1.0, "1"), (0.5, "&#189;"), (0.0, "0")):
    body.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{L + PW}" y2="{py(v):.1f}" '
                f'stroke="var(--rule)" stroke-dasharray="2 4" opacity="0.7"/>')
    body.append(f'<text class="tick" x="{L - 8}" y="{py(v) + 3:.1f}" text-anchor="end">{lab}</text>')
for lo, hi in ((-8, -4.2), (4.2, 8)):  # saturated zones
    body.append(f'<rect x="{px(lo):.1f}" y="{T}" width="{px(hi) - px(lo):.1f}" height="{PH}" '
                f'fill="var(--dim)" opacity="0.07"/>')
body.append('<polyline class="contour" stroke="var(--c1)" fill="none" points="' +
            " ".join(f"{px(z):.1f},{py(sig(z)):.1f}" for z in zs) + '"/>')
body.append('<polyline fill="none" stroke="var(--c0)" stroke-width="1.6" stroke-dasharray="4 3" '
            'points="' + " ".join(f"{px(z):.1f},{py(sig(z) * (1 - sig(z))):.1f}" for z in zs) + '"/>')
for z in (-8, -4, 0, 4, 8):
    body.append(f'<text class="tick" x="{px(z):.1f}" y="{T + PH + 15}" text-anchor="middle">{z}</text>')
body += [f'<text class="ttl" x="{L}" y="15">the sigmoid, and the gradient it buys</text>',
         f'<text class="tick" x="{px(6.5):.0f}" y="{py(0.90):.0f}" fill="var(--c1)">&#963;(z)</text>',
         f'<text class="tick" x="{px(0.9):.0f}" y="{py(0.30):.0f}" fill="var(--c0)">'
         '&#963;&#8242;(z) = &#963;(1&#8722;&#963;)</text>',
         f'<text class="tick" x="{px(-6.1):.0f}" y="{T + 13}" text-anchor="middle">saturated</text>',
         f'<text class="tick" x="{px(6.1):.0f}" y="{T + 13}" text-anchor="middle">saturated</text>',
         f'<text class="sub" x="{L}" y="{Hh - 7}">'
         'in the shaded zones the derivative is ~0: points far from the boundary, right or wrong, '
         'stop pulling on the weights</text>',
         f'<text class="tick" x="{L + PW + 6}" y="{py(0) + 3:.0f}">z</text>']
write("sigmoid.svg", svg(W, Hh, "\n".join(body)))

print("\ndone.")
