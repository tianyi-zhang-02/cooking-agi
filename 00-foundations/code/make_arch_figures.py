"""Architecture diagrams for the foundations chapter.

Each figure answers exactly one visual question (EDITORIAL.md's rule):

  transformer-block.svg  where does the norm sit, and what does that do to the
                         residual path?
  attention-sites.svg    in the 2017 encoder-decoder, where do Q, K and V come
                         from at each of the three attention sites?
  kv-cache.svg           what is actually recomputed during incremental decoding,
                         and why is the mask no longer square?

Run: python make_arch_figures.py
"""

import os

from svgkit import arrow, box, bracket, curve, elbow, svg, text, write

OUT = os.path.join(os.path.dirname(__file__), "..", "assets")

# --------------------------------------------------------------------------- #
# 1. post-norm vs pre-norm
# --------------------------------------------------------------------------- #
def transformer_block():
    """Post-norm vs pre-norm, on a shared grid so the two read as one comparison."""
    W, H = 706, 548
    BW, BH = 190, 34
    ROWS = [155, 209, 263, 317, 371, 425]
    IN_Y, OUT_Y = 462, 120
    R = 12                                            # radius of the (+) nodes
    b = []

    def col(x, title, formula, stack, taps, unbroken):
        cx, lane = x + BW / 2, x - 28
        kinds = [k for k, _ in stack]
        b.append(text(x, 76, title, "ttl"))
        b.append(text(x, 92, formula, "sub"))

        for row, (kind, label) in zip(ROWS, stack):
            if kind == "add":
                b.append(f'<circle class="box-1" cx="{cx}" cy="{row}" r="{R}"/>')
                b.append(text(cx, row + 4.5, "+", "lbl", "middle"))
            else:
                b.append(box(x, row - BH / 2, BW, BH, label, None,
                             "box-0" if kind == "norm" else "box"))

        half = lambda i: R if kinds[i] == "add" else BH / 2
        for i in range(len(ROWS) - 1):                # spine, bottom row -> top row
            b.append(arrow(cx, ROWS[i + 1] - half(i + 1), cx, ROWS[i] + half(i)))
        b.append(arrow(cx, ROWS[0] - half(0), cx, OUT_Y + 8))
        b.append(arrow(cx, IN_Y, cx, ROWS[-1] + half(len(ROWS) - 1)))
        b.append(text(cx, OUT_Y, "block output", "lbl-s", "middle"))
        b.append(text(cx, IN_Y + 20, "block input", "lbl-s", "middle"))

        for src, dst in taps:                          # residual paths
            b.append(elbow([(cx - 15, src), (lane, src), (lane, dst), (cx - R - 1, dst)],
                           "resid", "a1"))
        if unbroken:
            b.append(f'<path class="resid" stroke-width="2.6" opacity=".9" '
                     f'd="M{lane} {IN_Y} L{lane} {ROWS[0] - R - 2}" '
                     f'marker-end="url(#a1)"/>')

    # post-norm: LayerNorm(x + Sublayer(x)) -- the norm sits ON the residual path
    col(96, "vanilla (2017) · post-norm", "LayerNorm(x + Sublayer(x))",
        [("norm", "LayerNorm"), ("add", "+"), ("sub", "FFN"),
         ("norm", "LayerNorm"), ("add", "+"), ("sub", "Attention")],
        taps=[(IN_Y, 371), (300, 209)], unbroken=False)

    # pre-norm: x + Sublayer(Norm(x)) -- the norm moved inside the branch
    col(452, "modern · pre-norm", "x + Sublayer(Norm(x))",
        [("add", "+"), ("sub", "FFN"), ("norm", "RMSNorm"),
         ("add", "+"), ("sub", "Attention"), ("norm", "RMSNorm")],
        taps=[(IN_Y, 317), (305, 155)], unbroken=True)

    b.insert(0, text(28, 24, "Where the normalisation sits", "ttl"))
    b.insert(1, text(28, 41, "the same two sublayers, wired two ways", "sub"))
    b.append(f'<line class="divider" x1="369" y1="62" x2="369" y2="496"/>')
    b.append(text(96, 508, "Every residual add is followed by a norm, so the gradient",
                  "sub"))
    b.append(text(96, 524, "crosses one on every layer. Hence warmup.", "sub"))
    b.append(text(424, 508, "The ember line runs input to output without", "sub"))
    b.append(text(424, 524, "passing through a norm or a sublayer.", "sub"))
    return svg(W, H, "\n".join(b))


# --------------------------------------------------------------------------- #
# 2. the three attention sites
# --------------------------------------------------------------------------- #
def attention_sites():
    W, H = 700, 300
    b = [text(24, 22, "One module, three wirings", "ttl"),
         text(24, 38, "self-attention and cross-attention are the same class — "
                      "only the three inputs differ", "sub")]

    PW, PH = 204, 172
    for i, (title, q, kv, mask, shape, tint) in enumerate([
        ("encoder self-attention", "src", "src", "padding only · bidirectional",
         "(B, h, S, S)", "box-0"),
        ("decoder self-attention", "tgt", "tgt", "padding ∨ causal",
         "(B, h, T, T)", "box-0"),
        ("cross-attention", "tgt", "encoder output", "src padding",
         "(B, h, T, S)  not square", "box-1"),
    ]):
        x = 24 + i * (PW + 22)
        y = 58
        b.append(f'<rect class="frame" x="{x}" y="{y}" width="{PW}" height="{PH}" rx="9"/>')
        b.append(text(x + PW / 2, y + 20, title, "lbl", "middle"))

        # source is written inside the box, so nothing collides with the arrows
        b.append(box(x + 12, y + 34, 80, 40, "Q", q, tint, 6))
        b.append(box(x + PW - 92, y + 34, 80, 40, "K, V", kv,
                     "box-1" if i == 2 else tint, 6))

        b.append(box(x + 40, y + 96, PW - 80, 28, "attention", None, "box", 6))
        b.append(arrow(x + 52, y + 76, x + 64, y + 94))
        b.append(arrow(x + PW - 52, y + 76, x + PW - 64, y + 94))
        b.append(text(x + PW / 2, y + 142, mask, "lbl-s", "middle"))
        b.append(text(x + PW / 2, y + 157, shape, "mono", "middle"))

    b.append(text(24, 262,
                  "Cross-attention is the only place the two towers touch: the decoder "
                  "asks, the encoder's output answers.", "sub"))
    b.append(text(24, 280,
                  "It is also the only one whose attention matrix is not square — "
                  "T queries against S keys.", "sub"))
    return svg(W, H, "\n".join(b))


# --------------------------------------------------------------------------- #
# 3. what the KV cache changes
# --------------------------------------------------------------------------- #
def kv_cache():
    W, H = 700, 330
    CELL = 17
    b = [text(24, 22, "What the KV cache actually saves", "ttl"),
         text(24, 38, "shaded = attention scores computed at this step", "sub")]

    def grid(x, y, rows, cols, filled, title, sub):
        out = [text(x, y - 26, title, "lbl"), text(x, y - 12, sub, "lbl-s")]
        for r in range(rows):
            for c in range(cols):
                state = filled(r, c)
                cls = {"new": "box-1", "old": "box-q", "none": ""}[state]
                if state == "none":
                    continue
                out.append(f'<rect class="{cls}" x="{x + c * CELL}" y="{y + r * CELL}" '
                           f'width="{CELL - 2}" height="{CELL - 2}" rx="2"/>')
        out.append(f'<rect class="frame" x="{x - 3}" y="{y - 3}" '
                   f'width="{cols * CELL + 4}" height="{rows * CELL + 4}" rx="3"/>')
        out.append(text(x + cols * CELL / 2, y + rows * CELL + 16, "keys →", "lbl-s", "middle"))
        return "\n".join(out)

    T = 8
    b.append(grid(70, 84, T, T, lambda r, c: "new" if c <= r else "none",
                  "prefill · one forward over the whole prompt",
                  f"{T}×{T} causal mask, {T * (T + 1) // 2} scores"))
    b.append(text(56, 84 + T * CELL / 2, "queries", "lbl-s", "end"))

    b.append(grid(330, 84, T + 1, T + 1,
                  lambda r, c: ("new" if r == T and c <= T else
                                ("old" if c <= r and r < T else "none")),
                  "one decode step, with the cache",
                  "1 query × 9 keys — the row in ember is all that is computed"))
    b.append(text(316, 84 + (T + 1) * CELL / 2, "queries", "lbl-s", "end"))

    b.append(text(560, 100, "cached", "lbl-s"))
    b.append(f'<rect class="box-q" x="530" y="90" width="14" height="12" rx="2"/>')
    b.append(text(560, 122, "computed now", "lbl-s"))
    b.append(f'<rect class="box-1" x="530" y="112" width="14" height="12" rx="2"/>')

    b.append(text(24, 262,
                  "Without the cache each new token re-runs the whole prefix: the "
                  "greyed cells get recomputed every step.", "sub"))
    b.append(text(24, 280,
                  "The subtle part is the mask. Query 9 sits at absolute position 9 "
                  "while keys run 1..9, so the mask is (1, 9) — not square —", "sub"))
    b.append(text(24, 296,
                  "and RoPE's cos/sin must be sliced from position 9, not from 0.", "sub"))
    return svg(W, H, "\n".join(b))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("drawing architecture figures...")
    write(OUT, "transformer-block.svg", transformer_block())
    write(OUT, "attention-sites.svg", attention_sites())
    write(OUT, "kv-cache.svg", kv_cache())
    print("done.")
