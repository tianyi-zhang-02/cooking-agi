"""A tiny box-and-arrow SVG toolkit shared by the diagram scripts.

Why not draw these by hand, or paste the figures from the papers?

  - Paper figures are someone else's copyright, and they carry someone else's
    visual language. A diagram that matches the site reads as part of the text
    rather than as a screenshot dropped into it.
  - Hand-authored SVG drifts. When the prose changes, nobody re-draws the
    picture, and the two quietly start disagreeing. A generated figure is
    regenerated.
  - These come out theme-aware for free: every colour is a CSS variable with a
    prefers-color-scheme block, so one file reads correctly on GitHub and on the
    site, in light mode and dark.

Colours are the site's own tokens: ink-blue #2f5d7c and ember #c8501e.
"""

import os

# --------------------------------------------------------------------------- #
STYLE = """
  :root {
    --ink: #1b1e23; --dim: #6c727b; --faint: #949aa2;
    --rule: #dee1e5; --panel: #ffffff; --paper: #f5f6f7;
    --c0: #2f5d7c; --c1: #c8501e;
    --f0: rgba(47,93,124,0.09); --f1: rgba(200,80,30,0.10);
    --f2: rgba(120,120,128,0.07);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ink: #e3e5e8; --dim: #8c9199; --faint: #6e747c;
      --rule: #343941; --panel: #1c2026; --paper: #14161a;
      --c0: #74a9d8; --c1: #e8794f;
      --f0: rgba(116,169,216,0.13); --f1: rgba(232,121,79,0.14);
      --f2: rgba(160,168,180,0.07);
    }
  }
  text { font-family: Charter, "Bitstream Charter", Cambria, Georgia, serif;
         fill: var(--ink); }
  .ttl   { font-size: 13.5px; font-weight: 700; }
  .sub   { font-size: 10.5px; fill: var(--dim); font-style: italic; }
  .lbl   { font-size: 11.5px; }
  .lbl-s { font-size: 10px; fill: var(--dim); }
  .mono  { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 10px;
           fill: var(--dim); }
  .box   { fill: var(--panel); stroke: var(--rule); stroke-width: 1.2; }
  .box-0 { fill: var(--f0); stroke: var(--c0); stroke-width: 1.2; }
  .box-1 { fill: var(--f1); stroke: var(--c1); stroke-width: 1.2; }
  .box-q { fill: var(--f2); stroke: var(--rule); stroke-width: 1;
           stroke-dasharray: 4 3; }
  /* An outline-only rect. Needed as its own class: in SVG a CSS rule beats a
     presentation attribute, so class="box" fill="none" still paints the fill
     and would cover whatever it frames. */
  .frame { fill: none; stroke: var(--rule); stroke-width: 1.2; }
  .arrow { stroke: var(--dim); stroke-width: 1.3; fill: none; }
  .arrow-1 { stroke: var(--c1); stroke-width: 1.5; fill: none; }
  .arrow-0 { stroke: var(--c0); stroke-width: 1.5; fill: none; }
  .resid { stroke: var(--c1); stroke-width: 1.5; fill: none; stroke-dasharray: 5 3; }
  .divider { stroke: var(--rule); stroke-width: 1; }
"""

MARKERS = """
<defs>
  <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5"
          orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--dim)"/></marker>
  <marker id="a1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5"
          orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--c1)"/></marker>
  <marker id="a0" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5"
          orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--c0)"/></marker>
</defs>
"""


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img">\n<style>{STYLE}</style>{MARKERS}\n'
            f'{body}\n</svg>\n')


def write(out_dir, name, content):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote {name:32s} {len(content) / 1024:5.1f} KB")


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("→", "&#8594;").replace("×", "&#215;").replace("·", "&#183;")
            .replace("‖", "&#8214;").replace("≈", "&#8776;").replace("−", "&#8722;"))


def box(x, y, w, h, label, sub=None, kind="box", r=7, cls="lbl"):
    """A rounded box with a centred label and an optional second line."""
    out = [f'<rect class="{kind}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/>']
    if sub:
        out.append(f'<text class="{cls}" x="{x + w / 2}" y="{y + h / 2 - 3}" '
                   f'text-anchor="middle">{esc(label)}</text>')
        out.append(f'<text class="lbl-s" x="{x + w / 2}" y="{y + h / 2 + 11}" '
                   f'text-anchor="middle">{esc(sub)}</text>')
    else:
        out.append(f'<text class="{cls}" x="{x + w / 2}" y="{y + h / 2 + 4}" '
                   f'text-anchor="middle">{esc(label)}</text>')
    return "\n".join(out)


def text(x, y, s, cls="lbl", anchor="start"):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{esc(s)}</text>'


def arrow(x1, y1, x2, y2, cls="arrow", marker="a"):
    return (f'<path class="{cls}" d="M{x1} {y1} L{x2} {y2}" '
            f'marker-end="url(#{marker})"/>')


def elbow(pts, cls="arrow", marker="a"):
    """Poly-line arrow through a list of (x, y)."""
    d = " ".join(("M" if i == 0 else "L") + f"{x} {y}" for i, (x, y) in enumerate(pts))
    return f'<path class="{cls}" d="{d}" marker-end="url(#{marker})"/>'


def curve(x1, y1, x2, y2, bulge, cls="resid", marker="a1"):
    """A quadratic curve bulging sideways -- used for residual paths."""
    mx, my = (x1 + x2) / 2 + bulge, (y1 + y2) / 2
    return (f'<path class="{cls}" d="M{x1} {y1} Q{mx} {my} {x2} {y2}" '
            f'marker-end="url(#{marker})"/>')


def bracket(x, y1, y2, label, side="left"):
    """A vertical brace-ish bracket labelling a span."""
    d = 6 if side == "left" else -6
    anchor = "end" if side == "left" else "start"
    return (f'<path class="divider" d="M{x + d} {y1} L{x} {y1} L{x} {y2} L{x + d} {y2}" '
            f'fill="none"/>'
            f'<text class="lbl-s" x="{x - 4 if side == "left" else x + 4}" '
            f'y="{(y1 + y2) / 2 + 3}" text-anchor="{anchor}">{esc(label)}</text>')
