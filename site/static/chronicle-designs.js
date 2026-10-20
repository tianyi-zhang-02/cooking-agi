/* One object per milestone, drawn by chronicle-artifacts.js. Each design is built from the parts on
   its model (lines, rings, boxes, grids, arcs ...) and shows what the work was about: a station
   wiring a street for Pearl Street, a butterfly diagram for the FFT, a balance for DPO. `r` is the
   milestone's own seeded random source, for small variations only. Coordinates are rough; the
   model is centred and scaled afterwards. */
(function (scope) {
  'use strict';
  const TAU = Math.PI * 2;
  const D = {};
  const range = (n) => Array.from({length: n}, (_, i) => i);
  const spread = (n, w = 1.8) => range(n).map((i) => -w / 2 + w * (n === 1 ? .5 : i / (n - 1)));
  const layers = (m, sizes, gap = .7, link = 1, r = Math.random) => {  // units in columns, linked forward
    const cols = sizes.map((n, l) => spread(n, 1.5).map((y) => m.add((l - (sizes.length - 1) / 2) * gap, y, 0)));
    for (let l = 0; l + 1 < cols.length; l += 1) for (const a of cols[l]) for (const b of cols[l + 1]) if (r() < link) m.link(a, b);
    return cols;
  };
  const house = (m, x, z, s = .3) => { m.box(x, -s / 2, z, s, s, s); m.path([[x - s / 2, 0, z], [x, s * .6, z], [x + s / 2, 0, z]]); };
  const chipBody = (m, w, d, pins, h = .12) => {
    m.box(0, 0, 0, w, h, d);
    spread(pins, d * .8).forEach((z) => { m.line([-w / 2, 0, z], [-w / 2 - .18, -.12, z]); m.line([w / 2, 0, z], [w / 2 + .18, -.12, z]); });
  };

  /* ------------------------------------------------ 1882–1955: origins */
  D['energy-pearl-street-1882'] = (m) => {  // one station, cables under the street to several buildings
    m.box(-.9, 0, 0, .5, .6, .5); m.cyl(-.8, .5, .1, .07, .4, 8);
    [[.2, -.5], [.6, .1], [.2, .6], [.9, -.3]].forEach(([x, z]) => { house(m, x, z, .28); m.path([[-.65, -.3, 0], [x - .1, -.3, z], [x, -.15, z]]); });
  };
  D['turing-computability-1936'] = (m) => {  // a tape of cells under a read/write head
    range(7).forEach((i) => m.cube(-1.2 + .4 * i, 0, 0, .34));
    m.path([[0, .25, 0], [-.18, .65, 0], [.18, .65, 0]], true); m.box(0, .85, 0, .5, .3, .3);
  };
  D['transistor-1947'] = (m) => {  // point contacts pressed onto a germanium slab
    m.box(0, -.5, 0, 1.2, .3, .8);
    m.path([[-.5, .8, 0], [-.05, -.3, 0], [.05, -.3, 0], [.5, .8, 0]]);
    m.line([-.05, -.3, 0], [-.4, .9, .3], 5); m.line([.05, -.3, 0], [.4, .9, -.3], 5);
    m.line([0, -.65, 0], [0, -1.1, 0], 3);
  };
  D['shannon-information-theory-1948'] = (m) => {  // source → noisy channel → receiver, carrying bits
    m.cube(-1, 0, 0, .4); m.cube(1, 0, 0, .4);
    m.path(range(15).map((i) => { const x = -.75 + 1.5 * i / 14; return [x, .08 * Math.sin(i * 2.3), 0]; }));
    range(5).forEach((i) => m.add(-.6 + .3 * i, .25, 0));
    m.path(range(8).map((i) => [-.2 + .06 * i, -.35 + (i % 2 ? .12 : -.12), 0]));  // noise
  };
  D['metropolis-monte-carlo-1953'] = (m, r) => {  // particles in a box; one tries a random move
    m.cube(0, 0, 0, 1.8);
    range(18).forEach(() => m.add(r() * 1.6 - .8, r() * 1.6 - .8, r() * 1.6 - .8));
    m.node(-.2, .1, 0, .09); m.arrow([-.2, .1, 0], [.35, .4, .2]);
  };
  D['memory-whirlwind-core-1953'] = (m) => {  // a plane of magnetic cores threaded by wires
    const n = 5;
    for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) m.ringXY(-.8 + .4 * i, -.8 + .4 * j, 0, .09, 8);
    spread(n, 1.6).forEach((u) => { m.line([u, -1, 0], [u, 1, 0], 6); m.line([-1, u, .02], [1, u, .02], 6); });
  };
  D['bellman-dynamic-programming-1954'] = (m) => {  // stages of states; the best path through them
    const cols = [1, 3, 3, 3, 1].map((n, s) => spread(n, 1.4).map((y) => m.add(-1 + .5 * s, y, 0)));
    for (let s = 0; s + 1 < cols.length; s += 1) for (const a of cols[s]) for (const b of cols[s + 1]) m.link(a, b);
    m.path([[-1, 0, .05], [-.5, .7, .05], [0, 0, .05], [.5, -.7, .05], [1, 0, .05]]);
  };
  D['dartmouth'] = (m) => {  // a summer workshop: people around one table
    m.circle(28, .6, 0); m.circle(28, .55, -.05);
    range(10).forEach((i) => { const a = i / 10 * TAU; m.ringXY(Math.cos(a) * .85, .25, Math.sin(a) * .85, .08, 8); m.line([Math.cos(a) * .85, .15, Math.sin(a) * .85], [Math.cos(a) * .85, -.25, Math.sin(a) * .85]); });
  };
  D['symbolic'] = (m) => {  // a proof growing upward from axioms to a theorem
    const leaf = spread(4, 1.6).map((x) => m.add(x, -.8, 0)), mid = spread(2, .8).map((x) => m.add(x, 0, 0)), top = m.add(0, .8, 0);
    leaf.forEach((id, i) => m.link(id, mid[i < 2 ? 0 : 1])); mid.forEach((id) => m.link(id, top));
    [[0, .8], ...spread(2, .8).map((x) => [x, 0]), ...spread(4, 1.6).map((x) => [x, -.8])].forEach(([x, y]) => m.rectXY(x, y, 0, .22, .14));
  };

  /* ------------------------------------------------ 1956–1985: early AI and computing */
  D['memory-ibm-ramac-disk-1956'] = (m) => {  // a tall stack of platters and an access arm
    range(12).forEach((i) => m.circle(24, .8, -.9 + i * .16));
    m.line([0, -1, 0], [0, .95, 0], 6); m.path([[1.1, -.6, 0], [1.1, .5, 0]]); m.line([1.1, .1, 0], [.3, .1, 0], 4);
  };
  D['software-fortran-1957'] = (m) => {  // a punched card
    m.path([[-1, -.5, 0], [.85, -.5, 0], [1, -.35, 0], [1, .5, 0], [-1, .5, 0]], true);
    for (let i = 0; i < 16; i += 1) for (let j = 0; j < 4; j += 1) if ((i * 7 + j * 3) % 5 < 2) m.add(-.85 + i * .11, -.3 + j * .2, 0);
    m.dashes(-.85, .38, 0, 1.4, 1);
  };
  D['perceptron-1958'] = (m, r) => {  // a retina of sensors, random wires, one response
    m.gridXY(4, 4, .8, .8, 0, -.9, 0);
    const assoc = spread(5, 1.2).map((y) => m.add(0, y, 0)), out = m.add(.9, 0, 0);
    for (let k = 0; k < 14; k += 1) m.link(m.add(-1.2 + r() * .6, -.4 + r() * .8, 0), assoc[Math.floor(r() * assoc.length)]);
    assoc.forEach((id) => m.link(id, out)); m.node(.9, 0, 0, .12);
  };
  D['dijkstra-shortest-path-1959'] = (m, r) => {  // settled distance growing out from the source
    [.35, .7, 1.05].forEach((rad) => m.ringXY(0, 0, 0, rad, 28));
    const pts = range(10).map((i) => { const a = r() * TAU, d = .2 + r() * .9; return m.add(Math.cos(a) * d, Math.sin(a) * d, 0); });
    const src = m.add(0, 0, 0); pts.slice(0, 4).forEach((id) => m.link(src, id));
    pts.forEach((id, i) => { if (i >= 4) m.link(pts[i % 4], id); });
  };
  D['chips-ibm-system360-1964'] = (m) => {  // six models of growing size, one instruction set across the top
    range(6).forEach((i) => { const h = .4 + .25 * i; m.box(-1.25 + .5 * i, -.9 + h / 2, 0, .36, h, .4); });
    m.line([-1.3, .95, 0], [1.3, .95, 0], 8);
    range(6).forEach((i) => m.line([-1.25 + .5 * i, .95, 0], [-1.25 + .5 * i, -.9 + .4 + .25 * i, 0]));
  };
  D['cooley-tukey-fft-1965'] = (m) => {  // the butterfly: eight inputs, three stages
    const ys = spread(8, 1.6), cols = range(4).map((s) => ys.map((y) => m.add(-1.2 + .8 * s, y, 0)));
    for (let s = 0; s < 3; s += 1) {
      const span = 1 << s;
      ys.forEach((_, i) => { const j = i ^ span; m.link(cols[s][i], cols[s + 1][i]); m.link(cols[s][i], cols[s + 1][j]); });
    }
  };
  D['chips-moore-cramming-1965'] = (m) => {  // component counts doubling on a log scale
    m.line([-1, -1, 0], [1, -1, 0], 6); m.line([-1, -1, 0], [-1, 1, 0], 6);
    m.path(range(10).map((i) => [-.9 + .2 * i, -.95 + .19 * i, 0]));
    range(10).forEach((i) => m.ringXY(-.9 + .2 * i, -.95 + .19 * i, 0, .04, 6));
  };
  D['ibm-model85-cache-1968'] = (m) => {  // a small fast store beside the processor, a large slow one behind
    m.box(-.8, 0, 0, .5, .5, .5); m.box(0, 0, 0, .35, .35, .35); m.box(.9, 0, 0, .9, .9, .9);
    m.line([-.55, 0, 0], [-.18, 0, 0]); m.line([.18, 0, 0], [.45, 0, 0]);
  };
  D['networks-arpanet-1969'] = (m) => {  // the first four hosts
    const at = [[-.8, .5, 0], [.8, .6, .1], [.7, -.6, -.1], [-.6, -.6, .1]];
    at.forEach(([x, y, z]) => { m.ringXY(x, y, z, .18, 12); m.cube(x, y, z, .12); });
    [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]].forEach(([a, b]) => m.line(at[a], at[b], 8));
  };
  D['software-unix-1969'] = (m) => {  // small programs joined by pipes
    [-1, 0, 1].forEach((x) => m.box(x, 0, 0, .45, .45, .45));
    [-.5, .5].forEach((x) => { m.ringXY(x - .27, 0, 0, .08, 10); m.ringXY(x + .27, 0, 0, .08, 10); m.line([x - .27, 0, 0], [x + .27, 0, 0], 4); });
    m.dashes(-1.15, -.45, 0, 2.3, 1);
  };
  D['dendral-1969'] = (m) => {  // a molecule, ball and stick
    const ring = range(6).map((i) => { const a = i / 6 * TAU; return m.add(Math.cos(a) * .5, Math.sin(a) * .5, 0); });
    ring.forEach((id, i) => m.link(id, ring[(i + 1) % 6]));
    range(6).forEach((i) => { const a = i / 6 * TAU; m.ringXY(Math.cos(a) * .5, Math.sin(a) * .5, 0, .08, 8); if (i % 2) m.line([Math.cos(a) * .5, Math.sin(a) * .5, 0], [Math.cos(a) * .95, Math.sin(a) * .95, .2], 4); });
  };
  D['databases-relational-model-1970'] = (m) => {  // two tables related by a key
    m.gridXY(3, 4, .9, 1, 0, -.6, 0); m.gridXY(2, 3, .6, .75, .3, .75, .1);
    m.arc([-.15, .3, 0], [.45, .2, .3], .4, 9, 1);
  };
  D['intel-4004-1971'] = (m) => chipBody(m, .9, 1.6, 8);  // a 16-pin package
  D['shakey-1971'] = (m) => {  // a wheeled robot with a camera mast, and the box it pushes
    m.box(-.3, -.5, 0, .7, .5, .6); m.ringXY(-.55, -.8, .3, .12, 10); m.ringXY(-.05, -.8, .3, .12, 10);
    m.line([-.3, -.25, 0], [-.3, .7, 0], 5); m.box(-.3, .85, 0, .35, .25, .3);
    m.cube(.75, -.6, 0, .4);
  };
  D['software-c-language-1973'] = (m) => {  // braces around code, and a pointer
    for (const side of [-1, 1]) m.path(range(20).map((i) => { const t = i / 19 * 2 - 1, notch = Math.max(0, .18 - Math.abs(t) * .9); return [side * (.9 + .1 * t * t - notch), t, 0]; }));
    m.dashes(-.55, .45, 0, .9, 4, .3, (i) => [1, .6, .8, .4][i]);
    m.star(.55, -.45, 0, .12); m.arrow([.55, -.45, 0], [.2, .45, 0]);
  };
  D['mycin'] = (m) => {  // if-then rules leading to advice
    [.7, 0, -.7].forEach((y, i) => { m.path([[-1, y, 0], [-.75, y + .2, 0], [-.5, y, 0], [-.75, y - .2, 0]], true); m.arrow([-.45, y, 0], [.3, -.1 * i, 0]); });
    m.rectXY(.65, -.1, 0, .6, .45); m.dashes(.43, 0, 0, .4, 2, .14);
  };
  D['networks-ethernet-paper-1976'] = (m) => {  // one shared cable, stations tapped onto it
    m.cyl(0, 0, 0, .06, .001, 8); m.line([-1.3, 0, 0], [1.3, 0, 0], 14);
    spread(5, 2.2).forEach((x, i) => { const up = i % 2 ? 1 : -1; m.line([x, 0, 0], [x, up * .5, 0], 3); m.box(x, up * .7, 0, .3, .22, .25); });
  };
  D['hopfield-network-1982'] = (m) => {  // every unit linked to every other
    const pts = range(7).map((i) => { const a = i / 7 * TAU; return m.add(Math.cos(a) * .9, Math.sin(a) * .9, (i % 2 ? .15 : -.15)); });
    pts.forEach((a, i) => pts.forEach((b, j) => { if (j > i) m.link(a, b); }));
    range(7).forEach((i) => { const a = i / 7 * TAU; m.ringXY(Math.cos(a) * .9, Math.sin(a) * .9, (i % 2 ? .15 : -.15), .08, 8); });
  };
  /* ------------------------------------------------ 1986–1997: learning */
  D['backprop'] = (m, r) => {  // layers forward, errors arcing back
    layers(m, [3, 4, 4, 2], .65, .8, r);
    m.arc([1, .3, .3], [-.3, .3, .3], .5, 10, 1); m.arc([1, -.3, .3], [-1, -.3, .3], -.6, 12, 1);
  };
  D['alvinn-1988'] = (m) => {  // a road in perspective, a van, a camera image feeding a network
    m.path([[-1, -1, 0], [-.15, .6, 0]]); m.path([[1, -1, 0], [.15, .6, 0]]);
    range(4).forEach((i) => m.line([0, -.9 + .4 * i, 0], [0, -.75 + .4 * i, 0]));
    m.box(0, -.65, .2, .45, .3, .5); m.gridXY(3, 3, .45, .45, 0, .85, .75);
  };
  D['networks-web-proposal-1989'] = (m, r) => {  // documents linked by hypertext
    const docs = [[-.8, .5], [.7, .7], [.1, -.1], [-.7, -.7], [.8, -.6]].map(([x, y]) => { m.rectXY(x, y, 0, .32, .42); m.dashes(x - .1, y + .1, .01, .2, 3, .1); return [x, y, 0]; });
    [[0, 2], [1, 2], [2, 3], [2, 4], [0, 1], [3, 4]].forEach(([a, b]) => m.arc(docs[a], docs[b], .35 * (r() < .5 ? 1 : -1), 8));
  };
  D['hmm-rabiner-tutorial-1989'] = (m) => {  // hidden states above, what you hear below
    const xs = spread(4, 1.8);
    xs.forEach((x, i) => { m.ringXY(x, .45, 0, .16, 12); m.rectXY(x, -.55, 0, .22, .22); m.line([x, .29, 0], [x, -.44, 0], 4); if (i) m.arrow([xs[i - 1] + .17, .45, 0], [x - .17, .45, 0], .08); });
    m.arc([xs[1] - .1, .62, 0], [xs[1] + .1, .62, 0], .3, 8, 1);
  };
  D['software-python-0-9-1991'] = (m) => {  // a coiled snake: a flat spiral with a head
    m.path(range(40).map((i) => { const a = i / 39 * 3 * TAU, rad = .15 + .8 * i / 39; return [Math.cos(a) * rad, Math.sin(a) * rad, .05 * Math.sin(i)]; }));
    m.ringXY(.95, 0, 0, .1, 10);
  };
  D['arxiv-1991'] = (m) => {  // one server, papers fanning out to many readers
    m.box(-.9, 0, 0, .45, .7, .45);
    spread(5, 1.8).forEach((y) => { m.rectXY(.85, y, 0, .22, .28); m.arc([-.65, 0, 0], [.72, y, 0], .25 * y, 8); });
  };
  D['qlearning'] = (m) => {  // a grid world with a learned arrow in each cell
    m.gridXY(4, 4, 1.8, 1.8);
    for (let i = 0; i < 4; i += 1) for (let j = 0; j < 4; j += 1) {
      const x = -.675 + .45 * i, y = -.675 + .45 * j, toRight = (i + j) % 3 !== 0;
      m.arrow([x - (toRight ? .12 : 0), y - (toRight ? 0 : .12), 0], [x + (toRight ? .12 : 0), y + (toRight ? 0 : .12), 0], .06);
    }
    m.star(.675, .675, .02, .1);
  };
  D['ibm-smt-1993'] = (m) => {  // two sentences and the alignment between their words
    const top = m.tokens(6, .5), bottom = m.tokens(6, -.5);
    [[0, 0], [1, 2], [2, 1], [3, 3], [4, 5], [5, 4], [3, 2]].forEach(([a, b]) => m.line([top[a][0], .42, 0], [bottom[b][0], -.42, 0], 5));
  };
  D['penn-treebank-1993'] = (m) => {  // a parse tree over a row of words
    const words = m.tokens(5, -.8, 0, 1.8, .14);
    const np = m.add(-.6, 0, 0), vp = m.add(.45, 0, 0), s = m.add(0, .8, 0), pp = m.add(.8, -.4, 0);
    m.link(s, np); m.link(s, vp); m.link(vp, pp);
    [[np, 0], [np, 1], [vp, 2], [pp, 3], [pp, 4]].forEach(([id, w]) => m.link(id, m.add(words[w][0], -.72, 0)));
  };
  D['distributed-mpi-1994'] = (m) => {  // processes in a ring passing messages
    const n = 6, at = range(n).map((i) => { const a = i / n * TAU; return [Math.cos(a) * .85, Math.sin(a) * .85, 0]; });
    at.forEach(([x, y]) => m.cube(x, y, 0, .22));
    at.forEach((a, i) => m.arc(a, at[(i + 1) % n], .2, 7));
    m.rectXY(0, 0, 0, .3, .2); m.line([-.15, .1, 0], [0, 0, 0]); m.line([.15, .1, 0], [0, 0, 0]);  // an envelope
  };
  D['grouplens-1994'] = (m, r) => {  // readers and posts: similar readers recommend
    const users = spread(4, 1.6).map((y) => (m.ringXY(-.8, y, 0, .1, 10), m.add(-.7, y, 0)));
    const items = spread(5, 1.8).map((y) => (m.rectXY(.8, y, 0, .2, .2), m.add(.7, y, 0)));
    users.forEach((u, i) => items.forEach((it, j) => { if ((i + j * 2) % 3 === 0 || r() < .15) m.link(u, it); }));
  };
  D['okapi-bm25-1994'] = (m) => {  // a score that grows with repeats, but less each time
    m.line([-1, -.8, 0], [1, -.8, 0], 6); m.line([-1, -.8, 0], [-1, .9, 0], 6);
    m.path(range(16).map((i) => { const x = i / 15 * 1.9; return [-1 + x, -.8 + 1.5 * x / (x + .5), 0]; }));
    m.line([-1, .7, 0], [.9, .7, 0], 10);
  };
  D['svm'] = (m, r) => {  // two classes, a separating plane, the margins either side
    for (const [s, z] of [[-1, -.5], [1, .5]]) { m.rectXY(0, 0, z, 1.8, 1.6); }
    m.rectXY(0, 0, 0, 1.8, 1.6);
    range(9).forEach(() => m.ringXY(r() * 1.4 - .7, r() * 1.2 - .6, -.7 - r() * .3, .05, 6));
    range(9).forEach(() => m.cube(r() * 1.4 - .7, r() * 1.2 - .6, .7 + r() * .3, .08));
  };
  D['lstm'] = (m) => {  // a memory line running through, three gates
    m.rectXY(0, 0, 0, 1.6, 1.1); m.line([-1.2, .35, 0], [1.2, .35, 0], 12);
    spread(3, 1).forEach((x) => { m.ringXY(x, -.2, 0, .13, 12); m.line([x, -.07, 0], [x, .35, 0], 3); });
    m.arc([.8, -.55, 0], [-.8, -.55, 0], -.35, 12, 1);
  };
  D['applications-sojourner-1997'] = (m) => {  // six wheels, a body and a solar panel
    m.box(0, 0, 0, 1.1, .25, .7); m.rectXZ(0, .2, 0, 1.2, .8);
    [-.45, 0, .45].forEach((x) => [-.42, .42].forEach((z) => m.ringXY(x, -.3, z, .16, 12)));
    m.line([.45, .15, .2], [.6, .5, .2]); m.cube(.6, .55, .2, .1);
  };
  D['deep-blue-1997'] = (m) => {  // a chessboard and a king
    m.gridXZ(8, 8, 1.8, 1.8, -.4);
    [[0, .05], [0, .25]].forEach(([y]) => m.circle(16, .14, y - .35));
    m.cyl(0, -.1, 0, .1, .4, 12); m.line([0, .1, 0], [0, .35, 0], 3); m.line([-.08, .27, 0], [.08, .27, 0]);
  };

  /* ------------------------------------------------ 1998–2011: connected computing */
  D['lenet5-1998'] = (m) => {  // a handwritten digit read through small feature maps
    m.gridXY(5, 5, .7, .7, 0, -1, 0);
    m.path([[-1.25, .25, .01], [-.8, .25, .01], [-1.05, -.3, .01]]);  // a "7"
    [[-.3, .55], [.25, .4], [.7, .25]].forEach(([x, s], k) => range(2).forEach((j) => m.rectXY(x, (j - .5) * (s + .1), k * .02, s, s)));
    m.line([.9, 0, 0], [1.2, 0, 0], 3); m.node(1.25, 0, 0, .06);
  };
  D['google-search-prototype-1998'] = (m, r) => {  // pages ranked by the links pointing at them
    const hub = [0, 0, 0]; m.ringXY(0, 0, 0, .25, 18);
    range(8).forEach((i) => { const a = i / 8 * TAU, at = [Math.cos(a) * .9, Math.sin(a) * .9, r() * .4 - .2]; m.ringXY(...at, .07 + .05 * r(), 8); m.arrow(at, [Math.cos(a) * .3, Math.sin(a) * .3, 0], .07); if (i % 3 === 0) m.arc(at, [Math.cos(a + .8) * .9, Math.sin(a + .8) * .9, 0], .2, 6); });
  };
  D['distributed-xen-paper-2003'] = (m) => {  // one server split into guests above a thin hypervisor
    m.box(0, 0, 0, 1.8, 1.2, .9); m.rectXZ(0, -.35, 0, 1.7, .85);
    spread(3, 1.1).forEach((x) => m.box(x, .15, 0, .45, .7, .6));
  };
  D['distributed-mapreduce-paper-2004'] = (m) => {  // many map tasks, a shuffle, a few reducers
    const maps = spread(6, 1.8).map((y) => (m.cube(-.9, y, 0, .16), [-.82, y, 0]));
    const reds = spread(3, 1.2).map((y) => (m.cube(.9, y, 0, .22), [.79, y, 0]));
    maps.forEach((a, i) => reds.forEach((b, j) => { if ((i + j) % 2 === 0 || i === j) m.line(a, b, 5); }));
  };
  D['software-llvm-paper-2004'] = (m) => {  // many languages in, one representation, many machines out
    spread(4, 1.6).forEach((y) => { m.rectXY(-1, y, 0, .25, .2); m.line([-.87, y, 0], [-.15, 0, 0], 5); });
    m.ringXY(0, 0, 0, .16, 14); m.ringXY(0, 0, 0, .1, 10);
    spread(4, 1.6).forEach((y) => { m.cube(1, y, 0, .2); m.line([.15, 0, 0], [.88, y, 0], 5); });
  };
  D['sift-2004'] = (m) => {  // a keypoint: its scale, its orientation, and a 4×4 descriptor
    m.ringXY(0, 0, 0, .9, 30); m.arrow([0, 0, 0], [.8, .35, 0]);
    for (let i = 0; i < 4; i += 1) for (let j = 0; j < 4; j += 1) { const x = -.45 + .3 * i, y = -.45 + .3 * j, a = (i * 3 + j * 5) % 8 / 8 * TAU; m.line([x, y, .05], [x + Math.cos(a) * .1, y + Math.sin(a) * .1, .05]); }
    m.gridXY(4, 4, 1.2, 1.2, .04);
  };
  D['nvidia-cuda-2006'] = (m) => {  // a grid of blocks, each a grid of threads
    for (let i = 0; i < 3; i += 1) for (let j = 0; j < 2; j += 1) {
      const x = -.8 + .8 * i, y = -.4 + .8 * j; m.rectXY(x, y, 0, .65, .65);
      for (let a = 0; a < 3; a += 1) for (let b = 0; b < 3; b += 1) m.add(x - .2 + .2 * a, y - .2 + .2 * b, 0);
    }
  };
  D['ecosystem-differential-privacy-2006'] = (m, r) => {  // two crowds that differ by one person look the same
    const crowd = range(14).map(() => [r() * 1.2 - .6, r() * 1.2 - .6]);
    crowd.forEach(([x, y]) => { m.add(x - .3, y, 0); m.add(x + .3, y, .3); });
    m.ringXY(-.3 + .55, .55, 0, .1, 10); m.ringXY(-.3, 0, 0, .95, 28); m.ringXY(.3, 0, .3, .95, 28);
  };
  D['networks-rdmap-rfc-2007'] = (m) => {  // memory to memory between two machines, past the processors
    for (const s of [-1, 1]) { m.box(s * .8, 0, 0, .6, .9, .5); m.cube(s * .8, .25, .1, .2); m.gridXY(2, 2, .3, .25, .26, s * .8, -.2); }
    m.arc([-.5, -.2, .26], [.5, -.2, .26], -.35, 10, 1); m.arrow([.3, -.4, .26], [.5, -.2, .26], .08);
  };
  D['energy-proportional-computing-2007'] = (m) => {  // power should follow load: proportional versus flat
    m.line([-1, -.9, 0], [1, -.9, 0], 6); m.line([-1, -.9, 0], [-1, 1, 0], 6);
    m.line([-1, -.9, 0], [.9, .9, 0], 10); m.line([-1, .3, .05], [.9, .55, .05], 10);
    range(4).forEach((i) => m.ringXY(-.6 + .45 * i, -.9 + (.4 + .45 * i) * .95, 0, .04, 6));
  };
  D['energy-pue-metric-2007'] = (m) => {  // total facility power around the share that reaches IT
    m.box(0, 0, 0, 1.8, 1.2, 1); m.box(-.2, -.15, 0, 1, .7, .6);
    spread(3, .7).forEach((x) => m.box(x - .2, -.15, 0, .2, .6, .45));
    m.cyl(.7, .35, .3, .12, .3, 10);  // the cooling plant
  };
  D['chips-roofline-report-2008'] = (m) => {  // a sloped bandwidth line meeting a flat compute roof
    m.line([-1, -.9, 0], [1, -.9, 0], 6); m.line([-1, -.9, 0], [-1, .9, 0], 6);
    m.line([-1, -.9, 0], [-.1, .6, 0], 8); m.line([-.1, .6, 0], [1, .6, 0], 8);
    [[-.6, -.4], [-.2, .1], [.4, .3], [.7, .5]].forEach(([x, y]) => m.ringXY(x, y, 0, .05, 6));
  };
  D['networks-fat-tree-paper-2008'] = (m) => {  // identical switches in three tiers
    const tiers = [4, 8, 8].map((n, l) => spread(n, 1.8).map((x) => (m.cube(x, .8 - .8 * l, 0, .12), m.add(x, .8 - .8 * l, 0))));
    tiers[1].forEach((id, i) => { m.link(id, tiers[0][Math.floor(i / 2)]); m.link(id, tiers[0][(Math.floor(i / 2) + 2) % 4]); });
    tiers[2].forEach((id, i) => { m.link(id, tiers[1][i]); m.link(id, tiers[1][i ^ 1]); });
  };
  D['energy-warehouse-scale-computing-2009'] = (m) => {  // a warehouse full of racks, built as one machine
    m.box(0, 0, 0, 2, .9, 1.2); m.path([[-1, .45, -.6], [0, .75, -.6], [1, .45, -.6]]);
    for (let i = 0; i < 4; i += 1) m.box(-.75 + .5 * i, -.1, 0, .2, .55, .8);
  };
  D['imagenet-2009'] = (m) => {  // a concept tree above a wall of labelled images
    for (let i = 0; i < 4; i += 1) for (let j = 0; j < 3; j += 1) m.rectXY(-.75 + .5 * i, -.75 + .38 * j, 0, .38, .28);
    const root = m.add(0, .95, 0); spread(3, 1.2).forEach((x) => { const c = m.add(x, .55, 0); m.link(root, c); m.link(c, m.add(x - .15, .35, 0)); m.link(c, m.add(x + .15, .35, 0)); });
  };
  D['netflix-prize-matrix-factorization-2009'] = (m) => {  // a rating matrix as a tall times a wide factor
    m.gridXY(5, 5, .9, .9, 0, -.6, 0);
    m.gridXY(1, 5, .2, .9, 0, .25, 0); m.gridXY(5, 1, .9, .2, 0, .9, .55);
    m.line([-.08, 0, 0], [.08, 0, 0]); m.line([-.08, .06, 0], [.08, .06, 0]);
  };
  D['storage-hdfs-paper-2010'] = (m) => {  // one name node, many data nodes holding copied blocks
    m.box(0, .8, 0, .5, .3, .4);
    spread(4, 1.8).forEach((x, i) => { m.box(x, -.4, 0, .36, .7, .4); [0, 1].forEach((k) => m.cube(x, -.55 + .3 * k, .21, .1)); m.line([0, .65, 0], [x, -.05, 0], 5); });
  };
  D['amd-gcn-2011'] = (m) => {  // a compute unit: four SIMD blocks, a scheduler on top
    m.rectXY(0, 0, 0, 1.8, 1.4); m.rectXY(0, .52, 0, 1.5, .2);
    spread(4, 1.3).forEach((x) => { m.rectXY(x, -.15, 0, .28, .8); m.dashes(x - .1, .1, .01, .2, 4, .15); });
  };
  D['chips-intel-trigate-22nm-2011'] = (m) => {  // upright fins with the gate wrapped over three sides
    spread(3, 1).forEach((z) => m.box(0, 0, z, 1.8, .45, .14));
    m.path([[-.25, -.25, -.75], [-.25, .35, -.75], [-.25, .35, .75], [-.25, -.25, .75]]);
    m.path([[.25, -.25, -.75], [.25, .35, -.75], [.25, .35, .75], [.25, -.25, .75]]);
    m.line([-.25, .35, 0], [.25, .35, 0], 3);
  };

  /* ------------------------------------------------ 2012–2015: deep learning */
  D['dnn-acoustic-2012'] = (m, r) => {  // a spectrogram feeding a deep stack in place of the old model
    for (let i = 0; i < 8; i += 1) { const h = .2 + .6 * Math.abs(Math.sin(i * 1.3)); m.line([-1.2 + .12 * i, -h / 2, 0], [-1.2 + .12 * i, h / 2, 0], 3); }
    range(5).forEach((k) => m.rectXY(-.1 + .3 * k, 0, 0, .18, 1.2 - .1 * k));
  };
  D['distributed-spark-rdd-2012'] = (m) => {  // a lineage of datasets kept in memory
    const at = [[-1, .4], [-1, -.4], [-.3, 0], [.3, .45], [.3, -.45], [1, 0]];
    at.forEach(([x, y]) => m.box(x, y, 0, .3, .3, .3));
    [[0, 2], [1, 2], [2, 3], [2, 4], [3, 5], [4, 5]].forEach(([a, b]) => m.arrow([at[a][0] + .16, at[a][1], 0], [at[b][0] - .16, at[b][1], 0], .07));
  };
  D['dqn'] = (m) => {  // an Atari screen and a joystick
    m.rectXY(0, .25, 0, 1.6, 1.1); m.rectXY(0, .25, 0, 1.45, .95);
    [[-.4, .5], [.1, .3], [.4, .55], [-.2, .05]].forEach(([x, y]) => m.rectXY(x, y, 0, .12, .08));
    m.box(0, -.65, .2, .5, .15, .35); m.line([0, -.58, .2], [.08, -.3, .2], 3); m.ringXY(.08, -.27, .2, .05, 8);
  };
  D['word2vec-2013'] = (m) => {  // king − man + woman ≈ queen
    const o = [-.9, -.8, 0], king = [.1, .6, 0], man = [-.2, -.2, .2], woman = [.5, -.3, -.2], queen = [.8, .5, -.4];
    [king, man, woman, queen].forEach((p) => m.arrow(o, p, .08));
    m.line(king, queen, 5); m.line(man, woman, 5);
  };
  D['seq2seq'] = (m) => {  // an encoder squeezes a sentence into one vector; a decoder unrolls it
    const enc = spread(4, .9).map((x) => x - .65), dec = spread(4, .9).map((x) => x + .65);
    enc.forEach((x, i) => { m.rectXY(x, -.3, 0, .2, .2); if (i) m.line([enc[i - 1] + .1, -.3, 0], [x - .1, -.3, 0]); m.line([x, -.2, 0], [x, .1, 0]); });
    m.ringXY(0, .2, 0, .14, 12); m.line([enc[3], .1, 0], [-.14, .2, 0]);
    dec.forEach((x, i) => { m.rectXY(x, .55, 0, .2, .2); if (i) m.line([dec[i - 1] + .1, .55, 0], [x - .1, .55, 0]); });
    m.line([.14, .2, 0], [dec[0], .45, 0]);
  };
  D['distributed-parameter-server-2014'] = (m) => {  // servers holding slices of the model, workers pushing and pulling
    const servers = spread(3, 1.2).map((x) => (m.cyl(x, .7, 0, .16, .3, 12), [x, .55, 0]));
    spread(6, 1.9).forEach((x, i) => { m.cube(x, -.7, 0, .18); servers.forEach((s, j) => { if ((i + j) % 2 === 0) m.line([x, -.6, 0], s, 5); }); });
  };
  D['bahdanau-attention-2014'] = (m) => {  // one output word looking back over every source word
    const src = m.tokens(6, -.6); m.rectXY(.3, .7, 0, .22, .22);
    src.forEach(([x], i) => m.arc([.3, .58, 0], [x, -.5, 0], .15 + .1 * (i % 3), 8));
    spread(3, .9).forEach((x) => m.rectXY(x - .5, .7, 0, .18, .18));
  };
  D['adam-optimizer-2014'] = (m) => {  // contours of a loss and a path that adapts its step per direction
    [[1, .35], [.75, .26], [.5, .17], [.25, .09]].forEach(([a, b]) => m.path(range(28).map((i) => [Math.cos(i / 28 * TAU) * a, Math.sin(i / 28 * TAU) * b * 2.2, 0]), true));
    m.path([[-.95, .55, .1], [-.6, .1, .1], [-.35, .2, .1], [-.15, .02, .1], [-.03, .03, .1]]);
    m.arrow([-.95, .55, .1], [-.75, .55, .1], .06); m.arrow([-.95, .55, .1], [-.95, .35, .1], .06);
  };
  D['resnet'] = (m) => {  // a stack of layers with shortcuts around each block
    spread(6, 1.8).forEach((y) => m.rectXZ(0, y, 0, 1, .5));
    [-.9, -.18, .54].forEach((y) => m.arc([.5, y, 0], [.5, y + .72, 0], .45, 9, 0));
  };
  D['distributed-kubernetes-v1-2015'] = (m) => {  // a ship's wheel steering a cluster of containers
    m.ringXY(0, 0, 0, .75, 32); m.ringXY(0, 0, 0, .2, 12);
    range(7).forEach((i) => { const a = i / 7 * TAU; m.line([Math.cos(a) * .2, Math.sin(a) * .2, 0], [Math.cos(a) * 1, Math.sin(a) * 1, 0], 5); });
    range(3).forEach((i) => m.cube(-.25 + .25 * i, -1.15, 0, .18));
  };
  D['hbm-gpu-packaging-2015'] = (m) => {  // a GPU die with memory stacks beside it on an interposer
    m.rectXZ(0, -.25, 0, 2, 1.3); m.box(0, -.1, 0, .7, .2, .7);
    [[-.7, -.35], [-.7, .35], [.7, -.35], [.7, .35]].forEach(([x, z]) => range(4).forEach((k) => m.rectXZ(x, -.2 + .1 * k, z, .3, .25)));
  };
  D['software-tensorflow-2015'] = (m) => {  // a dataflow graph: operations, parameters and updates as nodes
    const at = [[-.9, .5], [-.9, -.5], [-.2, 0], [.4, .5], [.4, -.5], [1, 0]];
    at.forEach(([x, y], i) => (i % 2 ? m.ringXY(x, y, 0, .13, 12) : m.rectXY(x, y, 0, .24, .2)));
    [[0, 2], [1, 2], [2, 3], [2, 4], [3, 5], [4, 5]].forEach(([a, b]) => m.arrow([at[a][0] + .13, at[a][1], 0], [at[b][0] - .13, at[b][1], 0], .06));
    m.arc([1, -.15, 0], [-.9, -.65, 0], -.35, 12, 1);
  };
  D['distributed-borg-paper-2015'] = (m) => {  // machines shared: services first, batch work in the gaps
    for (let i = 0; i < 4; i += 1) {
      const x = -.75 + .5 * i, svc = .3 + .15 * (i % 3);
      m.rectXY(x, 0, 0, .4, 1.6); m.box(x, -.8 + svc / 2, 0, .34, svc, .2);
      m.rectXY(x, -.8 + svc + .25, .1, .34, .45);
    }
  };

  /* ------------------------------------------------ 2016–2018: many directions at once */
  D['gcn'] = (m, r) => {  // each node gathers its neighbours' features
    const at = range(9).map((i) => { const a = i / 9 * TAU + r() * .3, d = .5 + r() * .45; return [Math.cos(a) * d, Math.sin(a) * d, r() * .4 - .2]; });
    at.forEach((p, i) => { m.ringXY(...p, .06, 8); m.line(p, at[(i + 2) % 9], 5); });
    m.ringXY(0, 0, 0, .14, 12); at.slice(0, 5).forEach((p) => m.arrow(p, [p[0] * .2, p[1] * .2, 0], .06));
  };
  D['alphago'] = (m, r) => {  // a Go board, stones, and a few candidate moves searched above it
    m.gridXZ(9, 9, 1.8, 1.8, -.5);
    range(14).forEach(() => m.circle(8, .07, -.45, -.9 + .2 * Math.floor(r() * 10), -.9 + .2 * Math.floor(r() * 10)));
    const top = m.add(0, .9, 0); spread(3, 1.2).forEach((x) => { const c = m.add(x, .35, 0); m.link(top, c); m.line([x, .35, 0], [x * .9, -.45, x * .3], 4); });
  };
  D['wavenet-2016'] = (m) => {  // dilated layers: each skips twice as far as the one below
    const rows = 4, n = 9, ids = range(rows).map((l) => spread(n, 1.9).map((x) => m.add(x, -.8 + .55 * l, 0)));
    for (let l = 1; l < rows; l += 1) { const d = 1 << (l - 1); for (let i = d; i < n; i += 1) if ((n - 1 - i) % (1 << l) === 0) { m.link(ids[l][i], ids[l - 1][i]); m.link(ids[l][i], ids[l - 1][i - d]); } }
    m.path(spread(n, 1.9).map((x, i) => [x, -1.05 + .08 * Math.sin(i * 1.7), 0]));
  };
  D['gnmt-2016'] = (m) => {  // deep stacked encoder and decoder joined by attention
    for (const side of [-1, 1]) range(8).forEach((k) => m.rectXZ(side * .7, -.9 + .25 * k, 0, .55, .4));
    m.arc([-.4, .3, 0], [.4, .5, 0], .4, 9, 1); m.arc([-.4, -.3, 0], [.4, .5, 0], .3, 9, 1);
  };
  D['youtube-dnn-recsys-2016'] = (m, r) => {  // millions of videos narrowed to hundreds, then ranked
    [[1, 40, .9], [.6, 16, .2], [.3, 6, -.5]].forEach(([rad, n, y]) => { m.circle(28, rad, y); range(n).forEach(() => { const a = r() * TAU, d = r() * rad * .9; m.add(Math.cos(a) * d, y, Math.sin(a) * d); }); });
    m.line([1, .9, 0], [.3, -.5, 0], 5); m.line([-1, .9, 0], [-.3, -.5, 0], 5);
    spread(3, .5).forEach((x, i) => m.rectXY(x, -.95, 0, .12, .1 + .08 * (2 - i)));
  };
  D['ecosystem-federated-learning-2016'] = (m) => {  // phones keep their data and send back only updates
    m.box(0, 0, 0, .4, .5, .4);
    range(6).forEach((i) => { const a = i / 6 * TAU, x = Math.cos(a) * 1, y = Math.sin(a) * .8; m.rectXY(x, y, 0, .16, .3); m.arrow([x * .8, y * .8, 0], [x * .3, y * .3, 0], .06); });
  };
  D['squad-2016'] = (m) => {  // a passage with the answer span marked
    m.rectXY(0, 0, 0, 1.8, 1.4); m.dashes(-.75, .5, 0, 1.5, 6, .2, (i) => [1, .9, 1, .7, .95, .6][i]);
    m.rectXY(-.05, .1, .02, .7, .14); m.path([[.8, .9, 0], [.95, 1.05, 0], [1.05, .9, 0], [.95, .75, 0], [.95, .65, 0]]);
  };
  D['networks-nvlink-p100-2016'] = (m) => {  // four GPUs linked directly in a mesh, off the shared bus
    const at = [[-.6, .6], [.6, .6], [.6, -.6], [-.6, -.6]].map(([x, y]) => (m.rectXY(x, y, 0, .45, .45), m.gridXY(2, 2, .3, .3, .01, x, y), [x, y, 0]));
    [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]].forEach(([a, b]) => { m.line(at[a], at[b], 8); m.line(at[a].map((v, k) => v + (k === 2 ? .06 : 0)), at[b].map((v, k) => v + (k === 2 ? .06 : 0)), 8); });
  };
  D['transformer'] = (m, r) => {  // positions linked directly by attention, layer above layer
    const rows = [.8, 0, -.8].map((y) => m.tokens(6, y, 0, 1.8, .14));
    for (let l = 0; l < 2; l += 1) rows[l].forEach(([x], i) => [i, (i + 2) % 6, (i + 5) % 6].forEach((j) => m.arc([x, rows[l][0][1] - .07, 0], [rows[l + 1][j][0], rows[l + 1][0][1] + .07, 0], .15 + .2 * r(), 7)));
  };
  D['graphsage'] = (m, r) => {  // a target node and a sample of its two-hop neighbourhood
    m.ringXY(0, 0, 0, .12, 12);
    range(3).forEach((i) => { const a = i / 3 * TAU, p = [Math.cos(a) * .5, Math.sin(a) * .5, 0]; m.ringXY(...p, .07, 8); m.line(p, [0, 0, 0], 4);
      range(3).forEach((k) => { const b = a + (k - 1) * .5, q = [Math.cos(b) * 1, Math.sin(b) * 1, r() * .3 - .15]; m.add(...q); m.line(q, p, 4); }); });
    m.ringXY(0, 0, 0, .75, 30);
  };
  D['gat'] = (m) => {  // neighbours weighted differently: thicker links count more
    m.ringXY(0, 0, 0, .14, 12);
    [.9, .5, .25, .7, .4, .15].forEach((w, i) => { const a = i / 6 * TAU, p = [Math.cos(a) * .9, Math.sin(a) * .9, 0]; m.ringXY(...p, .08, 8); range(1 + Math.round(w * 3)).forEach((k) => m.line([p[0] * .85, p[1] * .85, (k - 1) * .03], [p[0] * .17, p[1] * .17, (k - 1) * .03], 5)); });
  };
  D['nvidia-volta-2017'] = (m) => {  // a Tensor Core: D = A × B + C on small matrices
    m.gridXY(4, 4, .6, .6, 0, -.85, .3); m.gridXY(4, 4, .6, .6, 0, 0, .3); m.gridXY(4, 4, .6, .6, 0, .85, .3);
    m.gridXY(4, 4, .6, .6, 0, 0, -.5);
    m.line([-.48, .3, 0], [-.38, .3, 0]); m.arrow([0, -.05, 0], [0, -.15, 0], .05);
  };
  D['sparse-moe-2017'] = (m) => {  // a router sends each token to only a few of many experts
    m.ringXY(0, .8, 0, .14, 12);
    spread(8, 2).forEach((x, i) => { const on = i === 2 || i === 5; m.box(x, -.5, 0, .18, on ? .4 : .25, .18); if (on) m.line([0, .66, 0], [x, -.3, 0], 6); });
  };
  D['software-pytorch-2017'] = (m) => {  // operations run as written; the tape replays them backwards for gradients
    const at = spread(5, 1.8).map((x, i) => [x, .2 * Math.sin(i * 1.4), 0]);
    at.forEach((p, i) => { m.ringXY(...p, .12, 10); if (i) m.arrow([at[i - 1][0] + .13, at[i - 1][1], 0], [p[0] - .13, p[1], 0], .06); });
    m.arc([at[4][0], -.25, 0], [at[0][0], -.25, 0], -.45, 12, 1);
  };
  D['chips-tpu-v1-paper-2017'] = (m) => {  // a systolic array: data pulses through a grid of multipliers
    m.gridXY(6, 6, 1.5, 1.5);
    range(6).forEach((i) => { m.arrow([-1.05, -.625 + .25 * i, 0], [-.8, -.625 + .25 * i, 0], .05); m.arrow([-.625 + .25 * i, 1.05, 0], [-.625 + .25 * i, .8, 0], .05); });
    m.box(0, 0, -.3, 1.6, 1.6, .1);
  };
  D['bert'] = (m) => {  // a hidden word guessed from both sides
    const row = m.tokens(7, 0, 0, 1.9, .16); m.rectXY(row[3][0], 0, 0, .1, .1);
    row.forEach(([x], i) => { if (i !== 3) m.arc([x, .1, 0], [row[3][0], .12, 0], .25 + .05 * Math.abs(i - 3), 8, 1); });
    m.path([[row[3][0] - .06, .45, 0], [row[3][0], .55, 0], [row[3][0] + .06, .45, 0]]);
  };
  D['sac'] = (m) => {  // a wide distribution over continuous actions, kept wide on purpose
    m.line([-1.1, -.6, 0], [1.1, -.6, 0], 8);
    m.path(range(25).map((i) => { const x = -1 + 2 * i / 24; return [x, -.6 + 1.1 * Math.exp(-x * x / .45), 0]; }));
    m.path(range(25).map((i) => { const x = -1 + 2 * i / 24; return [x, -.6 + 1.5 * Math.exp(-x * x / .08), .2]; }));
  };
  D['datasheets-2018'] = (m) => {  // a datasheet: why, how, what for, as a checklist
    m.rectXY(0, 0, 0, 1.3, 1.7);
    range(5).forEach((i) => { const y = .55 - .27 * i; m.rectXY(-.4, y, .01, .12, .12); if (i < 4) m.path([[-.45, y, .02], [-.4, y - .04, .02], [-.33, y + .05, .02]]); m.line([-.25, y, .01], [.45, y, .01], 4); });
  };
  D['model-cards-2018'] = (m) => {  // a model card: uses, conditions, scores by group
    m.rectXY(0, 0, 0, 1.6, 1.2); m.dashes(-.65, .4, .01, .7, 2, .16);
    spread(4, .9).forEach((x, i) => m.rectXY(x + .2, -.3 + .1 * i, .01, .14, .3 + .2 * i));
  };
  D['elmo-2018'] = (m) => {  // the same word, a different vector in each sentence
    for (const y of [.55, -.55]) { const row = m.tokens(5, y, 0, 1.6, .14); m.rectXY(row[2][0], y, 0, .2, .2); }
    m.arrow([0, .65, 0], [.45, 1.05, .2], .07); m.arrow([0, -.45, 0], [-.4, -.05, -.2], .07);
    m.line([-.9, .55, -.2], [.9, .55, -.2], 8); m.line([.9, -.55, -.2], [-.9, -.55, -.2], 8);
  };
  D['distributed-horovod-2018'] = (m) => {  // ring all-reduce: each GPU passes a chunk to the next
    const n = 6, at = range(n).map((i) => { const a = i / n * TAU; return [Math.cos(a) * .85, Math.sin(a) * .85, 0]; });
    at.forEach((p, i) => { m.rectXY(...p, .26, .26); const q = at[(i + 1) % n]; m.arrow(p.map((v, k) => v + (q[k] - v) * .22), p.map((v, k) => v + (q[k] - v) * .78), .06); });
  };
  D['software-tvm-paper-2018'] = (m) => {  // one model graph compiled to code for many devices
    [[-.2, .8], [.2, .8], [0, .45]].forEach(([x, y]) => m.ringXY(x, y, 0, .09, 8)); m.line([-.2, .72, 0], [0, .52, 0]); m.line([.2, .72, 0], [0, .52, 0]);
    m.rectXY(0, 0, 0, .9, .25); m.line([0, .36, 0], [0, .12, 0]);
    [[-.8, 'chip'], [0, 'phone'], [.8, 'gpu']].forEach(([x, kind]) => { if (kind === 'phone') m.rectXY(x, -.75, 0, .18, .32); else m.box(x, -.75, 0, .3, .2, .3); m.line([x * .5, -.12, 0], [x, -.6, 0], 4); });
  };
  D['glue-2018'] = (m) => {  // nine tasks, one average
    spread(9, 1.8).forEach((x, i) => m.rectXY(x, -.6 + (.3 + .08 * ((i * 5) % 7)) / 2, 0, .14, .3 + .08 * ((i * 5) % 7)));
    m.line([-1, -.6, 0], [1, -.6, 0], 8); m.line([-1, -.13, .05], [1, -.13, .05], 12);
  };
  D['zero'] = (m) => {  // training state cut into slices, one slice per GPU
    range(4).forEach((k) => { m.box(-.75 + .5 * k, -.6, 0, .38, .3, .38); m.box(-.75 + .5 * k, .35, 0, .38, .55, .3); });
    m.rectXY(0, .35, .2, 2, .6); spread(3, 1).forEach((x) => m.line([x, .05, .2], [x, .65, .2], 4));
  };
  D['amd-rdna-2019'] = (m) => {  // two compute units paired as one group
    m.rectXY(0, 0, 0, 1.9, 1.3);
    for (const x of [-.48, .48]) { m.rectXY(x, -.05, 0, .8, 1); range(2).forEach((k) => m.rectXY(x, -.25 + .4 * k, .01, .6, .3)); }
    m.rectXY(0, .55, .01, 1.6, .12);
  };
  D['distributed-megatron-lm-2019'] = (m) => {  // one weight matrix split by columns across the GPUs of a server
    m.gridXY(8, 6, 1.6, 1.2, 0); spread(4, 1.6).forEach((x, k) => m.box(x, 0, -.3, .36, 1.2, .1));
    m.line([-1.1, .8, 0], [1.1, .8, 0], 8);
  };
  D['hf-transformers-2019'] = (m) => {  // a shelf of pretrained models, each one call away
    m.line([-1.1, -.6, 0], [1.1, -.6, 0], 8); m.line([-1.1, .55, 0], [1.1, .55, 0], 8);
    spread(9, 1.9).forEach((x, i) => { const h = .55 + .15 * ((i * 3) % 4); m.box(x, -.6 + h / 2, 0, .14, h, .3); });
  };
  D['software-triton-mapl-2019'] = (m) => {  // a matrix worked on one tile at a time
    m.gridXY(4, 4, 1.6, 1.6);
    m.box(-.2, .2, .1, .4, .4, .1); m.arrow([.15, .2, .2], [.55, .2, .2], .06); m.arrow([-.2, -.15, .2], [-.2, -.55, .2], .06);
  };
  D['chips-euv-volume-production-2019'] = (m) => {  // 13.5 nm light bouncing off mirrors onto a wafer
    m.circle(30, .7, -.8);
    m.path([[-1.1, .9, 0], [-.4, .3, 0], [.5, .8, 0], [.1, -.75, 0]]);
    [[-.4, .3], [.5, .8]].forEach(([x, y]) => m.arc([x - .2, y + .05, 0], [x + .2, y - .05, 0], .08, 6, 1));
  };

  /* ------------------------------------------------ 2019–2021: scale and science */
  D['gpt3'] = (m) => {  // a prompt of worked examples, then the model completes the next one
    [.7, .25, -.2].forEach((y) => { m.rectXY(-.55, y, 0, .7, .25); m.arrow([-.15, y, 0], [.15, y, 0], .06); m.rectXY(.55, y, 0, .5, .25); });
    m.rectXY(-.55, -.7, 0, .7, .25); m.arrow([-.15, -.7, 0], [.15, -.7, 0], .06); m.ringXY(.55, -.7, 0, .14, 12);
  };
  D['vit'] = (m) => {  // an image cut into patches, read as a sequence
    m.gridXY(3, 3, .9, .9, 0, -.55, .4);
    spread(9, 1.9).forEach((x, i) => { m.rectXY(x, -.65, 0, .16, .16); if (i % 4 === 0) m.arc([-.55 + .3 * ((i % 3) - 1), .15, 0], [x, -.55, 0], .2, 7); });
  };
  D['diffusion'] = (m, r) => {  // noise on one side, a clean sample on the other, steps between
    range(5).forEach((k) => {
      const x = -1 + .5 * k, mess = 1 - k / 4;
      range(14).forEach((i) => { const a = i / 14 * TAU; m.add(x, Math.sin(a) * .5 + (r() - .5) * mess, Math.cos(a) * .5 + (r() - .5) * mess); });
    });
    m.path(range(14).map((i) => { const a = i / 14 * TAU; return [1, Math.sin(a) * .5, Math.cos(a) * .5]; }), true);
  };
  D['nvidia-ampere-2020'] = (m) => {  // one A100 split into seven separate instances
    m.box(0, 0, 0, 1.9, .1, 1.1);
    spread(7, 1.7).forEach((x) => m.box(x, .12, 0, .2, .14, .9));
  };
  D['amd-cdna-2020'] = (m) => {  // a compute chip: the graphics blocks taken out, compute units left
    m.rectXY(0, 0, 0, 1.9, 1.3);
    for (let i = 0; i < 6; i += 1) for (let j = 0; j < 4; j += 1) m.rectXY(-.75 + .3 * i, -.45 + .3 * j, .01, .22, .22);
    m.line([-.95, .65, .02], [-.6, .3, .02], 4); m.line([-.95, .3, .02], [-.6, .65, .02], 4);  // the removed display block
  };
  D['mmlu-2020'] = (m) => {  // 57 subjects around four answer choices
    range(57).forEach((i) => { const a = i / 57 * TAU, rad = .85 + .1 * (i % 3); m.add(Math.cos(a) * rad, Math.sin(a) * rad, (i % 2) * .08); });
    [[-.25, .25], [.25, .25], [-.25, -.25], [.25, -.25]].forEach(([x, y], k) => (k === 2 ? m.ringXY(x, y, 0, .13, 12) : m.rectXY(x, y, 0, .2, .2)));
  };
  D['scaling-laws-2020'] = (m) => {  // loss falling on a straight line in log–log axes
    m.line([-1, -.9, 0], [1, -.9, 0], 6); m.line([-1, -.9, 0], [-1, 1, 0], 6);
    m.line([-.9, .8, 0], [.9, -.6, 0], 12);
    range(6).forEach((i) => { const t = i / 5; m.ringXY(-.8 + 1.6 * t, .72 - 1.25 * t + (i % 2 ? .05 : -.04), 0, .04, 6); });
    spread(3, 1.5).forEach((x) => m.line([x, -.95, 0], [x, -.85, 0]));
  };
  D['clip'] = (m) => {  // an image tower and a text tower meeting in one shared space
    m.gridXY(2, 2, .4, .4, 0, -.8, .7); m.dashes(.6, .8, 0, .4, 3, .12);
    m.line([-.8, .45, 0], [-.2, -.2, 0], 5); m.line([.8, .45, 0], [.2, -.2, 0], 5);
    m.gridXY(4, 4, .9, .9, 0, 0, -.65);
  };
  D['copilot-codex-2021'] = (m) => {  // an editor; the suggested lines appear in grey ahead of the cursor
    m.rectXY(0, 0, 0, 1.9, 1.4); m.line([-.95, .55, 0], [.95, .55, 0], 8);
    m.dashes(-.75, .35, 0, 1.1, 3, .18, (i) => [1, .7, .85][i]);
    range(3).forEach((i) => range(4).forEach((k) => m.add(-.6 + .25 * k + .05 * i, -.2 - .18 * i, .02)));
    m.line([-.65, -.08, 0], [-.65, .04, 0]);
  };
  D['instructgpt'] = (m) => {  // demonstrations, ranked answers, then a feedback loop
    m.rectXY(-.9, .4, 0, .4, .5); m.dashes(-1.03, .55, .01, .26, 3, .12);
    spread(3, .6).forEach((y, i) => m.rectXY(0, y + .2, 0, .5 - .1 * i, .16));
    m.path(range(24).map((i) => { const a = i / 24 * TAU; return [.85 + Math.cos(a) * .3, .2 + Math.sin(a) * .3, 0]; }), true);
    m.arrow([-.65, .4, 0], [-.3, .4, 0], .06); m.arrow([.28, .2, 0], [.5, .2, 0], .06);
  };

  /* ------------------------------------------------ 2022–2023: public tools */
  D['flash'] = (m) => {  // attention computed tile by tile inside fast on-chip memory
    m.gridXY(4, 4, 1.2, 1.2, 0, -.35, 0);
    m.box(.85, .3, 0, .5, .5, .4); m.rectXY(-.5, .15, .05, .3, .3); m.arrow([-.35, .15, .05], [.6, .3, 0], .06);
  };
  D['chatgpt'] = (m) => {  // a conversation: a question, an answer, a follow-up
    for (const [x, y, w] of [[-.35, .55, 1.1], [.3, -.05, 1.2], [-.35, -.65, .8]]) { m.rectXY(x, y, 0, w, .38); m.dashes(x - w / 2 + .12, y + .06, .01, w - .3, 2, .12); }
    m.path([[-.65, .36, 0], [-.75, .22, 0], [-.55, .36, 0]]); m.path([[.65, -.24, 0], [.75, -.38, 0], [.55, -.24, 0]]);
  };
  D['stable-diffusion'] = (m, r) => {  // a square of noise turning into an image, in a compressed space
    m.rectXY(-.75, 0, 0, .8, .8); range(30).forEach(() => m.add(-.75 + r() * .7 - .35, r() * .7 - .35, 0));
    m.rectXY(.75, 0, 0, .8, .8); m.ringXY(.6, .1, 0, .15, 12); m.path([[.4, -.3, 0], [.65, -.05, 0], [.85, -.25, 0], [1.1, .05, 0]]);
    range(3).forEach((k) => m.rectXY(-.2 + .2 * k, 0, 0, .1, .4 - .08 * k));
  };
  D['whisper-2022'] = (m) => {  // a waveform turned into a line of text
    m.path(range(30).map((i) => { const t = i / 29; return [-1 + 1 * t, .45 * Math.sin(t * 30) * Math.sin(t * Math.PI), 0]; }));
    m.arrow([.05, 0, 0], [.3, 0, 0], .06); m.tokens(4, 0, 0, .55, .1).forEach(([x]) => m.line([x + .4, -.12, 0], [x + .6, -.12, 0]));
  };
  D['laion-5b-2022'] = (m) => {  // billions of image–text pairs, spiralling out
    range(36).forEach((i) => { const a = i * .55, rad = .15 + i * .025, x = Math.cos(a) * rad, y = Math.sin(a) * rad; m.rectXY(x, y, 0, .07, .07); m.line([x + .05, y, 0], [x + .12, y, 0]); });
  };
  D['dpo'] = (m) => {  // a balance: the preferred answer outweighs the other
    m.line([0, -.9, 0], [0, .5, 0], 6); m.line([-.9, .65, 0], [.9, .35, 0], 8); m.path([[-.15, -.9, 0], [.15, -.9, 0]]);
    for (const [x, y] of [[-.85, .65], [.85, .35]]) { m.line([x, y, 0], [x - .2, y - .45, 0]); m.line([x, y, 0], [x + .2, y - .45, 0]); m.circle(12, .22, y - .45, x, 0); }
    m.rectXY(-.85, .35, 0, .2, .14); m.rectXY(.85, .05, 0, .2, .14);
  };
  D['vllm'] = (m) => {  // the KV cache in pages, handed out as requests need them
    m.gridXY(6, 4, 1.5, 1, 0, .3, .1);
    [[0, 0], [3, 1], [5, 2], [1, 3]].forEach(([i, j]) => m.box(.3 - .75 + .125 + .25 * i, .1 - .5 + .125 + .25 * j, .05, .2, .2, .1));
    spread(3, .9).forEach((y, k) => { m.rectXY(-.95, y, 0, .3, .2); m.arc([-.8, y, 0], [.3 - .75 + .125 + .25 * [0, 3, 5][k], .1 - .5 + .125 + .25 * [0, 1, 2][k], 0], .2, 7); });
  };
  D['amd-cdna3-mi300-2023'] = (m) => {  // CPU and GPU chiplets stacked on base dies, memory around
    m.rectXZ(0, -.3, 0, 2, 1.3);
    [[-.4, -.25], [.4, -.25], [-.4, .25], [.4, .25]].forEach(([x, z], k) => { m.box(x, -.2, z, .55, .1, .4); m.box(x, -.08, z, .4, .1, .3); if (k === 0) m.gridXZ(2, 2, .3, .2, -.02, x, z); });
    [-.9, .9].forEach((x) => [-.35, .35].forEach((z) => m.box(x, -.15, z, .15, .25, .25)));
  };
  D['segment-anything-2023'] = (m) => {  // one click, and the object's outline is cut out
    m.rectXY(0, 0, 0, 1.9, 1.4);
    m.path(range(24).map((i) => { const a = i / 24 * TAU, rad = .45 + .12 * Math.sin(a * 3); return [.1 + Math.cos(a) * rad, Math.sin(a) * rad * .8, .02]; }), true);
    m.star(.05, .05, .04, .1); m.ringXY(.05, .05, .04, .05, 8);
  };
  D['graphcast-2023'] = (m) => {  // the globe as a mesh of triangles, weather passed between them
    const g = (1 + Math.sqrt(5)) / 2, v = [[-1, g, 0], [1, g, 0], [-1, -g, 0], [1, -g, 0], [0, -1, g], [0, 1, g], [0, -1, -g], [0, 1, -g], [g, 0, -1], [g, 0, 1], [-g, 0, -1], [-g, 0, 1]].map((p) => p.map((x) => x / Math.hypot(...p)));
    const ids = m.dots(v); v.forEach((a, i) => v.forEach((b, j) => { if (j > i && Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1.1) { m.link(ids[i], ids[j]); const mid = a.map((x, k) => (x + b[k]) / 2), l = Math.hypot(...mid); m.add(...mid.map((x) => x / l)); } }));
  };

  /* ------------------------------------------------ 2024–2026: thinking longer, starting to act */
  D['grpo'] = (m) => {  // several answers to one prompt, compared against their group's average
    m.ringXY(0, .8, 0, .14, 12);
    spread(5, 1.8).forEach((x, i) => { const score = [.3, .55, .2, .7, .4][i]; m.rectXY(x, -.1, 0, .25, .3); m.line([0, .66, 0], [x, .05, 0], 5); m.rectXY(x, -.55 - score / 2 + .25, 0, .1, score); });
    m.line([-1, -.62, .05], [1, -.62, .05], 10);
  };
  D['o1-2024'] = (m) => {  // a long chain of thought winding towards the answer
    m.path(range(60).map((i) => { const t = i / 59, a = t * 4 * TAU; return [-1 + 1.7 * t, .45 * Math.sin(a) * (1 - t * .7), .45 * Math.cos(a) * (1 - t * .7)]; }));
    m.ringXY(.85, 0, 0, .14, 12); m.star(.85, 0, 0, .09);
  };
  D['gemini-1-5-2024'] = (m) => {  // a million tokens: a very long scroll, rolled up
    m.path(range(70).map((i) => { const a = i / 69 * 5 * TAU, rad = .2 + .7 * i / 69; return [Math.cos(a) * rad, (i / 69 - .5) * .5, Math.sin(a) * rad]; }));
    m.path(range(70).map((i) => { const a = i / 69 * 5 * TAU, rad = .2 + .7 * i / 69; return [Math.cos(a) * rad, (i / 69 - .5) * .5 + .3, Math.sin(a) * rad]; }));
    m.star(.3, .4, .5, .08);
  };
  D['nvidia-gb200-nvl72-2024'] = (m) => {  // a rack of trays joined by an NVLink spine
    m.box(0, 0, 0, .9, 2, .7);
    range(9).forEach((i) => m.rectXZ(0, -.85 + .21 * i, .02, .8, .6));
    m.line([.38, -.9, .36], [.38, .9, .36], 10); m.line([.32, -.9, .36], [.32, .9, .36], 10);
    range(4).forEach((i) => m.line([.32, -.6 + .4 * i, .36], [-.38, -.6 + .4 * i, .36], 5));
  };
  D['alphafold3-2024'] = (m) => {  // a protein, a strand of DNA and a small molecule fitted together
    m.path(range(50).map((i) => { const t = i / 49, a = t * 3 * TAU; return [Math.cos(a) * .3, -1 + 2 * t, Math.sin(a) * .3]; }));
    m.path(range(50).map((i) => { const t = i / 49, a = t * 3 * TAU + Math.PI; return [Math.cos(a) * .3, -1 + 2 * t, Math.sin(a) * .3]; }));
    m.path(range(30).map((i) => { const t = i / 29; return [.6 + .2 * Math.sin(t * 14), -.7 + 1.4 * t, .3 * Math.cos(t * 9)]; }));
    const ring = range(6).map((i) => { const a = i / 6 * TAU; return m.add(-.65 + Math.cos(a) * .15, .1 + Math.sin(a) * .15, .3); }); ring.forEach((id, i) => m.link(id, ring[(i + 1) % 6]));
  };
  D['llama-3-1-2024'] = (m) => {  // an open crate: the weights inside, layer on layer
    m.box(0, -.3, 0, 1.4, .8, 1);
    m.path([[-.7, .1, -.5], [-.95, .55, -.75], [.45, .55, -.75], [.7, .1, -.5]]);
    range(5).forEach((k) => m.rectXZ(0, -.6 + .15 * k, 0, 1.1, .7));
  };
  D['nobel-ai-2024'] = (m) => {  // two medals on ribbons: physics and chemistry
    for (const x of [-.55, .55]) {
      m.ringXY(x, -.15, 0, .45, 28); m.ringXY(x, -.15, 0, .33, 24);
      m.path([[x - .2, .25, 0], [x - .1, .95, 0], [x, .35, 0], [x + .1, .95, 0], [x + .2, .25, 0]]);
    }
  };
  D['mcp-2024'] = (m) => {  // one connector: an AI app plugged into many data and tool servers
    m.box(0, 0, 0, .45, .45, .45);
    range(6).forEach((i) => { const a = i / 6 * TAU, p = [Math.cos(a) * 1, Math.sin(a) * .8, 0]; m.cyl(p[0], p[1], 0, .1, .12, 10); m.line([Math.cos(a) * .28, Math.sin(a) * .28, 0], [p[0] * .88, p[1] * .88, 0], 5); m.rectXY(Math.cos(a) * .28, Math.sin(a) * .28, .23, .08, .05); });
  };
  D['deepseek-v3-2024'] = (m) => {  // a router and a ring of many small experts, a few lit
    m.ringXY(0, 0, 0, .15, 12);
    range(32).forEach((i) => { const a = i / 32 * TAU, on = i % 9 === 0 || i === 13; m.cube(Math.cos(a) * .9, Math.sin(a) * .9, 0, on ? .12 : .06); if (on) m.line([Math.cos(a) * .17, Math.sin(a) * .17, 0], [Math.cos(a) * .82, Math.sin(a) * .82, 0], 5); });
  };
  D['posttraining-open'] = (m) => {  // still an open question
    m.path(range(20).map((i) => { const a = Math.PI * 1.1 - i / 19 * Math.PI * 1.5; return [Math.cos(a) * .5, .45 + Math.sin(a) * .45, 0]; }));
    m.path([[.18, .02, 0], [0, -.15, 0], [0, -.45, 0]]); m.ringXY(0, -.8, 0, .07, 10);
    ['SFT', 'DPO', 'PPO', 'GRPO'].forEach((_, i) => { const a = i / 4 * TAU + .4; m.cube(Math.cos(a) * 1, Math.sin(a) * .8, 0, .14); });
  };
  D['deepseek-r1'] = (m) => {  // answers checked and rewarded, round and round
    m.path(range(40).map((i) => { const a = i / 40 * TAU; return [Math.cos(a) * .8, Math.sin(a) * .8, .25 * Math.sin(a * 3)]; }), true);
    m.arrow([.78, -.15, 0], [.75, .2, 0], .08);
    m.path([[-.25, 0, 0], [-.05, -.2, 0], [.3, .25, 0]]);
  };
  D['iea-energy-ai-2025'] = (m) => {  // a transmission tower feeding a data centre
    m.path([[-.7, -1, 0], [-.45, .8, 0], [-.2, -1, 0]]); m.line([-.75, .5, 0], [-.15, .5, 0], 4); m.line([-.7, .2, 0], [-.2, .2, 0], 4);
    m.arc([-.15, .5, 0], [.7, .1, 0], -.2, 8, 1); m.arc([-.75, .5, 0], [-1.2, .2, 0], -.15, 6, 1);
    m.box(.7, -.55, 0, .8, .7, .6); spread(3, .5).forEach((x) => m.line([.7 + x, -.85, .31], [.7 + x, -.3, .31], 4));
  };
  D['coding-agents-2025'] = (m) => {  // a terminal the agent works in: edits, a test run, a pass
    m.rectXY(0, 0, 0, 1.9, 1.3); m.line([-.95, .5, 0], [.95, .5, 0], 8);
    m.path([[-.8, .3, 0], [-.7, .22, 0], [-.8, .14, 0]]); m.dashes(-.6, .22, 0, 1, 3, .2, (i) => [.8, 1, .5][i]);
    m.path([[-.75, -.45, 0], [-.6, -.58, 0], [-.35, -.3, 0]]);
    range(8).forEach((i) => { const a = i / 8 * TAU; m.line([.6 + Math.cos(a) * .12, -.4 + Math.sin(a) * .12, 0], [.6 + Math.cos(a) * .2, -.4 + Math.sin(a) * .2, 0]); }); m.ringXY(.6, -.4, 0, .12, 10);
  };
  D['imo-2025-gold'] = (m) => {  // an olympiad geometry figure: a circle, a triangle, its centres
    m.ringXY(0, 0, 0, .9, 36);
    const at = [[Math.cos(1.9) * .9, Math.sin(1.9) * .9, 0], [Math.cos(3.9) * .9, Math.sin(3.9) * .9, 0], [Math.cos(5.6) * .9, Math.sin(5.6) * .9, 0]];
    at.forEach((p, i) => m.line(p, at[(i + 1) % 3], 8)); const c = at.reduce((s, p) => s.map((v, k) => v + p[k] / 3), [0, 0, 0]);
    at.forEach((p) => m.line(p, c, 5)); m.ringXY(...c, .05, 8);
  };
  D['genie-3-2025'] = (m) => {  // rolling ground generated around a path you walk
    for (let i = 0; i < 9; i += 1) m.path(range(9).map((j) => { const x = -1 + .25 * i, z = -1 + .25 * j; return [x, .25 * Math.sin(x * 2.4 + 1) * Math.cos(z * 2.1) + .15 * Math.sin((x + z) * 2), z]; }));
    for (let j = 0; j < 9; j += 1) m.path(range(9).map((i) => { const x = -1 + .25 * i, z = -1 + .25 * j; return [x, .25 * Math.sin(x * 2.4 + 1) * Math.cos(z * 2.1) + .15 * Math.sin((x + z) * 2), z]; }));
    m.ringXY(0, .5, 0, .08, 8); m.line([0, .42, 0], [0, .2, 0], 3);
  };
  D['ai-research-math-2026'] = (m) => {  // the Moser spindle: a graph whose every edge has length one
    const s3 = Math.sqrt(3), a = [0, 0], rot = (p, t) => [p[0] * Math.cos(t) - p[1] * Math.sin(t), p[0] * Math.sin(t) + p[1] * Math.cos(t)];
    const rhombus = [[0, 0], [1, 0], [.5, s3 / 2], [1.5, s3 / 2]], t = 2 * Math.asin(1 / (2 * s3));
    const left = rhombus, right = rhombus.map((p) => rot(p, t));
    const pts = [...left, ...right.slice(1)].map(([x, y]) => [x - .75, y - .5, 0]), ids = m.dots(pts);
    [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [0, 4], [0, 5], [4, 5], [4, 6], [5, 6], [3, 6]].forEach(([i, j]) => { const p = pts[i], q = pts[j]; const mid = range(4).map((k) => m.add(...p.map((v, d) => v + (q[d] - v) * (k + 1) / 5))); [ids[i], ...mid, ids[j]].reduce((x, y) => { m.link(x, y); return y; }); });
    pts.forEach((p) => m.ringXY(...p, .06, 8));
  };
  D['deepseek-v4-2026'] = (m) => {  // a long context folded small: a ribbon pleated into a block
    m.path(range(48).map((i) => { const t = i / 47; return [-1 + 2 * t, (i % 2 ? .45 : -.45) * (t < .55 ? 1 : .25), 0]; }));
    m.box(.75, 0, 0, .45, .35, .35); m.arrow([.15, .55, 0], [.55, .25, 0], .06);
  };
  D['tpu-8-2026'] = (m) => {  // two chips from one generation: a wide one for training, a memory-heavy one for serving
    m.box(-.6, 0, 0, .9, .12, .9); m.gridXZ(3, 3, .7, .7, .07, -.6, 0);
    m.box(.7, 0, 0, .5, .12, .6); [-.2, .2].forEach((z) => range(4).forEach((k) => m.rectXZ(1.05, -.05 + .07 * k, z, .18, .15)));
  };
  D['cyber-release-2026'] = (m) => {  // a shield with a lock: capability handed to defenders first
    for (const z of [-.1, .1]) m.path([[-.8, .9, z], [.8, .9, z], [.8, .1, z], [.45, -.5, z], [0, -.95, z], [-.45, -.5, z], [-.8, .1, z]], true);
    m.rectXY(0, -.05, .12, .5, .4); m.arc([-.18, .15, .12], [.18, .15, .12], .3, 9, 1); m.ringXY(0, -.05, .13, .06, 8);
  };

  /* ------------------------------------------------ redrawn after review: clearer forms */
  D['robbins-monro-stochastic-approximation-1951'] = (m) => {  // noisy steps, each smaller, closing in on the target
    m.ringXY(0, 0, 0, .1, 12); m.ringXY(0, 0, 0, .22, 16); m.star(0, 0, 0, .06);
    const steps = range(9).map((i) => { const s = Math.pow(.72, i), a = i * 2.2 + .6; return [Math.cos(a) * s, Math.sin(a) * s, 0]; });
    steps.forEach((p, i) => { m.ringXY(...p, .045, 8); if (i) m.arrow(steps[i - 1], p, .07 * Math.pow(.85, i)); });
  };
  D['dennard-dram-patent-1967'] = (m) => {  // one cell: a transistor switch and a capacitor, on the word and bit lines
    m.line([-1.1, .75, 0], [1.1, .75, 0], 8); m.line([-.75, -1, 0], [-.75, 1, 0], 8);
    m.box(-.15, .25, 0, .45, .3, .25); m.line([-.75, .25, 0], [-.375, .25, 0], 3); m.line([-.15, .4, 0], [-.15, .75, 0], 3);
    m.line([.075, .25, 0], [.45, .25, 0], 3);
    m.box(.45, -.15, 0, .5, .05, .4); m.box(.45, -.4, 0, .5, .05, .4); m.line([.45, .25, 0], [.45, -.13, 0], 3); m.line([.45, -.42, 0], [.45, -.8, 0], 3);
  };
  D['software-gnu-announcement-1983'] = (m) => {  // copyleft: a reversed C inside a ring, copies passed on
    m.ringXY(0, 0, 0, .55, 30); m.ringXY(0, 0, .06, .55, 30);
    m.path(range(18).map((i) => { const a = -1.15 + 2.3 * i / 17; return [-Math.cos(a) * .3 + .04, Math.sin(a) * .3, 0]; }));
    for (const x of [-1.05, 1.05]) { m.rectXY(x, -.15, -.2, .32, .42); m.dashes(x - .1, -.02, -.19, .2, 3, .1); }
    m.arc([.6, .1, 0], [.9, .1, -.2], .15, 6, 1); m.arc([-.6, .1, 0], [-.9, .1, -.2], .15, 6, 1);
  };
  D['networks-tcp-ip-transition-1983'] = (m) => {  // four protocol layers, a packet wrapped in headers as it goes down
    range(4).forEach((i) => { const y = .75 - .5 * i; m.rectXY(-.45, y, 0, .9, .32); m.dashes(-.8, y + .05, .01, .3, 1); });
    range(4).forEach((i) => { const y = .75 - .5 * i, w = .18 + .1 * i; m.rectXY(.55, y, 0, w, .18); if (i) m.arrow([.55, y + .38, 0], [.55, y + .12, 0], .05); });
  };
  D['gpl-v1-1989'] = (m) => {  // the licence travels with the source: a page linked to every copy
    m.rectXY(0, .25, 0, .7, .9); m.dashes(-.25, .5, .01, .5, 4, .14, (i) => [1, .8, 1, .6][i]);
    [-.9, 0, .9].forEach((x) => { m.rectXY(x, -.75, 0, .38, .45); m.dashes(x - .12, -.65, .01, .24, 2, .12); m.line([x * .3, -.2, 0], [x, -.52, 0], 4); });
  };
  D['software-linux-kernel-1991'] = (m) => {  // a kernel at the centre, the system built around it in rings
    m.sphere(0, 0, 0, .3, 2, 3, 16);
    m.circle(36, .62, 0); m.circle(36, .95, 0);
    range(8).forEach((i) => { const a = i / 8 * TAU; m.cube(Math.cos(a) * .95, 0, Math.sin(a) * .95, .1); m.line([Math.cos(a) * .3, 0, Math.sin(a) * .3], [Math.cos(a) * .62, 0, Math.sin(a) * .62], 3); });
  };
  D['software-numpy-2005'] = (m) => {  // a three-dimensional array, its axes marked
    m.box(0, 0, 0, 1.2, 1.2, 1.2);
    for (const v of [-.2, .2]) { m.line([v, -.6, .6], [v, .6, .6], 5); m.line([-.6, v, .6], [.6, v, .6], 5); m.line([.6, v, -.6], [.6, v, .6], 5); m.line([.6, -.6, v], [.6, .6, v], 5); m.line([-.6, .6, v], [.6, .6, v], 5); m.line([v, .6, -.6], [v, .6, .6], 5); }
    m.arrow([-.6, -.6, .6], [-1, -.6, .6], .07); m.arrow([-.6, -.6, .6], [-.6, -1, .6], .07); m.arrow([-.6, -.6, .6], [-.6, -.6, 1], .07);
  };
  D['stanley-grand-challenge-2005'] = (m) => {  // a car with a laser scanner on its roof, sweeping the track ahead
    m.box(0, -.25, 0, 1.2, .3, .6); m.box(-.15, .02, 0, .6, .25, .55);
    [-.4, .4].forEach((x) => [-.32, .32].forEach((z) => m.ringXY(x, -.45, z, .13, 10)));
    m.box(-.05, .22, 0, .14, .1, .14);
    range(5).forEach((i) => { const z = -.5 + .25 * i; m.line([-.05, .22, 0], [1.2, -.55, z], 6); });
    m.line([1.2, -.55, -.5], [1.2, -.55, .5], 5);
  };
  D['distributed-amazon-ec2-beta-2006'] = (m) => {  // servers rented by the hour: a row of machines under one call
    spread(4, 1.6).forEach((x, i) => { m.box(x, -.25, 0, .3, .55, .35); m.line([x - .1, -.1, .18], [x + .1, -.1, .18]); m.line([x - .1, -.3, .18], [x + .1, -.3, .18]); m.line([0, .55, 0], [x, .03, 0], 4); });
    m.rectXY(0, .65, 0, .5, .2); m.path([[-.12, .65, .01], [.12, .65, .01]]);
    m.ringXY(.75, .7, 0, .16, 14); m.line([.75, .7, 0], [.75, .8, 0]); m.line([.75, .7, 0], [.83, .7, 0]);  // a clock: paid by the hour
  };
  D['alexnet'] = (m) => {  // two GPUs side by side, each with half the network, crossing at a few layers
    for (const z of [-.35, .35]) [0, 1, 2, 3, 4].forEach((i) => { const s = .8 - .14 * i; m.rectXY(-1 + .5 * i, 0, z, .12, s); m.rectXY(-1 + .5 * i, s / 2 - .05, z, .12, .1); });
    for (const i of [1, 3]) { m.line([-1 + .5 * i, .1, -.35], [-.5 + .5 * i, -.1, .35], 4); m.line([-1 + .5 * i, -.1, .35], [-.5 + .5 * i, .1, -.35], 4); }
    for (const z of [-.35, .35]) m.box(-.1, -.75, z, 2, .08, .25);
  };
  D['faster-rcnn-2015'] = (m) => {  // the network proposes boxes; a car and a person are found
    m.rectXY(0, 0, 0, 2, 1.4);
    m.box(-.45, -.25, .05, .7, .25, .2); [-.65, -.25].forEach((x) => m.ringXY(x, -.4, .15, .08, 8));
    m.rectXY(-.45, -.22, .02, .9, .5); m.ringXY(.55, .25, .05, .09, 10); m.line([.55, .16, .05], [.55, -.25, .05], 4); m.rectXY(.55, 0, .02, .35, .75);
    [[.2, .3, .3, .25], [-.1, .45, .4, .2]].forEach(([x, y, w, h]) => m.rectXY(x, y, .03, w, h));
  };
  D['ppo'] = (m) => {  // a step held inside a trust band: small updates keep the policy close
    m.path(range(25).map((i) => { const x = -1 + 2 * i / 24; return [x, -.5 + 1 * Math.exp(-x * x / .3), 0]; }));
    m.path(range(25).map((i) => { const x = -1 + 2 * i / 24; return [x, -.5 + 1 * Math.exp(-(x - .25) * (x - .25) / .3), .15]; }));
    m.line([-.25, -.7, .08], [-.25, .7, .08], 6); m.line([.5, -.7, .08], [.5, .7, .08], 6);
    m.arrow([0, .62, .08], [.25, .62, .08], .06); m.line([-1.05, -.5, 0], [1.05, -.5, 0], 8);
  };
  D['energy-tpu-v3-liquid-cooling-2018'] = (m) => {  // cold plates on two chips, water flowing in and out through them
    for (const x of [-.5, .5]) { m.box(x, -.3, 0, .55, .1, .55); m.box(x, -.15, 0, .5, .15, .5); m.path(range(9).map((i) => [x - .2 + .05 * i, -.06, (i % 2 ? .18 : -.18)])); }
    m.path([[-1.1, .3, 0], [-.7, .3, 0], [-.7, -.06, -.18]]); m.path([[.7, -.06, .18], [.7, .3, 0], [1.1, .3, 0]]); m.line([-.3, -.06, .18], [.3, -.06, -.18], 4);
    m.arrow([-1.1, .45, 0], [-.8, .45, 0], .06); m.arrow([.8, .45, 0], [1.1, .45, 0], .06);
  };
  D['alphafold2'] = (m) => {  // a protein chain: two helices and a sheet, folded together
    for (const [x0, z0] of [[-.55, .1], [.1, -.15]]) m.path(range(36).map((i) => { const t = i / 35, a = t * 4 * TAU; return [x0 + .18 * Math.cos(a), -.7 + 1.4 * t, z0 + .18 * Math.sin(a)]; }));
    range(3).forEach((k) => { const x = .55 + .17 * k; m.path([[x, -.7, .2], [x + .05, -.35, .2], [x, 0, .2], [x + .05, .35, .2], [x, .7, .2]]); });
    m.arc([-.55, .7, .1], [.1, .7, -.15], .3, 7, 1); m.arc([.1, -.7, -.15], [.55, -.7, .2], -.25, 7, 1);
  };
  D['nvidia-hopper-2022'] = (m) => {  // the H100: one die between six memory stacks on a package
    m.rectXZ(0, -.2, 0, 2, 1.4); m.box(0, -.1, 0, .8, .14, 1);
    m.gridXZ(4, 6, .6, .85, -.02, 0, 0);
    [-.7, .7].forEach((x) => spread(3, 1).forEach((z) => { m.box(x, -.08, z, .32, .18, .28); m.line([x - .16, -.02, z], [x + .16, -.02, z]); }));
  };
  D['rt2'] = (m) => {  // words from a vision-language model become the arm's next move
    m.rectXY(-.75, .55, 0, .7, .45); m.dashes(-1, .65, .01, .5, 3, .12, (i) => [1, .7, .85][i]);
    m.ringXY(-.75, .55, 0, .08, 8);
    m.circle(16, .3, -.95, .5, 0); m.line([.5, -.95, 0], [.5, -.35, 0], 4); m.ringXY(.5, -.35, 0, .06, 8);
    m.line([.5, -.35, 0], [.95, .05, 0], 4); m.ringXY(.95, .05, 0, .06, 8); m.line([.95, .05, 0], [.75, .3, 0], 3);
    m.line([.75, .3, 0], [.62, .38, 0]); m.line([.75, .3, 0], [.7, .45, 0]);
    m.arrow([-.35, .45, 0], [.55, .35, 0], .07);
  };

  /* ------------------------------------------------ pivotal works added after the gap review */
  const ringX = (m, x, cy, cz, radius, n = 12) => m.path(range(n).map((i) => [x, cy + Math.cos(i / n * TAU) * radius, cz + Math.sin(i / n * TAU) * radius]), true);
  const ellipse = (m, cx, cy, z, rx, ry, turn = 0, n = 18) => m.path(range(n).map((i) => {
    const a = i / n * TAU, x = Math.cos(a) * rx, y = Math.sin(a) * ry;
    return [cx + x * Math.cos(turn) - y * Math.sin(turn), cy + x * Math.sin(turn) + y * Math.cos(turn), z];
  }), true);
  const prism = (m, outline, z0, z1) => {  // an outline in the x-y plane, extruded between two depths
    const front = m.path(outline.map(([x, y]) => [x, y, z0]), true), back = m.path(outline.map(([x, y]) => [x, y, z1]), true);
    front.forEach((id, i) => m.link(id, back[i]));
  };
  const toward = (a, b, cut = .14) => [a.map((v, k) => v + (b[k] - v) * cut), b.map((v, k) => v - (b[k] - v) * cut)];

  D['ac-niagara-1895'] = (m) => {  // a powerhouse at the falls; towers carry AC lines into the distance
    m.box(-1.15, -.35, 0, .5, .5, .6); m.path([[-1.4, -.1, -.3], [-1.15, .12, -.3], [-.9, -.1, -.3]]); m.path([[-1.4, -.1, .3], [-1.15, .12, .3], [-.9, -.1, .3]]);
    spread(5, .5).forEach((z) => m.line([-1.48, .1, z], [-1.48, -.6, z], 4));  // falling water
    const towers = [[-.55, 0], [0, -.35], [.55, -.7], [1.1, -1.05]];
    towers.forEach(([x, z]) => { m.path([[x - .12, -.6, z], [x, .45, z], [x + .12, -.6, z]]); m.line([x - .22, .3, z], [x + .22, .3, z]); });
    for (const s of [-.22, .22]) {
      m.arc([-.9, -.1, 0], [towers[0][0] + s, .3, towers[0][1]], -.08, 6, 1);
      for (let i = 0; i + 1 < towers.length; i += 1) m.arc([towers[i][0] + s, .3, towers[i][1]], [towers[i + 1][0] + s, .3, towers[i + 1][1]], -.1, 7, 1);
    }
    m.path(range(13).map((i) => [-1.4 + i * .05, .55 + .1 * Math.sin(i / 12 * TAU), 0]));
  };
  D['edvac-1945'] = (m) => {  // five organs; the memory holds the instructions as well as the numbers
    m.box(0, -.45, 0, 1.2, .45, .5); m.dashes(-.45, -.33, .26, .9, 3, .11, (i) => [1, .55, .8][i]);
    m.box(-.35, .4, 0, .5, .35, .4); m.box(.35, .4, 0, .5, .35, .4);  // control, arithmetic
    m.box(-1.15, -.05, 0, .35, .3, .35); m.box(1.15, -.05, 0, .35, .3, .35);  // input, output
    m.arrow([-.95, -.1, 0], [-.62, -.32, 0], .08); m.arrow([.62, -.32, 0], [.95, -.1, 0], .08);
    m.line([-.35, .22, 0], [-.35, -.22, 0], 3); m.line([.35, .22, 0], [.35, -.22, 0], 3); m.line([-.1, .4, 0], [.1, .4, 0]);
  };
  D['turing-test-1950'] = (m) => {  // an interrogator types to two hidden players through a wall
    m.box(0, 0, 0, .06, 1.2, 1.6);
    m.box(-.95, -.35, 0, .4, .25, .35); m.ringXY(-1.05, .15, 0, .14, 10);
    m.box(.95, -.35, -.55, .35, .25, .3); m.ringXY(.95, .08, -.55, .12, 10);  // a person
    m.box(.95, -.35, .55, .35, .25, .3); m.cube(.95, .1, .55, .22);  // a machine
    m.arc([-.75, -.2, 0], [.78, -.2, -.55], .3, 9, 1); m.arc([-.75, -.2, 0], [.78, -.2, .55], .3, 9, 1);
  };
  D['integrated-circuit-1959'] = (m) => {  // one slab of semiconductor; the parts are made in it and joined on top
    m.box(0, -.15, 0, 2, .2, 1.1);
    const y = -.05;
    m.path(range(9).map((i) => [-.8 + .07 * i, y, -.2 + (i % 2 ? .12 : -.12)]));  // resistor
    m.rectXZ(.1, y, .25, .3, .3); m.rectXZ(.1, y + .08, .25, .3, .3);  // capacitor
    m.box(.65, y + .06, -.15, .25, .12, .25);  // transistor
    m.path([[-.24, y, -.2], [-.24, y, .25], [-.05, y, .25]]); m.path([[.25, y + .08, .25], [.65, y, .25], [.65, y, -.02]]);
    m.arc([-.85, y, -.25], [-1.3, -.35, -.25], .3, 7, 1); m.arc([.78, y, -.15], [1.3, -.35, -.15], .3, 7, 1);
  };
  D['mosfet-1960'] = (m) => {  // a gate over a thin oxide; its field opens a channel from source to drain
    m.box(0, -.45, 0, 2, .6, .9);
    m.box(-.7, -.25, 0, .45, .2, .7); m.box(.7, -.25, 0, .45, .2, .7);  // source, drain
    m.box(0, -.12, 0, 1, .04, .7);  // oxide
    m.box(0, .1, 0, .9, .36, .6);  // gate
    spread(5, .7).forEach((x) => m.line([x, -.08, .36], [x, -.3, .36], 4));
    m.arrow([-.5, -.36, .4], [.5, -.36, .4], .1);
    m.line([-.7, -.15, 0], [-.7, .55, 0], 3); m.line([.7, -.15, 0], [.7, .55, 0], 3); m.line([0, .28, 0], [0, .65, 0], 3);
  };
  D['lisp-1960'] = (m) => {  // a list of cons cells, one holding a nested list, between parentheses
    const cell = (x, y, z) => { m.rectXY(x - .1, y, z, .2, .2); m.rectXY(x + .1, y, z, .2, .2); };
    const top = [-.8, -.1, .6];
    top.forEach((x, i) => { cell(x, .5, 0); if (i < 2) m.arrow([x + .1, .5, 0], [x + .5, .5, 0], .07); });
    m.line([.6, .4, 0], [.8, .6, 0]);  // nil
    m.arrow([-.9, .4, 0], [-.9, -.02, 0], .07); m.ringXY(-.9, -.14, 0, .11, 10);
    m.arrow([.5, .4, 0], [.5, -.02, 0], .07); m.ringXY(.5, -.14, 0, .11, 10);
    m.arrow([-.2, .4, 0], [-.3, -.3, .5], .07);
    cell(-.3, -.42, .5); cell(.4, -.42, .5); m.arrow([-.2, -.42, .5], [.2, -.42, .5], .07);
    m.ringXY(-.4, -.85, .5, .09, 10); m.line([-.4, -.52, .5], [-.4, -.75, .5]);
    m.arc([-1.2, .85, 0], [-1.2, -.95, 0], -.18, 11, 0); m.arc([1.05, .85, 0], [1.05, -.95, 0], .18, 11, 0);
  };
  D['kalman-filter-1960'] = (m) => {  // along a track: predict (wide), measure (×), correct (narrow)
    const xs = [-1.1, -.4, .3, 1], ys = [-.35, .05, .2, .5];
    xs.forEach((x, i) => {
      ellipse(m, x, ys[i], 0, .12, .09);
      if (i) { ellipse(m, x - .04, ys[i] - .06, -.3, .28, .2); m.star(x + .1, ys[i] + .14, .3, .07); }
    });
    m.path(xs.map((x, i) => [x, ys[i], 0]));
    for (let i = 0; i + 1 < xs.length; i += 1) m.arc([xs[i], ys[i] - .1, 0], [xs[i + 1] - .04, ys[i + 1] - .26, -.3], -.12, 6, 1);
  };
  D['atlas-virtual-memory-1962'] = (m) => {  // one long address space above; each page lives in core or on the drum
    const pages = m.tokens(8, .75, 0, 2, .18);
    m.gridXY(2, 2, .6, .6, .1, -.7, -.35);  // core: four page frames
    m.cyl(.7, -.4, -.2, .38, .6, 16); m.circle(16, .38, -.25, .7, -.2);  // drum
    pages.forEach(([x, y, z], i) => {
      const core = i % 3 === 0, target = core ? [-.85 + .3 * (i % 2), -.2, .1] : [.7 + Math.cos(i) * .3, -.1, -.2 + Math.sin(i) * .3];
      m.line([x, y - .09, z], target);
    });
    m.arrow([-.35, -.25, .1], [.28, -.25, 0], .07); m.arrow([.28, -.5, 0], [-.35, -.5, .1], .07);
  };
  D['packet-switching-1964'] = (m) => {  // a distributed mesh; message blocks route around a lost node
    const at = (i, j) => [-1 + i * .66, -.6 + j * .6, (i + j) % 2 ? .25 : -.25], lost = '1,1';
    for (let i = 0; i < 4; i += 1) for (let j = 0; j < 3; j += 1) {
      if (`${i},${j}` !== lost) m.node(...at(i, j), .06);
      for (const [a, b] of [[1, 0], [0, 1], [1, 1]]) {
        const k = `${i + a},${j + b}`;
        if (i + a < 4 && j + b < 3 && k !== lost && `${i},${j}` !== lost) m.line(...toward(at(i, j), at(i + a, j + b), .1), 3);
      }
    }
    const [x, y, z] = at(1, 1); m.line([x - .1, y - .1, z], [x + .1, y + .1, z]); m.line([x - .1, y + .1, z], [x + .1, y - .1, z]);
    [[0, 0, 1, 0], [1, 0, 2, 1], [0, 1, 0, 2], [0, 2, 1, 2], [2, 1, 3, 2]].forEach(([a, b, c, d]) => {
      const p = at(a, b), q = at(c, d); m.rectXY((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + .07, (p[2] + q[2]) / 2, .1, .08);
    });
  };
  D['optical-fibre-1966'] = (m) => {  // light bouncing along a glass fibre: a core inside its cladding
    spread(7, 2.2).forEach((x) => { ringX(m, x, 0, 0, .3); ringX(m, x, 0, 0, .14, 10); });
    for (const s of [-.3, .3]) { m.line([-1.1, s, 0], [1.1, s, 0], 8); m.line([-1.1, 0, s], [1.1, 0, s], 8); }
    m.path(range(12).map((i) => [-1.1 + i * .2, i % 2 ? .12 : -.12, 0]));
    m.star(-1.35, -.12, 0, .12);
  };
  D['dennard-scaling-1974'] = (m) => {  // the same area holds more, smaller transistors at the same power density
    [-1.05, 0, 1.05].forEach((cx, k) => {
      m.rectXZ(cx, -.2, 0, .8, .8);
      const n = k + 1, s = .6 / n;
      for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) {
        const x = cx - .3 + s * (i + .5), z = -.3 + s * (j + .5);
        m.box(x, -.2 + s * .2, z, s * .7, s * .4, s * .7); m.line([x, -.2 + s * .4, z], [x, -.2 + s * .75, z]);
      }
    });
    m.arrow([-.75, .5, 0], [.75, .5, 0], .1);
  };
  D['em-algorithm-1977'] = (m, r) => {  // points from two hidden groups; the fit alternates E and M steps
    [[-.5, .2, .5], [.45, -.25, -.6]].forEach(([cx, cy, turn]) => {
      range(14).forEach(() => {
        const u = (r() - .5) * .9, v = (r() - .5) * .35;
        m.add(cx + u * Math.cos(turn) - v * Math.sin(turn), cy + u * Math.sin(turn) + v * Math.cos(turn), (r() - .5) * .3);
      });
      ellipse(m, cx, cy, 0, .5, .2, turn); ellipse(m, cx + .12, cy - .06, -.25, .36, .3, turn * .4);
    });
    m.arc([.75, .75, 0], [1.3, .75, 0], .22, 8, 1); m.arc([1.3, .75, 0], [.75, .75, 0], -.22, 8, 1);
    m.arrow([1.22, .82, 0], [1.3, .75, 0], .06); m.arrow([.83, .68, 0], [.75, .75, 0], .06);
  };
  D['risc-1980'] = (m) => {  // simple instructions overlapped in a pipeline, on one chip
    range(4).forEach((i) => range(5).forEach((s) => m.rectXY(-.95 + .3 * (i + s), .5 - .3 * i, -.18 * i, .26, .22)));
    m.arrow([-1.1, -.55, 0], [1.3, -.55, -.5], .08);
    m.rectXZ(.15, -.8, -.3, 2.2, .9);
  };
  D['flash-memory-1984'] = (m) => {  // a block of floating-gate cells, all erased together
    for (let i = 0; i < 4; i += 1) for (let j = 0; j < 3; j += 1) {
      const x = -.75 + .5 * i, z = -.4 + .4 * j;
      m.box(x, -.3, z, .3, .08, .22); m.box(x, -.13, z, .3, .1, .22);  // floating gate, control gate
    }
    m.rectXZ(0, -.42, 0, 2.1, 1.3); m.rectXZ(0, -.42, 0, 2.2, 1.4);
    spread(4, 1.5).forEach((x) => { m.arrow([x, .02, 0], [x, .55, 0], .08); m.add(x - .08, .3, .15); m.add(x + .08, .4, -.15); });
  };
  D['bayesian-networks-1985'] = (m) => {  // a dependency graph; beliefs pass along the arrows, tables at the nodes
    const P = [[-.7, .7, .2], [.6, .75, -.2], [0, .1, 0], [-.85, -.45, -.25], [.8, -.25, .3], [0, -.85, 0]];
    P.forEach((p) => m.node(...p, .1));
    [[0, 2], [1, 2], [0, 3], [2, 4], [2, 5], [3, 5], [1, 4]].forEach(([a, b]) => {
      const [s, e] = toward(P[a], P[b], .16); m.arrow(s, e, .07);
      const mid = s.map((v, k) => v + (e[k] - v) * .45); m.ringXY(mid[0] + .06, mid[1], mid[2], .03, 6);
    });
    m.gridXY(2, 2, .26, .2, .2, -1.15, .85); m.gridXY(2, 4, .26, .34, 0, .45, .2);
  };
  D['tsmc-foundry-1987'] = (m) => {  // designs from many companies go into one fab; wafers come out
    m.rectXY(-1.15, .45, 0, .32, .3); m.gridXY(2, 2, .2, .18, 0, -1.15, .45);
    m.rectXY(-1.15, 0, 0, .32, .3); m.dashes(-1.27, .07, 0, .24, 3, .07);
    m.rectXY(-1.15, -.45, 0, .32, .3); m.ringXY(-1.15, -.45, 0, .09, 10);
    [.45, 0, -.45].forEach((y) => m.arrow([-.95, y, 0], [-.5, -.1, 0], .06));
    m.box(0, -.2, 0, .9, .5, .8);
    m.path(range(7).map((i) => [-.45 + .15 * i, .05 + (i % 2 ? .18 : 0), .4])); m.path(range(7).map((i) => [-.45 + .15 * i, .05 + (i % 2 ? .18 : 0), -.4]));
    m.arrow([.5, -.25, 0], [.78, -.35, .1], .07);
    [[1.1, .15], [1.25, -.15]].forEach(([x, z]) => m.ringXY(x, -.35, z, .3, 20)); m.gridXY(4, 4, .36, .36, .15, 1.1, -.35);
  };
  D['neural-language-model-2003'] = (m) => {  // three previous words become vectors; a network scores every next word
    const words = m.tokens(3, -.95, 0, 1.2, .2);
    m.gridXY(6, 4, 1.4, .55, -.45, 0, -.6);  // the shared table of word vectors
    words.forEach(([x]) => { range(4).forEach((k) => m.rectXY(x, -.6 + .13 * k, 0, .1, .1)); m.line([x, -.85, 0], [x, -.67, 0]); });
    const hidden = spread(5, 1.3).map((x) => m.add(x, .3, .1));
    words.forEach(([x]) => hidden.forEach((h, i) => { if ((i + x * 10) % 2 < 1.2) m.link(m.add(x, -.15, 0), h); }));
    const bars = [.1, .25, .5, .15, .35, .8, .2, .1, .3, .15, .05, .2];
    spread(12, 2.1).forEach((x, i) => m.line([x, .6, .1], [x, .6 + bars[i] * .5, .1], 3));
    hidden.forEach((h) => m.link(h, m.add(0, .58, .1)));
  };
  D['gan-2014'] = (m, r) => {  // noise → generator → a sample; the discriminator judges it against real data
    range(10).forEach(() => m.add(-1.35 + r() * .25, -.35 + r() * .3, (r() - .5) * .3));
    prism(m, [[-.95, -.3], [-.5, -.5], [-.5, .1], [-.95, -.1]], -.18, .18);  // generator widens
    m.rectXY(-.05, -.2, 0, .3, .3); m.ringXY(-.05, -.2, 0, .08, 8);  // its sample
    range(3).forEach((i) => m.rectXY(-.05 + .04 * i, .45, -.12 * i, .3, .3)); m.ringXY(-.05, .45, 0, .1, 10);  // real data
    prism(m, [[.35, -.4], [.8, -.15], [.8, .15], [.35, .55]], -.18, .18);  // discriminator narrows
    m.arrow([.15, -.2, 0], [.33, -.15, 0], .06); m.arrow([.15, .45, 0], [.33, .4, 0], .06);
    m.node(1.1, 0, 0, .09); m.line([.82, 0, 0], [1.01, 0, 0]);
    m.arc([1.05, -.15, 0], [-.7, -.55, 0], -.35, 13, 1);  // the generator learns from the judge
  };
  D['gender-shades-2018'] = (m) => {  // faces in four groups; one group's error bar towers over the rest
    const heights = [.06, .3, .45, 1.25];
    heights.forEach((h, g) => {
      const cx = -1 + g * .66;
      for (let i = 0; i < 2; i += 1) for (let j = 0; j < 2; j += 1) {
        m.ringXY(cx - .12 + .24 * i, -.55 + .26 * j, .3, .09, 10); m.arc([cx - .2 + .24 * i, -.68 + .26 * j, .3], [cx - .04 + .24 * i, -.68 + .26 * j, .3], -.04, 4, 1);
      }
      m.box(cx, -.75 + h / 2, -.2, .3, h, .25);
    });
    m.line([-1.3, -.75, -.2], [1.3, -.75, -.2], 6);
  };
  D['rag-2020'] = (m) => {  // a question pulls a few passages from an index; the generator answers from them
    range(6).forEach((i) => m.rectXZ(-.65, -.75 + .1 * i, 0, .75, .5));  // the index
    m.rectXY(-.65, .6, 0, .6, .16); m.dashes(-.88, .6, .01, .45, 1);  // the question
    m.arrow([-.65, .5, 0], [-.65, -.15, 0], .08);
    [[.05, .35], [.05, -.05]].forEach(([x, y]) => { m.rectXY(x, y, .1, .4, .3); m.dashes(x - .14, y + .08, .11, .28, 3, .08); });
    m.arc([-.3, -.3, 0], [-.15, .2, .1], .15, 5, 0);
    m.box(.85, .15, 0, .4, .55, .4);  // the generator
    m.arrow([.27, .3, .1], [.63, .25, 0], .06); m.arrow([.27, -.05, .1], [.63, .05, 0], .06);
    m.dashes(1.1, .2, 0, .3, 3, .1, (i) => [1, .8, .5][i]);
  };
  D['waymo-driverless-2020'] = (m) => {  // a car with an empty driver's seat and a spinning sensor on the roof
    m.box(0, -.35, 0, 1.6, .32, .8);
    for (const z of [-.4, .4]) m.path([[-.55, -.19, z], [-.3, .18, z], [.35, .18, z], [.6, -.19, z]]);
    [[-.3, .18], [.35, .18]].forEach(([x, y]) => m.line([x, y, -.4], [x, y, .4]));
    m.path([[-.25, -.18, .2], [.05, -.18, .2], [.05, .08, .2]]);  // the driver's seat, empty
    m.path([[-.25, -.18, -.2], [.05, -.18, -.2], [.05, .08, -.2]]);
    m.cyl(0, .27, 0, .1, .12, 10);
    range(10).forEach((i) => { const a = i / 10 * TAU; m.line([Math.cos(a) * .14, .28, Math.sin(a) * .14], [Math.cos(a) * 1.1, .05, Math.sin(a) * 1.1], 4); });
    [[-.5, -.42], [-.5, .42], [.5, -.42], [.5, .42]].forEach(([x, z]) => m.ringXY(x, -.52, z, .15, 10));
  };
  D['lora-2021'] = (m) => {  // a large frozen weight matrix, plus a small trained update B·A
    m.gridXY(6, 6, 1.1, 1.1, 0, -.6, 0);
    m.box(-1.05, .65, .02, .2, .14, .06); m.arc([-1.12, .72, .02], [-.98, .72, .02], .1, 6, 1);  // a lock: frozen
    m.line([.03, 0, 0], [.23, 0, 0]); m.line([.13, -.1, 0], [.13, .1, 0]);
    m.gridXY(1, 6, .18, 1.1, .25, .45, 0);  // B: tall and thin
    m.gridXY(6, 1, 1.1, .18, -.25, 1.15, .46);  // A: short and wide
    m.line([.54, .46, .25], [.6, .46, -.25]);
  };
  D['chain-of-thought-2022'] = (m) => {  // a question, then written steps one after another, then the answer
    m.rectXY(-1, .8, 0, .6, .24); m.dashes(-1.22, .82, .01, .44, 1);
    let prev = [-1, .68, 0];
    range(4).forEach((i) => {
      const c = [-.55 + .33 * i, .35 - .33 * i, .18 * i];
      m.rectXY(c[0], c[1], c[2], .5, .2); m.dashes(c[0] - .18, c[1] + .03, c[2] + .01, .36, 2, .07, (k) => [1, .6][k]);
      m.arrow(prev, [c[0] - .1, c[1] + .1, c[2]], .06); prev = [c[0] + .1, c[1] - .1, c[2]];
    });
    m.arrow(prev, [.98, -.9, .72], .06); m.ringXY(1.08, -1, .72, .12, 12); m.ringXY(1.08, -1, .72, .07, 10);
  };
  D['eu-ai-act-2024'] = (m) => {  // a pyramid of risk tiers beside a legal text under a ring of twelve stars
    const base = -.8, apex = [-.45, .6, 0], size = 1.3;
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => [apex[0] + a * size / 2, base, b * size / 2]);
    m.square(apex[0], base, 0, size); corners.forEach((c) => m.line(c, apex, 5));
    [.35, .6, .82].forEach((t) => m.square(apex[0], base + (apex[1] - base) * t, 0, size * (1 - t)));
    m.rectXY(.85, -.25, .2, .6, .8); m.dashes(.62, -.02, .21, .46, 5, .1, (i) => [1, .9, 1, .7, .5][i]);
    range(12).forEach((i) => { const a = i / 12 * TAU; m.star(.85 + Math.cos(a) * .26, .55 + Math.sin(a) * .26, .2, .035); });
  };

  /* ------------------------------------------------ underdogs: works that later milestones quietly rest on */
  D['reverse-mode-ad-1970'] = (m) => {  // each operation written to a tape on the way down, then swept back up
    const ys = spread(5, 1.8).reverse();
    ys.forEach((y, i) => {
      m.ringXY(-.55, y, 0, .11, 10); m.rectXY(.35, y, 0, .34, .2); m.line([-.43, y, 0], [.18, y, 0], 3);
      if (i < 4) m.arrow([-.55, y - .12, 0], [-.55, ys[i + 1] + .13, 0], .06);
    });
    m.ringXY(-1.05, ys[0] - .2, 0, .08, 8); m.line([-.98, ys[0] - .15, 0], [-.65, ys[1] + .05, 0]);
    m.arc([.6, ys[4], 0], [.6, ys[0], 0], .55, 13, 2); m.arrow([.62, ys[0] - .12, .12], [.6, ys[0], 0], .07);
  };
  D['blas-1979'] = (m) => {  // three tiers of routines: vector, matrix-vector, matrix-matrix
    m.tokens(5, -.75, .3, 1.1, .14); m.tokens(5, -.95, .3, 1.1, .14);  // level 1: two vectors
    m.gridXY(4, 4, .55, .55, 0, -.25, .05); range(4).forEach((k) => m.rectXY(.3, -.15 + .14 * k, 0, .12, .12));  // level 2
    m.gridXY(4, 4, .5, .5, -.3, -.4, .75); m.gridXY(4, 4, .5, .5, -.3, .35, .75);  // level 3
    [-.85, .05, .75].forEach((y, i) => m.line([-1.05, y - .3, .3 - .3 * i], [-1.05, y + .2, .3 - .3 * i], 3));
  };
  D['wordnet-1985'] = (m) => {  // synonym sets joined into a tree of broader and narrower meanings
    const syn = (x, y, z) => { m.rectXY(x, y, z, .42, .22); m.dashes(x - .15, y + .05, z + .01, .3, 2, .09, (k) => [1, .6][k]); return [x, y, z]; };
    const top = syn(0, .85, 0), mid = [syn(-.6, .2, .25), syn(.6, .2, -.25)];
    const low = [syn(-.95, -.5, .4), syn(-.3, -.5, .1), syn(.35, -.5, -.1), syn(.95, -.5, -.4)];
    mid.forEach((p) => m.arrow([p[0], p[1] + .12, p[2]], [top[0] + p[0] * .3, top[1] - .12, top[2]], .06));
    low.forEach((p, i) => { const q = mid[i < 2 ? 0 : 1]; m.arrow([p[0], p[1] + .12, p[2]], [q[0] + (p[0] - q[0]) * .3, q[1] - .12, q[2]], .06); });
    m.arc([-.3, -.62, .1], [.35, -.62, -.1], -.2, 7, 1);  // a part-of link across the tree
  };
  D['neocognitron-1980'] = (m) => {  // an input pattern, then feature (S) and shift-tolerant (C) layers in turn
    m.rectXY(0, 0, -1.1, 1, 1); m.path([[-.25, -.3, -1.1], [0, .3, -1.1], [.25, -.3, -1.1]]); m.line([-.13, -.05, -1.1], [.13, -.05, -1.1]);
    [[-.55, .36, 3, false], [0, .26, 3, true], [.55, .26, 2, false], [1.05, .16, 2, true]].forEach(([z, s, n, pool]) => {
      spread(n, n === 3 ? .9 : .5).forEach((x) => { m.rectXY(x, 0, z, s, s); if (pool) m.gridXY(2, 2, s, s, z, x, 0); m.line([x * .6, 0, z - .5], [x, 0, z], 2); });
    });
    m.node(0, 0, 1.4, .08); m.line([0, 0, 1.13], [0, 0, 1.32]);
  };
  D['mixture-of-experts-1991'] = (m) => {  // a gating network weighs three small expert networks for each case
    m.rectXY(-1.15, 0, 0, .2, .5);  // the input
    const experts = [[-.45, .45], [0, 0], [.45, -.45]];
    experts.forEach(([z, y]) => {
      const cols = [spread(3, .4), spread(2, .25)].map((ys, l) => ys.map((dy) => m.add(-.35 + .45 * l, y + dy, z)));
      cols[0].forEach((a) => cols[1].forEach((b) => m.link(a, b))); m.line([-1.05, 0, 0], [-.4, y, z], 3); m.line([.1, y, z], [.75, 0, 0], 3);
    });
    m.ringXY(-.2, .95, 0, .16, 12); m.line([-1.05, .2, 0], [-.35, .9, 0], 3);  // the gate
    [.7, .2, .1].forEach((w, i) => { const [z, y] = experts[i]; m.line([-.1, .82, 0], [-.05, y + .25, z], 3); m.line([.95, .7 - .12 * i, 0], [.95 + w * .5, .7 - .12 * i, 0]); });
    m.node(.85, 0, 0, .1);
  };
  D['paxos-1998'] = (m) => {  // five replicas; with one down, a majority still agrees on the same value
    const at = range(5).map((i) => { const a = i / 5 * TAU + .3; return [Math.cos(a) * .95, -.3, Math.sin(a) * .95]; });
    at.forEach(([x, y, z], i) => {
      m.box(x, y, z, .28, .32, .28);
      if (i === 3) { m.line([x - .2, y - .2, z], [x + .2, y + .2, z]); m.line([x - .2, y + .2, z], [x + .2, y - .2, z]); return; }
      m.ringXY(x, y + .35, z, .08, 8); m.line([0, .45, 0], [x, y + .44, z], 4);
    });
    m.cube(0, .55, 0, .22); m.circle(30, .95, -.48);
  };
  D['bleu-2002'] = (m) => {  // a candidate translation matched word by word against two references, and a gauge
    const cand = m.tokens(6, .65, 0, 1.6, .17), refs = [m.tokens(6, -.05, .25, 1.6, .17), m.tokens(5, -.55, -.25, 1.4, .17)];
    [[0, 0], [1, 2], [2, 1], [4, 3], [5, 5]].forEach(([a, b]) => m.line([cand[a][0], .56, 0], [refs[0][b][0], .04, .25], 3));
    [[0, 0], [3, 2], [5, 4]].forEach(([a, b]) => m.line([cand[a][0], .56, 0], [refs[1][b][0], -.46, -.25], 4));
    m.path(range(11).map((i) => { const a = Math.PI * i / 10; return [1.3 + Math.cos(a) * .35, -.35 + Math.sin(a) * .35, 0]; }));
    m.line([1.3, -.35, 0], [1.3 + Math.cos(.9) * .3, -.35 + Math.sin(.9) * .3, 0], 3);
  };
  D['uct-2006'] = (m, r) => {  // a search tree grown unevenly toward good moves, with one random playout below
    const kids = (p, n, depth, w) => spread(n, w).map((dx, i) => {
      const q = [p[0] + dx, p[1] - .4, p[2] + (i - (n - 1) / 2) * .15 * depth];
      m.node(...q, .06); m.line([p[0], p[1] - .06, p[2]], [q[0], q[1] + .06, q[2]]); return q;
    });
    const root = [0, .9, 0]; m.node(...root, .08);
    const l1 = kids(root, 3, 1, 1.4), l2 = kids(l1[1], 3, 2, .7), l3 = kids(l2[0], 2, 3, .4); kids(l1[2], 1, 2, 0);
    m.path(range(6).map((i) => [l3[1][0] + (r() - .5) * .15, l3[1][1] - .14 - .1 * i, l3[1][2]]));  // the playout
    [.2, .7, .35].forEach((h, i) => m.line([l1[i][0] + .12, l1[i][1] - .1, l1[i][2]], [l1[i][0] + .12, l1[i][1] - .1 + h * .5, l1[i][2]]));
  };
  D['common-crawl-2007'] = (m) => {  // pages gathered from the web and kept in a public archive
    m.sphere(-.7, .1, 0, .55, 3, 4, 16);
    [[-.05, .55], [.1, .2], [-.1, -.15]].forEach(([x, y], i) => { m.rectXY(x, y, .1 * i, .2, .26); m.dashes(x - .06, y + .06, .1 * i + .01, .12, 3, .06); });
    m.arrow([.15, .2, 0], [.45, .05, 0], .06);
    [-.45, -.15, .15].forEach((y) => m.cyl(.85, y, 0, .35, .22, 16));
  };
  D['theano-2010'] = (m) => {  // an expression graph, its derivative, compiled into code for a GPU
    const g = (z, s) => { const o = [-.7, .6, z], a = [-1, .1, z], b = [-.4, .1, z]; [o, a, b].forEach((p) => m.ringXY(...p, .1 * s, 8)); m.line([-.95, .18, z], [-.75, .52, z]); m.line([-.45, .18, z], [-.65, .52, z]); m.ringXY(-1.15, -.35, z, .08 * s, 8); m.line([-1.12, -.27, z], [-1.02, .02, z]); };
    g(0, 1); g(-.45, .8); m.line([-.7, .6, 0], [-.7, .6, -.45], 3);  // the graph and its symbolic derivative
    m.arrow([-.25, .2, -.2], [.15, .2, -.1], .07);
    m.rectXY(.55, .45, 0, .6, .5); m.dashes(.3, .62, .01, .45, 4, .1, (i) => [1, .7, .85, .5][i]);
    m.box(.55, -.45, 0, .7, .1, .5); spread(4, .4).forEach((z) => { m.line([.2, -.45, z], [.08, -.55, z]); m.line([.9, -.45, z], [1.02, -.55, z]); });
    m.line([.55, .2, 0], [.55, -.39, 0], 3);
  };
  D['multi-column-cnn-2012'] = (m) => {  // several deep columns see differently prepared inputs; their votes are averaged
    spread(4, 2).forEach((x, c) => {
      m.rectXY(x, .7, 0, .34, .34); m.line([x - .12 + .04 * c, .58, 0], [x + .1 - .03 * c, .82, 0]);  // the same input, prepared differently
      [.45, .3, .18].forEach((s, l) => m.rectXY(x, .7 - .45 * (l + 1), .15 * (l + 1), s, s));
      m.line([x, -.75, .45], [0, -1.05, .7], 3);
    });
    m.node(0, -1.12, .7, .1);
  };
  D['unet-2015'] = (m) => {  // the U: shrink to see context, grow back to place every pixel, skips across
    const levels = [[.8, .55], [.4, .4], [0, .28], [-.4, .2]];
    levels.forEach(([y, h], i) => {
      const x = .9 - .2 * i;
      m.box(-x, y, 0, .08 + .04 * i, h, h); m.box(x, y, 0, .08 + .04 * i, h, h);
      if (i < 3) { m.arrow([-x, y - h / 2 - .02, 0], [-x + .2, levels[i + 1][0] + levels[i + 1][1] / 2 + .02, 0], .05); m.arrow([x - .2, levels[i + 1][0] + levels[i + 1][1] / 2 + .02, 0], [x, y - h / 2 - .02, 0], .05); m.line([-x + .1, y, h / 2], [x - .1, y, h / 2], 6); }
    });
    m.box(0, -.75, 0, .3, .16, .16);
  };
  D['hnsw-2016'] = (m, r) => {  // layered graphs, sparse on top: a search drops down layer by layer
    const layer = (y, n) => {
      const pts = range(n).map(() => [r() * 1.8 - .9, y, r() * 1.2 - .6]); m.rectXZ(0, y, 0, 2.1, 1.5);
      pts.forEach((p) => m.add(...p));
      pts.forEach((p, i) => { const near = pts.map((q, k) => [Math.hypot(p[0] - q[0], p[2] - q[2]), k]).sort((a, b) => a[0] - b[0]).slice(1, 3); near.forEach(([, k]) => { if (k > i) m.line(p, pts[k]); }); });
      return pts;
    };
    const top = layer(.8, 3), mid = layer(0, 7), low = layer(-.8, 14);
    const path = [top[0], top[1], [top[1][0], 0, top[1][2]], mid[0], [mid[0][0], -.8, mid[0][2]], low[0]];
    for (let i = 0; i + 1 < path.length; i += 1) m.arrow(path[i].map((v) => v + .02), path[i + 1].map((v) => v + .02), .06);
  };
  D['human-preferences-2017'] = (m) => {  // a person picks the better of two clips; a reward model learns from the picks
    [-.55, .55].forEach((x, i) => {
      m.rectXY(x, .6, 0, .7, .5);
      m.path([[x - .15, .45, 0], [x - .05 + .1 * i, .65, 0], [x + .15, .45, 0]]); m.ringXY(x - .05 + .1 * i, .74, 0, .06, 8);
    });
    m.path([[-.08, .7, 0], [.06, .6, 0], [-.08, .5, 0]]);  // "better than"
    m.arrow([-.55, .32, 0], [-.2, -.15, .2], .07);
    m.box(0, -.35, .2, .6, .35, .35); m.dashes(-.2, -.3, .38, .4, 2, .1);
    m.arc([.32, -.35, .2], [.6, .3, 0], .4, 9, 0);  // the policy improves and makes new clips
  };
  D['ring-allreduce-2017'] = (m) => {  // GPUs in a ring, gradients cut into chunks, passed round twice
    const n = 4, at = range(n).map((i) => { const a = i / n * TAU + .4; return [Math.cos(a) * .9, -.4, Math.sin(a) * .9]; });
    at.forEach(([x, y, z], i) => {
      m.box(x, y, z, .34, .1, .26);
      range(n).forEach((k) => m.cube(x, y + .14 + .14 * k, z, k === i ? .14 : .1));  // its gradient, in chunks
      const q = at[(i + 1) % n];
      for (const h of [.05, .75]) m.arc([x, y + h, z], [q[0], y + h, q[2]], .18, 7, 1);
    });
  };

  const api = {designs: D};
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; return; }
  scope.ChronicleDesigns = D;
})(typeof window === 'undefined' ? globalThis : window);
