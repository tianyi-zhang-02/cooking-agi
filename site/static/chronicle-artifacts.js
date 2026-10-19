/* Chronicle artifacts: a small 3D object beside every milestone on the timeline.

   Each object is a point model: its field picks a shape family (a die with pins for chips, a globe
   for the internet, layers of units for neural networks ...), a few landmark works get their own
   shape (a tape for Turing machines, a bulb for Pearl Street), and a seed taken from the milestone's
   id varies the rest, so no two cards carry the same object. At rest it turns slowly and reads as a
   solid: depth-shaded points joined by thin edges. On hover, focus or touch the edges fade and the
   points loosen into twinkling stars. Drag to turn it.

   Canvas 2D only, no library. Only cards on screen draw; with reduced motion nothing moves on its
   own and the star state appears without animation. Without JavaScript the canvases stay hidden. */
(function (scope) {
  'use strict';

  /* ---------------------------------------------------------------- seeded randomness */
  function hash(text) {
    let h = 2166136261;
    for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }
  function random(seed) {  // mulberry32
    return () => {
      seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const TAU = Math.PI * 2;
  const between = (r, lo, hi) => lo + (hi - lo) * r();
  const whole = (r, lo, hi) => Math.floor(between(r, lo, hi + 1));

  /* ---------------------------------------------------------------- models: points and edges */
  class Model {
    constructor() { this.p = []; this.e = []; }
    add(x, y, z) { this.p.push([x, y, z]); return this.p.length - 1; }
    link(a, b) { this.e.push([a, b]); }
    path(points, closed = false) {
      const ids = points.map(([x, y, z]) => this.add(x, y, z));
      for (let i = 0; i + 1 < ids.length; i += 1) this.link(ids[i], ids[i + 1]);
      if (closed && ids.length > 2) this.link(ids[ids.length - 1], ids[0]);
      return ids;
    }
    dots(points) { return points.map(([x, y, z]) => this.add(x, y, z)); }
    line(a, b, n = 2) { return this.path(Array.from({length: n}, (_, i) => a.map((v, k) => v + (b[k] - v) * i / (n - 1)))); }
    circle(n, radius, y = 0, cx = 0, cz = 0) {
      return this.path(Array.from({length: n}, (_, i) => [cx + Math.cos(i / n * TAU) * radius, y, cz + Math.sin(i / n * TAU) * radius]), true);
    }
    box(cx, cy, cz, w, h, d) {
      const c = [];
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) c.push(this.add(cx + sx * w / 2, cy + sy * h / 2, cz + sz * d / 2));
      [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(([a, b]) => this.link(c[a], c[b]));
      return c;
    }
    square(cx, y, cz, size) {
      const s = size / 2;
      return this.path([[cx - s, y, cz - s], [cx + s, y, cz - s], [cx + s, y, cz + s], [cx - s, y, cz + s]], true);
    }
    // parts facing the viewer (in the x-y plane) and simple solids, used by the per-milestone designs
    ringXY(cx, cy, cz, radius, n = 16) {
      return this.path(Array.from({length: n}, (_, i) => [cx + Math.cos(i / n * TAU) * radius, cy + Math.sin(i / n * TAU) * radius, cz]), true);
    }
    rectXY(cx, cy, cz, w, h) { return this.path([[cx - w / 2, cy - h / 2, cz], [cx + w / 2, cy - h / 2, cz], [cx + w / 2, cy + h / 2, cz], [cx - w / 2, cy + h / 2, cz]], true); }
    rectXZ(cx, y, cz, w, d) { return this.path([[cx - w / 2, y, cz - d / 2], [cx + w / 2, y, cz - d / 2], [cx + w / 2, y, cz + d / 2], [cx - w / 2, y, cz + d / 2]], true); }
    cyl(cx, cy, cz, radius, h, n = 14) {
      const bottom = this.circle(n, radius, cy - h / 2, cx, cz), top = this.circle(n, radius, cy + h / 2, cx, cz);
      for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 4))) this.link(bottom[i], top[i]);
      return [bottom, top];
    }
    sphere(cx, cy, cz, radius, lat = 3, lon = 4, n = 18) {
      for (let i = 1; i <= lat; i += 1) {
        const y = -1 + 2 * i / (lat + 1);
        this.circle(n, radius * Math.sqrt(1 - y * y), cy + y * radius, cx, cz);
      }
      for (let k = 0; k < lon; k += 1) {
        const a = k / lon * Math.PI;
        this.path(Array.from({length: n}, (_, i) => { const u = i / n * TAU; return [cx + Math.cos(u) * Math.cos(a) * radius, cy + Math.sin(u) * radius, cz + Math.cos(u) * Math.sin(a) * radius]; }), true);
      }
    }
    arc(a, b, lift = .3, n = 9, axis = 2) {  // a curve from a to b bulging along one axis
      return this.path(Array.from({length: n}, (_, i) => {
        const u = i / (n - 1), q = a.map((v, k) => v + (b[k] - v) * u);
        q[axis] += Math.sin(Math.PI * u) * lift;
        return q;
      }));
    }
    arrow(a, b, head = .12) {
      const ids = this.line(a, b, 3), d = b.map((v, k) => v - a[k]), len = Math.hypot(...d) || 1, u = d.map((v) => v / len);
      const side = Math.abs(u[1]) < .9 ? [-u[1], u[0], 0] : [0, -u[2], u[1]], sl = Math.hypot(...side) || 1;
      for (const s of [-1, 1]) this.link(ids[2], this.add(...b.map((v, k) => v - u[k] * head + s * side[k] / sl * head * .6)));
      return ids;
    }
    dashes(x, y, z, width, rows, gap = .16, ragged = null) {  // lines of text
      for (let i = 0; i < rows; i += 1) this.line([x, y - i * gap, z], [x + width * (ragged ? ragged(i) : 1), y - i * gap, z], 4);
    }
    gridXY(nx, ny, w, h, z = 0, cx = 0, cy = 0) {
      for (let i = 0; i <= nx; i += 1) this.line([cx - w / 2 + w * i / nx, cy - h / 2, z], [cx - w / 2 + w * i / nx, cy + h / 2, z], ny + 1);
      for (let j = 0; j <= ny; j += 1) this.line([cx - w / 2, cy - h / 2 + h * j / ny, z], [cx + w / 2, cy - h / 2 + h * j / ny, z], nx + 1);
    }
    gridXZ(nx, nz, w, d, y = 0, cx = 0, cz = 0) {
      for (let i = 0; i <= nx; i += 1) this.line([cx - w / 2 + w * i / nx, y, cz - d / 2], [cx - w / 2 + w * i / nx, y, cz + d / 2], nz + 1);
      for (let j = 0; j <= nz; j += 1) this.line([cx - w / 2, y, cz - d / 2 + d * j / nz], [cx + w / 2, y, cz - d / 2 + d * j / nz], nx + 1);
    }
    cube(cx, cy, cz, s) { return this.box(cx, cy, cz, s, s, s); }
    star(cx, cy, cz, s) { this.line([cx - s, cy, cz], [cx + s, cy, cz], 3); this.line([cx, cy - s, cz], [cx, cy + s, cz], 3); return this.add(cx, cy, cz); }
    tokens(n, y, z = 0, w = 1.8, size = .16) {  // a row of small squares; returns their centres
      return Array.from({length: n}, (_, i) => { const x = -w / 2 + w * (n === 1 ? .5 : i / (n - 1)); this.rectXY(x, y, z, size, size); return [x, y, z]; });
    }
    node(x, y, z, radius = .07) { this.ringXY(x, y, z, radius, 8); return [x, y, z]; }
    normalize() {  // centre on the bounding box and fit inside the unit sphere
      const lo = [0, 1, 2].map((k) => Math.min(...this.p.map((q) => q[k])));
      const hi = [0, 1, 2].map((k) => Math.max(...this.p.map((q) => q[k])));
      const mid = lo.map((v, k) => (v + hi[k]) / 2);
      this.p = this.p.map((q) => q.map((v, k) => v - mid[k]));
      const radius = Math.max(...this.p.map(([x, y, z]) => Math.hypot(x, y, z))) || 1;
      this.p = this.p.map((q) => q.map((v) => v / radius));
      return this;
    }
  }

  /* ---------------------------------------------------------------- shape families */
  const SHAPES = {
    chip(m, r) {  // a die with core blocks on top and pins on every side
      m.box(0, 0, 0, 1.6, .18, 1.6);
      const k = whole(r, 2, 4), size = 1.2 / k;
      for (let i = 0; i < k; i += 1) for (let j = 0; j < k; j += 1) m.square(-.6 + size * (i + .5), .1, -.6 + size * (j + .5), size * .7);
      const pins = whole(r, 4, 7);
      for (let i = 0; i < pins; i += 1) {
        const u = -.65 + 1.3 * i / (pins - 1);
        m.line([u, -.05, .8], [u, -.05, 1.05]); m.line([u, -.05, -.8], [u, -.05, -1.05]);
        m.line([.8, -.05, u], [1.05, -.05, u]); m.line([-.8, -.05, u], [-1.05, -.05, u]);
      }
    },
    wafer(m, r) {  // a disc of dies
      m.circle(48, 1, 0);
      const k = whole(r, 4, 6), size = 1.7 / k;
      for (let i = 0; i < k; i += 1) for (let j = 0; j < k; j += 1) {
        const x = -.85 + size * (i + .5), z = -.85 + size * (j + .5);
        if (Math.hypot(x, z) < .88) m.square(x, 0, z, size * .8);
      }
    },
    stack(m, r) {  // stacked plates joined by vias, as in a memory stack
      const n = whole(r, 3, 7), h = 1.6 / n;
      for (let i = 0; i < n; i += 1) m.square(0, -.8 + i * h, 0, 1.3 - (i === n - 1 ? .25 : 0));
      for (const [x, z] of [[-.35, -.35], [.35, -.35], [.35, .35], [-.35, .35], [0, 0]]) m.line([x, -.8, z], [x, -.8 + (n - 1) * h, z], n);
    },
    cylinder(m, r) {  // a database: rings with two sides
      const n = whole(r, 3, 5);
      for (let i = 0; i < n; i += 1) m.circle(28, .8, -.8 + 1.6 * i / (n - 1));
      m.line([-.8, -.8, 0], [-.8, .8, 0]); m.line([.8, -.8, 0], [.8, .8, 0]);
    },
    globe(m, r) {  // latitude and longitude, with arcs between places
      for (const y of [-.6, 0, .6]) m.circle(32, Math.sqrt(1 - y * y), y);
      const meridians = whole(r, 3, 5);
      for (let k = 0; k < meridians; k += 1) {
        const a = k / meridians * Math.PI;
        m.path(Array.from({length: 32}, (_, i) => { const t = i / 32 * TAU; return [Math.cos(t) * Math.cos(a), Math.sin(t), Math.cos(t) * Math.sin(a)]; }), true);
      }
      const arcs = whole(r, 2, 4);
      for (let k = 0; k < arcs; k += 1) {
        const a = [between(r, 0, TAU), between(r, -.8, .8)], b = [between(r, 0, TAU), between(r, -.8, .8)];
        const at = ([lon, lat]) => [Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)];
        const pa = at(a), pb = at(b);
        m.path(Array.from({length: 14}, (_, i) => {
          const t = i / 13, v = pa.map((x, j) => x + (pb[j] - x) * t), len = Math.hypot(...v) || 1, lift = 1 + .35 * Math.sin(Math.PI * t);
          return v.map((x) => x / len * lift);
        }));
      }
    },
    fattree(m, r) {  // switches in three levels, each linked to the level above
      const levels = [2, 4, whole(r, 6, 8)], ids = levels.map((n, l) => Array.from({length: n}, (_, i) => m.add(-1 + 2 * (i + .5) / n, .9 - .9 * l, between(r, -.25, .25))));
      for (let l = 1; l < levels.length; l += 1) ids[l].forEach((id, i) => {
        const up = ids[l - 1], a = Math.floor(i * up.length / ids[l].length);
        m.link(id, up[a]); m.link(id, up[(a + 1) % up.length]);
      });
    },
    coil(m, r) {  // a coil around a core, as in a generator or transformer
      const turns = between(r, 3, 6), n = Math.round(turns * 18);
      m.path(Array.from({length: n}, (_, i) => { const t = i / (n - 1); return [Math.cos(t * turns * TAU) * .55, -1 + 2 * t, Math.sin(t * turns * TAU) * .55]; }));
      m.line([0, -1.15, 0], [0, 1.15, 0], 6);
      m.circle(20, .75, -1.1); m.circle(20, .75, 1.1);
    },
    rack(m, r) {  // server racks with their slots
      const racks = whole(r, 1, 3);
      for (let k = 0; k < racks; k += 1) {
        const x = (k - (racks - 1) / 2) * .7;
        m.box(x, 0, 0, .55, 1.9, .55);
        const slots = whole(r, 5, 9);
        for (let i = 1; i < slots; i += 1) m.line([x - .27, -.95 + 1.9 * i / slots, .28], [x + .27, -.95 + 1.9 * i / slots, .28]);
      }
    },
    braces(m, r) {  // { } around a few lines of code
      const brace = (side) => m.path(Array.from({length: 24}, (_, i) => {
        const t = i / 23 * 2 - 1, notch = Math.max(0, .18 - Math.abs(t) * .9);
        return [side * (.75 + .1 * t * t - notch), t, 0];
      }));
      brace(-1); brace(1);
      const lines = whole(r, 3, 5);
      for (let i = 0; i < lines; i += 1) {
        const y = .6 - 1.2 * i / (lines - 1), len = between(r, .4, 1);
        m.line([-.45, y, 0], [-.45 + len, y, 0], 5);
      }
    },
    gear(m, r) {  // a gear: toothed rim on two faces
      const teeth = whole(r, 8, 13), n = teeth * 4;
      for (const z of [-.15, .15]) {
        m.path(Array.from({length: n}, (_, i) => { const a = i / n * TAU, rad = (i % 4 < 2) ? 1 : .78; return [Math.cos(a) * rad, Math.sin(a) * rad, z]; }), true);
        m.path(Array.from({length: 16}, (_, i) => [Math.cos(i / 16 * TAU) * .3, Math.sin(i / 16 * TAU) * .3, z]), true);
      }
      for (let i = 0; i < teeth; i += 1) { const a = i / teeth * TAU; m.line([Math.cos(a) * .3, Math.sin(a) * .3, 0], [Math.cos(a) * .78, Math.sin(a) * .78, 0]); }
    },
    layers(m, r) {  // units in layers, each linked to the next layer
      const count = whole(r, 3, 5), ids = [];
      for (let l = 0; l < count; l += 1) {
        const n = whole(r, 3, 5);
        ids.push(Array.from({length: n}, (_, i) => m.add(-1 + 2 * l / (count - 1), -.8 + 1.6 * (i + .5) / n, between(r, -.3, .3))));
      }
      for (let l = 0; l + 1 < ids.length; l += 1) for (const a of ids[l]) for (const b of ids[l + 1]) if (r() < .7) m.link(a, b);
    },
    cubes(m, r) {  // devices working in parallel: a block of small cubes
      const [a, b, c] = [whole(r, 2, 3), whole(r, 1, 3), whole(r, 2, 3)];
      for (let i = 0; i < a; i += 1) for (let j = 0; j < b; j += 1) for (let k = 0; k < c; k += 1) m.box((i - (a - 1) / 2) * .62, (j - (b - 1) / 2) * .62, (k - (c - 1) / 2) * .62, .42, .42, .42);
    },
    funnel(m, r) {  // requests narrowing into a server
      const rings = whole(r, 4, 6);
      const ids = Array.from({length: rings}, (_, i) => m.circle(24, 1 - .75 * i / (rings - 1), .9 - 1.8 * i / (rings - 1)));
      for (let k = 0; k < 6; k += 1) m.line([Math.cos(k / 6 * TAU), .9, Math.sin(k / 6 * TAU)], [Math.cos(k / 6 * TAU) * .25, -.9, Math.sin(k / 6 * TAU) * .25]);
      m.dots(Array.from({length: 10}, () => { const a = between(r, 0, TAU), y = between(r, -.8, .8), rad = .25 + .75 * (y + .9) / 1.8; return [Math.cos(a) * rad * .8, y, Math.sin(a) * rad * .8]; }));
      return ids;
    },
    surface(m, r, height) {  // a wire grid over z = height(x, y)
      const n = 9, ids = [];
      for (let i = 0; i < n; i += 1) {
        ids.push([]);
        for (let j = 0; j < n; j += 1) { const x = -1 + 2 * i / (n - 1), z = -1 + 2 * j / (n - 1); ids[i].push(m.add(x, height(x, z), z)); }
      }
      for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) { if (i + 1 < n) m.link(ids[i][j], ids[i + 1][j]); if (j + 1 < n) m.link(ids[i][j], ids[i][j + 1]); }
    },
    bowl(m, r) {  // a loss surface with a path down to its minimum
      const a = between(r, .7, 1.1);
      SHAPES.surface(m, r, (x, z) => a * (x * x + z * z) * .6 - .4);
      const start = between(r, 0, TAU);
      m.path(Array.from({length: 12}, (_, i) => { const t = 1 - i / 11, ang = start + i * .5; const x = Math.cos(ang) * t * .9, z = Math.sin(ang) * t * .9; return [x, a * (x * x + z * z) * .6 - .4 + .05, z]; }));
    },
    bell(m, r) {  // a probability surface
      const s = between(r, .25, .5);
      SHAPES.surface(m, r, (x, z) => Math.exp(-(x * x + z * z) / s) * 1.1 - .4);
    },
    scatter(m, r) {  // data points around a fitted line
      for (let i = 0; i < 26; i += 1) { const t = between(r, -1, 1); m.add(t, t * .6 + between(r, -.3, .3), between(r, -.4, .4)); }
      m.line([-1.1, -.66, 0], [1.1, .66, 0], 8);
    },
    lattice(m, r) {  // a grid of numbers in three dimensions
      const n = whole(r, 3, 4), ids = {};
      for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) for (let k = 0; k < n; k += 1) ids[[i, j, k]] = m.add(-1 + 2 * i / (n - 1), -1 + 2 * j / (n - 1), -1 + 2 * k / (n - 1));
      for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) for (let k = 0; k < n; k += 1) {
        if (i + 1 < n) m.link(ids[[i, j, k]], ids[[i + 1, j, k]]);
        if (j + 1 < n) m.link(ids[[i, j, k]], ids[[i, j + 1, k]]);
        if (k + 1 < n) m.link(ids[[i, j, k]], ids[[i, j, k + 1]]);
      }
    },
    polyhedron(m, r) {  // a regular solid: icosahedron or octahedron
      const g = (1 + Math.sqrt(5)) / 2;
      const ico = [[-1, g, 0], [1, g, 0], [-1, -g, 0], [1, -g, 0], [0, -1, g], [0, 1, g], [0, -1, -g], [0, 1, -g], [g, 0, -1], [g, 0, 1], [-g, 0, -1], [-g, 0, 1]];
      const oct = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
      const verts = r() < .6 ? ico : oct, ids = m.dots(verts);
      const edge = Math.min(...verts.flatMap((a, i) => verts.slice(i + 1).map((b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]))));
      verts.forEach((a, i) => verts.forEach((b, j) => { if (j > i && Math.abs(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) - edge) < 1e-6) m.link(ids[i], ids[j]); }));
    },
    tree(m, r, depth = whole(r, 3, 4)) {  // a search or rule tree branching downwards
      const grow = (id, [x, y, z], d, spread) => {
        if (!d) return;
        const kids = r() < .3 ? 3 : 2;
        for (let k = 0; k < kids; k += 1) {
          const a = between(r, 0, TAU), child = [x + Math.cos(a) * spread, y - .5, z + Math.sin(a) * spread];
          const c = m.add(...child); m.link(id, c); grow(c, child, d - 1, spread * .6);
        }
      };
      grow(m.add(0, .9, 0), [0, .9, 0], depth, .7);
    },
    tiles(m, r) {  // a grid of labelled images
      const n = whole(r, 3, 4), size = 1.8 / n;
      for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) {
        const x = -.9 + size * (i + .5), y = -.9 + size * (j + .5), z = between(r, -.15, .15), s = size * .4;
        m.path([[x - s, y - s, z], [x + s, y - s, z], [x + s, y + s, z], [x - s, y + s, z]], true);
      }
    },
    bars(m, r) {  // a benchmark table as bars
      const n = 3;
      for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) { const h = between(r, .3, 1.6); m.box(-.7 + .7 * i, -.8 + h / 2, -.7 + .7 * j, .35, h, .35); }
    },
    pages(m, r) {  // pages fanned out
      const n = whole(r, 3, 5);
      for (let k = 0; k < n; k += 1) {
        const a = (k - (n - 1) / 2) * .35;
        const corner = (x, y) => [Math.cos(a) * x, y, Math.sin(a) * x];
        m.path([corner(0, -1), corner(1.1, -1), corner(1.1, 1), corner(0, 1)], true);
        for (let i = 1; i < 5; i += 1) m.line(corner(.15, 1 - i * .4), corner(.9, 1 - i * .4));
      }
    },
    attention(m, r, rows = 2) {  // tokens in rows, each looking at others in the row below
      const n = whole(r, 5, 7), ids = [];
      for (let row = 0; row < rows; row += 1) ids.push(Array.from({length: n}, (_, i) => m.add(-1 + 2 * i / (n - 1), .8 - 1.6 * row / (rows - 1), 0)));
      for (let row = 0; row + 1 < rows; row += 1) ids[row].forEach((id, i) => {
        const targets = new Set([i, whole(r, 0, n - 1)]);
        for (const j of targets) {
          const [a, b] = [m.p[id], m.p[ids[row + 1][j]]];
          const arc = Array.from({length: 7}, (_, k) => { const t = k / 6; return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, Math.sin(Math.PI * t) * between(r, .25, .6)]; });
          const mid = arc.slice(1, -1).map(([x, y, z]) => m.add(x, y, z));
          [id, ...mid, ids[row + 1][j]].reduce((p, q) => { m.link(p, q); return q; });
        }
      });
    },
    pyramid(m, r) {  // feature maps shrinking through a convolutional network
      const n = whole(r, 3, 5), ids = [];
      for (let i = 0; i < n; i += 1) { const s = 1.6 * (1 - i / (n + .5)), z = -1 + 2 * i / (n - 1); const h = s / 2; ids.push(m.path([[-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z]], true)); }
      for (let i = 0; i + 1 < n; i += 1) for (let c = 0; c < 4; c += 1) if (r() < .6) m.link(ids[i][c], ids[i + 1][c]);
    },
    wave(m, r) {  // a sound or signal as a ribbon
      const f = between(r, 1.5, 3.5), n = 40, top = [], bottom = [];
      for (let i = 0; i < n; i += 1) {
        const x = -1 + 2 * i / (n - 1), y = Math.sin(x * f * Math.PI) * .5 * Math.exp(-x * x);
        top.push(m.add(x, y, .25)); bottom.push(m.add(x, y, -.25));
        if (i) { m.link(top[i - 1], top[i]); m.link(bottom[i - 1], bottom[i]); }
        if (i % 4 === 0) m.link(top[i], bottom[i]);
      }
    },
    board(m, r, size = whole(r, 7, 9)) {  // a game board with stones
      for (let i = 0; i < size; i += 1) {
        const u = -1 + 2 * i / (size - 1);
        m.line([u, 0, -1], [u, 0, 1], size); m.line([-1, 0, u], [1, 0, u], size);
      }
      for (let k = 0; k < whole(r, 6, 12); k += 1) m.add(-1 + 2 * whole(r, 0, size - 1) / (size - 1), .08, -1 + 2 * whole(r, 0, size - 1) / (size - 1));
    },
    graph(m, r, count) {  // nodes and their nearest links
      const n = count || whole(r, 10, 15), ids = [], at = [];
      for (let i = 0; i < n; i += 1) { const v = [between(r, -1, 1), between(r, -1, 1), between(r, -1, 1)]; at.push(v); ids.push(m.add(...v)); }
      at.forEach((a, i) => at.map((b, j) => [Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]), j]).sort((x, y) => x[0] - y[0]).slice(1, 3).forEach(([, j]) => { if (j > i) m.link(ids[i], ids[j]); }));
    },
    torus(m, r) {  // a smooth form, as generated from noise
      const R = 1, a = between(r, .3, .45), rings = 14;
      for (let i = 0; i < rings; i += 1) {
        const u = i / rings * TAU;
        m.path(Array.from({length: 12}, (_, k) => { const v = k / 12 * TAU; return [(R + a * Math.cos(v)) * Math.cos(u), a * Math.sin(v), (R + a * Math.cos(v)) * Math.sin(u)]; }), true);
      }
    },
    knot(m, r) {  // a loop that feeds back into itself
      const p = 2, q = whole(r, 3, 5), n = 120;
      m.path(Array.from({length: n}, (_, i) => { const t = i / n * TAU, rad = Math.cos(q * t) + 2; return [rad * Math.cos(p * t), -Math.sin(q * t), rad * Math.sin(p * t)]; }), true);
    },
    bubble(m, r) {  // a speech bubble with a few lines of text
      for (const z of [-.12, .12]) {
        const pts = [];
        for (let i = 0; i <= 40; i += 1) { const a = i / 40 * TAU; pts.push([Math.cos(a) * 1, Math.sin(a) * .65 + .15, z]); }
        m.path(pts.slice(0, 36)); m.path([pts[35], [-.55, -.8, z], pts[29]]);
      }
      for (let i = 0; i < 3; i += 1) m.line([-.55, .45 - .28 * i, 0], [.55 - .25 * r(), .45 - .28 * i, 0], 5);
    },
    arm(m, r) {  // a robot arm: base, joints and a gripper
      m.circle(18, .7, -1);
      let at = [0, -1, 0], angle = between(r, .5, 1.2);
      for (let k = 0; k < 3; k += 1) {
        const len = between(r, .5, .8), next = [at[0] + Math.cos(angle) * len, at[1] + Math.sin(angle) * len, at[2] + between(r, -.2, .2)];
        m.line(at, next, 4); m.circle(10, .1, next[1], next[0], next[2]); at = next; angle -= between(r, .6, 1.1);
      }
      m.line(at, [at[0] + .25, at[1] + .2, at[2]]); m.line(at, [at[0] + .25, at[1] - .2, at[2]]);
    },
    helix(m, r) {  // a double helix with its rungs
      const turns = between(r, 1.2, 2), n = 60, a = [], b = [];
      for (let i = 0; i < n; i += 1) {
        const t = i / (n - 1), ang = t * turns * TAU, y = -1 + 2 * t;
        a.push(m.add(Math.cos(ang) * .5, y, Math.sin(ang) * .5)); b.push(m.add(Math.cos(ang + Math.PI) * .5, y, Math.sin(ang + Math.PI) * .5));
        if (i) { m.link(a[i - 1], a[i]); m.link(b[i - 1], b[i]); }
        if (i % 5 === 0) m.link(a[i], b[i]);
      }
    },
    people(m, r) {  // a circle of people around one shared thing
      const n = whole(r, 6, 9), centre = m.add(0, 0, 0);
      for (let i = 0; i < n; i += 1) {
        const a = i / n * TAU, x = Math.cos(a) * .9, z = Math.sin(a) * .9;
        const head = m.add(x, .25, z); m.circle(8, .12, .25, x, z); m.link(head, centre); m.line([x, .1, z], [x, -.4, z], 3);
      }
    },
    branch(m, r) {  // a commit history: a main line with branches merging back
      const main = m.line([0, -1, 0], [0, 1, 0], 7), branches = whole(r, 1, 3);
      for (let k = 0; k < branches; k += 1) {
        const from = whole(r, 0, 3), to = Math.min(6, from + whole(r, 2, 3)), side = r() < .5 ? -1 : 1, z = between(r, -.4, .4);
        const mid = m.path(Array.from({length: to - from - 1}, (_, i) => [side * .55, m.p[main[from + 1 + i]][1], z]));
        m.link(main[from], mid[0]); m.link(mid[mid.length - 1], main[to]);
      }
    },
    shield(m, r) {  // a shield: guarding and accountability
      for (const z of [-.12, .12]) m.path([[-.8, .9, z], [.8, .9, z], [.8, .1, z], [.45, -.5, z], [0, -.95, z], [-.45, -.5, z], [-.8, .1, z]], true);
      m.line([0, .7, 0], [0, -.6, 0], 5); m.line([-.5, .3, 0], [.5, .3, 0], 5);
      if (r() < .5) m.line([-.3, -.1, 0], [.3, -.1, 0], 3);
    },
    lens(m, r) {  // a magnifying glass over results
      m.circle(30, .65, 0); m.circle(30, .5, 0);
      m.line([.46, -.46, 0].map((v, i) => i === 1 ? v : v), [1.1, -1.1, 0], 5);
      for (let i = 0; i < 3; i += 1) m.line([-.35, .25 - .25 * i, 0], [.3 - .2 * r(), .25 - .25 * i, 0], 3);
    },
    // ------------------------------------------------ landmark shapes
    tape(m, r) {  // a Turing machine: cells on a tape under a head
      for (let i = 0; i < 7; i += 1) m.box(-1.2 + .4 * i, 0, 0, .34, .34, .34);
      m.path([[0, .35, 0], [-.18, .75, 0], [.18, .75, 0]], true); m.box(0, .95, 0, .5, .3, .3);
    },
    bulb(m, r) {  // a light bulb on its screw base
      for (let i = 0; i < 7; i += 1) { const y = -.1 + i * .17, rad = Math.sqrt(Math.max(0, .5 ** 2 - (y - .45) ** 2)) * 1.4 + (y < .3 ? .05 : 0); m.circle(20, Math.max(.18, rad), y); }
      for (let i = 0; i < 4; i += 1) m.circle(14, .26, -.2 - i * .15);
      m.path([[-.12, .05, 0], [-.06, .45, 0], [0, .3, 0], [.06, .45, 0], [.12, .05, 0]]);
    },
    medal(m, r) {  // a medal on a ribbon
      m.circle(32, .6, 0); m.circle(32, .45, 0);
      for (const z of [-.06, .06]) m.path([[-.3, .5, z], [-.15, 1.2, z], [0, .6, z], [.15, 1.2, z], [.3, .5, z]]);
      m.path(Array.from({length: 10}, (_, i) => { const a = i / 10 * TAU - Math.PI / 2, rad = i % 2 ? .12 : .28; return [Math.cos(a) * rad, Math.sin(a) * rad, 0]; }), true);
    },
    terrain(m, r) {  // a world you can walk through: rolling ground
      const k = [between(r, 1.5, 3), between(r, 1.5, 3)];
      SHAPES.surface(m, r, (x, z) => .25 * Math.sin(x * k[0] + 1) * Math.cos(z * k[1]) + .2 * Math.sin((x + z) * 2));
    },
    transistor(m, r) {  // a transistor: a body with three legs
      m.box(0, .45, 0, .8, .6, .5);
      for (const x of [-.25, 0, .25]) m.line([x, .15, 0], [x, -1, 0], 5);
    },
  };

  // field (topic) and direction (track) to a shape family; a few landmark works get their own
  const BY_TRACK = {
    supply: 'coil', facilities: 'rack', devices: 'wafer', processors: 'chip', accelerators: 'chip', memory: 'stack', storage: 'cylinder',
    internet: 'globe', interconnect: 'fattree', languages: 'braces', systems: 'gear', frameworks: 'lattice',
    'data-systems': 'cylinder', parallel: 'cubes', clusters: 'rack', serving: 'funnel',
    theory: 'polyhedron', probability: 'bell', numerical: 'lattice', optimization: 'bowl', search: 'tree',
    datasets: 'tiles', benchmarks: 'bars', documentation: 'pages',
    symbolic: 'tree', statistical: 'scatter', neural: 'layers', vision: 'pyramid', language: 'attention', speech: 'wave',
    rl: 'board', graphs: 'graph', generative: 'torus', 'post-training': 'knot',
    games: 'board', 'search-recsys': 'lens', translation: 'wave', robotics: 'arm', 'science-health': 'helix', assistants: 'bubble',
    community: 'people', 'open-source': 'branch', responsibility: 'shield',
  };
  const BY_TOPIC = {energy: 'coil', chips: 'chip', memory: 'stack', networks: 'globe', software: 'braces', distributed: 'cubes',
    algorithms: 'polyhedron', data: 'tiles', models: 'layers', applications: 'bubble', ecosystem: 'people'};
  const LANDMARKS = {
    'turing-computability-1936': 'tape', 'energy-pearl-street-1882': 'bulb', 'nobel-ai-2024': 'medal', 'genie-3-2025': 'terrain',
    'transistor-1947': 'transistor', 'networks-arpanet-1969': 'graph', 'imagenet-2009': 'tiles', 'deep-blue-1997': 'board',
  };

  const DESIGNS = (typeof module !== 'undefined' && module.exports) ? require('./chronicle-designs.js').designs : (scope.ChronicleDesigns || {});
  function family(id, topic, track) { return LANDMARKS[id] || BY_TRACK[track] || BY_TOPIC[topic] || 'polyhedron'; }
  function build(id, topic, track) {
    const r = random(hash(id)), m = new Model(), name = DESIGNS[id] ? id : family(id, topic, track);
    if (DESIGNS[id]) DESIGNS[id](m, r);  // the milestone's own object (chronicle-designs.js)
    else if (id === 'transformer') SHAPES.attention(m, r, 3);
    else if (id === 'networks-arpanet-1969') {  // the first four hosts, each drawn as a ring
      SHAPES.graph(m, r, 4);
      m.p.slice(0, 4).forEach(([x, y, z], i) => { m.circle(10, .16, y, x, z); m.link(i, (i + 1) % 4); });
    }
    else if (id === 'deep-blue-1997') SHAPES.board(m, r, 8);
    else SHAPES[name](m, r);
    m.normalize();
    // stars sit along every edge as well as on the corners, so the star form keeps the outline
    const length = m.e.reduce((sum, [a, b]) => sum + Math.hypot(...m.p[a].map((v, k) => v - m.p[b][k])), 0);
    const step = Math.max(.035, length / 520), samples = m.p.map((q) => q.slice());
    for (const [a, b] of m.e) {
      const d = m.p[b].map((v, k) => v - m.p[a][k]), n = Math.floor(Math.hypot(...d) / step);
      for (let i = 1; i < n; i += 1) samples.push(m.p[a].map((v, k) => v + d[k] * i / n));
    }
    // the same seed also fixes how the object sits and turns. Objects drawn flat towards the viewer sway
    // rather than spin, so they never turn edge-on; solid ones turn all the way round.
    const depth = Math.max(...m.p.map((q) => q[2])) - Math.min(...m.p.map((q) => q[2]));
    const sway = depth < .45;
    return {name, p: m.p, e: m.e, sway, tilt: sway ? between(r, .12, .32) : between(r, .35, .75),
      spin: between(r, .22, .42) * (r() < .5 ? -1 : 1), turn: sway ? between(r, -.4, .4) : between(r, 0, TAU),
      twinkle: m.p.map(() => [between(r, 1.5, 4.5), between(r, 0, TAU), [between(r, -1, 1), between(r, -1, 1), between(r, -1, 1)]]),
      // per star: twinkle speed, phase, drift direction, brightness, when it lights up during the change
      s: samples, stars: samples.map(() => [between(r, 1.2, 4), between(r, 0, TAU), [between(r, -1, 1), between(r, -1, 1), between(r, -1, 1)], Math.pow(r(), 3), r() * .55])};
  }

  const api = {build, family, SHAPES, BY_TRACK, BY_TOPIC, LANDMARKS};
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; return; }
  scope.ChronicleArtifacts = api;

  /* ---------------------------------------------------------------- drawing */
  const reduce = () => scope.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const dark = () => document.documentElement.dataset.theme !== 'light';
  function rgb(value) {
    const hex = (value || '').trim().replace('#', '');
    return hex.length === 6 ? [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)) : [201, 133, 0];
  }
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

  class Artifact {
    constructor(canvas) {
      this.canvas = canvas; this.li = canvas.closest('li');
      this.model = build(canvas.dataset.art, canvas.dataset.topic || '', canvas.dataset.track || '');
      this.angle = this.model.turn; this.tilt = this.model.tilt; this.phase = this.model.turn * 3; this.offset = 0;
      this.star = 0; this.target = 0; this.drag = null; this.colour = [201, 133, 0];
    }
    size() {
      const box = this.canvas.getBoundingClientRect(), ratio = Math.min(2, scope.devicePixelRatio || 1);
      const w = Math.round(box.width * ratio), h = Math.round(box.height * ratio);
      if (w && (this.canvas.width !== w || this.canvas.height !== h)) { this.canvas.width = w; this.canvas.height = h; }
      this.colour = rgb(getComputedStyle(this.li || this.canvas).getPropertyValue('--layer'));
    }
    draw(time) {
      const c = this.canvas.getContext('2d'), w = this.canvas.width, h = this.canvas.height;
      if (!w) return;
      c.clearRect(0, 0, w, h);
      const t = this.star * this.star * (3 - 2 * this.star), still = reduce(), night = dark();
      const [ca, sa, ct, st] = [Math.cos(this.angle), Math.sin(this.angle), Math.cos(this.tilt), Math.sin(this.tilt)];
      const bob = still ? 0 : Math.sin(time * .7 + this.model.turn) * .03;
      const scale = Math.min(w, h) * .38, cx = w / 2, cy = h / 2, unit = w / 64;
      // ivory and silver, tinted by the field; the field's own colour stays in the glow
      const metal = night ? mix([226, 220, 208], this.colour, .28) : mix([86, 78, 66], this.colour, .35);
      const starlight = night ? [255, 246, 226] : [120, 82, 30];
      const light = [-.45, .65, -.62];  // from the upper left, in front
      const ink = night ? 1 : 1.8;  // dark lines on a light page need more weight than light lines on a dark one
      const view = ([x, y, z]) => {
        const x1 = x * ca + z * sa, z1 = -x * sa + z * ca;
        const y2 = y * ct - z1 * st, z2 = y * st + z1 * ct;
        const k = 3.2 / (3.2 + z2), lit = .42 + .58 * Math.max(0, (x1 * light[0] + y2 * light[1] + z2 * light[2]) / (Math.hypot(x1, y2, z2) || 1));
        return [cx + x1 * k * scale, cy - (y2 + bob) * k * scale, (1 - z2) / 2, lit];
      };
      const glow = c.createRadialGradient(cx, cy * .92, 0, cx, cy, Math.min(w, h) * .52);
      glow.addColorStop(0, `rgba(${this.colour.join(',')},${(night ? .16 : .1) + .1 * t})`);
      glow.addColorStop(1, `rgba(${this.colour.join(',')},0)`);
      c.fillStyle = glow; c.fillRect(0, 0, w, h);
      c.globalCompositeOperation = night ? 'lighter' : 'source-over';
      if (t < .995) {  // at rest: fine luminous lines, brighter where they face the light and the viewer
        const points = this.model.p.map(view);
        c.lineCap = 'round'; c.lineWidth = unit * (night ? .75 : .9);
        for (const [a, b] of this.model.e) {
          const pa = points[a], pb = points[b], near = (pa[2] + pb[2]) / 2, lit = (pa[3] + pb[3]) / 2;
          c.strokeStyle = `rgba(${metal.join(',')},${Math.min(1, ink * (1 - t) * (.12 + .5 * near) * (.45 + .55 * lit)).toFixed(3)})`;
          c.beginPath(); c.moveTo(pa[0], pa[1]); c.lineTo(pb[0], pb[1]); c.stroke();
        }
        for (const [x, y, near, lit] of points) {  // a small highlight at each corner
          c.fillStyle = `rgba(${mix(metal, starlight, .5).join(',')},${Math.min(1, ink * (1 - t) * (.2 + .6 * near) * lit).toFixed(3)})`;
          c.beginPath(); c.arc(x, y, unit * (.45 + .45 * near), 0, TAU); c.fill();
        }
      }
      if (t > .005) {  // a constellation: stars along the same outline light up one by one and drift a little
        const samples = this.model.s;
        for (let i = 0; i < samples.length; i += 1) {
          const [speed, phase, dir, magnitude, delay] = this.model.stars[i];
          const shown = Math.min(1, Math.max(0, (t - delay) / .45));
          if (!shown) continue;
          const loose = (.02 + (still ? 0 : .02 * Math.sin(time * .9 + phase))) * t;
          const q = samples[i], [x, y, near, lit] = view([q[0] + dir[0] * loose, q[1] + dir[1] * loose, q[2] + dir[2] * loose]);
          const twinkle = still ? .8 : .55 + .45 * Math.sin(time * speed + phase);
          const bright = Math.min(1, (night ? 1 : 1.5) * shown * (.25 + .75 * twinkle) * (.45 + .55 * near) * (.6 + .4 * lit));
          const colour = mix(metal, starlight, .55 + .45 * magnitude);
          const radius = unit * (.32 + .5 * magnitude) * (.75 + .45 * near);
          if (magnitude > .55) {  // the brightest few get a soft halo and a four-point glint
            c.fillStyle = `rgba(${colour.join(',')},${(bright * .16).toFixed(3)})`;
            c.beginPath(); c.arc(x, y, radius * 3.2, 0, TAU); c.fill();
            if (magnitude > .82) {
              c.strokeStyle = `rgba(${colour.join(',')},${(bright * .55).toFixed(3)})`; c.lineWidth = unit * .35;
              const g = radius * (3 + 2 * twinkle);
              c.beginPath(); c.moveTo(x - g, y); c.lineTo(x + g, y); c.moveTo(x, y - g); c.lineTo(x, y + g); c.stroke();
            }
          }
          c.fillStyle = `rgba(${colour.join(',')},${bright.toFixed(3)})`;
          c.beginPath(); c.arc(x, y, radius, 0, TAU); c.fill();
        }
      }
      c.globalCompositeOperation = 'source-over';
    }
  }

  /* ---------------------------------------------------------------- one loop for the cards on screen */
  const shown = new Set();
  let last = 0, running = false;
  function frame(now) {
    // 30 frames a second are plenty for slow turning, and halve the work of a screenful of cards
    if (last && now - last < 32) { scope.requestAnimationFrame(frame); return; }
    const dt = Math.min(.05, (now - (last || now)) / 1000); last = now;
    const still = reduce();
    for (const art of shown) {
      art.star = still ? art.target : art.star + Math.sign(art.target - art.star) * Math.min(Math.abs(art.target - art.star), dt / .45);
      if (!art.drag && !still) {
        if (art.model.sway) { art.phase += Math.abs(art.model.spin) * dt * (1 + 1.2 * art.star); art.angle = art.offset + Math.sin(art.phase) * .75; }
        else art.angle += art.model.spin * dt * (1 + 1.2 * art.star);
      }
      art.draw(now / 1000);
    }
    running = shown.size > 0 && !still;
    if (running) scope.requestAnimationFrame(frame); else last = 0;
  }
  function wake(art) {
    if (reduce()) { art.star = art.target; art.draw(0); return; }
    if (!running) { running = true; scope.requestAnimationFrame(frame); }
  }

  function setUp(root) {
    const canvases = [...root.querySelectorAll('canvas.ch-art[data-art]')];
    if (!canvases.length || typeof IntersectionObserver === 'undefined') return;
    root.classList.add('ch-art-ready');
    const arts = new Map();
    const get = (canvas) => {
      if (!arts.has(canvas)) { canvas.hidden = false; const art = new Artifact(canvas); arts.set(canvas, art); art.size(); }
      return arts.get(canvas);
    };
    canvases.forEach((canvas) => { canvas.hidden = false; });
    const watch = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const art = get(entry.target);
        if (entry.isIntersecting) { art.size(); shown.add(art); wake(art); } else shown.delete(art);
      }
    }, {rootMargin: '120px 0px'});
    canvases.forEach((canvas) => {
      watch.observe(canvas);
      const li = canvas.closest('li');
      const focus = (on) => () => { const art = get(canvas); art.target = on || art.drag ? 1 : 0; wake(art); };
      li?.addEventListener('pointerenter', focus(true));
      li?.addEventListener('pointerleave', focus(false));
      li?.addEventListener('focusin', focus(true));
      li?.addEventListener('focusout', focus(false));
      canvas.addEventListener('pointerdown', (event) => {
        const art = get(canvas);
        art.drag = {x: event.clientX, y: event.clientY, moved: 0};
        art.target = 1; canvas.setPointerCapture(event.pointerId); wake(art);
      });
      canvas.addEventListener('pointermove', (event) => {
        const art = arts.get(canvas);
        if (!art || !art.drag) return;
        const dx = event.clientX - art.drag.x, dy = event.clientY - art.drag.y;
        art.drag.x = event.clientX; art.drag.y = event.clientY; art.drag.moved += Math.abs(dx) + Math.abs(dy);
        art.angle += dx * .02; art.offset += dx * .02; art.tilt = Math.max(-1.2, Math.min(1.2, art.tilt + dy * .02));
        if (reduce()) art.draw(0);
      });
      const release = (event) => {
        const art = arts.get(canvas);
        if (!art || !art.drag) return;
        const moved = art.drag.moved; art.drag = null;
        if (event.pointerType !== 'mouse') art.target = 0;
        wake(art);
        if (moved < 4 && event.type === 'pointerup') li?.querySelector('a.ch-item')?.click();  // a tap opens the milestone, like the card
      };
      canvas.addEventListener('pointerup', release);
      canvas.addEventListener('pointercancel', release);
    });
    scope.addEventListener('resize', () => { for (const art of arts.values()) art.size(); });
    new MutationObserver(() => { for (const art of arts.values()) { art.size(); if (reduce()) art.draw(0); } })
      .observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
  }

  const start = () => document.querySelectorAll('[data-chronicle]').forEach(setUp);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})(typeof window === 'undefined' ? globalThis : window);
