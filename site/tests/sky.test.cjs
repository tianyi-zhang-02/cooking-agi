const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/sky.js'), 'utf8');

function skyHarness(initialBounds, reduced = false) {
  let bounds = initialBounds;
  let gradients = 0;
  const listeners = {};
  const frames = new Set();
  const gradient = {addColorStop() {}};
  const context = new Proxy({
    createRadialGradient(startX, startY, startRadius, endX, endY, endRadius) {
      assert.ok(startRadius >= 0 && endRadius >= 0, 'Canvas gradient radii must be non-negative');
      gradients++;
      return gradient;
    },
    createLinearGradient() { return gradient; }
  }, {get(target, key) { return target[key] || (() => {}); }});
  const canvas = {getContext: () => context, getBoundingClientRect: () => bounds};
  const document = {
    hidden: false,
    body: {classList: {contains: () => false}},
    documentElement: {classList: {add() {}, contains: () => false}},
    querySelector: selector => selector === 'canvas.sky' ? canvas : null,
    createElement: () => ({getContext: () => context}),
    addEventListener(name, callback) { listeners[name] = callback; }
  };
  vm.runInNewContext(source, {
    document,
    window: {
      devicePixelRatio: 2,
      matchMedia: () => ({matches: reduced, addEventListener() {}}),
      addEventListener(name, callback) { listeners[name] = callback; }
    },
    performance: {now: () => 0},
    requestAnimationFrame(callback) { frames.add(callback); return callback; },
    cancelAnimationFrame(callback) { frames.delete(callback); }
  });
  return {
    canvas, frames,
    gradients: () => gradients,
    resize(next) { bounds = next; listeners.resize(); },
    theme(next) { bounds = next; listeners.themechange(); }
  };
}

test('hidden and one-pixel canvases wait until a visible theme is available', () => {
  for (const width of [0, 1]) {
    const sky = skyHarness({width, height: 420});
    assert.equal(sky.gradients(), 0);
    assert.equal(sky.frames.size, 0);
    sky.theme({width: 390, height: 420});
    assert.ok(sky.gradients() > 0);
    assert.equal(sky.canvas.width, 780);
    assert.equal(sky.frames.size, 1);
  }
});

test('hiding the sky stops rendering and restoring it restarts safely', () => {
  const sky = skyHarness({width: 390, height: 420});
  sky.theme({width: 0, height: 0});
  assert.equal(sky.frames.size, 0);
  sky.theme({width: 1440, height: 480});
  assert.equal(sky.canvas.height, 960);
  assert.equal(sky.frames.size, 1);
});

test('narrow, tall canvases use valid gradient radii', () => {
  const sky = skyHarness({width: 2, height: 1000});
  sky.resize({width: 64, height: 2000});
  assert.ok(sky.gradients() > 0);
});

test('reduced motion draws a still sky without starting an animation', () => {
  const sky = skyHarness({width: 390, height: 420}, true);
  assert.ok(sky.gradients() > 0);
  assert.equal(sky.frames.size, 0);
  sky.theme({width: 1440, height: 480});
  assert.equal(sky.frames.size, 0);
});
