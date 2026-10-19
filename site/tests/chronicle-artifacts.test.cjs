'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const art = require('../static/chronicle-artifacts.js');

const data = JSON.parse(fs.readFileSync(path.join(__dirname, '../static/technology-tree-data.json'), 'utf8'));

test('every milestone gets a model that fits the unit sphere', () => {
  for (const node of data.nodes) {
    const model = art.build(node.id, node.topic, node.track || '');
    assert.ok(model.p.length >= 6, `${node.id}: too few points`);
    assert.ok(model.p.length <= 700, `${node.id}: too many points (${model.p.length})`);
    for (const point of model.p) {
      assert.ok(point.every(Number.isFinite), `${node.id}: bad point`);
      assert.ok(Math.hypot(...point) <= 1 + 1e-9, `${node.id}: outside the unit sphere`);
    }
    for (const [a, b] of model.e) assert.ok(model.p[a] && model.p[b], `${node.id}: edge to a missing point`);
    assert.equal(model.twinkle.length, model.p.length);
  }
});

test('the same milestone always gets the same object, and no two milestones share one', () => {
  const seen = new Map();
  for (const node of data.nodes) {
    const one = art.build(node.id, node.topic, node.track || ''), two = art.build(node.id, node.topic, node.track || '');
    assert.deepEqual(one.p, two.p);
    const signature = JSON.stringify([one.name, one.p.length, one.p.slice(0, 4).map((q) => q.map((v) => v.toFixed(4))), one.tilt.toFixed(4)]);
    assert.ok(!seen.has(signature), `${node.id} looks the same as ${seen.get(signature)}`);
    seen.set(signature, node.id);
  }
});

test('every milestone has a design of its own', () => {
  const designs = require('../static/chronicle-designs.js').designs;
  for (const node of data.nodes) {
    assert.equal(typeof designs[node.id], 'function', `${node.id} has no design`);
    assert.equal(art.build(node.id, node.topic, node.track || '').name, node.id);
  }
  assert.deepEqual(Object.keys(designs).filter((id) => !data.nodes.some((node) => node.id === id)), [], 'designs for milestones that no longer exist');
  for (const node of data.nodes) assert.ok(art.build(node.id, node.topic, node.track || '').s.length <= 900, `${node.id}: too many stars`);
});

test('every field and direction in the data has a shape family', () => {
  for (const topic of data.topics) {
    assert.ok(art.BY_TOPIC[topic.id], `no family for field ${topic.id}`);
    for (const track of topic.tracks || []) assert.ok(art.SHAPES[art.BY_TRACK[track.id]], `no family for ${topic.id}/${track.id}`);
  }
  for (const [id, name] of Object.entries(art.LANDMARKS)) {
    assert.ok(art.SHAPES[name], name);
    assert.ok(data.nodes.some((node) => node.id === id), `landmark ${id} is not a milestone`);
  }
});
