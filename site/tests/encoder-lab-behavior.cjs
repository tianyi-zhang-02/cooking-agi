const assert = require("node:assert/strict");
const {mlmExample, attentionExample} = require("../static/encoder-lab.js");

const baseline = mlmExample();
assert.deepEqual(baseline.input, ["[CLS]", "the", "tea", "is", "[MASK]", "[SEP]", "[PAD]"]);
assert.deepEqual(baseline.labels, [-100, -100, 5, -100, 7, -100, -100]);
assert.deepEqual(baseline.attention, [1, 1, 1, 1, 1, 1, 0]);
assert.ok(Math.abs(baseline.loss - 0.8047189562170501) < 1e-12);
assert.ok(Math.abs(baseline.wrongLoss - baseline.loss * 2 / 7) < 1e-12);
for (const replacement of ["mask", "random", "unchanged"]) {
  for (const padding of [0, 1, 5, 9, 32]) {
    const example = mlmExample(replacement, padding);
    assert.equal(example.loss, baseline.loss);
    assert.equal(example.labels.filter(label => label !== -100).length, 2);
    assert.equal(example.attention.filter(Boolean).length, 6);
    assert.equal(example.input.length, 6 + padding);
    assert.equal(example.labels[4], 7);
    assert.equal(example.wrongLoss, baseline.loss * 2 / (6 + padding));
  }
}
assert.equal(mlmExample("random").input[4], "hot");
assert.equal(mlmExample("unchanged").input[4], "cold");
assert.ok(mlmExample("mask", 1, 0.9).loss < baseline.loss);
assert.ok(mlmExample("mask", 1, 0.01).loss > baseline.loss);
assert.equal(mlmExample("mask", 1, 1).terms[1], -0);
for (const bad of [0, -1, 1.1, NaN, Infinity, "0.5"]) assert.throws(() => mlmExample("mask", 1, bad), RangeError);
for (const bad of [-1, 33, 0.5, NaN, "5"]) assert.throws(() => mlmExample("mask", bad), RangeError);
assert.throws(() => mlmExample("unknown"), RangeError);

for (const mode of ["encoder", "decoder", "cross"]) {
  for (let query = 0; query < 3; query++) {
    const example = attentionExample(mode, query);
    assert.equal(example.matrix.length, 3);
    assert.equal(example.keys.length, 4);
    assert.equal(example.visible.length, mode === "decoder" ? query + 1 : 3);
    for (let row = 0; row < 3; row++) {
      assert.equal(example.matrix[row][3], false);
      assert.equal(example.matrix[row][row], true);
      for (let column = 0; column < 3; column++) {
        assert.equal(example.matrix[row][column], mode !== "decoder" || column <= row);
      }
    }
  }
}
assert.deepEqual(attentionExample("decoder", 1).visible, ["[BOS]", "ich"]);
assert.deepEqual(attentionExample("cross", 0).visible, ["I", "like", "tea"]);
assert.deepEqual(attentionExample("encoder").queries, ["I", "like", "tea"]);
assert.deepEqual(attentionExample("cross").queries, ["[BOS]", "ich", "mag"]);
for (const bad of [-1, 3, 0.5, NaN, "1"]) assert.throws(() => attentionExample("encoder", bad), RangeError);
assert.throws(() => attentionExample("unknown"), RangeError);
process.stdout.write("Encoder labs: loss, padding, corruption, visibility, and invalid inputs pass.\n");
