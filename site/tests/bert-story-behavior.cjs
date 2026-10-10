const assert = require("node:assert/strict");
const {storyState} = require("../static/bert-story.js");

const initial = storyState();
assert.deepEqual(initial.inputs, ["[CLS]", "the", "[MASK]", "is", "cold", "[SEP]"]);
assert.deepEqual(initial.outputs, [2]);
assert.equal(initial.query, 2);

for (const step of [0, 1, 2, 3]) {
  for (const query of [0, 1, 2, 3, 4, 5]) {
    for (const task of ["mlm", "classify", "tag"]) {
      const state = storyState(step, task, query);
      assert.equal(state.step, step);
      assert.equal(state.query, query);
      assert.equal(state.inputs[2], task === "mlm" ? "[MASK]" : "tea");
      assert.equal(state.inputs[4], "cold");
      assert.deepEqual(state.positions, [0, 1, 2, 3, 4, 5]);
      assert.deepEqual(state.segments, [0, 0, 0, 0, 0, 0]);
      assert.deepEqual(state.visible, [0, 1, 2, 3, 4, 5]);
      assert.deepEqual(state.outputs, {mlm: [2], classify: [0], tag: [1, 2, 3, 4]}[task]);
    }
  }
}

initial.inputs[2] = "changed";
initial.outputs.push(0);
assert.equal(storyState().inputs[2], "[MASK]");
assert.deepEqual(storyState().outputs, [2]);

for (const args of [[-1], [4], [0.5], [NaN], [0, "unknown"], [0, "mlm", -1],
                    [0, "mlm", 6], [0, "mlm", 1.5], ["0"], [0, "mlm", "4"]]) {
  assert.throws(() => storyState(...args), RangeError);
}
process.stdout.write("BERT walkthrough: 72 valid states, output sites, bidirectional visibility, and invalid inputs pass\n");
