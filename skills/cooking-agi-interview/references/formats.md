# Session formats

Pick one at the start, or switch when the user asks.

## Mock loop (default)

~5 questions (or a time box) across the user's focus topics, mixing difficulty: start with a
fundamentals question, then go deeper. Full loop for each: ask → wait → probe → feedback →
record. End with `interview.py report` and a review list.

## Topic drill

Stay on one topic until the user scores 3+ on three questions in a row. Use bank questions
first, then write your own from the topic's notes, moving from definition → mechanism →
numbers → failure modes → implementation.

## Rapid-fire review

10-20 short questions, one-line answers, minimal follow-ups. Grade each in one line
("3/4 — yes, but GQA shares K/V per group, not per layer"). Record every score. Good for the
day before an interview.

## Weak-area review

`interview.py ask --weak -n 5`. Re-ask each from a new angle rather than verbatim, so the user
is not just recalling your earlier feedback.

## ML system design

Use the notes under `learn/system-design/`, `practice/ml-system/`, `practice/rag/`,
`practice/recommender-systems/` and `practice/post-training/`. Give a prompt ("design a
retrieval-augmented support assistant for 10M documents"), then play the interviewer across
stages: requirements and metrics → data → model / retrieval → training and evaluation →
serving, latency and cost → monitoring and iteration. Push on one stage at a time. Score the
whole design with the rubric and point at the stage that was weakest.

## Coding

Use `interview/algorithms/` (LeetCode patterns), `interview/coding-checks.en.md`,
`interview/python*.en.md`, `00-foundations/hand-write-kit.en.md` and
`00-foundations/pytorch/`. Ask for an implementation (e.g. scaled dot-product attention with a
causal mask, top-k sampling, a pattern problem), let the user write it, then review: correctness,
complexity, edge cases, and a test. If the user wants to run the code, let them; do not write it
for them first.

## Teach-back

For learning rather than testing: ask the user to explain a concept as if to a junior engineer,
then correct and fill gaps from the note. `interview.py ask --kind concept -t TOPIC` gives
the notes' one-paragraph summary cards, a good starting point. Afterwards ask one check question and record it.
