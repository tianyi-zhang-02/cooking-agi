# Self-improvement: changing code, not weights (AlphaEvolve and the Darwin Gödel Machine)

[中文](code-not-weights.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

The loop in the previous note changes weights, so every round requires training. Another approach keeps the weights fixed and changes the program around the model. A coding agent might edit its own prompts, tools and task decomposition, or iteratively improve the function it is writing, with tests scoring each change.

As of October 2026, most public, checkable "self-improvement" uses approaches of this kind. Progress depends on the quality of the automated evaluator and remains constrained by the capabilities of the underlying model.

## Why most demonstrations change code

Change some code, run the benchmark once, and you know whether it helped; bad changes are thrown away, good ones are archived, and you can always roll back. Changing weights means a training run for every candidate you evaluate. The authors of the [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) (DGM) say so directly (§1): their framework envisions agents that rewrite their own training scripts, even training a new foundation model (FM), but FM training is computationally intensive and adds substantial complexity, so they leave it as future work and improve only the design of agents built on frozen FMs.

[AlphaEvolve](https://arxiv.org/abs/2506.13131) also changes programs. Its link to the next Gemini is indirect: it finds faster kernels, which make training cheaper. In the guide's terms, DGM is type (b) and AlphaEvolve is type (c).

## AlphaEvolve: let the evaluator pick programs

The following rewrites the controller loop in Fig. 2 of the white paper and adds a few settings from §2.1–2.5:

```text
# Beforehand, the user marks editable code with # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END
# and supplies an evaluation function h(program) -> scalar metrics (higher is better)
repeat:
    parent, inspirations ← sample from the program database   # MAP-Elites plus islands, balancing exploration and exploitation
    prompt ← parent + inspirations + their scores
    diff ← code edits from an LLM ensemble                    # Gemini 2.0 Flash for volume, Gemini 2.0 Pro for quality
    child ← apply diff to parent
    scores ← evaluation cascade: cheap small tests first, harder tiers only if they pass
    store (child, scores) in the database
```

In the coding example, the thing evolved is the function body, and $h$ might be "hidden-test pass rate plus run time". People write $h$ and AlphaEvolve does not change it, so the previous note's problem of tests accepting wrong solutions carries over unchanged: whatever $h$ lets through is where evolution may head.

<!-- paper-figure:alphaevolve-loop -->

The evaluation cascade (§2.4) means the user supplies test cases of increasing difficulty, and a new program moves to the next tier only if it does well enough on all earlier tiers; new programs are also run at small scale first, so faulty ones are filtered early. Properties hard to write into $h$ can be judged by LLM-generated feedback. The program database (§2.5) borrows from MAP-Elites and island-based population models, so the population does not crowd around one local optimum.

### Reported results, and where the "self" comes in

All numbers below are Google DeepMind reporting on its own systems, from the white paper (2025-06) and two blog posts (2025-05-14 and 2026-05-07):

- **Data-centre scheduling (§3.3.1).** Candidates were evaluated in a simulator built from historical snapshots, checked on held-out recent data, then rolled out fleet-wide; post-deployment measurements found an average of 0.7% of fleet-wide compute continuously recovered.
- **Gemini kernels (§3.3.2).** Half of the collected input shapes drove evolution and half were held out for evaluation; the tiling heuristic found gives a 23% average kernel speedup and a corresponding 1% reduction in Gemini's overall training time, and kernel tuning went from months of engineering to days of automated experiments. The FlashAttention kernel speedup is 32% in the paper and "up to 32.5%" on the Google DeepMind [blog](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/).
- **Mathematics (§3.1–3.2).** Two 4×4 complex matrices multiplied with 48 scalar multiplications; applying Strassen's algorithm recursively takes 49, and this is the first result below 49 over a field of characteristic 0 in 56 years. Across more than 50 open math problems, it matched the best known constructions on about 75% and improved on about 20%.
- **One year later.** In a 2026-05-07 [post](https://deepmind.google/blog/alphaevolve-impact/), Google DeepMind says AlphaEvolve has moved from pilot to a core component of its infrastructure, is a regular tool for next-generation TPU design, and cut write amplification in Spanner by 20% (self-reported).

The "self" is in the kernel item: the white paper calls it an instance of Gemini, through AlphaEvolve, optimising its own training process, and the blog adds that this includes training the LLMs behind AlphaEvolve. But what changes is a program, not the LLM's weights; the loop speeds up the infrastructure that trains the next model. In §6 the authors write that the gains are currently moderate and that the feedback loop for improving the next version of AlphaEvolve is on the order of months. The next step they propose is to distil AlphaEvolve-augmented performance into the next generation of base models; that is a proposal, not something the paper did.

§6 also states the main limitation: it handles only problems for which an automated evaluator can be written. Many problems in mathematics and computational science qualify; in the natural sciences, often only some experiments can be simulated or automated.

## Darwin Gödel Machine: an agent edits its own repository

DGM (Zhang, Hu, Lu, Lange, Clune; UBC, Vector, Sakana AI; ICLR 2026) replaces the Gödel Machine's "rewrite only after proving it helps" with experiment: test every change on a benchmark, and keep every agent that still works in an archive. Rewritten from the paper's Algorithm 1 (Appendix C.4):

```text
Input: initial coding agent g0, benchmark suite B, iterations T
A ← {(g0, score of g0 on B)}                    # archive: keep every agent that still works
for t = 1..T:
    draw k parents from A with probabilities p_i, with replacement   # only agents scoring < 1 are eligible
    for each parent p:
        read p's evaluation logs, find the next thing to improve      # a separate FM diagnoses this (App. C.3)
        c ← p edits its own code repository
        evaluate c on B
        if c can still edit code: add (c, score) to A                 # no need to beat the parent
return A
```

Why call it self-referential: self-improvement is posed as a coding task, so a better coding agent is also a better self-modifier. Two things are not modified: the open-ended exploration process itself (parent selection, archive upkeep) and the FMs. Code edits are made by Claude 3.5 Sonnet (New); for evaluation, SWE-bench uses the same model and Polyglot uses o3-mini.

Keeping children that score below their parents allows exploration to continue along paths with an initial drop in performance. On SWE-bench, the paper reports that agents at iterations 4 and 56 temporarily scored below their parents, yet further modifications along that path eventually produced an agent that beat all its predecessors.

<!-- paper-figure:dgm-archive -->

### Choosing a parent: work it once

The sampling rule in Appendix C.2 (Eqs. 1–7):

$$
s_i=\frac{1}{1+e^{-\lambda(\alpha_i-\alpha_0)}},\qquad h_i=\frac{1}{1+n_i},\qquad w_i=s_i\,h_i,\qquad p_i=\frac{w_i}{\sum_j w_j}
$$

$\alpha_i$ is agent $i$'s benchmark score and $n_i$ its number of children that can still edit code. $s_i$ favours high scorers; $h_i$ favours agents that have not been explored much. The paper uses $\lambda=10$ and $\alpha_0=0.5$, and draws $k$ parents with replacement each iteration ($k=2$ on SWE-bench, $k=4$ on Polyglot).

A teaching example with three agents in the archive; the numbers are made up:

| Agent | $\alpha_i$ | $n_i$ | $s_i$ | $h_i$ | $w_i$ | $p_i$ |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.20 | 0 | 0.047 | 1 | 0.047 | 0.083 |
| B | 0.45 | 0 | 0.378 | 1 | 0.378 | 0.661 |
| C | 0.60 | 4 | 0.731 | 0.20 | 0.146 | 0.256 |

Take C: $s_C=1/(1+e^{-10\times0.10})=1/(1+0.368)=0.731$, $h_C=1/(1+4)=0.20$, $w_C=0.146$. The three weights sum to 0.571; divide to get $p$. The top scorer C already has 4 children, so B is drawn most often. The weakest agent A still has an 8.3% chance; with $k=2$ draws, the probability that A is drawn at least once is $1-(1-0.083)^2\approx0.16$.

Change one condition: suppose C has no children yet. Then $w_C=0.731$, the total becomes 1.156, and the three probabilities become 0.041, 0.327 and 0.632. The number of children moves selection as much as the score does. Note that the rule counts children but ignores whether they were any good; that is exactly where the HGM critique below starts.

### Results, cost and ablations

- **Results (§4.4).** After 80 iterations, SWE-bench goes from 20.0% to 50.0% and the full Polyglot benchmark from 14.2% to 30.7%. Gains transfer to other FMs and languages: the best agent evolved on Polyglot reaches 24.5% on SWE-bench, above the initial 20.0%.
- **Cost (Appendix E.1).** One SWE-bench run takes about 2 weeks and about US$22,000; each of the two ablation baselines costs about US$10,000.
- **Ablations (§4).** Without self-improvement (the initial agent always does the modifying, which replicates ADAS), early gains taper off quickly. Without open-ended exploration (only the latest agent is kept), one bad modification makes later improvement harder. Both fall short of the full DGM on both benchmarks.
- **Limits (§6).** The authors state that DGM "is inherently limited by the capabilities of the underlying FM" ([§6](https://arxiv.org/html/2505.22954v3#S6)); experiments ran in sandboxes under human oversight. Appendix H records objective hacking; see the [previous note](generate-verify-train.en.md).

## Later variants: SICA, HGM, AIDE²

[SICA](https://arxiv.org/abs/2504.15228) (University of Bristol and iGent AI, 2025-04) is simpler: no archive; the best agent so far always acts as the meta-agent that edits the codebase. Its utility combines score, cost and time, $U=w_{\rm score}\,p_{\rm score}+w_{\rm cost}\,(1-\min(1,p_{\rm cost}/\$10))+w_{\rm time}\,(1-\min(1,p_{\rm time}/300\,{\rm s}))$. On a random 50-problem subset of SWE-bench Verified it goes from 17% to 53%, with 15 iterations costing about US$7,000. The paper also records "agent framework saturation": on reasoning-heavy tasks where the base model is already strong, editing the scaffold brings almost no gain.

The [Huxley-Gödel Machine](https://arxiv.org/abs/2510.21614) (KAUST, 2025-10) identifies a problem with DGM-style selection that it calls the metaproductivity–performance mismatch: an agent's own benchmark score predicts poorly how good its descendants will be. HGM instead estimates CMP (clade-metaproductivity) by aggregating the performance of all of an agent's descendants, and uses Thompson sampling to decide which branch to expand. The abstract reports that it beats prior self-improving-agent methods on SWE-bench Verified and Polyglot with fewer CPU hours, and that an agent optimised with GPT-5-mini, when run with GPT-5, matches the best human-engineered agents the authors checked on SWE-bench Lite.

[AIDE²](https://arxiv.org/abs/2609.26457) (Weco AI, 2026-09-22) applies the same idea to a production ML-research agent: an outer loop rewrites the inner agent's search policy, memory and context-compression code, and a change is accepted only if it improves results on held-out data the rewritten agent never sees. One 8-day run produced 99 rewrite proposals and accepted 7, raising the grade from 0.703 to 0.778 (the human-engineered AIDE_human sits at 0.749); two further full runs accepted 2 and 4. The strongest agent matches or exceeds AIDE_human on 4 external benchmarks that never influenced selection, including the out-of-distribution WeatherBench 2; on a separate held-out task family, the reward-hacking rate fell from 55% to 32% (AIDE_human: 39%). The authors also ran an "ignition test": use a discovered agent as the outer-loop optimiser and see whether it improves agents better than the agent that created it. It still produced accepted improvements, but because noise compounds across both loops and more seeds were too costly, its effect could not be decisively distinguished from the strong baseline.

## Side by side

| System | What changes | Evaluator | What was measured | What stays fixed |
| --- | --- | --- | --- | --- |
| AlphaEvolve | User-marked code blocks: heuristics, kernels, algorithms | User-written evaluation function $h$, optionally with a cascade | 0.7% fleet compute recovered; 23% kernel speedup; 48 multiplications | LLM weights, evaluation function |
| DGM | The agent's own Python repository | SWE-bench and Polyglot scores | SWE-bench 20.0% → 50.0% after 80 iterations | FM weights, parent selection and archive |
| SICA | The agent's own code, edited by the best agent so far | Utility combining score, cost and time | 50-problem subset 17% → 53% | FM weights |
| HGM | Agent code; which branch to expand chosen by estimated CMP | SWE-bench Verified, Polyglot | Beats prior methods with fewer CPU hours | FM weights |
| AIDE² | A research agent's search-policy, memory and context-compression code | Held-out data the rewritten agent never sees | 7 rewrites accepted in 8 days, 0.703 → 0.778 | The LLMs used, the outer acceptance rule |
| STaR (contrast, type (d)) | Model weights | Ground-truth answers | CommonsenseQA dev 72.5% | Human-written training procedure |

What the first five rows share: the evaluator is written by people, the changes are executable code, and the underlying model does not move. They show that within the range where the evaluator can be trusted, systems keep finding improvements; what they have not shown is whether improvement gets faster as it goes.

## What would have to change for "a model changes its own weights"

Using the guide's three questions, here is our analysis, not the conclusion of any paper:

- **The object that changes** would be weights instead of code. Every candidate needs a training run, so evaluation costs far more than editing code; DGM, editing code only, already takes about 2 weeks and about US$22,000 per run.
- **Who designs the update** would move from people to the model: the model writes the data pipeline, loss and training code. DGM lists rewriting its own training script to update the FM as a future direction (§6); the AlphaEvolve authors propose distilling the augmented performance into the next generation of models (§6). Both are proposals.
- **The acceptance signal** must be out of the edited system's reach. DGM saw more objective hacking when its detection functions were not hidden; AIDE² accepts changes only on held-out data the rewritten agent never sees.

The closest public precursors: [PostTrainBench](https://arxiv.org/abs/2603.08640) (2026-03-09) has agents post-train a different base LLM in 10 hours on one H100; the best agent reaches 23.2% against 51.1% for the official instruct models, and reward hacking appeared along the way; OpenAI's GPT-5.6 system card of 2026-07-09 put a Lite version into its self-improvement eval suite. Karpathy's [autoresearch](https://github.com/karpathy/autoresearch), released in March 2026, has an agent edit the `train.py` of a small single-GPU model, train for 5 minutes, and keep or discard the change by val_bpb; the README reports no results. As of 2026-10-15, no public demonstration was found of a model autonomously improving its own weights and passing held-out evaluation.

## What to watch on this line next

These are open questions and the kind of result that would change the current reading; they are not predictions.

1. **Does improvement compound?** HGM (2025-10) found benchmark scores predict descendants poorly; AIDE²'s ignition test (2026-09-22) could not be decisively distinguished from the baseline. A system in which, across several seeds, a discovered agent clearly improves agents better than its creator would call for re-evaluating "bounded gains".
2. **Does the evaluator hold?** DGM (2025-05) recorded objective hacking; AIDE² saw reward hacking fall on one held-out task family, in a single run. Routine reporting of gaming rates on held-out task families across more systems, with independent replication, is what it would take to tell whether such loops are getting safer or learning to route around checks.
3. **Will AlphaEvolve's cycle for improving its own base shorten?** The authors (2025-06) described the feedback loop as months long with moderate gains; Google DeepMind said on 2026-05-07 that it is now a core infrastructure component. A published measurement showing that cycle clearly shortening, or the proposed distillation step actually done and reported, would mean the loop has tightened.
4. **From code to weights.** On PostTrainBench (2026-03-09) the best agent scores 23.2% against 51.1% for the official models. Agents closing that gap, or a public result of a system training its own weights and passing held-out evaluation, would push this line from type (b) toward type (a).

Next: [How to measure "AI speeding up AI"](measuring.en.md).
