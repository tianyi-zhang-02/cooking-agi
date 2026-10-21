# One agent or several: when splitting pays off

[中文](single-vs-multi-agent.md) · **English**

> Reading time: ~13 min · Last reviewed: 2026-10

A research task runs slowly and tool results keep filling the context. Could a lead agent split the task and send several subagents to search in parallel? This can reduce wall-clock time and keep each subtask's context separate. It also requires more tokens and hand-offs, with a risk of losing information at each hand-off.

Splitting is more useful when subquestions are independent and mostly involve reading. When steps depend on each other or must share unstated decisions, a single agent is usually more reliable. Compare the two designs at the same token and tool budget.

## What multi-agent actually changes

[Common building blocks](patterns.en.md) already lists orchestrator-workers, parallelization and related structures, so this note does not repeat them. It asks one question: when the same task moves from "one agent finishes it in one context" to "a coordinator plus several subagents", what concretely changes?

| Change | What it may buy | New cost or risk |
| --- | --- | --- |
| Separate contexts | Each subagent sees only its own task description (brief) and tool results, so its context is shorter and more focused | Subagents cannot see the whole picture; the coordinator usually sees summaries, not the sources |
| Parallel exploration | Wall-clock time depends on the slowest branch, not on the sum of all steps | More calls at the same moment, so higher peak spend and rate-limit pressure |
| Specialised prompts and tools | Read-only search, write actions and review can get different instructions and permissions | Vague roles lead to overreach, duplicated work or gaps |
| A coordinator | It can split the task on the fly, merge results and decide whether to send another round | One more layer of planning and hand-off; the coordinator can split or merge wrongly too |

What does not change matters as well. Subagents usually run on the same model and tend to share its blind spots, so sending more of them does not make their errors independent. More agents also do not mean more verification unless you design a review step on purpose.

## The cost is not only tokens

The example below calculates token usage and latency: every context carries its own fixed prompt, and planning and synthesis can only run serially. First, consider four costs that are harder to quantify:

- **Hand-off loss.** A subagent compresses more than ten thousand tokens of raw results into a 1k summary, and the fact the coordinator needs may be in the part that was cut.
- **Conflicting implicit decisions.** Every action carries unstated choices, such as how to define "holiday" or which version of a notice to use. When several subagents choose independently, the combined result can contradict itself.
- **Error propagation.** When a subagent returns a wrong fact, the coordinator usually has no source text to check it against and writes it straight into the conclusion.
- **Harder evaluation and debugging.** The same input can take a different path each run; when the final answer is wrong, you first have to work out which agent made the mistake, and at which step.

## What the primary sources say

### Anthropic's Research system (2025-06-13)

Anthropic's [engineering post](https://www.anthropic.com/engineering/multi-agent-research-system) describes its multi-agent research system: a lead agent makes a plan, sends several subagents to search in parallel, then synthesises. The most quoted sentence, in the [section on the benefits of a multi-agent system](https://www.anthropic.com/engineering/multi-agent-research-system#benefits-of-a-multi-agent-system), says that a multi-agent system with Claude Opus 4 as lead and Claude Sonnet 4 subagents "outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval". The example given is finding all board members of the companies in the S&P 500 Information Technology sector, which the single agent failed to do with slow, sequential searches.

The **unspecified conditions** matter too: the sentence gives no absolute scores for either side, does not say whether 90.2% is a relative gain or percentage points, and does not say whether the two sides had the same token budget or number of tool calls. The evaluation set is not public either.

By the post's own account, the two sides probably did not spend the same tokens: on BrowseComp, three factors explained 95% of the performance variance, and token usage alone explained 80% (the other two were the number of tool calls and the model choice); agents typically use about 4× more tokens than chat, and multi-agent systems about 15×. The authors conclude that multi-agent systems work mainly because they can spend enough tokens, so the task's value must justify that cost. The post also says that domains requiring all agents to share the same context, or with many dependencies between agents, are not a good fit today. Most coding tasks have fewer truly parallel parts than research.

### Cognition: don't split yet (2025-06-12)

In [Don't Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents), Walden Yan of Cognition makes the opposite engineering call. He imagines splitting "build a Flappy Bird clone" between two subagents, one for the background and one for the bird: one might produce a Super Mario-style background and the other a bird that neither looks nor moves like the game's, leaving the coordinator two halves that do not fit. He proposes two principles: share context, and share full agent traces rather than single messages; actions carry implicit decisions, and conflicting decisions produce bad results. His default is a single-threaded linear agent, with a model that compresses history when the task gets too long (which he admits is hard to get right), rather than a parallel split.

This is an argument from engineering experience with no controlled numbers. It does not fully contradict the Anthropic post: one is about read-heavy research that can run in parallel, the other about long tasks that must produce one consistent artifact.

### MAST: how multi-agent systems fail

Cemri et al., [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) (v1 March 2025, v3 October 2025; listed among NeurIPS 2025 Datasets and Benchmarks Track papers, though the arXiv page does not record it), collected 1642 execution traces from 7 open-source multi-agent frameworks and annotated why they failed. The MAST taxonomy was built by 6 experts through a grounded-theory analysis of more than 150 traces; agreement was checked by 3 annotators over three rounds on 15 traces, reaching κ = 0.88; an LLM annotator (o1) then labelled the full dataset, agreeing with human labels at κ = 0.77.

MAST has 14 failure modes in three categories. According to Figure 1 of v3: system design issues (ignoring task or role specifications, repeating steps, not knowing when to stop, and so on) make up 44.2%; inter-agent misalignment (not asking for clarification, derailing from the task, withholding key information, and so on) 32.3%; task verification (premature termination, missing or incorrect verification) 23.5%. The introduction reports failure rates between 41% and 86.7% across these systems. It measures why multi-agent systems fail; it does not compare them with a single agent at equal budget.

<!-- paper-figure:mast-taxonomy -->

### Budget-controlled comparisons (2025–2026)

- Kim et al. of Google Research, Google DeepMind and MIT, in v3 (April 2026) of [Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296) (preprint), ran 260 configurations across 6 agent benchmarks, 5 architectures (a single agent and 4 multi-agent variants) and 3 model families, and state that they standardised tools, prompts and compute (v1 and v2 had 4 benchmarks and 180 configurations, so cite the version). The change relative to a single agent ranges from +80.8% on decomposable financial analysis to −70.0% on the sequential planning task PlanCraft; on PlanCraft, all 4 multi-agent variants got worse (−39% to −70%). Their regression also shows that once the single-agent baseline exceeds roughly 45%, adding agents has negative returns.
- Tran and Kiela of Stanford, in [Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets](https://arxiv.org/abs/2604.02460) (preprint, April 2026), fix the "thinking token" budget and compare a single agent with 5 multi-agent architectures on FRAMES and MuSiQue 4-hop. Except at the lowest budget of 100 tokens, where almost no reasoning happens, the single agent is best or statistically indistinguishable from the best at every budget; the paper also says it spends fewer thinking tokens. A multi-agent design becomes competitive only after heavy masking or replacement of context tokens, and that test used a single model (Qwen3-30B-A3B) and compared only against the sequential multi-agent variant. They also point out that artifacts in API budget control (particularly for Gemini 2.5) and in the benchmarks themselves can inflate the apparent gains of multi-agent designs. Note that this is text-only multi-hop reasoning without tools.
- Gao et al., [Single-agent or Multi-agent Systems? Why Not Both?](https://arxiv.org/abs/2505.18286) (May 2025), compare the two across several agent applications and report that the advantage of multi-agent over single-agent shrinks as models get stronger.

<!-- paper-figure:agent-scaling-benchmarks -->

Taken together, the most consistent conclusion is: **when a task splits into independent information streams, a split may help; when steps are strongly sequential or a single agent already does well, a split often hurts.** As of 2026-10-15, neither of the first two preprints carries formal publication information on arXiv; their exact percentages depend on their own budget definitions and baselines, so do not transfer them directly to your task.

## Work it through: three subquestions, split or not

All numbers below are teaching assumptions for seeing the mechanism, not measurements of any system.

Task: find this year's holiday opening rules for three libraries, A, B and C (reusing the library example from [Deep Research](deep-research.en.md)). The three subquestions are independent.

- Each subquestion needs 4 tool calls; each call (one model step plus the tool's return) takes 10 seconds and returns 4k tokens.
- Each context carries a fixed 3k prompt (system prompt, tool descriptions, task); the final answer is 2k, and writing it is one more 10-second step.
- In the multi-agent version, the coordinator plans for 10 seconds and writes a 0.5k brief per subagent; each subagent returns a 1k summary; the coordinator then spends 10 seconds synthesising.
- Tokens count each piece of content once per context it enters. This compares the content held across contexts without accumulating history sent again on later calls. It is not an API billing model: a real bill distinguishes input, output, cache writes, and cache reads on every call. Cache hits need not be free, nor do they mean later generation stops attending to the history. [Prompt caching prices](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#pricing)

| Item | Single agent | Coordinator + 3 subagents |
| --- | --- | --- |
| Tool calls | 12, one after another | 12, three branches in parallel |
| Wall-clock time | 12×10+10 = 130 s | 10+4×10+10 = 60 s |
| Tokens | 3+12×4+2 = 53k | coordinator 3+1.5+3+2 = 9.5k, subagents 3×(3+0.5+16+1) = 61.5k, total 71k |
| Longest single context | 53k | 20.5k |

Time drops to 60 seconds, but not to a third: planning and synthesis cannot run in parallel. Tokens rise by about 34%, from the repeated fixed prompts, the briefs and the summaries.

Now the chance of a correct answer. Suppose the final answer needs one key fact from each subquestion. Let $s_1$ be the probability that the single agent correctly finds and keeps each fact in a long context mixing 12 results. Let $s_2$ be the probability that a subagent correctly finds and retains the fact until summarization, in a short context holding only its own 4 results. Let $d$ be the probability that the summary drops the fact, conditional on the subagent having found and retained it beforehand. Then $s_2(1-d)$ is the probability that one fact survives summarization.

This model considers only failed lookup and omission during summarization. To multiply across all three required facts, also assume that their successful retention events are mutually independent within each design, and that synthesis introduces no further errors. Separately searchable subquestions do not establish this independence: a shared model or source can still produce correlated errors. [Conditional probability and the multiplication rule](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class03-prep.pdf)

$$
P_{\rm single}=s_1^{\,3},\qquad P_{\rm multi}=\big(s_2\,(1-d)\big)^{3}.
$$

Take $s_1=0.90$ and $s_2=0.95$. The single agent gets $0.9^3=0.729$. With $d=0.02$, the multi-agent version gets $(0.95\times0.98)^3\approx0.807$ and the split wins; with $d=0.10$, it gets $0.855^3\approx0.625$ and the split loses. They break even when $s_2(1-d)=s_1$, that is $d^*=1-0.90/0.95\approx5.3\%$.

In this simplified model, **any accuracy gain from splitting comes entirely from $s_2>s_1$: shorter contexts are less likely to lose key facts.** If the single agent already uses its context well ($s_1=s_2$), any $d>0$ makes the multi-agent version worse. This is consistent with the direction of Tran and Kiela's argument.

Now change one condition: the subquestions become sequential. First check whether library A switched to the city's new notice, then use that notice to check library B, then library C. The parallel width drops to 1, and the multi-agent time becomes $10+40+10+40+10+40+10=160$ seconds, slower than the single agent's 130; tokens stay around 71k; and A's finding is retold twice, through "summary → coordinator → next brief". The split keeps only one benefit, context isolation, and costs an extra 30 seconds and about 18k tokens.

<!-- widget:step-subagent-timeline -->

Change a different condition: the three subquestions are independent again, but each must decide whether "holiday" includes the eve of a public holiday (many libraries close early that day). If the brief does not say, and each subagent independently picks one reading with probability 0.8 and the other with 0.2, the three results use a consistent definition with probability only $0.8^3+0.2^3=0.52$. A single agent makes this decision once, so its results are consistent by construction, though possibly consistently wrong. The fix is to put the definition into every brief, or to check definitions explicitly during synthesis, not to add another agent.

## How to choose: start from the shape of the task

| Task shape | Recommendation | What to measure |
| --- | --- | --- |
| Independent, read-heavy subquestions that do not fit in one context (breadth-first research) | Coordinator plus parallel read-only subagents; briefs state the objective, output format, sources and boundaries | Wall-clock time, tokens per successful task, share of duplicated searches, retention of key facts after hand-off |
| Steps that depend on earlier steps (step-by-step planning or derivation) | A single agent; compress history when the context gets too long | Success at equal budget; whether key decisions survive compression |
| Several places writing the same state (one codebase, one document) | Keep a single writer; subagents only answer well-scoped questions | Conflicts and rework; consistency of interfaces and style |
| Long tool outputs where only a small part matters | First make the tool return shorter results; otherwise let a subagent read them and return only the key points | Main context length; rate at which filtering drops key facts |
| Different permissions or tool sets needed | Split by permission and enforce tool limits in the execution layer, not only in the prompt | Number of out-of-permission calls |
| Independent review needed | Start a separate review context that does not see the generation process, with explicit checking criteria | Detection rate and false-alarm rate on deliberately injected errors |
| Single-agent baseline already high | Do not split yet | Whether the gain from splitting exceeds run-to-run variation |

The basis for each row:

- The first row is the breadth-first querying that Anthropic says suits multi-agent systems best; the second matches the degradation Kim et al. saw on PlanCraft; the last matches the saturation point of about 45% they report.
- The third row is Cognition's argument. They describe Claude Code as of June 2025: subagents usually only answered questions and did not write code in parallel with the main agent.
- Fourth row: Anthropic describes subagents as "intelligent filters", and Cognition also says a subagent's investigation need not stay in the main agent's history.
- The fifth row is engineering advice; I did not find a dedicated controlled experiment.
- The evidence for the sixth row is mixed: MAST finds that systems with explicit verifiers fail less overall, but many verifiers only do surface checks (such as whether code compiles); Kim et al. measure error amplification of 17.2× for independent agents versus 4.4× for a centralised design, but in their regression that controls for other metrics, the amplification factor itself is not significant.

## How to evaluate: equal budget, inspect hand-offs, follow traces

**Align the budget first.** At minimum, fix total tokens (input and output recorded separately), the number of tool calls and a wall-clock limit, then compare success; ideally compare at several budget points and plot a curve. If the multi-agent version lets each subagent search twice as much (8 calls per branch), it takes about 100 seconds and about 119k tokens; then give the single agent the same 24 calls (about 250 seconds, 101k tokens) before comparing, or the gain may simply be extra spending. Record actual consumption, not the requested budget; Tran and Kiela found that the two can disagree.

**Use the same task set for paired comparisons.** Run both designs on every task, several times each, and look at paired differences and their spread rather than two averages; see [Controlled comparisons and slices](../07-evaluation/ablation-and-slices.en.md). Anthropic's experience is that about 20 real queries were enough early on to see large changes; research outputs were graded by an LLM judge with a rubric covering factual accuracy, citation accuracy, completeness, source quality and tool efficiency, alongside manual testing, because people find edge cases that evals miss.

**Check every hand-off.** Record briefs and summaries as spans in the trace, and on a sample check whether each fact the answer needs appears in the subagent's raw tool results, and whether it appears in its summary.

```text
trace_id=demo-41
lead.plan            → brief_A, brief_B, brief_C
  sub_A  tool×4      → summary_A (1k)   key fact: in sources, not in summary   ← hand-off loss
  sub_B  tool×4      → summary_B (1k)   key fact: not in sources at all        ← not found
  sub_C  tool×4      → summary_C (1k)   key fact: in both
lead.synthesize      ← summary_A..C
```

"In the sources, not in the summary" is hand-off loss: change the summary format or pass file references instead (Anthropic has subagents store their outputs in external systems and return lightweight references). "Not in the retrieved material" calls for checking search coverage and tool retrieval first. A different query or additional source may uncover the missing fact; having more agents reread the same material cannot itself fill an evidence gap. The MAST categories make a usable labelling vocabulary.

**Debug by following the trace, not by asking the model.** Parent and child spans should connect the coordinator and every subagent; see [Agent observability](../06-systems/agent-observability.en.md) for the fields. Pinpointing which agent failed at which step is hard in itself: Zhang et al.'s [Who&When](https://arxiv.org/abs/2505.00212) (ICML 2025) collects failure logs from 127 multi-agent systems, yielding 184 failure-annotation tasks. The 127 counts systems, including task-specific, automatically generated agent configurations, not 127 independently developed frameworks; 184 is the number of annotation tasks. See [§3 of the paper](https://arxiv.org/html/2505.00212v1#S3).

The [paper's abstract](https://proceedings.mlr.press/v267/zhang25cq.html) reports best attribution accuracies of 53.5% for the responsible agent and 14.2% for the failing step. Treat an LLM's attribution as a lead; confirm the cause with a controlled replay, for example by swapping one subagent's summary for a correct version and checking whether the final answer becomes correct.

## What you can and cannot conclude

| Evidence | Supports | Does not show |
| --- | --- | --- |
| Anthropic's 90.2% | Under their unpublished internal eval and configuration, the multi-agent version was better | That multi-agent wins at equal budget; the absolute size of the gain; that your task will improve as much |
| Tokens explain 80% of variance | On BrowseComp, how many tokens are spent is the main explanatory factor | That the split structure itself caused the gain |
| Cognition's two principles | For tasks that must produce a consistent result, shared context is critical | That any parallelism is a bad idea |
| MAST | Failures cluster in design, coordination and verification, and the mix differs by system | That multi-agent systems are always worse than a single agent |
| Kim et al., Tran and Kiela | After aligning budgets, gains depend on task structure; on text-only multi-hop reasoning a single agent does not lose | That any specific percentage applies to your system |

Read next: how to write the split structures in [Common building blocks](patterns.en.md), question queues and evidence records in [Deep Research](deep-research.en.md), full cost accounting in [Choosing an agent model](model-choice.en.md), and when to bring in a person in [Human-in-the-Loop](../06-systems/human-in-the-loop.en.md).
