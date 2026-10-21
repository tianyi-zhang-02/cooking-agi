# Preparing for RE / MLE interviews

[中文](ai-roles-prep.md) · **English**

> Sources checked: 2026-10-11 · Public accounts and original exercises, not company questions or a universal hiring standard.

A paper can make sense while the implementation still feels out of reach. Writing one small module is often more revealing than saving another preparation list. This page connects topics from public job-search accounts to explanations and implementations on this site. You do not need to finish every row.

## Start with the work, not just the title {#routes}

RE means Research Engineer, MLE means Machine Learning Engineer, and RS means Research Scientist. The work behind each title varies across teams. Read the job description and ask your recruiter about the rounds, coding environment, documentation access, and AI policy. Someone else's experience of not being asked a topic is not a reason to rule it out.

| Target work | Start with | Then go deeper | Leave for later when time is short |
| --- | --- | --- | --- |
| RE / MLE focused on models and training | Algorithms, [ML implementations](#implement), training bugs, data and evaluation, project decisions | Data and objectives, distributed training, experiment design | Every model family's details |
| RS focused on research | Required coding and implementation skills, research questions, evidence, personal contributions, follow-up experiments | Papers, derivations, and failure cases in your field; not necessarily an LLM checklist | Implementing every popular architecture |
| Inference / serving / ML systems | Algorithms, general coding, generation loops and caches, memory estimates, scheduling | [Request paths and performance diagnosis](#inference), then one layer in depth with performance evidence | Memorizing every framework class |
| General SWE | Algorithms, [code reading](code-comprehension.en.md), tests, projects; system design if requested | [Python and algorithm patterns](leetcode.en.md) | Unrelated model derivations |

For internships or new-grad roles, an MLP, attention, a training loop, and one well-understood project are a manageable starting point. Experienced candidates can put more weight on performance decisions, recovery, and team interfaces. This is a suggested practice order, not a hiring threshold for each level.

## ML coding: leave something you can test {#implement}

Pick one row at a time: define the inputs and outputs, write a small test, implement, and change one assumption. These links include explanations and runnable code. Having read a chapter is not the same as being able to implement it unaided.

| Implementation | Where to practice | Check at least this |
| --- | --- | --- |
| Two-layer MLP forward / backward | [Implementation kit](../00-foundations/hand-write-kit.en.md#the-last-one-is-the-dividing-line), with `interview_kit.py` | Mean-loss normalization; compare analytical gradients with float64 central differences away from ReLU kinks |
| Multi-head causal attention | [Multi-head attention](../00-foundations/core/multi-head-attention.en.md), [NumPy implementation](../00-foundations/code/attention_numpy.py) | Label `[B,H,T,D]`; changing a future token must not affect earlier outputs |
| Stable softmax and cross-entropy | [Softmax example](../00-foundations/hand-write-kit.en.md#softmax-numerics), [cross-entropy](../00-foundations/pytorch/cross-entropy.en.md) | Large logits, masked classes, fully masked rows; avoid taking the log of an underflowed probability |
| Temperature, top-k, top-p, EOS | [Decoding](../00-foundations/core/decoding.en.md), [implementation kit](../00-foundations/hand-write-kit.en.md), [when generation stops](../00-foundations/core/decoder-only.en.md#1-inference-the-final-assistant-marker-is-the-generation-boundary) | Keep the token that crosses the top-p threshold; stop each sequence independently and enforce a length cap |
| Incremental decoding and KV cache | [Implementation lab](../00-foundations/code/README.en.md), [KV mechanisms](../00-foundations/deep-dives/kv-cache-and-inference.en.md) | With fixed weights and dropout off, compare per-position logits for the same token sequence against a full forward pass within tolerance, before comparing sampled text |
| A small MoE | [Routing](../00-foundations/moe/router.en.md), [load balancing](../00-foundations/moe/load-balancing.en.md) | How expert outputs combine; gate normalization, capacity limits, and unused experts |
| Break a training loop, then fix it | [PyTorch training](../00-foundations/pytorch/training-loop.en.md), [data and objectives](../practice/post-training/data-and-objectives.en.md) | Change label shift, gradient clearing, padding loss masks, or `train/eval` one at a time; inspect losses and gradients, not just exceptions |
| A small decoder-only Transformer | [Implementation lab](../00-foundations/code/README.en.md), [GPT family](../00-foundations/model-families/gpt.en.md) | Connect attention, FFNs, residuals, and norms; reduce loss on a tiny batch before checking generalization |

The last row needs a distinction: this site's `model.py` uses **RMSNorm, RoPE, GQA, and SwiGLU**, not GPT-2. For a GPT-2-style exercise, specify learned absolute positions, pre-norm LayerNorm placement, a GELU MLP, and the final LayerNorm. Renaming a modern decoder does not reproduce GPT-2. [Karpathy's GPT-2 implementation video](https://www.youtube.com/watch?v=l8pRSuU81PU) is a separate route for that exercise. This site does not currently provide a second, complete GPT-2 reproduction.

## Small algorithms and tensor operations still matter {#small-problems}

[Five short exercises](coding-checks.en.md) cover fundamentals that can disappear underneath a long LLM reading list:

| Exercise | Check more than the final answer |
| --- | --- |
| kNN: loops, vectorization, blocks | Matching neighbors and votes, distance ties, temporary array sizes |
| Two binary-search boundaries | First value ≥ target, last value ≤ target, empty arrays, duplicates |
| Broadcasting | Why `[B,1] - [B]` produces `[B,B]`, not a per-example error |
| Task dependencies | Topological order, cycles, tasks that can start together; submitted is not completed |
| A two-state Markov chain | Multi-step probabilities, stationarity, and first hitting times are different questions |

The main algorithm route remains [Python and LeetCode](leetcode.en.md). The Markov exercise is a small probability problem here, not a revived Quant preparation section.

## Explain the mechanism, not just its name {#concepts}

| Topic | A useful check | Read next |
| --- | --- | --- |
| Initialization | Why should weight scale change when input width doubles? Which independence assumptions enter the derivation? | [Activations and initialization](../00-foundations/deep-dives/activation-and-initialization.en.md) |
| Position encodings | Where do absolute positions and RoPE enter the computation? What changes beyond a max-length setting? | [Position and context](../00-foundations/deep-dives/position-and-context.en.md) |
| PPO / GRPO | Where do advantages come from? What signal remains when every reward in a group is equal? | [After RLHF](../05-post-training/rlhf/after-rlhf.en.md), [sparse reward groups](../05-post-training/deep-rl/llm-bridge.en.md#sparse-groups), [credit assignment for multi-turn agents](../11-frontier/agentic-rl/credit-and-training.en.md) |
| DP / TP / PP | Are you splitting examples, within-layer computation, or layers? Where does communication happen? | [Distributed training](../06-systems/distributed-training.en.md), [training frameworks](../06-systems/training-frameworks.en.md) |
| Contrastive learning | How are positives and negatives built? Does changing batch size change the loss normalization? | [CLIP](../03-multimodal-learning/clip.en.md), [SigLIP](../03-multimodal-learning/siglip.en.md) |

These are review prompts, not questions reserved for particular companies. For vision, speech, robotics, or another field, substitute the relevant data, models, and evaluation problems. An LLM list is not a replacement for understanding the role.

## Inference: follow a request to the GPU and back {#inference}

Read the [vLLM / SGLang stack](../06-systems/inference-frameworks.en.md#engine-map), then trace one request through it. The point is not to recite class names. Explain who owns each state, what crosses an interface, and where you would measure a slowdown.

| Follow-up | Make the answer concrete |
| --- | --- |
| Who is this service for? | Input/output lengths, arrival rate, TTFT / ITL targets, quality, and budget |
| Who decides the next computation? | Token budgets, waiting queues, KV capacity, cancellation, preemption |
| What occupies GPU memory? | [Separate weights, KV, and workspace](../00-foundations/deep-dives/kv-cache-and-inference.en.md#inference-budget), rather than multiplying parameters by precision alone |
| What does a shared prefix reuse? | Compatible tokens and model settings, cache ownership, routing across replicas, tenant isolation |
| Why can throughput rise while the experience worsens? | [Long prefills competing with decode](../06-systems/llm-serving.en.md#token-budget); tail latency as well as tokens/s |
| GPU utilization is low: what next? | Separate CPU input, queues, execution, HBM bandwidth, communication, and output buffering; [trace first](../12-infra/performance.en.md#profiling), change one variable |
| When would you separate prefill and decode? | Weigh observed interference against KV transfer, both queues, and failure recovery |

For inference roles, prepare at least one layer down to the details: scheduling policy, KV-cache management, or performance evidence you can show. Do not drop traditional system design either; confirm the actual interview format.

## Projects, research discussions, and behavioral preparation {#projects}

Start with a table for one real project: problem and baseline, your change, alternatives, results, and unresolved issues. Separate your contribution from the team's work. Revisit older projects and papers where your contribution was smaller; check your records instead of improvising details in the interview. [Andrey's account](https://reyzharkov.com/blog/posts/interviews-2025-ml-research-engineer-uk) describes being questioned about old projects and paper details.

Then try three original follow-ups:

- **Did the method really improve results?** Fix data splits and evaluation, check leakage, report seed variation and compute/tuning budgets. One best run is not a stable gain. See [experiments and release](../practice/post-training/experiments-and-release.en.md).
- **What if the budget were halved?** Decide which hypothesis to test before reducing data, model size, or run count. Explain which conclusions are no longer supported. “Use LoRA” is not a general answer.
- **What if the result contradicted your prediction?** Separate implementation errors, measurement problems, and a failed hypothesis. Propose a follow-up that distinguishes them rather than simply doing more tuning.

For behavioral preparation, use real examples of disagreement, failure, and ambiguous requirements: what options you had, what you did, what happened, and what changed afterward. Not every story needs a happy ending. Scope depends on the role; a student project can show ownership without an invented cross-team leadership story.

## Free resources, with a purpose for each {#resources}

You need not complete all of these. They serve two purposes: identifying gaps and practicing implementations.

| Resource | What it is useful for |
| --- | --- |
| [Alisa: Notes on the Industry Job Search](https://alisawuffles.github.io/blog/job-search/) · 2026-06 | A PhD perspective that separates ML coding, technical discussion, and research discussion rather than focusing only on presenting papers |
| [Silvia: ML Job Interviews](https://silviasapora.github.io/blog/ml-interviews.html) · 2026-06 | An ML PhD's RS search, including implementation and debugging topics; her topic range is not everyone's required syllabus |
| [Andrey: ML/Research Engineer interviews 2025](https://reyzharkov.com/blog/posts/interviews-2025-ml-research-engineer-uk) · published 2026-03 | An experienced engineer's 2025 UK / remote search; publication date and interview dates differ |
| [Stanford CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics) | Implement the student interfaces and connect tests; a substantial assignment, not a solution dump. Follow course collaboration rules if enrolled |
| [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html) → [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) | Move from implementation to training and efficiency. Rewrite after closing the video rather than only following along |
| [TinyZero](https://github.com/Jiayi-Pan/TinyZero) → [verl](https://github.com/volcengine/verl) | An optional RL experiment. TinyZero says it is no longer actively maintained and directs new experiments to verl; its 0.5B observation concerns that model and setup, not a limit on all small models |

The authors have different backgrounds. We use their accounts to spot omissions, with our own exercises and organization—not reproduced company questions, compensation claims, or purported success thresholds. Advice from friends is supplementary too, not an internal company standard.

For one starting task, turn off completion, time an attention or MLP implementation, add tests, and explain the part you trust least. AI can still help with learning; this unaided pass checks what you can do independently.
