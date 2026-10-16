# Study plan: one small exercise at a time

[中文](study-plan.md) · **English**

Start somewhere you can follow, and pair reading with practice. This is a suggested learning sequence, not a promise that a certain number of days will get you an offer.

[Back to interview preparation](README.en.md) · [Looking for a focused review? Try Speedrun](speedrun.en.md)

## Find a starting point {#build}

If ML is new to you, begin with [one prediction in the starter route](speedrun-start.en.md#prediction). After working it out, ask: what is the input, what does the model return, and what does the loss compare?

- Lists and functions are unfamiliar: try the [small Python examples](speedrun-start.en.md#first-two), then come back.
- You can code, but haven't studied training: skip the syntax and connect predictions and loss to [one parameter update](speedrun-start.en.md#update).
- All of this is familiar: skip the rereading and try the [gap checks below](#refresh).

You're choosing a starting point for **this topic**, not a label for yourself. Someone with years of systems experience may still be new to model training.

## If you have a foundation, find the gaps first {#refresh}

Pick a small task relevant to your role and try it before looking up a solution. If you get stuck, decide whether the gap is the idea, writing the code, or checking the result, then return to the matching note.

| Try this | If you get stuck, start here |
| --- | --- |
| Compute an average loss, then change the batch partition | [Aggregation and denominators](code-comprehension.en.md#aggregation): what exactly are you averaging? |
| Write causal attention and label each tensor shape | [ML coding](../learn/ml-exercises/README.en.md): shapes, masks, and tests |
| Trace DFS and BFS on a small graph | [Traversal](algorithms/traversal.en.md): stacks, queues, and visited nodes |
| Trace an LLM request from the queue to a returned token | [Inference frameworks](../06-systems/inference-frameworks.en.md): scheduling, caching, and execution |
| Explain why you rejected an alternative in a project | [Project preparation](../career/prepare.en.md): constraints, your contribution, and evidence |

Skip what you can already do independently. If you can recall a conclusion but not explain it, try a runnable example. That often makes the missing step easier to see.

## Arrange a few small sessions {#sessions}

You could set aside **30–60 minutes of focused learning time** per session. That's a planning choice, not an estimate for finishing each unit. An unfamiliar idea may take several sessions. Breaks don't count, and missed sessions don't create work you owe.

Pick the dates yourself; there is no daily commitment. Session length can differ too: perhaps 25 minutes on Tuesday and 70 on Saturday. Your own calendar or notes app is enough to keep track, and empty days are for rest.

| Sequence | Read a little | Try it yourself | Leave something to return to |
| --- | --- | --- | --- |
| 1 · Inputs and predictions | [One prediction](speedrun-start.en.md#prediction) | Calculate predictions and loss for two samples, then check in Python | A runnable function with explicit inputs and expected outputs |
| 2 · Parameter updates | [One update](speedrun-start.en.md#update) | Change the learning rate, watch the loss, and explain an oversized step | Before-and-after values and one open question |
| 3 · A training loop | [Tensors and PyTorch](../00-foundations/pytorch/README.en.md) → [Training loop](../00-foundations/pytorch/training-loop.en.md) | Run forward, backward, and update on a small batch; distinguish train from eval | A repeatable script and one checked result |
| 4 · An algorithm pattern | [Hash maps](algorithms/hash-and-prefix.en.md) or [DFS / BFS](algorithms/traversal.en.md) | Implement it, add duplicate values or a cycle as appropriate, and explain complexity | An edge-case test and the conditions where the method applies |
| 5 · Your target area | [Role-based practice](ai-roles-prep.en.md) | Choose one branch below and one exercise in it | An implementation, experiment log, or design sketch |
| 6 · Put it together | [Code comprehension](code-comprehension.en.md) and [Project preparation](../career/prepare.en.md) | Change one requirement in code; explain a decision from a project | Two things you couldn't explain clearly; revisit one next time |

Units 1–3 help if ML is new to you. Algorithms can run alongside them; you don't have to finish the model material first. Split any unit across sessions, or skip it when you can already do the work.

## Then choose the branch you need {#roles}

What to practice first and what to take further in each direction is in the table of the [role-based guide](ai-roles-prep.en.md#routes). Below is one next exercise for each direction:

- **MLE / Research Engineer:** choose one [ML implementation exercise](../learn/ml-exercises/README.en.md), such as stable softmax or attention; start with small inputs, then check outputs, gradients, and edge cases. For LLM work, continue with [post-training practice](../practice/post-training/README.en.md); for recommendation and retrieval, try the [recommender walkthroughs](../practice/recommender-systems/README.en.md).
- **Inference / ML Systems:** follow [one inference request](../06-systems/inference-frameworks.en.md) through each module's inputs and outputs, then pick a constraint: limited memory, long queues, or long requests delaying short ones. Write down two options, what each changes, and what you would need to measure. In the [system-design exercises](../learn/system-design/README.en.md), define the workload and latency target before picking components.
- **General Coding / SWE:** build your plan around [Python and algorithm patterns](leetcode.en.md) and [reading code](code-comprehension.en.md); practice one method, then change a condition and check whether it still applies. This site is organized around ML and AI systems, not every kind of SWE role; use the job description and interview guidance to decide which language, backend, or systems topics to add.
- **Research Scientist:** choose a paper you worked on or read carefully. Explain its question, controls, and limits, then propose an experiment that distinguishes two possible explanations. [Evaluation foundations](../07-evaluation/README.en.md) can help you check whether a metric improvement supports your interpretation. Practice the relevant implementation too; this route does not replace expertise in your research field.

Whatever the direction, keep time for algorithms and code reading; architecture diagrams are only part of preparation.

## Pick a set of exercises by topic {#routes}

Four topics, two sets each, three exercises per set. “Start” suits a topic that is new to you; “Review” suits one you have studied and want to check you can do on your own. Times are rough estimates for one session, and you can split them.

### ML foundations & coding

**Start · From shapes to attention**: You know some Python but have not built a model yourself.

| Exercise | Read first | What to do | Time |
| --- | --- | --- | --- |
| Trace broadcasting | [Operations and shapes: valid code can compute the wrong thing](../00-foundations/pytorch/operations-and-shapes.en.md) | Predict the shape of a broadcast addition, then check three small arrays. | 25 min |
| Trace one attention pass | [Multi-head attention: from equations to implementation](../00-foundations/core/multi-head-attention.en.md) | Label Q, K, V and output shapes for a few tokens before building a full model. | 40 min |
| Build a small kNN | [Five short exercises: calculate, implement, check](coding-checks.en.md) | Start with loop-based distances, then vectorize. Test duplicate points and ties. | 30 min |

**Review · Implement, debug, test**: You know the concepts and want to test your implementation skills.

| Exercise | Read first | What to do | Time |
| --- | --- | --- | --- |
| Implement attention from memory | [Multi-head attention: from equations to implementation](../00-foundations/core/multi-head-attention.en.md) | Track shapes, add a causal mask, and test that future tokens cannot affect earlier outputs. | 45 min |
| Catch a silent shape bug | [Five short exercises: calculate, implement, check](coding-checks.en.md) | Compare subtraction with [B,1] and [B] targets. Write a test that catches the mismatch. | 25 min |
| Fix an evaluation aggregation | [Code comprehension: understand, change, and verify](code-comprehension.en.md) | Trace inputs and outputs, then check the aggregate result across batch sizes. | 35 min |

### LLM post-training

**Start · Understand the training signal**: You know next-token prediction and want to understand post-training.

| Exercise | Read first | What to do | Time |
| --- | --- | --- | --- |
| Label one conversation | [SFT: how a conversation becomes a training signal](../05-post-training/sft-and-its-ceiling.en.md) | Write user and assistant tokens, shifted labels, and the positions included in the loss. | 30 min |
| Compare three updates | [Q-learning, PPO, and DPO: what is each learning?](../05-post-training/deep-rl/methods-map.en.md) | For Q-learning, PPO and DPO, identify the required data and the quantity being updated. | 30 min |
| Inspect a small training set | [Why post-training keeps coming back to data](../01-data-and-feedback/data-as-supervision.en.md) | Inspect a few examples: what behavior do they reward, and what remains untested? | 25 min |

**Review · From objectives to training checks**: You have trained models and want to connect data, objectives and implementation.

| Exercise | Read first | What to do | Time |
| --- | --- | --- | --- |
| Audit masks and shifts | [SFT: how a conversation becomes a training signal](../05-post-training/sft-and-its-ceiling.en.md) | Use one multi-turn conversation to verify loss masks, end tokens and label shifts. | 30 min |
| Distinguish reference and old policy | [Q-learning, PPO, and DPO: what is each learning?](../05-post-training/deep-rl/methods-map.en.md) | Explain each role, when it updates, and how confusing them changes the objective. | 35 min |
| Choose a training setup | [Choosing a training stack: DDP, FSDP2, and the surrounding tools](../06-systems/training-frameworks.en.md) | Given a memory budget and GPU count, compare DDP and FSDP and list correctness checks. | 40 min |

### Inference & ML systems

**Start · Follow a generation request**: You can call a model API and want to understand the service behind it.

| Exercise | Read first | What to do | Time |
| --- | --- | --- | --- |
| Draw prefill and decode | [KV cache: what do we stop computing, and start storing?](../00-foundations/deep-dives/kv-cache-and-inference.en.md) | Use a short prompt to draw the first pass and the cached state used for later tokens. | 30 min |
| Trace a generation request | [vLLM and SGLang: what a generation engine actually does](../06-systems/inference-frameworks.en.md) | Trace the entry point, scheduler, KV cache and model execution; give each layer one job. | 35 min |
| Separate training from inference | [Choosing a training stack: DDP, FSDP2, and the surrounding tools](../06-systems/training-frameworks.en.md) | Explain what DDP, FSDP and vLLM solve without treating them as interchangeable layers. | 30 min |

**Review · From architecture to bottlenecks**: You have deployment experience and want to practice system tradeoffs.

| Exercise | Read first | What to do | Time |
| --- | --- | --- | --- |
| Break down vLLM and SGLang | [vLLM and SGLang: what a generation engine actually does](../06-systems/inference-frameworks.en.md) | Compare scheduling and caching for the same request, then explain one difference in depth. | 40 min |
| Estimate KV cache memory | [KV cache: what do we stop computing, and start storing?](../00-foundations/deep-dives/kv-cache-and-inference.en.md) | State layers, KV heads, length, batch and dtype; check how GQA changes the estimate. | 30 min |
| Design a useful load test | [vLLM and SGLang: what a generation engine actually does](../06-systems/inference-frameworks.en.md) | Fix the length distribution, separate TTFT, decode latency and throughput, and plan bottleneck checks. | 35 min |

### Algorithms & data structures

**Start · Learn reusable patterns**: You are starting coding practice and want a few dependable patterns.

| Exercise | Read first | What to do | Time |
| --- | --- | --- | --- |
| Choose what a hash map stores | [Hash maps and prefix sums: what should you remember?](algorithms/hash-and-prefix.en.md) | Explain why Two Sum looks up before inserting, then test repeated values. | 25 min |
| Write binary-search boundaries | [Pointers, windows, and binary search: justify the move](algorithms/pointers-and-search.en.md) | Find the first value ≥ target; test empty input, duplicates and missing targets. | 30 min |
| Run DFS and BFS on one graph | [DFS / BFS: recursive or iterative?](algorithms/traversal.en.md) | Trace the stack and queue and explain when nodes become visited. | 30 min |

**Review · Change the conditions and retest**: You know common patterns and want to test boundaries, complexity and transfer.

| Exercise | Read first | What to do | Time |
| --- | --- | --- | --- |
| Count subarrays with prefix sums | [Hash maps and prefix sums: what should you remember?](algorithms/hash-and-prefix.en.md) | Define the map invariant; test negatives, zero and repeated prefixes. Explain why not a sliding window. | 30 min |
| Turn recursive DFS into iteration | [DFS / BFS: recursive or iterative?](algorithms/traversal.en.md) | Compare traversal order, saved state and space, then test on a cyclic graph. | 30 min |
| Order tasks with dependencies | [Five short exercises: calculate, implement, check](coding-checks.en.md) | Produce a topological order, detect cycles, and distinguish dependency layers from scheduling. | 35 min |

## Make it easy to return {#resume}

Leave three lines before stopping: **what you tried, what is still unclear, and what you will try next**. For example:

> The causal mask works, but I haven't tested padding. Next time, use two sequences of different lengths and write the expected mask by hand first.

Keep those lines with your code or in your own notes. To come back to a note later, choose “Save this page” under its title. “My reading” at the top right lists saved notes and what you read recently, and a note reopens where you stopped.

After a longer break, run your last example and decide whether to continue or revisit it.

## Keep a backup for your next device {#backups}

Saved notes, reading history, and reading positions live in this browser. Before switching computers or browsers, or clearing site data, export a copy:

1. Open “My reading” at the top right, go to “Settings,” and choose “Export” in the “Backup” row. This downloads one JSON file.
2. On the other device, open the same place, choose “Import,” and pick the file.
3. The two sets of records are merged. Where both have the same record, the newer one wins; nothing you already have is deleted.

- **No fork or clone required.** Records are separate from the site source, so normal site updates or `git pull` do not touch them. A different domain, protocol, or local port has separate browser storage, so import your backup there.
- **Browser storage is not a permanent backup.** Clearing site data, closing a private window, or browser storage cleanup can remove records. Keep the exported file somewhere else, perhaps in your own iCloud / Drive folder. This site does not connect to cloud storage or sync devices.
- **Imports are local.** The file is read in your browser, not uploaded. The light/dark theme is not included. Recent reading means this site's pages, not your browser history.
- **Older backups still import.** Files exported from the old study dashboard and older saved-notes files both work. Plans and memos in an old file are kept in your backup, though the site no longer shows them.
- **Renamed chapters keep their records.** Records match on a fixed ID, not the title, order, or language. Records without a matching page are kept, not deleted.

[Back to interview preparation](README.en.md) · [Practice one skill](README.en.md#prep-routes) · [Full library](../learn/README.en.md)
