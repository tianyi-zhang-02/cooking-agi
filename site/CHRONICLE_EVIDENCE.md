# Chronicle evidence ledger

Checked: **2026-10-12**. Scope: six infrastructure additions, Sojourner, and four earlier speech / vision / search / representation milestones. This is not a fresh audit of every existing node or external link. The data-level `reviewed` date records this bounded update, not complete coverage through that date.

## Selection and evidence rules

- Add missing questions, not recent-paper rankings: data independence; reusable compilation; protected remote data transfer; light-load power; CPU/memory performance bounds; durable distributed storage. Sojourner adds a non-learning autonomy example.
- `historical` describes a dated paper, report, specification, or mission. It does not claim present-day deployment, obsolescence, universal adoption, or optimality.
- A paper and its mirror are one intellectual source. An author's bibliography or institutional catalog can cross-check dates but is not an independent replication. Later system papers establish their own reported use, not an inevitable historical lineage.
- Each added node has multiple useful sources. The entries below distinguish original evidence, date checks, independently authored follow-up, and same-institution evidence. No new `connections` are inferred from chronology; the existing connections remain unchanged.
- Chinese and English node text are newly written summaries. Source titles identify works, not copied explanatory prose. Later AI examples stay explicitly later, rather than being attributed to the earlier publication.

## 1. Relational model — `databases-relational-model-1970`

Checked: **2026-10-12**. Placement: `early-ai / field-distributed`; topic `distributed`. This is the existing software/systems grouping, not a claim that the relational model is inherently distributed.

**Claims retained**

1. Codd's CACM article appeared in **June 1970** and described data through relations while addressing dependence on ordering, indexes, and access paths.
2. Data independence separates application logic from changes in storage representation; the article is not a SQL or product release.
3. **MADlib (2012)** is a concrete later ML/database intersection: regression and clustering implementations inside PostgreSQL and Greenplum.

**Evidence and roles**

- [Codd, original article](https://www.seas.upenn.edu/~zives/03f/cis550/codd.pdf): read the first-page issue date, §§1.1–1.2 and the relation discussion. This is original text hosted by a university, not a separate Penn study. The introduction explicitly excludes implementation details from its scope.
- [IBM Research publication record](https://research.ibm.com/publications/a-relational-model-of-data-for-large-shared-data-banks): institutional record gives June 1, 1970; the node conservatively uses the issue month/year. This checks metadata against the scan, not independent adoption.
- [MADlib, PVLDB 2012](https://dsf.berkeley.edu/cs286/papers/madlib-vldb2012.pdf): independently authored primary system report; §§3–4, especially §§4.2–4.3, document in-database logistic regression and k-means. It supports the later implementation example, not a direct causal edge from Codd to MADlib.

**Uncertainty / exclusions:** no claim that relational databases originated all AI data pipelines, that the model guarantees good data, or that all implementations share SQL, transaction, or physical execution semantics. The ACM DOI redirect failed during this check; technical claims were checked in the readable original scan instead.

## 2. LLVM — `software-llvm-paper-2004`

Checked: **2026-10-12**. Placement: `connected-computing / field-software`; topic `software`.

**Claims retained**

1. The **CGO 2004** paper describes a typed, low-level SSA representation and a framework supporting analysis and transformations across compilation stages.
2. This is not LLVM's project-start date: its official archive records **LLVM 1.0 on November 18, 2003**.
3. **TVM (OSDI 2018)** includes an LLVM IR code-generation path, while treating tensor graphs, operator transformations, and scheduling as additional compiler work.

**Evidence and roles**

- [LLVM paper](https://llvm.org/pubs/2004-01-30-CGO-LLVM.pdf): original technical account, especially §§2–3 and conclusion.
- [LLVM publication record](https://llvm.org/pubs/2004-01-30-CGO-LLVM.html): lists March 2004 at CGO and notes an earlier technical report. The `2004-01-30` URL component is not treated as the conference date.
- [LLVM historical announcements](https://www.llvm.org/OldNews.html): separate project record for the 2003 release boundary, not independent research corroboration.
- [TVM paper](https://www.usenix.org/system/files/osdi18-chen.pdf): independently authored primary implementation evidence, Figure 2 and §§3–4. Its graph/low-level-IR distinction prevents attributing a complete neural-network compiler to the 2004 framework.

**Uncertainty / exclusions:** no assertion that LLVM invented intermediate representations, that all AI compilers use it, or that its presence guarantees a speedup. The sources do not establish a LLVM-to-CUDA invention chain; their shared era branch is a reading arrangement only.

## 3. RDMAP — `networks-rdmap-rfc-2007`

Checked: **2026-10-12**. Placement: `connected-computing / field-networks`; topic `networks`.

**Claims retained**

1. **RFC 5040 (October 2007)** specifies protected access to advertised remote buffers, layered over DDP, with reducing copies and CPU overhead as design goals.
2. This dates a particular RDMA protocol specification, not RDMA's invention or the origin of InfiniBand/RoCE.
3. **TensorFlow (OSDI 2016), §5**, reports both gRPC/TCP and RoCE for inter-task communication. This establishes an ML use of the broader RDMA family, **not use of RFC 5040 RDMAP**.

**Evidence and roles**

- [RFC 5040](https://www.rfc-editor.org/rfc/rfc5040.html): primary normative document; header and §§1–3 establish date, protection, buffer advertisement, and layering. Statements of design goals are not benchmark measurements.
- [IETF document history](https://datatracker.ietf.org/doc/rfc5040/history/): records RFC publication on October 31, 2007, followed by a November 5 workflow-state update. These are different administrative events; the node retains the RFC's October publication month. Same standards process, not independent corroboration.
- [TensorFlow system paper](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf): separately authored primary ML implementation report; §5, printed p. 275, names the communication protocols. It does not identify the deployment as RFC 5040.

**Uncertainty / exclusions:** no RDMAP-to-TensorFlow dependency edge, claim of zero communication cost, or assertion that current TensorFlow versions retain the same transports. InfiniBand, RoCE, switched fabrics, and PCIe still need their own histories.

## 4. Energy proportionality — `energy-proportional-computing-2007`

Checked: **2026-10-12**. Placement: `connected-computing / field-energy`; topic `energy`.

**Claims retained**

1. Barroso and Hölzle's **December 2007** article argues that power should fall with workload, rather than evaluating only full-load efficiency; memory and disk power matter alongside CPUs.
2. **PowerNap (2009)** investigates a different approach to reducing idle power, rather than establishing that energy proportionality was already achieved everywhere.
3. The **2017 TPU report**, §§5–6, distinguishes power including the host from incremental accelerator power and examines changes with offered load. It supplies an AI-specific example, not current-device measurements.

**Evidence and roles**

- [Original IEEE Computer article](https://storage.googleapis.com/gweb-sustainability.appspot.com/pdf/33387.pdf): printed pp. 33–37; publication month is visible in the original page headers. Figures 2–4 and accompanying text distinguish existing server behavior from a proposed design goal.
- [Google Research publication record](https://research.google/pubs/the-case-for-energy-proportional-computing/): confirms authors, venue, and 2007; same institution, not independent verification.
- [PowerNap, ASPLOS 2009](https://web.eecs.umich.edu/~twenisch/papers/asplos09.pdf): independently authored Michigan/CMU primary research, abstract and §1. It investigates short idle intervals and active/idle transitions. Its reference [1] says **January 2007**, conflicting with the original's **December 2007** header; use the original for the month. It corroborates the year and problem, not the incorrect month or a universal solution. A first fetch failed; a retry returned readable text.
- [TPU report, fixed April 2017 v1](https://arxiv.org/pdf/1704.04760v1): §§5–6 and reference [Bar07]. This is later Google-authored implementation/measurement evidence, not institutionally independent replication of Google's 2007 observations.

**Uncertainty / exclusions:** no historic power percentage or speedup is generalized to today's fleet. Energy proportionality, full-load efficiency, facility provisioning, and electricity consumed over time are distinct questions. The shared era branch with warehouse-scale computing is not an asserted causal genealogy.

## 5. Roofline — `chips-roofline-report-2008`

Checked: **2026-10-12**. Placement: `connected-computing / field-chips`; topic `chips`. This fills a CPU/memory performance-analysis gap, not the still-missing CPU microarchitecture history.

**Claims retained**

1. **UCB/EECS-2008-134 is dated October 17, 2008**. The subsequent CACM article appeared in **April 2009**; the node uses the public report date rather than silently treating journal publication as the beginning.
2. The basic model bounds throughput by the smaller of peak computation and memory bandwidth times operations per DRAM byte. It distinguishes computation limits from memory-supply limits on multicore systems.
3. The **2017 TPU report, §4**, explicitly adapts Roofline to neural-network workloads and compares CPU, GPU, and TPU behavior.

**Evidence and roles**

- [Original Berkeley report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/Archive/EECS-2008-134.pdf): cover dates the report; §§2–3 give the intensity definition, basic bound, and additional ceilings. The model is an explanatory bound, not an exact runtime simulator.
- [Berkeley report catalog](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html): matching date and report number; an archive cross-check of the same work, not a second study.
- [Berkeley Lab publication list](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/ppan-roofline-publications/): inspected the 2008 and 2009 entries. It distinguishes the earlier report/presentation from the April 2009 journal publication; same research lineage, not independent corroboration.
- [TPU report, fixed April 2017 v1](https://arxiv.org/pdf/1704.04760v1): §4, Figures 5–7, and reference [Wil09] document the adaptation. David Patterson is an author of both works, so this is follow-on primary evidence, **not an independent replication**.

**Uncertainty / exclusions:** no claim of first-ever performance modeling or multicore invention. The report itself mentions earlier submission/revision dates; none is used as a product launch. No independent reproduction of the original performance measurements was established in this bounded review. The era grouping with AMD GCN does not imply GCN was caused by Roofline.

## 6. HDFS — `storage-hdfs-paper-2010`

Checked: **2026-10-12**. Placement: `connected-computing / field-memory`; topic `memory`, whose existing title includes storage.

**Claims retained**

1. The **MSST 2010** paper describes NameNode metadata, DataNode block storage, replication for failure tolerance, and storage/computation colocation.
2. This is a paper date, not HDFS's project start. The described file I/O model is single-writer/multiple-reader and supports append rather than arbitrary rewriting of existing bytes.
3. **Spark's RDD paper (NSDI 2012), §6.1**, reads initial input from HDFS in logistic-regression and k-means experiments, then benefits from in-memory reuse between iterations.

**Evidence and roles**

- [MSST original paper](https://msstconference.org/MSST-history/2010/Papers/MSST/Shvachko.pdf): original conference-hosted document, 2010 copyright, §§I–III. This distinguishes filesystem persistence/recovery from computation scheduling and database transaction semantics.
- [Shvachko publication list](https://shvachko.github.io/Publications.html): the author's separate bibliography identifies MSST and **May 2010**; it is a date cross-check, not independent adoption evidence.
- [RDDs, NSDI 2012](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf): independently authored primary ML systems evidence; §6 and §6.1 identify storage, workloads, and first versus subsequent iterations. This is an observed use in the reported experiments, not evidence that all training uses HDFS.

**Uncertainty / exclusions:** no universal throughput claim; no equation of replicated files with GPU memory, model caches, or database transactions. The note does not date Flash/SSD, object storage, or checkpoint-system origins; those remain open coverage gaps.

## 7. Sojourner — `applications-sojourner-1997`

Checked: **2026-10-12**. Added following the explicit supplemental request. Placement: `learning / field-applications`; topic `applications`. The existing era ID is chronological; placement there does not label the rover as a learning system. The bilingual node is intentionally short, without a separate `detail` expansion.

**Claims retained**

1. Sojourner arrived with Pathfinder on **July 4, 1997**.
2. Its onboard navigation used commanded goals plus terrain and position sensing, integrating limited onboard computation with mobility and communications.
3. These sources describe navigation behaviors and ground support, **not reinforcement-learning training**. Autonomy is not synonymous with RL.

**Evidence and roles**

- [JPL Robotics subsystem account](https://www-robotics.jpl.nasa.gov/what-we-do/flight-projects/pathfinder/software-electronics/): engineering-team account describes simple navigation behaviors, sensors, processor constraints, and operator corrections.
- [NASA mission history](https://www.nasa.gov/history/25-years-of-continuous-robotic-mars-exploration-from-pathfinder-to-perseverance/): retrospective dated September 13, 2022; its Pathfinder section dates the landing to July 4, 1997. The article publication year is not the event year.
- [JPL archived rover description](https://planetarydata.jpl.nasa.gov/img/data/mpfr-m-rvreng-2_3-edr_rdr-v1.0/mprv_0001/document/insthost.htm): mission archive, “Instrument Host Overview” and “Control and Navigation Subsystem,” independently checks the textual landing date within the same institutional lineage and describes navigation/hazard-avoidance experiments.

**Uncertainty / exclusions:** all three sources are NASA/JPL accounts, not institutionally independent adoption evidence. Landing is not the same event as driving off the lander. No first-rover claim, self-learning claim, or lineage to modern robot foundation models is added.

## Compatibility and validation scope

- Preserve version 3, all pre-existing node/era/topic/branch IDs, the original node contents, and the four existing evidence-backed connections.
- Each new node has one canonical, chronologically ordered era branch, bilingual required fields, an existing topic, and HTTPS sources. Relevant coverage descriptions acknowledge additions without declaring their broader histories complete.
- `site/tests/test_technology_tree.py` has no fixed total-node expectation to update and was left byte-for-byte unchanged.
- Data validation passed with **73 nodes**. A baseline comparison confirmed that all **66 original node objects**, existing canonical placements, featured selections, layers, and four connections were preserved. Each new event appears exactly once in each language's rendered markup.
- **14 focused tests passed**, using Python 3.12.2 and the existing `TechnologyTreeTests`: evidence/schema, world-tree fields, era summaries, optional explanations/escaping, invalid taxonomy/connections, branch targets/determinism, sparse-domain counts, exhibit data, exhibit behavior, reading routes, server-rendered stable IDs, date/coverage/source rejection, atlas behavior, and branch-navigation behavior. The behavior tests invoke the existing Node.js harnesses. No full site build was run.
- The first focused run caught an ASCII possessive apostrophe in new coverage text conflicting with a test's literal HTML assertion. Rephrasing that sentence fixed the failure without changing renderer behavior or tests. The system Python 3.9 was not used; Python 3.13 lacked `markdown`, so the existing Python 3.12 environment supplied the test dependencies without installation.
- Only the JSON data and this ledger were edited. No renderer/JS/CSS/dashboard/global checklist changes, commits, or pushes. Structural tests do not establish historical truth or reproduce published experiments.

## Follow-up: histories before current language models

The earlier validation log above describes the 73-node infrastructure update. This follow-up adds four nodes for **77 total**, while separately replacing the decorative tree with a navigable canvas. New entries have their own bilingual mechanisms and limits; no causal edges were inferred.

| Entry | Evidence actually checked | Boundary retained |
|---|---|---|
| `hmm-rabiner-tutorial-1989` | [Rabiner's original tutorial](https://web.mit.edu/6.435/www/Rabiner89.pdf), issue header, introduction, definitions and three-problem discussion; [CMU's 1994 software archive](https://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/speech/systems/hmm/0.html) supplies a separately authored implementation record | The tutorial itself says the theory and speech applications predate 1989. The software archive is evidence of an implementation, not a replication of all reported recognition results. |
| `lenet5-1998` | [Original document-recognition paper](https://leon.bottou.org/publications/pdf/ieee-1998.pdf), LeNet-5 architecture diagram and convolution/subsampling description; [coauthor's bibliography](https://bottou.org/papers/lecun-98h) confirms publication metadata | The bibliography is not an independent study. The PDF text extraction is partly garbled; no accuracy or speed number was carried over. Do not call 1998 the invention of CNNs or describe LeNet-5 using a modern max-pooling/softmax recipe. |
| `google-search-prototype-1998` | [Brin and Page's paper](https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf), §§2.1–2.2 and §4 architecture; [Google Research record](https://research.google/pubs/the-anatomy-of-a-large-scale-hypertextual-web-search-engine/) for year and venue | The catalog describes the same work, not independent validation. The node explains the prototype and does not repeat its subjective search-quality comparison or claim to describe current production Google. |
| `word2vec-2013` | [January 2013 v1](https://arxiv.org/pdf/1301.3781v1), §§3.1–3.2; [arXiv version history](https://arxiv.org/abs/1301.3781) for the January / March / September revision boundary | CBOW and Skip-gram are explained from the initial version. No claim that embeddings originated here, no later negative-sampling result backdated to January, and no semantic-understanding claim inferred from analogy examples. |

These are original short explanations, not reproductions of the sources. The additions repair a few gaps in the reading route; they do not complete speech, search, CNN, or representation-learning history. Canonical placements, topic summaries and gap statements were updated together.

## Rewrite: five representative milestones, era summaries, and the Jianmu framing (Claude Code, 2026-10-12)

Scope: the bodies, explanations, limits and sources of `alexnet`, `transformer`, `alphago`, `nvidia-cuda-2006` and `software-linux-kernel-1991`; all nine era summaries; one new connection; two `root` settings. Era summaries only restate facts carried by nodes in the same era. Each rewritten node answers, in prose rather than fixed headings: what people did before, where it got stuck, what changed, a small example, what came after, and what remains open.

| Entry | Evidence actually read | Claims retained | Boundary |
| --- | --- | --- | --- |
| `alexnet` | NeurIPS 2012 PDF: §1, §3.1 (ReLU vs tanh, Figure 1), §3.2 (two GTX 580 3GB), §4.1–4.2 (augmentation, dropout), §6 (Tables 1–2), §7; ResNet arXiv v1 abstract; cuDNN arXiv v1 abstract | Earlier pipelines used SIFT + Fisher vectors; 60M parameters, five convolutional + three fully connected layers; five to six days on two GPUs; ReLU reached 25% CIFAR-10 training error six times faster in a four-layer net; **15.3% is a seven-network average including two pretrained on Fall 2011 ImageNet; a single network scored 18.2% (validation)**; removing a middle convolutional layer cost about 2% top-1; ResNet reached 152 layers and 3.57% ensemble test error | Not the first CNN (LeNet-5 node). The authors' own expectation of gains from faster GPUs is reported as theirs. |
| `transformer` | arXiv v1 PDF (2017-06-12): §1–2, §3.1 masking, §4 and Table 1, §5.1–5.2, §7, Figures 3–5; NIPS 2017 proceedings page; BERT arXiv v1 §3 | Recurrence precludes parallelism within a training example; ConvS2S / ByteNet path lengths grow linearly / logarithmically; self-attention is O(n²·d) per layer with O(1) sequential steps and is cheaper than recurrence when n < d; restricted attention left as future work; WMT14 En–De 28.4 BLEU, more than 2 BLEU above previous results including ensembles; big model 3.5 days on 8 P100s; attention to “making … more difficult”; heads appear to do anaphora; BERT is an encoder and GPT a left-to-right Transformer | The v1 En–Fr score differs from later revisions, so the node does not quote it. Decoding stays sequential. |
| `alphago` | DeepMind-hosted Nature PDF: abstract, introduction (b ≈ 250, d ≈ 150; chess b ≈ 35, d ≈ 80), SL policy (30M KGS positions, 57.0% vs 44.4%), RL policy vs Pachi (85%), value network (correlated positions, 30M self-play positions, 15,000× less computation), Methods (Fan Hui, 5–9 October 2015, 5–0; distributed 1,202 CPUs / 176 GPUs; received 11 Nov 2015, accepted 5 Jan 2016); DeepMind AlphaGo page (Seoul, March 2016, 4–1); DeepMind AlphaGo Zero post (2017-10-18; three days of self-play; 100–0 against the version that beat Lee Sedol) | As listed | AlphaZero and MuZero are named only as successors; no claim about what each changed. |
| `nvidia-cuda-2006` | Brook for GPUs (Stanford PDF; listed as SIGGRAPH 2004): §1–2 on packing data into textures and launching kernels by rendering geometry; GeForce 8800 brief pp. 12–14 (C interface, Parallel Data Cache, drivers for Linux or Windows); CUDA Programming Guide 1.0 (2007-06-23): §1.1 (graphics-API-only programming, no scatter, bandwidth bottleneck), §1.2 (C extension, CUBLAS/CUFFT, supported products), Appendix A (16 KB shared memory per multiprocessor); cuDNN arXiv v1 abstract (no BLAS-like library; 36% faster in Caffe, less memory) | As listed. The vector-addition comparison is an illustration written for the site, not an experiment from a source. | The brief's “100× faster” is labelled as marketing. No claim that later frameworks all depend on CUDA. |
| `software-linux-kernel-1991` | kernel.org RELNOTES-0.01 (free Minix-like kernel for 386; needs Minix to bootstrap; shell, compilers and library not included; no fee-based distribution; “get it working fast”); RELNOTES-0.12 (license changes to the GNU copyleft, removing the no-fee condition, pending approval of people who helped write code); GNU “Linux and the GNU System” (by the early 1990s the whole system except the kernel; Torvalds freed Linux in 1992; asks readers to say GNU/Linux); MapReduce OSDI 2004 §3 (dual-processor x86 machines running Linux); GeForce 8800 brief p. 14 | As listed | The GNU article is cited as the GNU project's own position. The 0.01 month and the comp.os.minix announcement were not verified in this pass (Google Groups refused automated access), so neither is stated. |

New connection: `distributed-mapreduce-paper-2004` → `software-linux-kernel-1991` (`uses`), source = the MapReduce paper already listed on the origin node, §3.

`root` settings: `networks` from `networks-web-proposal-1989` and `software` from `software-linux-kernel-1991`. These are an editorial reading that the Web/internet and free and open-source software grew their own communities and industries; they are not historical claims about when those fields began. The owner may change or remove them.

Jianmu (design reference only; after owner feedback the page no longer quotes it, and only the nine roots remain in the drawing): Chinese Wikisource 《山海經·海內經》 reads “名曰建木，百仭無枝，上有九欘…，下有九枸”, and 《淮南子·墬形訓》 reads “建木在都廣，眾帝所自上下，日中無景，呼而無響，蓋天地之中也。” The page quotes the first in simplified characters with an ellipsis and paraphrases the second. ctext.org refused automated access and was not used.

Validation for this pass: see the pull request description. Structural tests check IDs, translations, links and geometry, not historical truth.

## Expansion to 143 nodes (2026-10-12)

Sixty-six nodes were added across all eleven fields, with a `track` (direction within a field) on every node. The per-node evidence for that batch is in [CHRONICLE_EVIDENCE_EXPANSION_2026-10-12.md](CHRONICLE_EVIDENCE_EXPANSION_2026-10-12.md).

## 2024–2026: sixteen nodes and the last period extended to 2026 (Claude Code, 2026-10-17)

The tree had three nodes after 2023. This pass adds sixteen, extends the last period (`reasoning`) to 2026 under the title "Thinking longer, starting to act" and gives it new branches: context and efficiency, agents and tools, compute and power, science, open weights, world models, staged release. Every fact below was read in the listed source during this pass. Several sources are the organisation's own announcement; those numbers stay attributed to it.

| Entry | Evidence actually read | Claims retained | Boundary |
| --- | --- | --- | --- |
| `o1-2024` | OpenAI, Learning to reason with LLMs (2024-09-12) | Large-scale RL for a long internal chain of thought; performance rises with train-time RL and test-time thinking; AIME 2024: GPT-4o 12%, o1 74% single sample | Not preferred on some writing tasks; users see a summary of the reasoning. The IOI-tuned model's results are a different model and are left out |
| `gemini-1-5-2024` | Google blog (2024-02-15); technical report arXiv v1 (2024-03-08) | Gemini 1.0 at 32K; 1.5 Pro MoE; 1M-token context in private preview, 10M in research; about an hour of video; near-perfect single-needle recall | The blog's "1 million tokens in production" conflicts with the 128K standard / 1M preview framing; the latter is used. With 100 needles recall is just over 60% |
| `nvidia-gb200-nvl72-2024` | NVIDIA newsroom (2024-03-18); GB200 NVL72 product page | 72 Blackwell GPUs in one liquid-cooled rack; fifth-generation NVLink, 1.8 TB/s bidirectional per GPU; 4-bit floating-point inference | The 30× inference and 25× cost claims are vendor estimates for set lengths and latency targets and are not used as results |
| `alphafold3-2024` | Nature paper (2024-05-08): pairformer, diffusion module, limitations; Google blog (same day) | Joint structure of complexes with proteins, DNA, RNA, ligands; diffusion from noised atom coordinates; at least 50% better on protein interactions with other molecule types (blog) | Static structures, not dynamics; hallucination in disordered regions (paper's own limitations) |
| `llama-3-1-2024` | Meta blog (2024-07-23); Llama 3.1 Community License | Weights in three sizes, largest 405B; over 15T training tokens; FP8 quantisation to fit one server | Called "open source" by Meta; the licence has a 700M-MAU clause and naming rules, so the node says open weights |
| `nobel-ai-2024` | nobelprize.org physics (2024-10-08) and chemistry (2024-10-09) press releases | Citations as published; AlphaFold2 used by more than two million people in 190 countries | The limit is an editorial line, not a fact from the releases |
| `mcp-2024` | Anthropic announcement (2024-11-25) | Servers and clients; spec, SDKs and six pre-built servers at launch | Local servers only at launch; adoption after launch is not claimed |
| `deepseek-v3-2024` | Technical report arXiv v1 (2024-12-27) | 671B total / 37B active; first validation of FP8 training at this scale; 2.788M H800 GPU hours for the official run | Excludes earlier research and ablations; large deployment unit |
| `iea-energy-ai-2025` | IEA executive summary and press release (2025-04-10) | ~415 TWh in 2024 (~1.5%); ~945 TWh by 2030; 2035 range 700–1,700 TWh | The 945 TWh sentence does not itself name a scenario, so the node does not either |
| `coding-agents-2025` | Anthropic, Claude 3.7 Sonnet and Claude Code (2025-02-24) and Claude 4 (2025-05-22); OpenAI, Introducing Codex (2025-05-16) | Terminal agent that searches, edits, tests and commits; cloud sandbox per task, parallel, 1–30 minutes, logs and test output attached | Human review of every change; tool-call reliability and long-running commands named as open problems |
| `imo-2025-gold` | Google DeepMind post (2025-07-21); Alexander Wei's thread for OpenAI (2025-07-19) and the published proofs | 2024 system needed Lean translation and 2–3 days; 2025: natural language, time limit, 35/42; DeepMind certified by IMO coordinators, OpenAI graded by three former medalists | The IMO did not validate either system; no official OpenAI blog post was found, so the thread is cited as the researcher's statement |
| `genie-3-2025` | Google DeepMind post (2025-08-05) | Frame-by-frame, real time at 24 fps, 720p; consistent for a few minutes; promptable world events | Limited actions; multi-agent interaction and hours-long sessions out of reach; limited research preview |
| `ai-research-math-2026` | OpenAI post (2026-05-20); arXiv 2605.22763 v1 (2026-05-21) | A general-purpose reasoning model disproved a longstanding conjecture on the planar unit-distance problem; proof checked by external mathematicians; LLM-written Lean proofs resolved 9 of 353 open Erdős problems | The preprint page does not name the lab or system, so the node does not; LLM proofs need checking; hit rate ~2.5% |
| `deepseek-v4-2026` | DeepSeek V4 Preview Release (2026-04-24); V4-Pro model card | At 1M tokens V4-Pro needs 27% of V3.2's per-token FLOPs and 10% of its KV cache; 1M context default in DeepSeek services; MIT licence | Preview; all scores self-reported |
| `tpu-8-2026` | Google post (2026-04-22) | TPU 8t for training, TPU 8i for low-latency inference with more memory bandwidth; 9,600-chip 8t superpod; up to 2× performance per watt over Ironwood | Not generally available at announcement; "up to" figures |
| `cyber-release-2026` | Anthropic Project Glasswing (2026-04-07); OpenAI GPT-6 Astra safety overview (2026-09-03); Google Gemini 4 Argon post (2026-09-30) | Mythos Preview not generally available, access for organisations maintaining critical software; Astra the first OpenAI model at the Critical cyber level, broadly released with strengthened protections; Argon rolled out first to trusted defenders | Capabilities and vulnerability counts are the labs' own. A research summary described Astra as restricted to defenders; the safety overview says it was broadly deployed, so the node follows the overview |

New connections (each sourced from its origin node): `deepseek-r1` → `deepseek-v3-2024` (R1 v1: DeepSeek-V3-Base as the base model), `alphafold3-2024` → `alphafold2` and → `diffusion` (Nature: pairformer and diffusion module replace the AF2 evoformer and structure module), `nobel-ai-2024` → `alphafold2` (chemistry press release), `nvidia-gb200-nvl72-2024` → `networks-nvlink-p100-2016` (product page).

Considered and left out for now: International AI Safety Report 2026, Samsung HBM4, NVIDIA Vera Rubin, IEA's April 2026 update, and a self-hosted personal agent project. Each was sourced, but the tree favours one node per idea. Claims seen only in press coverage (for example AI scores at IMO 2026) were not used.

## Forty links between existing milestones (Claude Code, 2026-10-19)

80 of 159 milestones had no connection. This pass adds 40, leaving 26 unlinked. No milestone was added. Each link is stated or clearly shown in a source already listed on its origin node, and the origin node's source was opened during this pass. Four were rechecked directly against the source text: TPU 8 → TPU v3 liquid cooling ("fourth-generation liquid cooling"), R1 → o1 ("comparable to OpenAI-o1-1217"), TCP/IP → ARPANET (RFC 801 is the NCP/TCP transition plan), and MPI → FORTRAN (MPI-1 bindings for Fortran 77 and C). Pages that refused automated fetches (openai.com, ai.meta.com) were read through Internet Archive captures of the same URLs.

| Link | Type | What the origin's source says | Source on the origin node |
| --- | --- | --- | --- |
| `instructgpt` → `gpt3` | adapts | It starts from the pretrained GPT-3 models and fine-tunes them with human demonstrations and rankings. | https://arxiv.org/abs/2203.02155 |
| `instructgpt` → `ppo` | uses | Its reinforcement-learning step maximises the reward model's score with PPO. | https://arxiv.org/abs/2203.02155 |
| `chatgpt` → `instructgpt` | adapts | Trained with RLHF using InstructGPT's methods, with a different data-collection setup. | https://openai.com/index/chatgpt/ |
| `grpo` → `ppo` | adapts | A PPO variant that drops the critic and uses group scores as the baseline, saving memory. | https://arxiv.org/html/2402.03300v3 |
| `deepseek-r1` → `o1-2024` | complements | The R1 report takes o1 as its reference point and reports parity with o1-1217 on reasoning tasks. | https://arxiv.org/html/2501.12948v1 |
| `dpo` → `instructgpt` | adapts | It keeps the RLHF objective of the InstructGPT line but replaces reward modelling plus RL with one direct loss. | https://arxiv.org/abs/2305.18290 |
| `posttraining-open` → `vllm` | uses | Tülu 3's verifiable-reward stage generates rollouts with vLLM and syncs weights to it after each update. | https://arxiv.org/html/2411.15124v3 |
| `llama-3-1-2024` → `dpo` | uses | Each alignment round combines supervised fine-tuning, rejection sampling and DPO. | https://ai.meta.com/blog/meta-llama-3-1/ |
| `o1-2024` → `mmlu-2020` | uses | OpenAI reports o1 ahead of GPT-4o on 54 of 57 MMLU subcategories. | https://openai.com/index/learning-to-reason-with-llms/ |
| `gemini-1-5-2024` → `sparse-moe-2017` | adapts | Its report describes Gemini 1.5 as built on the mixture-of-experts architecture of Shazeer et al. | https://arxiv.org/abs/2403.05530v1 |
| `flash` → `chips-roofline-report-2008` | uses | It sorts operations into compute- or memory-bound by arithmetic intensity, citing Roofline. | https://arxiv.org/abs/2205.14135 |
| `amd-cdna3-mi300-2023` → `hbm-gpu-packaging-2015` | uses | MI300X carries 192 GB of HBM3 and MI300A 128 GB. | https://www.amd.com/en/newsroom/press-releases/2023-12-6-amd-delivers-leadership-portfolio-of-data-center-a.html |
| `nvidia-hopper-2022` → `transformer` | complements | Hopper adds a Transformer Engine aimed at Transformer models. | https://nvidianews.nvidia.com/news/nvidia-announces-hopper-architecture-the-next-generation-of-accelerated-computing |
| `nvidia-ampere-2020` → `nvidia-volta-2017` | adapts | The A100 whitepaper presents its SM, Tensor Core and memory changes as gains over V100. | https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf |
| `amd-rdna-2019` → `amd-gcn-2011` | adapts | RDNA's dual compute unit replaces the GCN compute unit and stays compatible with GCN. | https://www.igorslab.de/wp-content/uploads/2019/08/rdna-whitepaper.pdf |
| `amd-cdna-2020` → `amd-gcn-2011` | adapts | CDNA's compute units derive from GCN, with added Matrix Core engines. | https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf |
| `chips-intel-trigate-22nm-2011` → `transistor-1947` | adapts | Intel calls Tri-Gate a reinvention of the transistor: a thin 3-D fin replaces the flat gate. | https://www.intc.com/news-events/press-releases/detail/655/intel-reinvents-transistors-using-new-3-d-structure |
| `chips-euv-volume-production-2019` → `chips-moore-cramming-1965` | complements | ASML presents EUV as what keeps Moore's Law going with ever smaller transistors. | https://www.asml.com/en/products/euv-lithography-systems |
| `tpu-8-2026` → `energy-tpu-v3-liquid-cooling-2018` | uses | Both eighth-generation TPUs rely on Google's fourth-generation liquid cooling. | https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/ |
| `dennard-dram-patent-1967` → `memory-whirlwind-core-1953` | complements | The patent contrasts its refreshed capacitor storage with remanent storage such as magnetic cores. | https://patents.google.com/patent/US3387286A/en |
| `networks-tcp-ip-transition-1983` → `networks-arpanet-1969` | adapts | RFC 801 plans moving the operational ARPANET from its original host protocol to TCP/IP. | https://www.rfc-editor.org/rfc/rfc801 |
| `networks-ethernet-paper-1976` → `networks-arpanet-1969` | complements | Metcalfe and Boggs place Ethernet after the ARPA network's work on packet switching and resource sharing. | https://www.cl.cam.ac.uk/teaching/1920/CompNet/files/p395-metcalfe.pdf |
| `networks-rdmap-rfc-2007` → `networks-tcp-ip-transition-1983` | uses | RDMAP runs over DDP/MPA on TCP to avoid the copies TCP/IP normally needs. | https://www.rfc-editor.org/rfc/rfc5040.html |
| `energy-warehouse-scale-computing-2009` → `energy-pue-metric-2007` | uses | Its efficiency breakdown starts with PUE, citing The Green Grid. | https://zhjwpku.com/assets/pdf/TheDatacenterAsaComputer.pdf |
| `distributed-mpi-1994` → `software-fortran-1957` | uses | MPI-1 standardised bindings for Fortran 77 and C. | https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node5.html |
| `dqn` → `qlearning` | adapts | DQN is a convolutional network trained with a variant of Q-learning. | https://arxiv.org/abs/1312.5602 |
| `sac` → `ppo` | complements | SAC is an off-policy alternative to on-policy PPO, which it also uses as a baseline. | https://arxiv.org/abs/1801.01290 |
| `graphsage` → `gcn` | adapts | Its convolutional aggregator is an inductive version of Kipf and Welling's GCN. | https://arxiv.org/abs/1706.02216 |
| `gat` → `transformer` | adapts | GAT brings Transformer-style self-attention to graph neighbourhoods and names Vaswani et al. as its inspiration. | https://arxiv.org/abs/1710.10903 |
| `svm` → `perceptron-1958` | adapts | Cortes and Vapnik start from Rosenblatt's perceptron: a fixed non-linear map, then a linear surface. | https://homepages.math.uic.edu/~lreyzin/papers/cortes95.pdf |
| `hopfield-network-1982` → `perceptron-1958` | complements | Hopfield sets his model against perceptrons: strong backward coupling and asynchronous units. | https://authors.library.caltech.edu/records/w41x7-8bn13/files/HOPpnas82.pdf |
| `lenet5-1998` → `backprop` | uses | It names backprop-trained multilayer networks as the leading example of gradient-based learning. | https://leon.bottou.org/publications/pdf/ieee-1998.pdf |
| `alphago` → `deep-blue-1997` | complements | AlphaGo evaluated thousands of times fewer positions than Deep Blue, chosen by its networks. | https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf |
| `alexnet` → `sift-2004` | complements | Its results tables compare against SIFT + Fisher Vector pipelines, the previous best. | https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf |
| `google-search-prototype-1998` → `networks-web-proposal-1989` | uses | The prototype crawls and indexes the Web and exploits its hypertext links. | https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf |
| `gnmt-2016` → `ibm-smt-1993` | complements | GNMT names statistical MT (citing Brown et al. 1993) as the dominant paradigm it is measured against. | https://arxiv.org/pdf/1609.08144v1 |
| `wavenet-2016` → `resnet` | adapts | WaveNet uses residual connections (He et al.) plus skip connections throughout. | https://arxiv.org/pdf/1609.03499v1 |
| `elmo-2018` → `penn-treebank-1993` | uses | ELMo tests its representations with part-of-speech tagging on the WSJ Penn Treebank. | https://arxiv.org/pdf/1802.05365v1 |
| `dartmouth` → `shannon-information-theory-1948` | complements | Shannon's part of the proposal applies information theory to computers and brain models. | https://www-formal.stanford.edu/jmc/history/dartmouth/dartmouth.html |
| `dendral-1969` → `mycin` | complements | Buchanan and Shortliffe call MYCIN an outgrowth of DENDRAL that reused its lessons on encoding knowledge. | https://www.shortliffe.net/Buchanan-Shortliffe-1984/Chapter-01.pdf |

Left unlinked on purpose: FFT, Dijkstra, Metropolis, RAMAC, `symbolic`, IEA *Energy and AI*, Genie 3 and others. No source read in this pass states a relation for them, and chronology alone is not a link. Candidate new milestones (24 pivotal works, 15 underdog works) were researched in the same pass but not added; the owner chose links only for now.
