# NVIDIA GPU generations: what each one changed, and why

[中文](nvidia-generations.md) · **English**

> Reading time: ~13 min · Last reviewed: 2026-10

When you read "the new generation is N times faster", ask three things first: which precision? Dense or sparse? Which SKU, and is it per GPU or per system?

Once those are aligned, the broad trend is growth in low-precision Tensor throughput, HBM capacity and the size of the NVLink domain. Memory bandwidth has grown too, but more slowly than compute. The number of operations needed per byte read to saturate compute went from about 140 on the V100 to about 1,900 on the GB300 with NVFP4. For read-heavy, compute-light work such as large-model decode, bandwidth often explains generational speedups better than peak compute does.

SMs, warps and Tensor Cores are explained in [GPU architecture](gpu-architecture.en.md). Every hardware number here comes from NVIDIA whitepapers, datasheets, product pages and technical blogs, checked on 2026-10-15; shipped and merely announced products are marked separately.

## Before reading a datasheet, align these

| What to align | Example | How this note handles it |
| --- | --- | --- |
| Dense or sparse | The [H100 product page](https://www.nvidia.com/en-us/data-center/h100/) lists BF16 at 1,979 TFLOPS with the footnote "with sparsity"; dense is half. But the [GB300 NVL72 page](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) lists FP4 as 1,440 (sparse) and 1,080 (dense), not a factor of 2 | Compare dense only. Divide by 2 only when the page gives sparse alone and its footnote says dense is half |
| Which SKU and form factor | Both are "H100": the SXM part has 80 GB and 3.35 TB/s; the NVL (PCIe) part has 94 GB and 3.9 TB/s, with lower Tensor throughput | Use SXM or rack parts throughout and say so |
| What is being counted | A [GB200 superchip](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) is 1 Grace CPU plus 2 GPUs; the [HGX page](https://www.nvidia.com/en-us/data-center/hgx/) gives 8-GPU totals | Convert everything to per GPU |
| Preliminary or final | The 2022 [Hopper blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) gives dense BF16 of 1,000 TFLOPS and memory of 3,000 GB/s, marked preliminary; the [H100 architecture whitepaper V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf) gives the final 989.4 TFLOPS (dense) and 3,352 GB/s, consistent with the product page's 1,979 (sparse) and 3.35 TB/s | Use final specs: whitepaper V1.04 or the current product page |

Official pages also disagree with each other. For Blackwell memory per GPU, the comparison table in the [Blackwell Ultra technical blog](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) says up to 192 GB, the GB200 page works out to 186 GB, and the [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)'s 1,440 GB divided by 8 is 180 GB. For Rubin's memory bandwidth, NVLink and throughput, different pages give different numbers too (see the Rubin section below). Where this happens, the note lists the competing figures and says where each comes from.

## Once aligned: one table

All throughput figures are dense and per GPU. The B-series figures are converted from HGX or NVL72 system totals.

| Product (form factor) | Process · transistors | SMs | Tensor Cores (new precisions) | Memory · bandwidth | NVLink (per GPU) | Power limit | BF16 | FP8 | FP4 |
| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| V100 SXM2 (Volta) | TSMC 12 nm FFN · 21.1B | 80 | 1st gen (FP16) | 32 GB HBM2 · 900 GB/s | NVLink 2: 300 GB/s | 300 W | 125 TFLOPS (FP16) | — | — |
| A100 80GB SXM (Ampere) | TSMC N7 · 54.2B | 108 | 3rd gen (TF32, BF16, 2:4 sparsity) | 80 GB HBM2e · 2,039 GB/s | NVLink 3: 600 GB/s | 400 W | 312 TFLOPS | — | — |
| H100 SXM (Hopper) | TSMC 4N · 80B | 132 | 4th gen (FP8) | 80 GB HBM3 · 3,352 GB/s | NVLink 4: 900 GB/s | 700 W | 989.4 TFLOPS | 1,978.9 TFLOPS | — |
| H200 SXM (Hopper) | TSMC 4N · 80B | — | 4th gen | 141 GB HBM3e · 4.8 TB/s | NVLink 4: 900 GB/s | 700 W | 989.4 TFLOPS | 1,978.9 TFLOPS | — |
| B200 (HGX B200) | TSMC 4NP · 208B (2 dies) | — | 5th gen (FP6, FP4) | 180 GB HBM3e · 8 TB/s | NVLink 5: 1.8 TB/s | up to 1,200 W* | 2.25 PFLOPS | 4.5 PFLOPS | 9 PFLOPS |
| GB200 (each GPU in NVL72) | Same | — | 5th gen | 186 GB HBM3e · 8 TB/s | NVLink 5: 1.8 TB/s | up to 1,200 W* | 2.5 PFLOPS | 5 PFLOPS | 10 PFLOPS |
| B300 (HGX B300, Blackwell Ultra) | TSMC 4NP · 208B (2 dies) | up to 160 | 5th gen (faster NVFP4) | about 262 GB HBM3e · 8 TB/s | NVLink 5: 1.8 TB/s | up to 1,400 W* | 2.25 PFLOPS | 4.5 PFLOPS | 13.5 PFLOPS |
| GB300 (each GPU in NVL72) | Same | up to 160 | Same | about 278 GB HBM3e · 8 TB/s | NVLink 5: 1.8 TB/s | up to 1,400 W* | 2.5 PFLOPS | 5 PFLOPS | 15 PFLOPS |
| Rubin (in production, see below) | Process not disclosed · 336B (2 dies) | 224 | 5th gen (per the platform blog); 3rd-gen Transformer Engine | 288 GB HBM4 · 22 or 19.2 TB/s | NVLink 6: 3.6 or 3 TB/s | Not disclosed | 4 PFLOPS | 17.5 PFLOPS | 35 PFLOPS (NVFP4 training) |

Sources by row: V100 from the [Volta whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) and the [V100 datasheet](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf); A100 from the [Ampere blog](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) and the [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf); H100 from whitepaper V1.04 and the product page; H200 from the [H200 product page](https://www.nvidia.com/en-us/data-center/h200/), whose Tensor rows match the H100 SXM product page exactly, so 989.4 / 1,978.9 carry over; B200 memory and bandwidth converted from the DGX B200's 8-GPU totals of 1,440 GB and 64 TB/s, and its throughput from the HGX page; GB200 as the NVL72 page's superchip figures divided by 2; B300 memory as the [DGX B300](https://www.nvidia.com/en-us/data-center/dgx-b300/)'s 2.1 TB divided by 8, GB300 as the NVL72's 20 TB divided by 72; process and transistors for the H200 and the B series, and the B300's bandwidth and SM count ("up to 160, varies by SKU"), from the Blackwell Ultra blog's generational comparison table; Rubin is covered below.

\* The Blackwell and Blackwell Ultra power figures come from the generational comparison table in the Blackwell Ultra blog ("up to", TGP), which does not name a SKU; the HGX and DGX pages give only system power (about 14.3 kW for a DGX B200). Also, per the GB200 press release a GB200 holds two B200 GPUs, yet per-GPU BF16 is 2.25 PFLOPS in HGX B200 and 2.5 in GB200: the same GPU is specified differently in different products.

## What each generation changed

### Volta (V100, 2017): Tensor Cores arrive

The Volta whitepaper (August 2017 edition) introduced the first Tensor Cores: each performs one 4×4×4 FP16 multiply-accumulate per clock with FP32 accumulation, and the V100 has 640 of them. On the same card, the ordinary FP32 peak is 15.7 TFLOPS and the Tensor peak is 125 TFLOPS. The whitepaper's reasoning: matrix multiplication is the core operation of training and inference, and dedicated units raise throughput dramatically at high energy efficiency. The same generation brought second-generation NVLink (6 links, 300 GB/s in total) and independent thread scheduling.

### Ampere (A100, 2020): Tensor Cores for FP32 code too

The A100 was announced in May 2020 (Ampere blog). Third-generation Tensor Cores added three things. TF32 reads FP32 data and computes with FP32 range and reduced internal precision, so training code written in FP32 can use Tensor Cores without changing formats; BF16 runs at the same rate as FP16 but has as many exponent bits as FP32, so it is less prone to overflow in training (the trade-off is in [Precision and memory](../00-foundations/deep-dives/precision-and-memory.en.md)); 2:4 structured sparsity doubles throughput for matrices that satisfy it, and from this generation on, datasheets list Tensor figures in separate dense and sparse columns. Beyond that, L2 grew to 40 MB, NVLink moved to its third generation (600 GB/s), and MIG can split one card into up to 7 isolated instances.

### Hopper (H100 2022, H200 2023): FP8 and the Transformer Engine

The H100 was announced in March 2022 (Hopper blog). Fourth-generation Tensor Cores double the A100's per-SM 16-bit MMA rate and quadruple it with FP8. FP8 comes as E4M3 and E5M2; with only 8 bits, its range is much narrower than 16-bit formats, so the Transformer Engine arrived alongside it: it gathers statistics on each layer's outputs, decides between FP8 and 16 bits, and rescales values dynamically into the representable range. This generation also added TMA, thread block clusters and distributed shared memory, all aimed at making data movement asynchronous (see [GPU architecture](gpu-architecture.en.md)). Memory moved to HBM3 (final value 3,352 GB/s in whitepaper V1.04), NVLink to its fourth generation (900 GB/s), and the power limit to 700 W.

The H200 was announced at SC23 on 13 November 2023, and NVIDIA called it the first GPU with HBM3e ([press release](https://nvidianews.nvidia.com/news/nvidia-supercharges-hopper-the-worlds-leading-ai-computing-platform)). On the product pages, every Tensor row of the H200 SXM matches the H100 SXM exactly; only memory changed, to 141 GB and 4.8 TB/s. **Same compute, more memory capacity and bandwidth**; the worked example below shows why such an upgrade matters for decode.

### Blackwell (B200 / GB200, 2024): two dies, FP4 and NVL72

Blackwell was announced at GTC on 18 March 2024 ([press release](https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing)). The [architecture page](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) states 208 billion transistors on TSMC 4NP, with two "reticle-limited" dies joined into one GPU by a 10 TB/s chip-to-chip link. The previous three die sizes show why: V100 815 mm², A100 826 mm², H100 814 mm² (each generation's whitepaper and the Hopper blog). A single die could not grow further, so more transistors meant two dies in one package. Process and packaging are in [How chips are made](chips-and-fabs.en.md).

The second-generation Transformer Engine adds FP4 and micro-tensor scaling, i.e. finer-grained scaling. NVFP4, which NVIDIA introduced in June 2025, is a format of this small-block-scaled kind: one FP8 scale per 16 values ([NVFP4 blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/); a worked example is in [GPU architecture](gpu-architecture.en.md)). Fifth-generation NVLink gives 1.8 TB/s per GPU and can connect up to 576 GPUs. GB200 NVL72 puts 36 Grace CPUs and 72 Blackwell GPUs in one liquid-cooled rack that forms a single NVLink domain with 130 TB/s in total.

### Blackwell Ultra (B300 / GB300, 2025): gains concentrated in NVFP4 and attention

Blackwell Ultra was announced at GTC on 18 March 2025, with partner products expected from the second half of 2025 ([press release](https://nvidianews.nvidia.com/news/nvidia-blackwell-ultra-ai-factory-platform-paves-way-for-age-of-ai-reasoning)). As of October 2026, an HGX page footnote says both HGX B300 and B200 are shipping, and the GB300 NVL72 page is headed "Available Now".

Transistors, process and die count match Blackwell; what changed is the allocation (Blackwell Ultra blog): dense NVFP4 rises from 10 to 15 PFLOPS; throughput for the exponentials that softmax in attention needs doubles; memory goes up to 288 GB; the power limit is 1,400 W. The cost is visible too. On the HGX page, the 8-GPU HGX B300 and HGX B200 have the same BF16 and FP8 (36 and 72 PFLOPS sparse), but INT8 drops from 72 POPS to 3 POPS (both sparse figures) and FP64 from 296 TFLOPS to 10 TFLOPS; in the [compute capability tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html), the B300's compute capability 10.3 also lists no FP64 Tensor Core input type. **This generation did not raise BF16 or FP8; its gains are concentrated in NVFP4, memory capacity and softmax exponentials, while most FP64 and INT8 throughput was cut.**

Memory is stated three ways: the blog says up to 288 GB; the DGX and HGX B300 pages say 2.1 TB for 8 GPUs, about 262 GB each; the GB300 NVL72 page says 20 TB for 72 GPUs, about 278 GB each. The pages do not explain the difference.

### Rubin / Vera Rubin (in production): HBM4 and NVLink 6

The timeline:

- GTC, March 2025 ([NVIDIA blog recap](https://blogs.nvidia.com/blog/nvidia-keynote-at-gtc-2025-ai-news-live-updates/)): Vera Rubin NVL144 set for the second half of 2026, Rubin Ultra systems for the second half of 2027.
- CES, 5 January 2026 ([press release](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer)): its Rubin Readiness section states "NVIDIA Rubin is in full production …", with partner products in the second half of the year; the rack was renamed Vera Rubin NVL72 (72 Rubin GPUs, 36 Vera CPUs). The GTC [press release](https://nvidianews.nvidia.com/news/nvidia-vera-rubin-platform) of 16 March added a seventh chip, the Groq 3 LPU, and the [POD blog](https://developer.nvidia.com/blog/nvidia-vera-rubin-pod-seven-chips-five-rack-scale-systems-one-ai-supercomputer/) described Rubin Ultra scaling to 576 GPUs and the Kyber rack with 144 per rack, without dates.
- GTC Taipei, 31 May 2026 ([press release](https://nvidianews.nvidia.com/news/vera-rubin-full-production-agentic-ai-factory)): production shipments set to begin "this fall"; an [NVIDIA blog post](https://blogs.nvidia.com/blog/vera-cpu-delivery/) updated on 27 August said AWS had received its first Vera CPU server and Vera Rubin GPU.
- 26 August 2026 ([second-quarter fiscal 2027 results release](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-second-quarter-fiscal-2027)): Jensen Huang said Vera Rubin is "now in full production", with racks running at partners including CoreWeave, Google Cloud, Microsoft Azure, Oracle Cloud Infrastructure and Nebius. An [NVIDIA blog post](https://blogs.nvidia.com/blog/coreweave-agentic-ai-vera-rubin/) of 30 September said CoreWeave had received its first Vera Rubin NVL72 production racks that month.

As of October 2026, the [Vera Rubin NVL72 product page](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) is headed "Available Now" and says server makers are "shipping Vera Rubin-based systems", so this note marks Rubin as "in production". Broad cloud availability is still rolling out: an [NVIDIA blog post](https://blogs.nvidia.com/blog/google-cloud-vera-rubin-infrastructure/) of 21 October says general availability of Google Cloud's A5X instances, built on Vera Rubin NVL72, comes next month. The numbers below are NVIDIA's published specifications.

The [Rubin GPU architecture blog](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/) (21 July 2026) gives 336 billion transistors, two compute dies, 224 SMs and up to 288 GB of HBM4, without naming the process; the January [platform blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/) calls the SMs' Tensor Cores fifth-generation. The table's throughput comes from the per-GPU table on the [HGX page](https://www.nvidia.com/en-us/data-center/hgx/) and from the NVL72 page: NVFP4 training 35, FP8 17.5 and BF16 4 PFLOPS, all marked dense. The frequently quoted "50 PFLOPS of NVFP4 inference" is marked sparse on those two pages, while the platform blog attributes it to "adaptive compression" in the third-generation Transformer Engine without saying dense or sparse.

Official pages disagree in at least three ways (read on 2026-10-15):

| Source | Memory · bandwidth per GPU | NVLink per GPU | NVFP4 training (dense, per GPU) |
| --- | --- | --- | ---: |
| CES press release (2026-01) | Not given | 3.6 TB/s, 260 TB/s per NVL72 | Not given |
| Platform blog (2026-01), GPU blog (2026-07), per-GPU table on the HGX page | 288 GB · 22 TB/s | 3.6 TB/s | 35 PFLOPS |
| Vera Rubin NVL72 product page, [Vera Rubin datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/56p68o47y6f1yucpubl0ump1c0aif422.pdf) (2026-10) | 288 GB · 19.2 TB/s (rack: 20.7 TB, 1,400 TB/s) | 3 TB/s, 216 TB/s per rack | 35 PFLOPS |
| HGX Rubin NVL8 system on the same HGX page and in the datasheet, divided by 8 | 2 TB and 130 TB/s, about 250 GB · 16 TB/s | 24 TB/s switch bandwidth, about 3 TB/s | 270 PFLOPS, about 33.75 |

The latest datasheet agrees with the NVL72 product page, but the HGX page's own per-GPU and system tables do not match each other either, and none of the pages explains the differences. The later tables in this note give a range using 22 and 19.2; once the parts ship in volume, the datasheet for the specific SKU should be taken as authoritative.

## Design logic: what kept changing

Divide each generation's dense peak at its lowest precision by memory bandwidth and you get "how many operations per byte read are needed to saturate compute". [Performance](performance.en.md) calls this ratio the ridge point:

| Product | Dense peak at lowest precision | Memory bandwidth | FLOPs per byte |
| --- | ---: | ---: | ---: |
| V100 SXM2 | FP16 125 TFLOPS | 0.9 TB/s | 139 |
| A100 80GB SXM | BF16 312 TFLOPS | 2.039 TB/s | 153 |
| H100 SXM | FP8 1,978.9 TFLOPS | 3.352 TB/s | 590 |
| H200 SXM | FP8 1,978.9 TFLOPS | 4.8 TB/s | 412 |
| GB200 (per GPU) | NVFP4 10 PFLOPS | 8 TB/s | 1,250 |
| GB300 (per GPU) | NVFP4 15 PFLOPS | 8 TB/s | 1,875 |
| Rubin (in production) | NVFP4 training 35 PFLOPS | 22 or 19.2 TB/s | 1,591 or 1,823 |

Several design threads show up in the table:

1. **Lower precision is the biggest lever.** From V100 FP16 to GB300 NVFP4, the dense peak rose 120×, of which 16-bit to 16-bit (V100 FP16 at 125 versus GB300 BF16 at 2,500 TFLOPS) is only 20×; the remaining factor of about 6 came from narrower formats. Narrow formats keep their accuracy only with the Transformer Engine's scaling: per-tensor statistics in the first generation, micro-blocks in the second, and "adaptive compression" added for Rubin.
2. **Bandwidth grows more slowly than compute.** Over the same span, memory bandwidth went from 0.9 to 8 TB/s, about 9×, so the ridge point keeps rising and each byte read in must be reused more and more. The H200 goes the other way: same compute as the H100, 43% more bandwidth, and the ridge point falls from 590 to 412.
3. **When one die cannot grow, put two in the package.** Three generations of dies all sit between 814 and 826 mm²; Blackwell and Rubin both moved to two dies with a fast die-to-die link.
4. **NVLink domains keep growing.** Per-GPU NVLink bandwidth rose every generation: 300, 600, 900, 1,800 GB/s, and 3,600 or 3,000 GB/s for Rubin (see above); the domain grew from one 8-GPU machine to a 72-GPU rack, and the Kyber rack has been announced with 144 per rack. Tensor parallelism communicates at every layer and MoE expert parallelism needs all-to-all, and both want to stay inside that domain; see [Tensor parallelism](../06-systems/tensor-parallel.en.md).
5. **More specialized, and more power-hungry.** Blackwell Ultra cut FP64 and INT8; the power limit went from 300 W to 1,400 W, and GB200 NVL72 is a liquid-cooled rack.

## Worked example: how many GPUs a 70B model needs, and the fastest possible token

The [series guide](README.en.md) worked through decode for an 8B model on one card; here we take a 70B-parameter dense model and see how many cards each generation and precision needs. Simplifying assumptions:

- Weights only: BF16 is 2 bytes per parameter, 140 GB in total; FP8 is 1 byte per parameter, 70 GB in total. Capacities are divided as printed on the datasheets, without distinguishing GB from GiB or subtracting what the driver and CUDA context use.
- Batch size 1, one token at a time, each token reads all weights once from HBM; KV cache reads and compute time are ignored.
- With several cards, weights are split evenly and read in parallel, communication takes zero time, and bandwidth runs at 100% of peak.

Then:

$$
n_{\min}=\left\lceil\frac{P\,b}{M}\right\rceil,\qquad
t_{\text{token}}\ \ge\ \frac{P\,b}{n\,B}.
$$

$P=70\times10^9$ is the parameter count, $b$ the bytes per parameter, $M$ the memory per card, $B$ the bandwidth per card, and $n$ the number of cards. For example, BF16 on two H100s: $140\ \text{GB}/(2\times3.352\ \text{TB/s})\approx20.9$ ms, i.e. at most about 48 tokens per second.

| Product | Memory · bandwidth per card | BF16 (140 GB): fewest cards · per-token floor | FP8 (70 GB): fewest cards · per-token floor |
| --- | --- | --- | --- |
| V100 SXM2 | 32 GB · 0.9 TB/s | 5 cards (FP16) · 31.1 ms | No FP8 Tensor Cores |
| A100 80GB SXM | 80 GB · 2.039 TB/s | 2 cards · 34.3 ms | No FP8 Tensor Cores |
| H100 SXM | 80 GB · 3.352 TB/s | 2 cards · 20.9 ms | 1 card (only 10 GB left) · 20.9 ms |
| H200 SXM | 141 GB · 4.8 TB/s | 1 card (almost no headroom) · 29.2 ms; 14.6 ms on 2 cards | 1 card · 14.6 ms |
| B200 (HGX) | 180 GB · 8 TB/s | 1 card · 17.5 ms | 1 card · 8.8 ms |
| B300 (HGX) | about 262 GB · 8 TB/s | 1 card · 17.5 ms | 1 card · 8.8 ms |
| Rubin (in production) | 288 GB · 19.2–22 TB/s | 1 card · 6.4–7.3 ms | 1 card · 3.2–3.6 ms |
| RTX PRO 6000 Workstation Edition | 96 GB · 1.792 TB/s | 2 cards · 39.1 ms | 1 card · 39.1 ms |
| DGX Spark (GB10) | 128 GB unified memory · 0.273 TB/s | Does not fit | 1 unit · 256 ms |

<!-- widget:step-decode-floor -->

The last two rows come from the [RTX PRO 6000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/) and [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) product pages. Reading row by row:

- **On H100, FP8 on one card has the same floor as BF16 on two**, 20.9 ms each: halving the bytes is like doubling the bandwidth, and it uses one card fewer.
- **Two A100s (34.3 ms) have a higher time floor than five V100s (31.1 ms).** In this model the floor depends only on the group's total bandwidth: 5 × 0.9 = 4.5 TB/s, more than 2 × 2.039 ≈ 4.1 TB/s. More cards also have a cost, namely the communication we ignored, so this does not establish their measured speed ordering.
- **One H200 holds the BF16 weights but has a higher time floor than two H100s** (29.2 versus 20.9 ms); under these assumptions it saves cards without lowering the time floor.
- **B300 and B200 have the same batch-1 time floor**, because this weights-only read model uses their identical bandwidth; that does not guarantee identical measured speeds. The extra memory goes to KV cache and larger batches.
- **A workstation part that fits the model is not necessarily fast.** One RTX PRO 6000 holds a 70B model in FP8, but with about half an H100 SXM5's bandwidth its floor is 39.1 ms; DGX Spark fits it too, but each token takes at least 256 ms, about 4 tokens per second.

Workstation spec pages hold two more traps. "AI TOPS" often states neither precision nor sparsity (the RTX PRO 6000's 4,000 AI TOPS carries a footnote saying FP4 with sparsity), so it cannot be compared with the dense figures above; and although they are called Blackwell, the RTX PRO 6000 is compute capability 12.0 and DGX Spark 12.1 ([CUDA GPUs list](https://developer.nvidia.com/cuda-gpus)), with at most 100 KB of shared memory per SM against 228 KB on the data-centre 10.0, and code compiled for an architecture-specific target such as sm_100a does not run on 12.x.

Now check that compute really is not the bottleneck: each token takes about $2P=1.4\times10^{11}$ operations, which the H100 SXM's 989.4 dense BF16 TFLOPS finish in 0.14 ms, more than two orders of magnitude less than the 20.9 ms spent reading weights. Batch-1 decode is deep in bandwidth-bound territory, and serving many requests at once is what puts the compute to use; see [How a request runs](../06-systems/llm-serving.en.md) and [KV cache](../00-foundations/deep-dives/kv-cache-and-inference.en.md).

| Change one condition | What you should see | What you can NOT conclude |
| --- | --- | --- |
| Batch from 1 to 32 | Weights are read once per step and shared by 32 requests; each step also reads 32 sets of KV | Each user's tokens come just as fast |
| Switch to MoE, still 70B total parameters, with only part active per token | Each step reads only the selected experts, so the floor follows the active parameters | Memory shrinks too: all experts still have to fit |
| On H100, tensor parallelism from 2 cards to 8 | The floor in this model drops to a quarter | It really gets 4× faster: every layer needs an all-reduce over NVLink, and that share grows with more cards |
| On B200, FP8 to NVFP4 | Weights become about $70\times10^9\times4.5/8\approx39$ GB, and the floor about 4.9 ms | Model quality is unchanged; check evaluations after quantization |

## What to read next

- The SMs, warps, memory hierarchy and Tensor Cores behind these specs: [GPU architecture](gpu-architecture.en.md).
- How the ridge point tells you whether a kernel is compute-bound or bandwidth-bound: [Performance](performance.en.md).
- Splitting across GPUs and the communication it needs: [Tensor parallelism](../06-systems/tensor-parallel.en.md), [Distributed training](../06-systems/distributed-training.en.md).
