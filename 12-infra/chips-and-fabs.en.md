# How chips are made: wafers, process nodes, packaging, and HBM

[中文](chips-and-fabs.md) · **English**

> Reading time: ~10 min · Last reviewed: 2026-10

The news keeps saying "AI chips are in short supply." Short of what, exactly? Is it that the fab cannot make the GPUs? Not quite. **The larger a die, the fewer fit on a wafer and the larger the fraction that is defective; and a single die cannot exceed the exposure limit. So AI chips increasingly package several dies together with HBM, a step that needs advanced packaging and HBM, both of which have limited capacity of their own.** This note starts with the production chain, works through the yield of one wafer, and then looks at packaging and HBM.

## From design to server: the steps one GPU goes through

| Step | What goes in, what comes out | What can get stuck here |
| --- | --- | --- |
| Design | Architecture and circuits → the layout handed to the fab (tape-out) | Long design cycles; every change means new masks |
| Wafer fabrication | A 300 mm silicon wafer → many steps of lithography, deposition, etching and more, yielding a wafer full of dies | Leading-edge process capacity, defect density |
| Wafer test | A whole wafer → a map of which dies work | Bad dies that escape test flow downstream and waste packaging |
| Dicing and packaging | Good dies (plus HBM and so on) → one packaged chip | Advanced-packaging capacity, packaging yield |
| Final test | Packaged chips → passing parts, binned by specification | Test time and equipment |
| Boards and systems | Chips → modules, servers, racks | Power, cooling, networking and system assembly |

The design company need not own a fab. NVIDIA states that the GH100 in the H100 is "fabricated using the TSMC 4N process customized for NVIDIA …" ([Hopper architecture blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/#nvidia_h100_gpu_architecture_in-depth), H100 architecture section, 2022-03). HBM comes from memory makers; Micron, for example, sells HBM3E publicly ([Micron product page](https://www.micron.com/products/memory/hbm/hbm3e)). An AI chip needs three things at once, good logic dies, HBM and packaging, and is not shippable if any one is missing.

## What a process node name does and does not tell you

Early node names roughly matched a physical dimension of the transistor; today a node name is closer to the name of a process generation. A 2020 IEEE Spectrum article points out that most of the critical features of a "7 nm" transistor are much larger than 7 nm, and that names and physical dimensions have been disconnected for about two decades; it also describes LMC, an alternative metric proposed by H.-S. Philip Wong and colleagues: logic density, memory density, and the density of connections between them ([IEEE Spectrum, 2020-07](https://spectrum.ieee.org/a-better-way-to-measure-progress-in-semiconductors)).

So when reading about a node, look at what the vendor compares against and what it compares. TSMC's own descriptions ([TSMC smartphone platform page](https://www.tsmc.com/english/dedicatedfoundry/technology/platform_smartphone_tech_advancedTech), [HPC platform page](https://www.tsmc.com/english/dedicatedFoundry/technology/platform_HPC_tech_advancedTech), read 2026-10):

| Name | How TSMC describes it | What to watch for |
| --- | --- | --- |
| N4 | An enhanced version of N5, in volume production since 2022 | The name went from 5 to 4, but the official wording is "enhanced version," not a new generation |
| N4P | 11% performance boost over N5, in volume production since 2023 | The baseline is N5, not N4 |
| N3E | About 20% faster than N5, over 30% power savings, about 1.6x logic density | For N2P, TSMC separates "faster at the same power" from "lower power at the same speed"; such numbers usually cannot be had at the same time |
| N2P | Versus N3E, 18% faster at the same power or 36% lower power at the same speed; 1.2x logic density, 1.15x chip density | Logic density and whole-chip density are different things; TSMC gives two separate numbers |

For NVIDIA products, use only NVIDIA's own statements: the H100's GH100 uses a custom TSMC 4N process, has 80 billion transistors and a die size of 814 mm² ([Hopper architecture blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)); Blackwell-architecture GPUs have 208 billion transistors on a custom TSMC 4NP process, and all Blackwell products feature two "reticle-limited" dies joined by a 10 TB/s chip-to-chip interconnect ([Blackwell architecture page](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)). This note found no public statement from either company on exactly how "4N" and "4NP" differ from TSMC's public N4 family.

When comparing two chips, the node name is only a starting point. More reliable are measured performance, power and cost on the same workload.

## Why large dies are expensive: the arithmetic of one wafer

A lithography tool can expose only a limited area at once. ASML's NXT:2000i supports a maximum field of 26 mm × 33 mm, or 858 mm² ([ASML product page](https://www.asml.com/en/products/duv-lithography-systems/twinscan-nxt2000i)). The GH100's 814 mm² is already 95% of that area. The "reticle limit" simply means a single die cannot be larger than one exposure field.

First, how many dies fit on a 300 mm wafer. Let $d$ be the wafer diameter and $A$ the die area. A common approximation is:

$$
\text{DPW}\approx\frac{\pi(d/2)^2}{A}-\frac{\pi d}{\sqrt{2A}}.
$$

The first term is the area ratio; the second removes the ring at the edge where full dies do not fit, roughly the circumference divided by the die's diagonal.

Next, how many of them are good. The simplest model assumes defects land on the wafer randomly and independently, $D_0$ per square centimeter on average, and that one defect kills a die. The number of defects in area $A$ is then Poisson, and the yield is the probability of zero defects:

$$
Y_{\rm Poisson}=e^{-D_0A},\qquad
Y_{\rm Murphy}=\left(\frac{1-e^{-D_0A}}{D_0A}\right)^2.
$$

Real defects tend to cluster, so Poisson is pessimistic for large dies. [Murphy (1964)](https://doi.org/10.1109/PROC.1964.3442) let the defect density itself vary from die to die; the right-hand formula is the one usually called the Murphy model.

**Teaching example**: $D_0=0.1$ per cm² (made up; fabs generally do not publish real values), comparing a 100 mm² small die with an 800 mm² large die (for example 25 mm × 32 mm, which fits in the 26 × 33 mm field).

| Die area | Per wafer (formula) | Poisson yield | Good dies | Murphy yield |
| ---: | ---: | ---: | ---: | ---: |
| 100 mm² (1 cm²) | 640 | $e^{-0.1}=0.905$ | ~579 | 0.906 |
| 400 mm² (4 cm²) | 143 | $e^{-0.4}=0.670$ | ~96 | 0.679 |
| 800 mm² (8 cm²) | 64 | $e^{-0.8}=0.449$ | ~29 | 0.474 |

<!-- widget:step-good-dies -->

An area 8x larger gives 20x fewer good dies: fewer dies per wafer is one factor, more of them bad is another. Shifting the grid in 0.5 mm steps and directly counting full dies that fit gives 657, 151 and 71 on an ideal wafer, slightly more than the formula; real wafers also lose an edge-exclusion ring and scribe lanes, so the counts are lower. The numbers move, but the conclusion that good dies fall faster than area grows does not.

## Cut it in two and join the halves: the chiplet arithmetic

Split the 800 mm² die into two 400 mm² dies: the same wafer gives about 96 good dies, or 48 pairs, against about 29 for the monolithic die. This assumes you can tell which dies are good before packaging (known good die), so bad ones never get packaged.

But this arithmetic leaves out several costs:

| Try | Observe | What it does not establish |
| --- | --- | --- |
| Lower $D_0$ to 0.05 | The monolithic 800 mm² die rises from ~29 to ~43 good dies; the advantage of splitting shrinks | Mature processes no longer need chiplets |
| Raise $D_0$ to 0.2 | The monolithic die drops to ~13; two halves still give ~64 dies, 32 pairs | Splitting finer always saves more |
| Add die-to-die interface area to each small die | Each die grows, so both the count and the yield drop slightly | The interface's power and latency can be ignored |
| Multiply by packaging yield | When a package fails, the good dies and HBM inside are scrapped with it | Packaging is cheap enough to ignore |

Blackwell shows the other side: once a die is already at the exposure limit, the only way to more transistors is to join two dies with a high-bandwidth chip-to-chip link into one GPU. After that, the pressure moves to packaging.

## Advanced packaging: what the interposer does

A GPU needs on the order of a thousand data lines to each HBM stack (1,024 bits for HBM3, 2,048 for HBM4). 2.5D packaging puts an interposer between the GPU die, the HBM and the substrate, and connects them with chip-grade fine wiring. An ordinary organic substrate cannot hold connections that dense, and a new JEDEC standard shows this from the other direction: SPHBM4 (JESD330-4, 2026-07) reuses the HBM4 DRAM dies but cuts the data signals to a quarter and recovers the bandwidth with 4:1 serialization, so that the bump pitch can be relaxed enough for organic substrates ([JEDEC, 2026-07-13](https://www.jedec.org/news/pressreleases/new-jedec%C2%AE-sphbm4-standard-enables-hbm4-class-bandwidth-organic-substrates)).

TSMC's CoWoS is this kind of packaging. According to TSMC ([CoWoS page](https://3dfabric.tsmc.com/english/dedicatedFoundry/technology/cowos.htm), read 2026-10), it integrates multiple SoCs and HBM stacks and has been in volume production since 2012:

| Type | What the interposer is | Specification TSMC gives |
| --- | --- | --- |
| CoWoS-S | Silicon interposer | Up to about 3.3x the reticle size (~2,700 mm²) |
| CoWoS-R | RDL interposer of polymer and copper traces, relatively flexible | Minimum routing pitch 4 µm (2 µm line width and spacing), in volume production since 2023 |
| CoWoS-L | RDL interposer with embedded local silicon interconnect (LSI) chips | The 3.5x-reticle version in volume production since 2024; TSMC recommends L or R beyond 3.3x |

In CoWoS-L, the LSI chips carry the high-density wiring between dies and between dies and HBM, and the rest uses RDL. As for which NVIDIA product uses which: none of the NVIDIA whitepapers, architecture blogs or product pages checked for this note say; there is only news. According to a Reuters report of 2025-01-16 ([reprinted by the Taipei Times](https://www.taipeitimes.com/News/biz/archives/2025/01/17/2003830329)), Jensen Huang said Blackwell would largely use CoWoS-L, while Hopper, still in production, uses CoWoS-S. That is a CEO statement relayed by a news report, not a specification.

## HBM: stacking memory next to the GPU

HBM stacks several DRAM dies vertically and connects the layers with through-silicon vias (TSVs). It does not rely on each wire being fast but on a very wide interface; an interface that wide in standard HBM needs the fine, short wiring of an interposer, which is why HBM and advanced packaging go together.

The bandwidth of one stack can be computed by hand:

$$
\text{bandwidth per stack}=\frac{\text{interface width}\times\text{per-pin rate}}{8}.
$$

| Generation | Interface width | Per-pin rate | Bandwidth per stack | Capacity | Source |
| --- | ---: | ---: | ---: | --- | --- |
| HBM3 | 1,024 bits | Up to 6.4 Gb/s | 819 GB/s | 4 GB (4-high × 8 Gb) to 64 GB (16-high × 32 Gb) | [JEDEC, 2022-01-27](https://www.jedec.org/news/pressreleases/jedec-publishes-hbm3-update-high-bandwidth-memory-hbm-standard) |
| HBM3E (Micron) | 1,024 bits | Over 9.2 Gb/s | Over 1.2 TB/s | 24 GB 8-high, 36 GB 12-high | [Micron product page](https://www.micron.com/products/memory/hbm/hbm3e) |
| HBM4 | 2,048 bits | Up to 8 Gb/s | Up to 2 TB/s | 16-high × 32 Gb, up to 64 GB | [JEDEC, 2025-04-16](https://www.jedec.org/news/pressreleases/jedec%C2%AE-and-industry-leaders-collaborate-release-jesd270-4-hbm4-standard-advancing) |

Check one row: $1024\times6.4/8=819.2$ GB/s. HBM4 doubles the width, so its per-pin rate need not exceed HBM3E's to deliver more bandwidth. Micron's two "over"s have to be read together: at exactly 9.2 Gb/s, $1024\times9.2/8\approx1{,}178$ GB/s, slightly under 1.2 TB/s; a pin has to reach about 9.4 Gb/s before one placement (stack) exceeds 1.2 TB/s.

Now work backward from a product: the H100 SXM5 has 5 HBM3 stacks and a 5,120-bit interface, and [whitepaper V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf) gives 3,352 GB/s of memory bandwidth. Each pin then runs at about $3.352\times10^{12}\times8/5120\approx5.24$ Gb/s, which matches the memory data rate the whitepaper lists, 2,619 MHz DDR (two transfers per clock, about 5.24 Gb/s), and is below the 6.4 Gb/s the HBM3 standard allows. Product bandwidth is not "number of stacks × the standard's maximum"; check the specification of the specific SKU.

## Why capacity got stuck at packaging and HBM

A finished GPU module needs all three inputs at once. With made-up numbers (not tied to any real product): in some month there are 12,000 good logic dies (2 per module), 60,000 HBM stacks (8 per module), and the packaging line can build 6,500 modules. What ships is

$$
\min\left(\frac{12000}{2},\ \frac{60000}{8},\ 6500\right)=\min(6000,\ 7500,\ 6500)=6000.
$$

This month the limit is logic dies; if the fab supplies 2,000 more, the bottleneck moves to packaging (6,500); expand packaging and the next limit may be HBM (7,500). Expanding only one of the three does not raise shipments.

Some signals from public sources, by date:

- **TSMC's own statement** (CoWoS page, read 2026-10): since generative AI arrived in late 2022, demand for CoWoS has grown further, and TSMC continues to expand its advanced-packaging capacity.
- **News, Reuters, 2024-05-02** ([reprinted by ThePrint](https://theprint.in/tech/nvidia-supplier-sk-hynix-says-hbm-chips-almost-sold-out-for-2025/2066101/)): SK hynix said its HBM for 2024 was sold out and its 2025 supply almost sold out.
- **News, Reuters, 2025-01-16** (reprinted by the Taipei Times): Jensen Huang said advanced-packaging capacity was probably four times what it was less than two years earlier; the report says packaging remained a bottleneck due to capacity constraints. That second part is the reporter's wording.

These are snapshots at particular dates, and capacity changes every quarter; cite them with their dates and do not treat them as conclusions that still hold today.

## Back to one request

In the [series guide](README.en.md), every decode step reads 16 GB of weights from HBM, and its speed is set by HBM bandwidth. Making that step faster needs more or faster HBM stacks, which means a larger interposer, more packaging capacity, and more good dies cut from wafers. The next note looks at what the CPU does in this machine: [CPU architecture](cpu-architecture.en.md); for how the GPU die uses that bandwidth, see [GPU architecture](gpu-architecture.en.md).
