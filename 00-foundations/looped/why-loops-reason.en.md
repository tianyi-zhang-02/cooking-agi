# Looped Transformers: why loops help reasoning

[中文](why-loops-reason.md) · **English**

> Reading time: ~4 min · Last reviewed: 2026-10

## Some problems need serial steps

"In which city is the university that A's manager graduated from?" One approach looks up the manager, then the university, then its city. Intermediate results can help, but this is not a computational lower bound: parallel algorithms, different representations, or existing knowledge may reduce serial work. Multi-digit addition is similar: the usual digit-by-digit carry is one algorithm, not the only way to add.

Within a forward pass, looping increases serial network depth, not necessarily correct reasoning steps. The illustration deliberately allows only one link per pass to show the budget change. Real attention can aggregate several positions at once; it is not constrained by this toy one-layer/one-hop rule.

<!-- widget:tx-loop-reach -->

You can count it without the sliders. Keep the figure's toy rule (one layer follows one hop) and suppose the question needs six hops:

| Model | Layers of weights stored | Layer evaluations | Hop reached |
| --- | ---: | ---: | --- |
| Fixed 4 layers, untied | 4 | 4 | hop 4, two short |
| 1-layer block looped 6 times | 1 | 6 | hop 6, answer reached |
| 6 layers, untied | 6 | 6 | hop 6, answer reached |

The last two rows run the same number of layer evaluations, but the third stores six times the weights of the second; the first stores more weights than the second yet runs out of depth. This is the case $k=1$, $L=6$ in the parameter accounting of the next section. Conversely, if the link needed for hop 3 is missing from the input, the second row stops at hop 2 even after 60 loops: looping adds steps, not missing information.

## What theory and experiments say

- [Giannou et al. (2023)](https://arxiv.org/abs/2301.13196) construct networks that execute programs. Representability does not establish that next-token training will discover those programs.
- [Yang et al. (2023)](https://arxiv.org/abs/2311.12424) show parameter efficiency on particular in-context learning tasks, not a universal compression ratio for language tasks.
- [Saunshi et al. (2025)](https://arxiv.org/abs/2502.17416) compare shared and independent depth and find parameter-efficient performance on tested reasoning tasks. First separate the budgets being compared:

$$D_{\text{shared}}=D_{\text{untied}}=kL,\qquad P_{\text{shared block}}=kP_{\text{layer}},\quad P_{\text{untied block}}=kLP_{\text{layer}}$$

This is parameter accounting at fixed width, excluding extra components, not an accuracy formula. The study also finds tradeoffs in perplexity and some memorization tasks. It does not establish that reasoning depends only on depth rather than parameters.

<!-- paper-figure:looped-mechanism -->

## How it relates to CoT

Chain-of-thought adds computation by **generating intermediate tokens**. With KV caching, each new position traverses the network without recomputing the whole prefix. Looping instead updates **hidden states**, without requiring an output token per loop. The two can coexist, and a visible CoT is not automatically a faithful account of the model's decision. Generating more tokens is the serial axis of test-time compute; for how it compares with the parallel axis of sampling many answers, see [Reasoning models and test-time compute](../../11-frontier/test-time-compute/README.en.md).

Saunshi et al. give an existence construction in which a looped model simulates fixed-length CoT, with added width, heads, and placeholder tokens. This does not make arbitrary pretrained weights equivalent to CoT when repeated, or guarantee that training finds the constructed parameters.

## Where it stops

Extra computation can help use available information, but supplies no new external facts. If the manager's university is missing, more loops do not look it up; retrieval or additional input is needed. Ouro's synthetic knowledge-capacity result is a measurement, not a theorem for all models; see [costs and limits](costs.en.md).
