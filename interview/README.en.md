# Tech interviews: how to use these notes

[中文](README.md) · **English**

> Reading time: ~2 min · Last reviewed: 2026-09

> **Read this first**: interview formats and emphasis change quickly, so this is time-sensitive. Only **public fundamentals** are collected here: no interview questions from any company and nothing about any company's process.

Technical interviews look like a grab bag, but most of them keep asking the same thing: do you really understand the mechanism, can you go from the formula to code, and can you explain why. These notes are ordered that way.

## Algorithm questions

- [How I practise LeetCode](leetcode.en.md) — what to practise, my routine, the list of question types, and how far to take it

## System design

- [System design](system-design.en.md) — no fixed formula, it is about trade-offs; public resources, plus a recommendation example

## A quick run-through before the interview

These pages cover the same knowledge as the “Learn” side, presented differently: there it is explained in full and you can play with it; here only the questions and answers are left, for the night before. The overlap is deliberate.

- [Quick review: run through the questions, block by block](questions.en.md): interview and self-check questions pulled automatically from the notes; open one to see the answer, or draw one at random
- [Transformer follow-ups, quick review](transformer-followups.en.md): the follow-ups on KV cache, MQA / GQA / MLA, RoPE, MoE and FlashAttention, one question, one answer

## Suggested order

1. [Interview basics: most of them ask the same thing](../00-foundations/interview-basics.en.md): get the through-line first
2. [The seven things you may be asked to write on a whiteboard](../00-foundations/hand-write-kit.en.md): check you can write them without looking anything up
3. [ML interview mathematics: probability to estimators](../00-foundations/ml-math-interview.en.md): the probability and statistics line
4. [The Transformer, interactively](../00-foundations/transformer-lab.en.md): for follow-ups on the KV cache, GQA / MLA, RoPE, MoE and FlashAttention, test yourself on figures you can drag and click

## When the follow-ups go into detail

- [Multi-head attention: from equations to implementation](../00-foundations/core/multi-head-attention.en.md)
- [Normalisation: BatchNorm, LayerNorm, and RMSNorm](../00-foundations/core/normalization.en.md)
- [Decoder-only: autoregressive generation](../00-foundations/core/decoder-only.en.md)
- [Post-training overview](../05-post-training/README.en.md): SFT, preference learning, RLHF and what came after

To order things by role, see [the roadmap on the home page](../README.en.md).
