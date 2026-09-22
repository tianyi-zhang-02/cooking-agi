# Career: a retrospective on one MLE interview

[中文](one-interview.md) · **English**

> Reading time: ~4 min · Last reviewed: 2026-09

> **Read this first**: a lot of this is time-sensitive and may no longer hold by the time you read it; I am an international student, so it may not apply to green-card holders or US citizens; there are no interview questions here, and nothing about any company's specific process. The full note is in [Career: read this page first](README.en.md).

I just came back from my internship and started preparing for full-time roles. This is a retrospective on an MLE interview I finished recently, written while it was still fresh.

I have blurred a lot of the details, and please don't overfit to this: I have interviewed for quite a few roles and each one wants something different. Everyone's time and energy are limited, so not being perfectly prepared is normal. Try not to let it get to you.

## Where I started

A master's new grad. I did some LeetCode while looking for an internship last year, but ML fundamentals and system design have always been weak spots for me, my LeetCode is mediocre, and I had never practised ML coding systematically.

Compared with some of the US tech companies I had interviewed with, this one put more weight on ML fundamentals and project experience; overall it was a fairly generalised, standardised interview. The interviewer was kind and gave me plenty of hints when I got stuck.

## First half: the project

We talked about the recommendation retrieval work from my internship. I spent a while on:

- what the project actually did;
- why it was designed that way;
- which system and serving constraints we faced;
- how compute was allocated in training, and the cost trade-offs;
- what I observed in the experiments and what I made of it;
- why we ended up with an architecture that balanced quality against online serving cost.

The interviewer also asked whether it shipped, and I said honestly that the final scope did not go to production.

## The middle: a question I did not catch

We got onto a direction that has been getting a lot of attention lately. I brought up the problems we ran into while exploring it, and why we did not take that route given the system constraints and the project timeline. The interviewer said they already had an approach for it in their own setting.

I froze for a second and could not put a good answer together. Thinking about it afterwards, the better reply would have been: I don't think the approach is unworkable, it is that data distributions, system architectures, compute budgets, latency, freshness, and serving requirements differ between companies, so what fits differs too — and if I had turned it around and asked how they handled those trade-offs, the discussion would have been far more useful.

## Second half: ML fundamentals

I had never interviewed with a large Chinese tech company before; I had only heard they weigh fundamentals and projects more heavily. The breadth in this round really was wider than at some of the US tech companies I had seen, and classical ML models were in scope too.

And then, predictably, I fell apart (laughing). I had studied plenty of it before, but my day-to-day work is fairly narrow and I could not pull it back up on the spot. The interviewer kept offering hints; that must have been work for him too.

The main lesson from this round: **if you are preparing for MLE at a large Chinese tech company, go through ML fundamentals properly and broadly**, not only the parts closest to your current project. Classical ML can still be tested widely.

On the project part, my takeaway is that it is enough to explain the problem, the reasoning behind the design, the system constraints, and the trade-offs, as long as it holds together.

## One more thing: language

I learned coding and ML almost entirely in English, and this technical round was mostly in Chinese, so I had to switch terms between the two languages on the spot and my delivery stumbled a bit. It was not the main reason things went badly, but it is something to practise deliberately.

## In short

This role was not a particularly good match for my background, and I was underprepared, so a weak showing has no excuses. Back to putting in the work.

I interviewed for two RS roles and will keep updating this. Overall, I think managing your own state of mind matters most.
