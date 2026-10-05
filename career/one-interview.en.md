# Career: a retrospective on one MLE interview

[中文](one-interview.md) · **English**

> Reading time: ~3 min · Last reviewed: 2026-10

I wrote this review in September 2026, shortly after returning from my internship and starting to prepare for full-time roles. An MLE interview was still fresh in my memory, so I wrote it down.

The project discussion went reasonably well. The ML fundamentals didn't. I had studied some of the material before, but explaining it on the spot was different from remembering that I used to know it. This is about my performance, not the specific questions or company process.

## How prepared I actually was

I had done some LeetCode during the previous internship search, but I wasn't especially strong at it. I hadn't reviewed ML fundamentals or system design thoroughly, and had barely practiced ML coding systematically.

The interview covered more than the area I'd been working on recently. The interviewer was helpful and offered hints when I got stuck. Sometimes I still couldn't get there, which showed how much preparation I was missing.

## The project was easier to discuss concretely

We discussed recommendation retrieval from my internship: the problem, the model and architecture choices, and how training resources, compute cost, and serving requirements affected them. I also talked about observations from the experiments and how I interpreted them.

Having work I'd actually done to talk about felt more natural than delivering a rehearsed “project story.” I still needed to distinguish observations from hypotheses, and my contribution from the rest of the team's work.

When asked whether it was deployed, I described the actual delivery scope: that architecture hadn't gone to production. There wasn't a reason to talk around it.

## A moment I didn't handle well

We got onto a direction that had been getting attention. I described problems we'd encountered while exploring it, and why we hadn't continued with it under our constraints and timeline. The interviewer said they already had an approach working in their setting.

I paused and didn't respond very well.

Afterward, I realized what I meant was “we didn't get it to work under those conditions,” not “this direction doesn't work.” The data, compute budget, latency, and content freshness requirements could all differ. Asking how they handled those trade-offs would have made for a much better conversation.

That was a problem with my explanation. I had the assumptions in my head but hadn't said them, so the interviewer could reasonably hear a different conclusion.

## The fundamentals needed work

When we moved to broader ML fundamentals, including some classical models, I started struggling. Having studied something before and rarely used it since wasn't enough to bring it back in an interview.

This was my first interview with a large Chinese tech company. I'd heard that the questions could be broad, but hadn't prepared thoroughly enough. One experience doesn't describe every company. It did tell me that reviewing only the topics closest to my project wasn't enough for the next round.

I want to get past recognizing an answer when I see it. I need to explain the basic idea, when I'd use it, and where it can fail without looking at the notes. Otherwise, a differently phrased question is likely to expose the same gap.

## Explaining technical work in Chinese takes practice too

I learned coding and ML mostly in English. This discussion was mainly in Chinese, so I sometimes had to translate a term on the spot. I knew what I wanted to say but took a moment to get it out.

That wasn't the main reason I struggled, but it's something I can practice separately. When reviewing a concept, I can try explaining it in both languages. Keeping familiar terms in English is fine; the explanation around them needs to flow.

## What I took from it

The role wasn't a particularly close match for my experience, and I was underprepared. The useful follow-ups were fairly clear: review a broader range of fundamentals, state the assumptions before discussing a design, and practice technical explanations in Chinese.

I also interviewed for two RS roles around that time and may add those reflections when I've put them together. For now, remembering the gaps this interview revealed is more useful than giving myself an overall grade.
