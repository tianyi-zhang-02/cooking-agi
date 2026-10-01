# Can the law of large numbers explain kindness?

[中文](kindness-and-probability.md) · **English**

A slightly ridiculous thought: can probability theory explain kindness? If helping has a negative expected return today, does doing it often enough make “what goes around comes around” work? Add a Markov model and suddenly it sounds respectable.

We can play with that. But **this is a mathematical analogy, not a proof that good people get rewarded. Every number below is an assumption, not a finding about human behavior.**

## 1 · The law of large numbers does not process karmic refunds

Suppose each favor costs you 1 point. There is a 20% chance of receiving 3 points in return, and otherwise nothing. Counting only this personal balance, the expected net payoff is:

$$\mathbb{E}[X]=0.2\times 3-1=-0.4.$$

For independent, identically distributed outcomes with a finite expectation, the law of large numbers says the sample average approaches that expectation. We mean convergence in probability here, not an exact guarantee for every batch. [MIT's lecture on the law of large numbers](https://math.mit.edu/~sheffield/2019600/Lecture29.pdf)

So **a negative expectation doesn't turn positive just because you repeat it**. In this model, the average heads toward −0.4. Receiving nothing the last few times doesn't mean you're due a reward next time, either.

The universe isn't necessarily keeping a reimbursement ticket open for you. At least, the law of large numbers doesn't expose that API.

## 2 · Maybe the ledger only counts what comes back to me

You spend 10 minutes saving someone an afternoon. A ledger that only asks whether they repay you might record a loss. But their afternoon was still saved.

If their well-being matters to me, it belongs among the things I value. That's a statement about my values, not a proof that my financial return has become positive—and not a reason to invent an upside to every bad experience.

There's an even simpler possibility: I know this won't come back to me, I can afford it, and I want to help anyway. Buying a friend a coffee doesn't always need an earnings forecast attached.

**Something can be worth doing without being a profitable transaction.**

## 3 · Enter Markov: could this interaction change the next one?

The first model treats favors as unrelated, one-off events. If we'll keep interacting, though, what happens today might affect how willing we are to cooperate tomorrow.

Let's compress the relationship into two states: **N, keeping to ourselves; C, helping each other.** These are interaction states, not categories of good and bad people or a trust score.

<figure>
<svg viewBox="0 0 480 240" role="img" aria-labelledby="kindness-states-title" style="display:block;width:100%;max-width:36rem;margin:1.5rem auto;color:var(--ink-soft)">
<title id="kindness-states-title">Two hypothetical states: keeping to ourselves and helping each other. a is the probability of entering mutual help; b is the probability of leaving it.</title>
<defs><marker id="kindness-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="currentColor"/></marker></defs>
<rect x="14" y="85" width="180" height="58" rx="14" fill="var(--panel)" stroke="currentColor"/>
<rect x="286" y="85" width="180" height="58" rx="14" fill="var(--panel)" stroke="currentColor"/>
<g fill="currentColor" font-family="sans-serif" font-size="18" text-anchor="middle"><text x="104" y="120">N · Keeping apart</text><text x="376" y="120">C · Mutual help</text><text x="240" y="44">a · Start cooperating</text><text x="240" y="194">b · Step back</text><text x="240" y="232" font-size="14">Stay in the same state: 1−a for N, 1−b for C</text></g>
<path d="M194 101Q240 20 286 101M286 132Q240 211 194 132" fill="none" stroke="currentColor" stroke-width="1.5" marker-end="url(#kindness-arrow)"/>
</svg>
</figure>

The Markov assumption is that the current state is enough to describe next-step probabilities without separately tracking the entire history. Real relationships are much messier. This is a sketch to help us think.

Let's invent two sets of parameters. Suppose helping within limits makes mutual help easier to establish and maintain. The comparison policy isn't a bad person; it just involves less active investment.

| Entirely hypothetical policy | a: apart → mutual help | b: mutual help → apart | Long-run share in mutual help |
| --- | --- | --- | --- |
| Less active investment | 0.10 | 0.30 | 25% |
| Help within limits | 0.30 | 0.20 | 60% |

**The 25% and 60% come from parameters we invented. They aren't evidence that kindness works.** Whether helping actually changes these probabilities is the question, not something we get to assume and then claim to have proved.

<details markdown="1">
<summary>Show the calculation—and when the comparison goes the other way</summary>

Order the states N, C. Rows represent the current state and columns the next state:

$$P=\begin{pmatrix}1-a&a\\b&1-b\end{pmatrix}.$$

For $0<a,b<1$, this two-state chain approaches a stationary distribution. At stationarity, the flow from N to C equals the reverse flow: $(1-\pi_C)a=\pi_C b$. Therefore:

$$\pi_C=\frac{a}{a+b}.$$

Now suppose state C brings 2 points of personal benefit per round and N brings 0. The helping policy has a fixed extra cost $c$ each round; the comparison policy has no extra cost. These are long-run average net payoffs, not promises over a finite stretch:

$$\bar r_0=2\times0.25=0.5,\qquad \bar r_1=2\times0.60-c=1.2-c.$$

- At $c=0.4$, the new policy yields 0.8 versus the baseline's 0.5.
- At $c=0.8$, it yields 0.4, below the baseline. **A positive payoff isn't the same as outperforming the alternative.**
- If a and b don't change at all, paying an extra cost only worsens this personal balance.

Under these assumptions, the new policy has a higher long-run average only when $c<0.7$. Change the assumptions and the answer changes. Early interactions don't necessarily resemble the stationary distribution, either.

The first example assumed independent outcomes. This one has state dependence, so we can't simply reuse the same i.i.d. assumption. With a policy fixed, we're comparing Markov chains. Choosing actions such as helping, declining, or leaving based on the state, with rewards attached, leads us toward a Markov decision process (MDP). [Standard definitions of fixed policies and value functions](https://gradml.mit.edu/reinforcement/value_bellman/)

</details>

## 4 · Don't remove the decline button from the model

What I like about the analogy isn't that it calculates an obligation to be kind. It separates immediate returns, future interactions, what I value, and what I can afford.

If repeated effort meets no mutual respect, calling it “long-term thinking” doesn't erase the cost. Declining an unreasonable request, doing less, or leaving are actions too. **A model without a no button won't tell you much about boundaries.**

My version would be: offer some goodwill, pay attention to what happens, and keep the freedom to say no. Not because the universe owes me a payout, but because I'd like the environment I share with others to be a little better. When I can't help, “being a good person” shouldn't become a way to pressure myself.

Math can separate the questions. It doesn't choose our values for us :)

## I'd like to hear your side

Is there a small kindness you're happy to offer even when you expect nothing back? How do you decide when to help and when to stop?

No identifying details needed. I'm curious about the measure you use.
