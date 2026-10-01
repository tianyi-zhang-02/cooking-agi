# Does kindness add up?

[中文](kindness-and-probability.md) · **English**

I wanted a mathematical argument for kindness. If helping someone costs me something and brings nothing back this time, surely doing it often enough will let the law of large numbers even things out?

The first step is already a problem: averaging a negative expectation doesn't turn it positive. Not the answer I was hoping for.

The numbers below are made up to play with this idea. This isn't a mathematical proof that kindness gets rewarded.

## 1 · Start with the losing calculation

Suppose each favor costs 1 point. There's a 20% chance of receiving 3 points back, and otherwise nothing. Counting only your own costs and returns, the expected net payoff is:

$$\mathbb{E}[X]=0.2\times 3-1=-0.4.$$

For independent, identically distributed outcomes with a finite expectation, the law of large numbers says the sample average converges in probability to that expectation. With more trials, the average becomes increasingly likely to be close to −0.4—not suddenly positive. [More on the law of large numbers: MIT lecture notes](https://math.mit.edu/~sheffield/2019600/Lecture29.pdf)

Doing it more often just makes the average loss more likely to settle near −0.4. Getting nothing the last few times doesn't improve my chances next time, either. The law of large numbers isn't in charge of returning favors.

## 2 · Would I do this before buying a friend coffee?

I spend 10 minutes saving someone an afternoon. In the calculation above, if they give me nothing back, I've lost out.

But maybe saving them the afternoon was what I wanted. I wouldn't buy a friend a coffee and chase them afterward for a return.

The calculation left out that I might be happy to do it. The time and money are still spent, of course. If I want to spend them and can afford to, that's fine with me.

## 3 · What if we'll meet again?

The first calculation treated favors as unrelated events. Friends, colleagues, and neighbors keep seeing each other: if you help me today, I may be more willing to help next time.

That calls for a different model, one in which how we interact can change. We'll use two states: **N, keeping to ourselves; C, helping each other.** These describe interactions, not categories of good and bad people.

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

The Markov model makes a strong simplification here: use how we're interacting now to estimate what happens next, without tracking every earlier encounter. Real relationships are messier, but this gives us something simple to work with.

Suppose offering some help makes a habit of mutual support easier to build and maintain. We'll invent two sets of parameters to explore that. Taking less initiative is another approach, not a judgment about who's a good person.

| Entirely hypothetical approach | a: apart → mutual help | b: mutual help → apart | Long-run share in mutual help |
| --- | --- | --- | --- |
| Less active investment | 0.10 | 0.30 | 25% |
| Help within your limits | 0.30 | 0.20 | 60% |

**The 25% and 60% come from assumptions, not observations.** Does helping actually change these probabilities? That's the question. We can't put the benefit into the parameters and then say the math proved it.

You can skip the formulas and read on. The calculation is here if you'd like to see where the numbers come from—and when helping costs more than it brings back.

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

## 4 · I'd still like to help sometimes

After all that, I haven't proved that kindness always pays. Fair enough. Something I can reverse by changing a few parameters probably shouldn't become a rule for living.

I'd still like to offer small favors when I can. If it saves someone trouble and I'm happy to do it, I'll do it. If we get to know each other better and help each other later, even better.

If someone keeps asking without so much as a thank-you, or I'm already too busy, I'd like to say no. Having helped before shouldn't mean I've signed up to do it forever.

That's probably enough math. I'll try not to bring this table out when I'm actually buying a friend coffee.
