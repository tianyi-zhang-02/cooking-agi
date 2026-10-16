# RLHF: mapping ordinary RL onto a language model

[中文](rl-for-language-models.md) · **English**

> Reading time: ~2 min · Level: core · Last reviewed: 2026-10

A language model only predicts the next token each time, so why can it be trained with reinforcement learning? Because one generation can be read as a sequence of decisions: at each step the model sees the current prefix and picks a token, and at the end a reward model, a test, or an environment scores the whole result. Supervised learning usually supplies a target answer. Reinforcement learning (**RL**) instead evaluates the outcome of a sequence of behavior. The model can learn the score of a rollout without being told which individual step caused it or what the correct replacement action was.

## Borrow a few RL terms first {#rl-vocabulary}

In RL, an agent in **state** $s_t$ picks an **action** $a_t$ according to its **policy** $\pi_\theta$; the environment's dynamics then produce the next state $s_{t+1}$ and a **reward** $r_t$. One transition is written $(s_t,a_t,r_t,s_{t+1})$, and many steps in a row form a trajectory:

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots).
$$

What the agent actually sees is an observation, which need not be the complete state; for states, observations, dynamics, and the Bellman recursion, see [MDPs and Bellman](../deep-rl/mdp-bellman.en.md). Return, value, Q, and advantage are quantities constructed or estimated for learning, not feedback the environment hands over; the [next note](value-and-advantage.en.md) defines each one.

## Mapping the pieces onto a language model

Language generation can be written directly as sequential decision-making:

| RL concept | In a language model |
| --- | --- |
| **Agent** | the language model being trained |
| **State** $s_t$ | the prompt plus generated prefix $(x, y_{<t})$ |
| **Action** $a_t$ | the next token $y_t$ |
| **Policy** $\pi_\theta(a_t\mid s_t)$ | the model's softmax next-token distribution |
| **Trajectory** $\tau$ | the token sequence from the start to the end of a response |
| **Reward** $r_t$ | scalar feedback from a Reward Model, verifier, or real environment |

The Actor is not a separate controller wrapped around the language model: **the language model itself is the policy**. At state $s_t=(x,y_{<t})$, its next-token probabilities define

$$a_t=y_t\sim\pi_\theta(\cdot\mid x,y_{<t}).$$

Appending that token creates the next state. In plain text generation this transition is deterministic token concatenation; tool-using or interactive agents also receive search results, execution outputs, or other observations from the environment.
