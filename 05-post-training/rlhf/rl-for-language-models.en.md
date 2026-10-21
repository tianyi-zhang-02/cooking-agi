# RLHF: mapping ordinary RL onto a language model

[中文](rl-for-language-models.md) · **English**

> Reading time: ~4 min · Last reviewed: 2026-10

A language model only predicts the next token each time, so why can it be trained with reinforcement learning? Because one generation can be read as a sequence of decisions: at each step the model sees the current prefix and picks a token, and at the end a reward model, a test, or an environment scores the whole result. Supervised learning usually supplies a target answer. Reinforcement learning (**RL**) instead evaluates the outcome of a sequence of behavior. The model can learn the score of a rollout without being told which individual step caused it or what the correct replacement action was.

<span id="environment-layer-how-the-world-works"></span>

## Borrow a few RL terms first {#rl-vocabulary}

In RL, an agent in **state** $s_t$ picks an **action** $a_t$ according to its **policy** $\pi_\theta$; the environment's dynamics then produce the next state $s_{t+1}$ and a **reward** $r_t$. One transition is written $(s_t,a_t,r_t,s_{t+1})$, and many steps in a row form a trajectory:

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots).
$$

What the agent actually sees is an observation, which need not be the complete state; for states, observations, dynamics, and the Bellman recursion, see [MDPs and Bellman](../deep-rl/mdp-bellman.en.md). <span id="learning-layer-how-policy-behavior-is-judged"></span>Return, value, Q, and advantage are quantities constructed or estimated for learning, not feedback the environment hands over; the [next note](value-and-advantage.en.md) defines each one.

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

## Walk through one small problem {#worked-episode}

The prompt is “2 + 3 =”. The model generates `5`, then the end token `<EOS>`. A correct answer earns 1, and that score arrives only when the answer ends. The probabilities below are teaching numbers:

| Step $t$ | State $s_t$ | Action $a_t$ | $\pi_\theta(a_t\mid s_t)$ | Reward $r_t$ |
| --- | --- | --- | --- | --- |
| 0 | `2 + 3 =` | `5` | 0.8 | 0 |
| 1 | `2 + 3 = 5` | `<EOS>` | 0.9 | 1 |

The probability of the whole answer is the product of the step probabilities, $0.8\times0.9=0.72$. Taking logs turns the product into a sum, $\log0.8+\log0.9\approx-0.329$. This is how training code adds up a “sequence log-prob”:

$$
\pi_\theta(y\mid x)=\prod_t\pi_\theta(y_t\mid x,y_{<t}).
$$

Now look at the reward. Step 0 picked the right token, `5`, yet received 0 on the spot; the 1 lands on the last step. The model has to learn which step deserves that point: this is the credit-assignment problem of the [next note](value-and-advantage.en.md). Conversely, if step 0 had sampled `6`, every ending would score 0: one early token can decide the outcome of the whole answer.

## Where does this mapping strain? {#where-mapping-strains}

| Situation | What happens | Common response |
| --- | --- | --- |
| The tokenizer sets the action size | The same `15` is 1 token for one tokenizer and two tokens, `1` and `5`, for another, so the number of “steps” changes | Sequence log-prob is still the sum of the steps; per-token loss averages, length normalization, and per-token advantages change with the split, so check them with the actual tokenizer |
| Reward mostly arrives at the end | Intermediate tokens get no direct feedback | Spread it with a Critic / GAE, or treat the whole answer as a single action |
| Plain text generation has almost no external randomness | The next state is just the token appended | Only tools, retrieval, or multi-turn dialogue return content the model does not control |

Another way to write it treats the whole answer as **one** action: the state is just the prompt, the action is the entire answer $y$, and the reward is one number. RL calls this a contextual bandit. It uses the same probability as the token-level view, namely the product above. [RLOO (Back to Basics §2.2–2.3)](https://arxiv.org/html/2402.14740v1#S2.SS2) takes the whole-answer view; when you need step-by-step credit (a Critic, GAE, process rewards), the token-level view is easier to work with. Both views optimize the same expected reward; they differ in how the reward is spread over the tokens.
