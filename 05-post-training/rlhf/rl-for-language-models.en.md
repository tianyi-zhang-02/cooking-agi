# RLHF: mapping ordinary RL onto a language model

[中文](rl-for-language-models.md) · **English**

> Reading time: ~3 min · Level: core · Last reviewed: 2026-09

The cleanest way to understand RL is to separate two layers: the **environment layer** describes how the world changes, while the **learning layer** describes how an agent improves its policy from experience. Supervised learning usually supplies a target answer. Reinforcement learning (**RL**) instead evaluates the outcome of a sequence of behaviour. The model can learn the score of a rollout without being told which individual step caused it or what the correct replacement action was.

## Environment layer: how the world works

| Component | Question it answers | Driving example |
| --- | --- | --- |
| **State** $s_t$ | What is the complete situation of the world? | position, speed, nearby traffic, and road conditions |
| **Observation** $o_t$ | What information can the agent actually see? | camera, LiDAR, and speedometer readings |
| **Policy** $\pi_\theta(a\mid o)$ | How should actions be distributed from this input? | assign braking high probability at a red light |
| **Action** $a_t$ | What did the agent actually do? | steer, accelerate, or brake |
| **Dynamics** $P$ | How does an action change the world? | how braking changes speed and position |
| **Reward** $r_t$ | What score did this step or outcome receive? | positive for safe progress, negative for a collision |

One transition or experience is commonly written as

$$
(s_t,a_t,r_t,s_{t+1}),
$$

and a sequence of them forms a trajectory:

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots).
$$

**State need not equal observation.** State is the complete information needed to determine how the world evolves; observation is the signal available to the agent. Textbook MDPs often assume $o_t=s_t$. Robots, games, and conversational agents are often partially observed and closer to POMDPs. Their policies must act from observations or a summary of history rather than an inaccessible true state.

## Learning layer: how policy behaviour is judged

| Quantity | Question it answers |
| --- | --- |
| Immediate reward $r_t$ | “What feedback did I receive now?” |
| Return $G_t$ | “What did this rollout actually receive from now onward?” |
| Value $V^\pi(s_t)$ | “What do I normally expect if the current policy continues from here?” |
| Q-value $Q^\pi(s_t,a_t)$ | “What do I expect after choosing this particular action here?” |
| Advantage $A_t$ | “Was this action better or worse than normal here?” |

The first layer produces experience; the second turns experience into a training signal. Reward is feedback from the environment or evaluator. Return, value, Q, and advantage are quantities constructed or estimated for learning.

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

Appending that token creates the next state. In plain text generation this transition is almost deterministic concatenation; tool-using or interactive agents also receive search results, execution outputs, or other observations from the environment.
