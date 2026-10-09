import torch


def actor_loss(log_prob, advantage):
    if log_prob.ndim != 1 or log_prob.shape != advantage.shape or log_prob.numel() == 0:
        raise ValueError("log_prob and advantage must be matching nonempty vectors")
    return -(log_prob * advantage.detach()).mean()


def double_dqn_loss(online, target, states, actions, rewards,
                    next_states, terminated, gamma=0.99):
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a nonempty vector")
    if not rewards.is_floating_point():
        raise ValueError("rewards must be floating-point")
    if actions.shape != rewards.shape or terminated.shape != rewards.shape:
        raise ValueError("actions, rewards, and terminated must have shape [batch]")
    if actions.dtype != torch.long or terminated.dtype != torch.bool:
        raise ValueError("actions must be int64 and terminated must be boolean")
    if states.ndim != 2 or states.shape != next_states.shape or states.shape[0] != rewards.numel():
        raise ValueError("states and next_states must have matching [batch, state_dim] shapes")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    prediction = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_q = torch.zeros_like(rewards)
        continuing = ~terminated
        if gamma > 0 and continuing.any():
            successors = next_states[continuing]
            next_actions = online(successors).argmax(dim=1, keepdim=True)
            next_q[continuing] = target(successors).gather(1, next_actions).squeeze(1)
        expected = rewards + gamma * next_q
    return torch.nn.functional.smooth_l1_loss(prediction, expected)


def train_bandit(seed=7, updates=150, batch_size=128):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.nn.Parameter(torch.zeros(2))
    rewards = torch.tensor([1.0, 3.0])
    optimizer = torch.optim.SGD([logits], lr=0.1)
    for update in range(updates):
        probabilities = logits.softmax(dim=0)
        actions = torch.multinomial(probabilities.detach(), batch_size,
                                    replacement=True, generator=generator)
        baseline = (probabilities.detach() * rewards).sum()
        advantage = rewards[actions] - baseline
        loss = actor_loss(logits.log_softmax(dim=0)[actions], advantage)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return logits.detach().softmax(dim=0)


def run_checks():
    logits = torch.zeros(2, requires_grad=True)
    advantage = torch.tensor([1.0], requires_grad=True)
    loss = actor_loss(logits.log_softmax(dim=0)[1:2], advantage)
    loss.backward()
    assert logits.grad[1] < 0 < logits.grad[0]
    assert advantage.grad is None

    with torch.random.fork_rng():
        torch.manual_seed(4)
        online = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        states = torch.ones(2, 3)
        actions = torch.tensor([0, 1])
        rewards = torch.tensor([1.0, 2.0])
        ended = torch.tensor([True, False])
        double_dqn_loss(online, target, states, actions, rewards,
                        states, ended).backward()
        assert all(parameter.grad is None for parameter in target.parameters())
        assert online.weight.grad is not None

    action = torch.tensor(0.2, requires_grad=True)
    critic_value = -(action - 0.7).square()
    (-critic_value).backward()
    assert action.grad < 0
    probabilities = train_bandit()
    assert probabilities[1] > 0.9
    print("PASS: Actor sign, detached advantage, detached TD target, action gradient")
    print("Toy bandit probabilities:", probabilities.tolist())


if __name__ == "__main__":
    run_checks()
