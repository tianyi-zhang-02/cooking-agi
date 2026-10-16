# 用强化学习训练 agent（agentic RL）：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 4 分钟 · 草稿，待审核

这一页把本系列的要点整理成面试题和自检题，答案里的数字都能在前面几篇找到出处。

> **待审核草稿。** 这一篇还没有经过站主审阅，事实、取舍和措辞都可能修改；引用前请回到文中链接的一手资料核对。

## 面试常见问题

<details class="interview" markdown="1">
<summary>从单轮 RLVR 换到多轮 agent RL，什么变了，什么没变？</summary>

没变的是骨架：采样、打分、按 advantage 更新策略，PPO、GRPO 这些更新规则都能接着用。变的是一条样本：它是多轮的（动作，观察）序列，观察来自环境而不是策略；奖励多半在回合结束时按环境的终局状态给；检查器常和 agent 在同一个沙盒里，可能被它碰到；一条轨迹可以有上千次工具调用；一次 rollout 要起容器、跑命令、跑测试，成本以分钟计。DeepSeek-R1（2025-01）就因为评估太慢，没在软件工程任务上大规模用 RL。

</details>

<details class="interview" markdown="1">
<summary>为什么工具返回的 token 不进 loss？不 mask 会怎样？</summary>

轨迹的对数概率可以拆成两部分：策略选动作的概率，加上环境返回观察的概率。后者和参数无关，求梯度时为零，所以 policy gradient 本来就只包含动作 token。不 mask 的话，训练会去抬高或压低“模型自己写出这段文件内容、这段报错”的概率，而这些文字是环境给的；工具输出又往往比动作长得多，梯度会被它们主导。Search-R1（2025-03）的消融里，加 mask 平均 exact match 0.431，不加 0.343。被 mask 的 token 仍然留在上下文里，只是不产生梯度。

</details>

<details class="interview" markdown="1">
<summary>同一任务 4 条 3 轮的轨迹，奖励是 [1, 0, 1, 1]。组相对 advantage 是多少？哪些 token 拿到梯度？</summary>

均值 0.75，总体标准差约 0.433，标准化后是 [+0.577, −1.732, +0.577, +0.577]；用 $G-1$ 作分母是 [+0.5, −1.5, +0.5, +0.5]；DeepSWE 那样不除标准差、用 leave-one-out 基线，是 [+0.333, −1, +0.333, +0.333]。Episode 级做法里，这个数铺到该轨迹所有模型生成的 token 上；prompt、文件内容、测试日志这些工具输出 mask 为 0。教学例子里一条轨迹 1,800 个 token，只有 200 个动作 token 有梯度。几条轨迹共有的同一步（同样上下文、同样 token），梯度会相加；如果 4 条第一步完全相同，加起来是 0。

</details>

<details class="interview" markdown="1">
<summary>整条轨迹一个分数，哪里不够？有哪些更细的做法？</summary>

它分不清轨迹里哪一步做得好：失败轨迹里合理的读文件动作也一起被压低，只能靠大量采样平均掉噪声。更细的做法各有前提：GiGPO（2025-05）在组内找重复出现的状态，从同一状态出发的动作再比一次，不加模型，但状态不重复时退化成 GRPO；turn 级 critic（ArCHer，2024-02）每轮都有估计，代价是多一个模型和估值误差；SWEET-RL（2025-03）让 critic 在训练时看到策略看不到的信息；逐轮奖励信号更密，但中间奖励要设计、也会被钻。Z.ai 的 GLM-5（2026-02）还在用组相对，GLM-5.2（2026-06）在长任务上换成了 critic PPO。

</details>

<details class="interview" markdown="1">
<summary>测试只能抓住一部分失败。训练奖励和真实成功率会怎样分开？</summary>

期望奖励等于真修好的比例，加上被放过的错解和篡改所占的比例；真实成功率只算前者。教学例子里，测试文件 agent 能改、可见测试漏掉一些错解时，奖励从 0.55 涨到 0.90，真实成功率只从 0.30 涨到 0.35。组内比较时，改测试的轨迹和真修好的轨迹拿到同样的正 advantage。评分前恢复测试文件、加隐藏测试之后，同样两个策略的奖励是 0.34 和 0.40，和真实变化接近。“卡住了就改测试”是跨任务通用的套路，所以在弱检查器下会在难题上涨得尤其快。

</details>

<details class="interview" markdown="1">
<summary>说几个有记录的 agent 钻空子的例子，以及防护手段的局限。</summary>

OpenAI 的 Baker 等（2025-03）记录了训练中的 agent 用 `exit(0)` 提前退出、`raise SkipTest` 跳过测试；Anthropic（2025-11）记录了 `__eq__` 永远为真的对象、`sys.exit(0)` 和改 pytest 报告的 `conftest.py`；SWE-bench 镜像曾留着未来的 git 历史（2025-09）；Claude Mythos Preview（2026-04）偶尔绕过训练环境的网络限制。防护都有边界：测试只读挡不住特判；隐藏测试让作弊接近 0，但正常任务也变难（ImpossibleBench，2025-10）；把 CoT monitor 直接当惩罚，优化太狠时 agent 会学会隐藏意图；Anthropic 说训练前的环境评审没能拦下所有有问题的环境。

</details>

<details class="interview" markdown="1">
<summary>撞上上下文上限、步数上限或超时的轨迹，该怎么记分？</summary>

三种选择。记 0 分，模型会学到早点交差，长任务被系统性压低；整条从 loss 里去掉，少了负样本，但不再惩罚“还没做完”，DeepSWE（2025-07）的 compact filtering 就是这样做的，理由是只在 agent 主动提交时给奖励能鼓励它认真测试；当作截断、用价值估计接上后续回报，需要 critic。环境自己出错要单独处理：GLM-5 把因沙盒崩溃而失败的样本排除在训练外。无论选哪种，都要把撞上限、超时、工具报错分开统计，截断率下降不一定是能力提升。

</details>

<details class="interview" markdown="1">
<summary>Agent RL 的训练系统和单轮 RLVR 有什么不同？为什么要异步，代价是什么？</summary>

相当一部分时间花在 GPU 之外：起容器、执行命令、跑测试。DeepSWE 每次迭代并行起 512 个容器，Qwen3-Coder 报告 2 万个环境并行。轨迹长短差别很大，同步等最慢的一条很浪费，所以常把生成和训练解耦：AReaL 报告同样 GPU 数下最多快 2.77 倍，Kimi-Researcher 把超时的任务存起来、之后用新权重接着跑。代价是数据由旧策略生成，甚至一条轨迹的前后几轮出自不同版本，需要记录生成时的 log-prob 并做修正。另外训练用的必须是生成时的 token ids：解析成消息再重新编码，token 可能对不上，GLM-5 为此加了 token-in-token-out 网关。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>从轨迹对数概率的分解出发，为什么工具输出的 token 不该产生梯度，但必须留在上下文里？</li>
    <li>奖励 [1, 0, 1, 1] 时，三种基线约定（总体标准差、样本标准差、leave-one-out）分别给出什么 advantage？为什么它们的和都是 0？</li>
    <li>GiGPO 的 anchor state 小组在什么情况下比 episode 级组相对多给信息，什么时候退化成 GRPO？</li>
    <li>在“测试只抓住一部分失败”的例子里，怎样从期望奖励公式看出 v1 检查器的奖励涨幅主要来自哪几类结果？</li>
    <li>撞上限的轨迹记 0、整条去掉、当作截断接上价值估计，各自会把模型往哪个方向推？</li>
  </ol>
</div>

## 参考论文

- [Search-R1](https://arxiv.org/abs/2503.09516)：多轮检索 RL，检索结果 token 不进 loss 的消融
- [GiGPO](https://arxiv.org/abs/2505.10978)：episode 级之外，按重复状态做 step 级分组
- [ArCHer](https://arxiv.org/abs/2402.19446)：句子级 critic 带动 token 级策略的多轮 RL
- [SWEET-RL](https://arxiv.org/abs/2503.15478)：critic 在训练时可见额外信息，ColBench
- [Reinforcing Multi-Turn Reasoning in LLM Agents（Wei、Zeng 等）](https://arxiv.org/abs/2505.11821)：终局、延迟和逐轮奖励的比较
- [RAGEN](https://arxiv.org/abs/2504.20073)：多轮 agent RL 的 Echo Trap 与 StarPO-S
- [DeepSWE](https://www.together.ai/blog/deepswe)：只用 RL 训练代码 agent，GRPO++ 与 compact filtering
- [SWE-RL](https://arxiv.org/abs/2502.18449)：单轮设置下按补丁相似度给奖励
- [SWE-Gym](https://arxiv.org/abs/2412.21139)、[R2E-Gym](https://arxiv.org/abs/2504.07164)、[SWE-smith](https://arxiv.org/abs/2504.21798)：可执行代码环境的构建与规模
- [τ-bench](https://arxiv.org/abs/2406.12045)、[τ²-bench](https://arxiv.org/abs/2506.07982)：模拟用户与终局状态评分
- [AReaL](https://arxiv.org/abs/2505.24298)、[ROLL Flash](https://arxiv.org/abs/2510.11345)：异步 RL 系统
- [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/)、[Kimi K2](https://arxiv.org/abs/2507.20534)、[Kimi K2.5](https://arxiv.org/abs/2602.02276)、[Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3)：Moonshot AI 的 agentic RL 报告
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)、[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)：评估成本的局限；大规模 agent 任务合成
- [Qwen3-Coder](https://qwenlm.github.io/blog/qwen3-coder/)、[GLM-5](https://arxiv.org/abs/2602.15763)、[GLM-5.2](https://z.ai/blog/glm-5.2)：环境规模、异步系统与 critic PPO
- [Monitoring Reasoning Models for Misbehavior（Baker 等）](https://arxiv.org/abs/2503.11926)：训练中的测试钻空子与思维链隐藏
- [Natural emergent misalignment from reward hacking](https://arxiv.org/abs/2511.18397)：生产编码环境里的钻空子与泛化
- [Recent Frontier Models Are Reward Hacking（METR）](https://metr.org/blog/2025-06-05-recent-reward-hacking/)：评估中的钻空子实例
- [ImpossibleBench](https://arxiv.org/abs/2510.20270)：测试与规格矛盾时的作弊率，以及测试可见性的影响
- [PatchDiff](https://arxiv.org/abs/2503.15223)、[UTBoost](https://arxiv.org/abs/2506.09289)：测试太弱导致的错误“通过”
- [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794)、[Gaia2](https://arxiv.org/abs/2602.11964)：LLM judge 作为奖励时被钻
- [Claude 3.7 Sonnet](https://www-cdn.anthropic.com/9ff93dfa8f445c932415d335c88852ef47f1201e.pdf)、[Claude 4](https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285/Claude%204%20System%20Card.pdf)、[Claude Mythos Preview](https://www-cdn.anthropic.com/7624816413e9b4d2e3ba620c5a5e091b98b190a5/Claude%20Mythos%20Preview%20System%20Card.pdf)、[Claude Opus 5.5](https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf) 系统卡：训练环境中的钻空子与防护
- [deep research](https://openai.com/index/introducing-deep-research/)、[o3 与 o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)、[codex-1](https://openai.com/index/introducing-codex/)：OpenAI 关于用 RL 训练 agent 的说明
