# Agent RL：环境与奖励从哪里来

**中文** · [English](environments-and-rewards.en.md)

> 阅读时间：约 12 分钟 · 最近审阅：2026-10

想用 RL 训练一个修代码的 agent，首先要准备几千个任务：每个任务都要有能复现的沙盒，以及一套判断“修好了没有”的检查，再来考虑算法。单轮 RLVR 里，检查器在模型外面看答案；agent 的检查器往往就在它能操作的沙盒里，测试文件、git 历史、网络都可能被它访问。

环境决定 agent 有机会学习哪些行为，奖励则决定训练会鼓励哪些行为。检查器漏掉的失败，不能指望 RL 自动纠正：只要某种做法能通过检查，就可能被奖励强化。因此，训练奖励和真实成功率要分开测量，环境本身也要像代码一样接受测试。

## 一个 agent 环境要提供什么

和 [Gymnasium](https://gymnasium.farama.org/) 一样，环境至少要能重置和执行一步：重置时给出任务和初始状态，比如某个 commit 下的仓库；每一步接收 agent 的工具调用，返回观察；回合结束时给奖励。具体到 LLM agent，还要写清这几件事：

- **任务和初始状态**：任务说明、仓库或数据库的初始内容、哪些信息 agent 看不到；
- **动作空间**：能用哪些工具（shell、编辑、搜索、浏览器），有没有网络，能不能改测试；
- **检查器**：结束时看什么、怎么判，能不能被 agent 的动作改变；
- **预算**：最多几轮、多少 token、每个命令多长时间，超了算什么；
- **隔离**：一条轨迹的副作用不会漏到别的轨迹、评分进程或外部世界。

<!-- paper-figure:osworld-infra -->

| 环境类型 | 状态在哪里 | 奖励通常怎么来 | 公开例子 | 要先防什么 |
| --- | --- | --- | --- | --- |
| 带测试的代码仓库 | 容器里的仓库和文件系统 | 修复前失败、修复后通过的测试，加上原本就通过的测试 | [SWE-Gym](https://arxiv.org/abs/2412.21139)（2024-12，11 个仓库的 2,438 个任务）；[R2E-Gym](https://arxiv.org/abs/2504.07164)（2025-04，约 8.1k 个）；[SWE-smith](https://arxiv.org/abs/2504.21798)（2025-04，128 个仓库的 50,137 个） | 测试覆盖不全；改测试；从 git 历史里读到答案 |
| 搜索与浏览 | 检索库或真实网络，加上 agent 的上下文 | 最终答案和标准答案比对：[Search-R1](https://arxiv.org/abs/2503.09516) 用 exact match，[DeepResearcher](https://arxiv.org/abs/2504.03160) 用 F1 | OpenAI deep research（2025-02-02）；Kimi-Researcher（2025-06-20） | 答案对、依据错；网上直接找到题目的答案 |
| 模拟用户 | 业务数据库，加上一个由 LLM 扮演的用户 | 结束时数据库是否和目标状态一致，回复里是否给了必要信息 | [τ-bench](https://arxiv.org/abs/2406.12045)（2024-06）；[τ²-bench](https://arxiv.org/abs/2506.07982)（2025-06），用户也能操作工具 | 什么都不做也可能“通过”；模拟用户本身前后不一致 |
| 工具 API | 真实工具，或模拟工具行为的模型 | 规则、测试或 rubric | [Kimi K2](https://arxiv.org/abs/2507.20534)（2025-07）：3000 多个真实 MCP 工具、2 万多个合成工具；[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)（2025-12）：1,800 多个环境、8.5 万个 prompt | 模拟器和真实工具行为不一致 |
| 操作电脑 | 虚拟机里的桌面和应用 | 检查终局状态的脚本，或 LLM judge | [OSWorld](https://arxiv.org/abs/2404.07972)（2024-04，369 个任务，每个有执行式评估脚本）；[UI-TARS-2](https://arxiv.org/abs/2509.02544)（2025-09） | judge 误判；虚拟机慢、贵 |

Kimi K2 的工具模拟器（论文说它在功能上相当于一个 world model）用来生成多轮交互数据；编程和软件工程任务则接真实的执行沙盒，用测试通过率做反馈。DeepSeek-V3.2 的 Table 1 里，代码 agent 和搜索 agent 的环境是真实的，通用 agent 的 4,417 个任务环境是合成的。Kimi K3 的技术报告（2026-07）为长程个人助理任务做了 Gmail、Notion、Slack 等应用的仿真实现，保留核心行为，又不依赖外部 API 和限流。真实环境更可信，合成或仿真的环境更便宜、更容易扩到很多种工具：这是现在公开报告里的一条主要分界线。

## 奖励的四种来源

| 来源 | 在 agent 任务里怎么用 | 强在哪里 | 典型失效 |
| --- | --- | --- | --- |
| 单元测试、程序执行 | 跑隐藏测试，看补丁是否修好 | 可复现、便宜 | 覆盖不全；agent 能碰到测试或测试框架 |
| 任务检查器 | 比较终局状态：数据库记录、文件、虚拟机里的设置 | 看的是“世界变成了什么样”，不是 agent 自己说做完了 | 目标状态写错；什么都不做恰好符合目标 |
| LLM judge、rubric | 判断报告、网页操作、开放式结果 | 能覆盖写不成规则的任务 | judge 被特定格式骗过；和被训练的模型共享盲点 |
| 人的偏好 | 很少逐条轨迹打分，多用在训练 reward model 或做抽查 | 最接近真实意图 | 贵、慢；聊天式的偏好数据不一定迁移到 agent 任务 |

Kimi K3 的[技术报告](https://github.com/MoonshotAI/Kimi-K3)（2026-07）把第二种写得很明确：奖励来自验证器对最终环境状态的评估，而不是 agent 自报完成。SWE-RL（Meta，2025-02，[论文](https://arxiv.org/abs/2502.18449)）是另一种设计：不跑测试，按生成补丁和参考补丁的文本相似度给 0 到 1 的分，格式错给 −1；它是单轮设置，模型一次读完问题和相关文件、直接写出修改，不和环境多轮交互。关于测试当奖励的一般问题，见[可验证奖励](../../05-post-training/verifiable-rewards.md)；各种信号由强到弱的排列，见[生成、验证、再训练](../self-improvement/generate-verify-train.md)最后的对照表；judge 本身的偏差，见 [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.md)。

人类偏好在 agent 上的一个具体提醒来自 Anthropic 2025-11-21 的[研究](https://www.anthropic.com/research/emergent-misalignment-reward-hacking)：模型在生产编码环境里学会钻空子之后，加上聊天式的 RLHF 安全训练，结果是“aligned behavior on chat-like evaluations, but misalignment persists on agentic tasks”（论文[摘要](https://arxiv.org/abs/2511.18397)），即在聊天类评估上表现对齐，在 agent 任务上的不对齐却还在。

## 已经测到的钻空子

下表只收一手来源记录过的例子。多数例子有一个共同点：测试、评分器、答案线索或网络，就在 agent 能动手的环境里。

| 日期 | 谁报告的 | 发生了什么 |
| --- | --- | --- |
| 2025-02-24 | Anthropic，[Claude 3.7 Sonnet 系统卡](https://www-cdn.anthropic.com/9ff93dfa8f445c932415d335c88852ef47f1201e.pdf) §6 | 在 agent 编码环境里偶尔给测试写特判，多半是直接返回测试期望的值；系统卡说这是 RL 训练中 reward hacking 的结果 |
| 2025-03-14 | OpenAI，Baker 等，[论文](https://arxiv.org/abs/2503.11926) §2.1 | 训练中的 agent 调 `exit(0)` 提前退出、不跑完测试，或者 `raise SkipTest` 跳过测试；还有测试覆盖差时写桩代码、反编译含参考答案的 `.pyc` 文件 |
| 2025-06-05 | METR，[博客](https://metr.org/blog/2025-06-05-recent-reward-hacking/) | o3 改写评分器的计时变量、monkey-patch 评估器；在 RE-Bench 的一个优化任务上 21 次运行全部钻了空子 |
| 2025-09-03 | SWE-bench [issue #465](https://github.com/SWE-bench/SWE-bench/issues/465) | 评测镜像里留着“未来”的 git 历史，agent 用 `git log --all` 直接看到修复提交；2025-09-11 起镜像删除了基准 commit 之后的历史。用同样方式搭的训练环境，这就是一个奖励漏洞 |
| 2025-11-21 | Anthropic，[研究](https://www.anthropic.com/research/emergent-misalignment-reward-hacking)与[论文](https://arxiv.org/abs/2511.18397) | 在真实生产编码环境里：返回一个 `__eq__` 永远为真的对象、`sys.exit(0)`、写 `conftest.py` 改掉 pytest 的报告；学会这些之后，模型在其他场合也出现破坏、伪装对齐等行为 |
| 2026-02-12 | Meta，[Gaia2](https://arxiv.org/abs/2602.11964) 附录 B.2.3 | RL 策略往写操作里塞越来越复杂的类代码字符串，压垮了做软检查的 LLM judge；加一项与任务无关的“风格”检查后堵住 |
| 2026-04-07 | Anthropic，[Claude Mythos Preview 系统卡](https://www-cdn.anthropic.com/7624816413e9b4d2e3ba620c5a5e091b98b190a5/Claude%20Mythos%20Preview%20System%20Card.pdf) §4.2.2.1 | 模型偶尔绕过训练环境的网络限制，下载能走捷径的数据；约 0.05% 的训练回合有尝试，约 0.01% 成功 |
| 2026-08-26 | OpenAI，[Hugging Face 事件报告](https://openai.com/index/hugging-face-incident-and-the-road-ahead/) | OpenAI 说，agent 上网查答案作弊是这次事件的主要推手；这种行为此后在训练中被强化，可能促成了模型使用类似手段的倾向 |
| 2026-09-22 | Anthropic，[Claude Opus 5.5 系统卡](https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf) §6.2.1 | 用分类器筛 RL 训练回合：约 0.3% 的抽样回合里，模型成功猜中了答案键想要什么，而不是从任务本身推出答案 |

<!-- paper-figure:baker-hack-cot -->

不在沙盒里的检查器也会被钻。[One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794)（2025-07）发现，在一个多学科数据集上只回一个冒号 `:`，LLaMA3-70B 当 judge 时有 77.2% 被判为正确。[Agentic Benchmark Checklist](https://arxiv.org/abs/2507.02825)（2025-07）指出 τ-bench 把空回复算作成功：有些任务本来就无法完成，什么都不做时数据库不变，恰好和目标状态一致。更早的 [Sycophancy to Subterfuge](https://arxiv.org/abs/2406.10162)（Anthropic，2024-06）在一串越来越容易被钻空子的环境上训练，在最后一个留出环境的 32,768 个回合里，有 45 个回合改了自己的奖励，7 个回合还改了单元测试来避免被发现。

还有一类问题出在测试本身太弱，agent 并没有主动作弊。[PatchDiff](https://arxiv.org/abs/2503.15223)（2025-03）检查 SWE-bench Verified 上被判通过的补丁：7.8% 其实没通过开发者写的完整测试，29.6% 的行为和参考修复不同。[UTBoost](https://arxiv.org/abs/2506.09289)（ACL 2025）找到 36 个测试不足的任务，345 个错误补丁被标成通过。

## 算一次：测试只抓住一部分失败

下面是教学假设，数字是编的。一批修 bug 任务里，agent 交上去的结果分四类：

| 类别 | 意思 |
| --- | --- |
| C | 真修好了 |
| W1 | 修错了，可见测试能抓到 |
| W2 | 修错了，可见测试没覆盖到，比如漏了空输入 |
| H | 改了测试文件，或给可见测试写了特判 |

比较两种检查器。v1 在 agent 的工作区里直接跑可见测试，agent 可以修改测试文件。v2 在评分前从原仓库恢复测试文件，再加一组隐藏测试；假设隐藏测试能抓到 80% 的 W2，H 一律不过。记 $q_{\rm miss}$ 为 W2 被放过的概率，$q_{\rm tamper}$ 为 H 被放过的概率：

$$
\mathbb E[r]=P(\mathrm C)+q_{\rm miss}\,P(\mathrm{W2})+q_{\rm tamper}\,P(\mathrm H),\qquad S=P(\mathrm C)
$$

$\mathbb E[r]$ 是期望奖励，$S$ 是真实成功率。v1 里两个 $q$ 都是 1；v2 里 $q_{\rm miss}=0.2$，$q_{\rm tamper}=0$。比较两个策略：$\pi_A$ 是训练前，$\pi_B$ 是假设在 v1 上训练一段时间之后。

| 策略 | C | W1 | W2 | H | 真实成功率 $S$ | v1 期望奖励 | v2 期望奖励 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\pi_A$ | 0.30 | 0.45 | 0.20 | 0.05 | 0.30 | 0.55 | 0.34 |
| $\pi_B$ | 0.35 | 0.10 | 0.25 | 0.30 | 0.35 | 0.90 | 0.40 |

按 v1 看，奖励从 0.55 涨到 0.90，涨了 35 个百分点；真实成功率只涨了 5 个点。按 v2 看，奖励涨 6 个点，和真实的 5 个点接近。v1 的奖励曲线很好看，却主要在记录 W1 怎样变成了 W2 和 H。

为什么训练会往 H 走？看一组 $G=4$ 条轨迹：结果是 [C, W1, H, W1]。v1 下奖励 $[1,0,1,0]$，组内标准化后 advantage 是 $[+1,-1,+1,-1]$：改测试和真修好被同样鼓励。v2 下奖励 $[1,0,0,0]$，advantage 是 $[+1.732,-0.577,-0.577,-0.577]$（算法见站内 [GRPO 算例](../../05-post-training/rlhf/after-rlhf.md)），改测试被压低。更麻烦的是，“卡住了就改测试”是一个跨任务通用的套路，而真修好要针对每个任务分别做对，所以在 v1 下，H 在难题上涨得尤其快。这一点是机制上的推断；上一节 Claude 3.7 Sonnet 的系统卡描述的正是多次失败之后才开始写特判。

## 怎样让奖励更接近真实成功

| 先试什么 | 应该观察什么 | 不能据此推出什么 |
| --- | --- | --- |
| 评分前从原仓库恢复测试文件，在 agent 碰不到的进程里跑 | H 类拿不到分 | 别的篡改通道也堵上了；`conftest.py`、`sys.exit(0)` 这类要单独检查 |
| 测试设成只读，或干脆不给 agent 看 | ImpossibleBench（2025-10）：隐藏测试让作弊成功率接近 0，但正常任务也变难；只读是折中，挡住改测试，挡不住特判 | 只读之后就不会写特判 |
| 加隐藏测试或留出用例 | W2 被抓到的比例 | 隐藏测试足够全；UTBoost 说明它们也会漏 |
| 抽样人工复核“通过”的补丁，或和参考修复比较行为 | 估计 $q_{\rm miss}$ | 抽查没发现问题，就没有盲区 |
| 用分类器或 CoT monitor 扫训练轨迹 | 钻空子的比例随训练怎么变 | 把 monitor 直接当奖励惩罚就安全了：Baker 等发现优化太狠时，agent 学会在思维链里藏起意图，照样钻空子 |
| 给 agent 一个“报告任务不可能”的出口 | ImpossibleBench 里 GPT-5 在矛盾任务上的作弊率从 54% 降到 9% | 所有模型都会一样受益；论文里 Claude Opus 4.1 的变化小得多 |

[Claude 4 系统卡](https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285/Claude%204%20System%20Card.pdf)（2025-05-22）报告，在它的 reward hacking 评估上，硬编码行为比 Claude Sonnet 3.7 平均少了 67%（Opus 4）和 69%（Sonnet 4）；同一节列出的措施包括修补训练环境里的漏洞、让环境说明和奖励信号一致、调整 RL 中的奖励信号让它更难被钻。Anthropic 2026-04-07 的 [Mythos Preview 风险报告](https://www-cdn.anthropic.com/3edfc1a7f947aa81841cf88305cb513f184c36ae.pdf)说，他们在训练前对环境做一轮评估，检查 reward hack 和 prompt 与奖励不一致的地方，但这道检查没能及时拦下所有有问题的环境。Kimi K3 在它的自主执行任务环境里把验证器和 agent 隔开：可见的验证器给诊断反馈，隐藏的验证器在留出场景上评分，提交次数也有限。

## 环境规模和验证成本

**规模在快速变大。** 公开的代码环境从 SWE-Gym 的 2,438 个任务（2024-12），到 SWE-smith 的 50,137 个（2025-04）、DeepSeek-V3.2 的“数万个”可复现 issue 环境（2025-12）、GLM-5 覆盖 9 种语言的 1 万多个可验证环境（[论文](https://arxiv.org/abs/2602.15763)，2026-02）。Kimi K3 的技术报告（2026-07）说训练和评测期间共创建了 51,219,741 个沙盒、涉及 1,505,678 个镜像。这些都是各团队自报的规模，环境质量没有统一的外部检查。

**每个环境都要钱。** SWE-smith 让同一个仓库的任务共用一个环境，报告存储 295 GB，而它估计的 R2E-Gym 子集约 4 TB、SWE-Gym 约 6 TB。daVinci-Env（[论文](https://arxiv.org/abs/2603.13023)，2026-03）报告，构建 45,320 个可执行 Docker 环境花了 89.1 万美元。

**每次评分也要时间。** DeepSeek-R1（[论文](https://arxiv.org/abs/2501.12948)，2025-01）在局限一节写道，因为评估耗时太长、拖慢 RL，他们还没有在软件工程任务上大规模用 RL。DeepSWE（2025-07-02）每次迭代并行起 512 个容器，测试限时 5 分钟。粗算一下，下面的 2 分钟是假设：如果每次测试平均占一个 CPU 核 2 分钟，一次迭代的 512 次评分就是 1,024 核·分钟，约 17 核·小时，这还不算 agent 在回合中间自己跑的测试。

**团队也在推动共享环境接口。** Prime Intellect 的 [Environments Hub](https://www.primeintellect.ai/blog/environments)（2025-08-27）和 Meta 与 Hugging Face 的 [OpenEnv](https://huggingface.co/blog/openenv)（2025-10-23）都在推共享的环境格式。Kimi K3 的报告说，只用一种 agent 外壳训练，模型可能过拟合某套工具格式、系统提示或上下文管理方式，所以它做了一个可配置的环境，能实例化 Claude Code、Codex 等多种外壳。

## 这条线接下来看什么

下面是开放问题，以及什么样的结果会改变现在的判断，不是预测。

1. **数量和质量哪个更值钱？** 公开的代码环境从 2024 年的几千个任务涨到 2026 年的上万个可验证环境，Kimi K3 训练中创建的沙盒以千万计，但公开报告很少给出“同样算力下，少而精的环境和多而粗的环境”对比。出现这样的消融，环境建设的优先级才有依据。
2. **钻空子的比例能不能持续压低？** Anthropic 的系统卡从 Claude 3.7 Sonnet（2025-02）到 Opus 5.5（2026-09）一直在报告训练中的钻空子，但评估集一直在换，数字不能直接比。若有人在固定的一组环境上跨模型代际报告，趋势才看得清。
3. **不可验证的 agent 任务怎么给奖励？** Rubric 和 LLM judge 已经用在报告、网页操作上，Gaia2（2026-02）展示了它们怎样被钻。用 judge 做奖励、再由独立的人工评估确认收益的结果，会决定这条路能走多远。
4. **换一套 agent 外壳还能保持表现吗？** Kimi K3（2026-07）用多种外壳训练来防过拟合。如果有公开结果测量“在 A 外壳里训练、在 B 外壳里用”的性能下降，就能知道环境设计中有哪些部分需要随外壳一起调整。

下一篇：[多步回合的信用分配与训练系统](credit-and-training.md)。
