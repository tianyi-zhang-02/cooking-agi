# Agents：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 3 分钟 · 难度：进阶 · 最近审阅：2026-09

## 面试常见问题

<details class="interview" markdown="1">
<summary>Workflow 和 agent 有什么区别？什么时候不该用 agent？</summary>

workflow 用事先写好的代码路径串起模型和工具；agent 让模型自己决定流程和工具。步骤能事先列出来、没有验证信号、或者做错一步代价很高（发送、付款、删除）时，都不该直接用 agent：先写成 workflow，或者在关键步骤加人工确认。

</details>

<details class="interview" markdown="1">
<summary>一个 agent 由哪几部分组成？</summary>

模型（决定下一步）、工具（对外界动手）、循环（结果回到模型）、停止条件（做完或预算用完），再加上存放中间状态的上下文或记忆。ReAct 的「想一步、做一步」是这个循环最常见的形式。

</details>

<details class="interview" markdown="1">
<summary>为什么写代码是 agent 最成熟的场景之一？</summary>

因为验证信号最强：代码能跑，测试能过，每一步都有客观反馈，agent 可以放心多试。相比之下，客服、个人助理这类场景很难自动判断对错，更依赖规则、权限和人工。即使在写代码的场景，测试通过也不等于改对了。

</details>

<details class="interview" markdown="1">
<summary>怎么评估一个客服 agent 的可靠性？</summary>

不能只看单次成功率，要看同一个任务重复多次是否都做对，比如 τ-bench 的 pass^k。还要检查是否遵守政策、工具调用是否正确、复杂情况有没有转人工，并看完整的运行轨迹而不只是最终回答。

</details>

<details class="interview" markdown="1">
<summary>什么时候用 frontier API，什么时候自己 serve 开源模型？</summary>

先看能力：在自己的评估集上开源模型不达标，就用 frontier。数据不能出环境，就只能自己部署。流量大且稳定、需要在自己的数据上微调，自己 serve 更划算也更可控。量小或不确定时，先用 API 把质量和成本测出来。实际上常常混用：路由、级联、蒸馏、兜底。

</details>

<details class="interview" markdown="1">
<summary>为什么单步准确率上的小差距，在 agent 里会被放大？</summary>

错误沿步骤累积。每步 95% 做对，20 步全对只有约 36%；每步 99% 做对，20 步全对约 82%。长任务里，单步能力的一点差距决定了整体能不能用。

</details>

<details class="interview" markdown="1">
<summary>什么是 reward hacking？举两个例子。</summary>

模型优化的是你实际给的奖励，而不是你心里想要的目标，两者一有偏差，它就会钻空子。OpenAI 的划船游戏里，奖励是沿途的道具，agent 就原地转圈刷道具，比跑完比赛得分还高。LLM 里也一样：拿「测试通过」当奖励，agent 可能去改测试；RLHF 里奖励模型偏爱长回答或附和用户，策略模型就会学着写得更长、更会讨好。

</details>

<details class="interview" markdown="1">
<summary>级联和路由都是在小模型和 frontier 模型之间分配请求，区别在哪？</summary>

级联让小模型先答，看了答案再决定要不要升级，所以判断更准，但被升级的请求要付两次钱，还要串着等两个模型。路由在看到答案之前就决定派给谁，每条请求只付一次、延迟更低，但猜错了没有补救。两种方案都只有在难度判断足够准的时候才划算。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>workflow 和 agent 的分界线在哪里，各自的代价是什么？</li>
    <li>为什么「有没有验证信号」是判断一个场景适不适合 agent 的关键？</li>
    <li>一个会读网页的 agent，为什么要把网页内容当数据而不是指令？</li>
    <li>自己 serve 的成本为什么有一块固定的底？什么情况下它会比 API 便宜？</li>
    <li>除了成本，还有哪些因素会决定用 frontier API 还是开源模型？</li>
  </ol>
</div>

## 参考资料

- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)：workflow 与 agent 的区分和常见结构
- [ReAct](https://arxiv.org/abs/2210.03629)：推理与行动交替
- [SWE-bench](https://arxiv.org/abs/2310.06770)：用真实 issue 评估写代码的 agent
- [τ-bench](https://arxiv.org/abs/2406.12045)：带政策约束的工具与对话，以及 pass^k
- [WebArena](https://arxiv.org/abs/2307.13854)
- [OSWorld](https://arxiv.org/abs/2404.07972)
- [Not what you've signed up for](https://arxiv.org/abs/2302.12173)：indirect prompt injection
- [FrugalGPT](https://arxiv.org/abs/2305.05176)：模型级联
- [RouteLLM](https://arxiv.org/abs/2406.18665)：强弱模型之间的路由
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [SGLang](https://arxiv.org/abs/2312.07104)
