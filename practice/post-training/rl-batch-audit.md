# 从一批回答核对 RL loss

**中文** · [English](rl-batch-audit.en.md)

> 原创教学实验 · 最近核验：2026-11 · 不下载模型，不代表完成了真实 RL 训练。

把 PPO 或 GRPO 的公式写对，只完成了一半。回答有没有分错组？EOS 算没算进去？记录的 old log-prob 是生成时的，还是更新后又算了一遍？这些地方出错，程序往往照样运行。

这篇不搭一个新的训练框架，而是检查一小批数据怎样变成 loss。先用 [RLOO](../../05-post-training/rloo.md) 的留一优势，避免同时引入 Critic 和 GAE；分组、位置对齐与概率版本的检查也适用于其他采样训练。本文只处理单轮、纯文本回答；工具交互的动作边界见[后训练基础设施](../../05-post-training/post-training-infrastructure.md#rollout-record)。

## 先留下能还原这次更新的信息 {#record}

从两个 prompt 开始，每题采三份回答。完整流程是：

<figure class="worked-update">
<ol>
<li><small>01 / 采样</small><strong>保留原始 token</strong><span>记录 prompt、回答标识、生成版本、采样配置和停止原因。</span></li>
<li><small>02 / 评分</small><strong>区分低分和未评分</strong><span>保留各项奖励、评分器版本与执行错误。</span></li>
<li><small>03 / 分组</small><strong>找到同题的三份回答</strong><span>按标识构造优势，不依赖异步任务的返回顺序。</span></li>
<li><small>04 / 重算</small><strong>对齐当前位置的概率</strong><span>用同一批 token 做前向，保留当前策略的计算图。</span></li>
<li><small>05 / 汇总</small><strong>核对 mask 和分母</strong><span>先手算，再比较单批次与拆分批次的结果。</span></li>
</ol>
<figcaption>每一步都能单独检查。跑通这些检查后，再接模型和分布式训练。</figcaption>
</figure>

| 记录 | 至少能回答什么 |
| --- | --- |
| `prompt_id`、`sample_id`、输入与回答 token IDs | 哪些回答属于同一道题？有没有重复消费同一条记录？ |
| 动作 mask、停止原因、实际长度 | 哪些位置是采样动作？是生成 EOS，还是达到长度上限？ |
| policy 版本、tokenizer / 模板版本、采样配置 | 当时用什么分布和输入生成？ |
| 奖励分量、评分状态、评分器版本 | 0 分是执行后的结果，还是评分服务没返回？ |
| 所需的 old / reference / sampler log-prob | 每个数组来自谁、对应哪些 token、什么时候计算？ |

不同算法需要的字段不完全相同。无需机械地存三份概率，但只要使用其中一份，就要说清它的含义。不要仅存解码后的文本，再假设重新分词一定能还原原始动作。

## 一个 shift，错一位就全变了 {#alignment}

下面用 5 个位置说明。`<P>` 代表已经准备好的 prompt token；例子不对应某个真实 tokenizer。

| 输入位置 | token | 是否为动作 | 由哪个位置的 logits 预测 |
| --- | --- | ---: | --- |
| 0 | `<P>` | 否 | 不在本例内 |
| 1 | `A` | 是 | 位置 0 |
| 2 | `B` | 是 | 位置 1 |
| 3 | `<EOS>` | 是 | 位置 2 |
| 4 | `<PAD>` | 否 | 不计入损失 |

因果模型位置 $t$ 的 logits 预测位置 $t+1$ 的 token。因此，移位后 `labels=[A,B,EOS,PAD]`，动作 mask 是 `[1,1,1,0]`。假设三个实际动作的条件概率是 0.5、0.25、0.8，序列 log-prob 就是 $\log(0.5\times0.25\times0.8)=\log0.1\approx-2.302585$。

PyTorch 中，模型已经返回 `logits` 后，可以这样取被采到的 token 的概率：

```python
import torch


def completion_log_probs(logits, input_ids, action_mask):
    if logits.ndim != 3 or input_ids.shape != logits.shape[:2]:
        raise ValueError("Expected logits [batch, length, vocabulary]")
    if action_mask.shape != input_ids.shape or action_mask.dtype != torch.bool:
        raise ValueError("Expected a boolean action mask matching input IDs")
    if action_mask[:, 0].any():
        raise ValueError("The first input position has no preceding logits")
    shifted_mask = action_mask[:, 1:]
    if not shifted_mask.any(dim=-1).all():
        raise ValueError("Every response needs at least one sampled action")
    safe_logits = logits[:, :-1].float().masked_fill(~shifted_mask[..., None], 0)
    token_log_probs = safe_logits.log_softmax(dim=-1).gather(
        -1, input_ids[:, 1:, None]
    ).squeeze(-1)
    if not torch.isfinite(token_log_probs[shifted_mask]).all():
        raise ValueError("Action log-probabilities must be finite")
    token_log_probs = token_log_probs.masked_fill(~shifted_mask, 0)
    return token_log_probs, token_log_probs.sum(dim=-1)
```

这里用 `log_softmax`，不先算 `softmax` 再取 `log`；前者避免这一步的数值问题，见 [PyTorch 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.log_softmax.html)。真实前向还要传正确的 attention mask、位置和模板。动作 mask 决定哪里计算策略损失，不等于 attention mask；prompt 虽不直接计损失，仍是回答的条件。

若 PAD 和 EOS 恰好共用一个 ID，也不要用 `token != pad_id` 构造动作 mask，那会把真正生成的 EOS 一起删掉。mask 应来自实际生成边界。达到长度上限的回答没有“生成过 EOS”，不能补一个 EOS 后把它当成采样动作；截断样本如何评分和是否保留，需要另行约定。

## 分组错了，优势也会看起来很合理 {#grouping}

两题各三份奖励是：

| Prompt | 奖励 | 正确的留一优势 |
| --- | --- | --- |
| Q1 | `[1, 0, 0.5]` | `[0.75, −0.75, 0]` |
| Q2 | `[0, 0, 0]` | `[0, 0, 0]` |

若异步返回顺序为 `Q1-A, Q2-A, Q1-B, Q2-B, Q1-C, Q2-C`，直接 `reshape(2, 3)` 会把第一组奖励写成 `[1,0,0]`，第二组写成 `[0,0.5,0]`。每组优势仍然和为 0，shape 也没错，却已经在比较不同问题。

先按 `prompt_id` 收齐同一轮、同一生成策略下的预定样本，再算统计量。超时或评分失败不是默认的 0 分；补采样、丢弃整组还是保留不完整组，都可能改变采样分布，需要显式记录。更不能只丢掉低分回答，再把剩下的叫作原始 on-policy 样本。

## 三种概率，不要混用 {#probabilities}

| 概率 | 什么时候变化 | 用途 |
| --- | --- | --- |
| 当前策略 log-prob | 随本次训练参数变化，保留梯度 | 计算当前更新 |
| old log-prob | 一批数据复用期间固定 | 作为相应策略比率的分母 |
| reference log-prob | 对应选定的参考策略；若刷新需显式记录 | 计算相对参考策略的约束 |

如果推理引擎的采样概率与训练端的 old 概率不同，还要区分真实行为分布。temperature、top-p、模板和数值精度都可能产生差异，不能只比较模型权重名称。详见[采样与训练的概率对齐](../../05-post-training/post-training-infrastructure.md#_4)。

调试时可以先用同一个训练后端评分，固定权重、输入和随机性：更新前的 current / old 比率应接近 1。真实生成引擎另做对照，不设一个适合所有精度与长度的统一阈值。第一步更新后若比率仍机械地全为 1，要查 old 是否跟着被覆盖；它不是每次 forward 都更新的一份副本。

奖励、优势和 old / reference 分数一般作为固定目标，current log-prob 才保留梯度。一次新鲜采样的 RLOO 奖励项可以写成 `-(advantages.detach() * sequence_log_probs).mean()`。多次使用旧数据则需要明确相应估计与更新规则，不能反复套这个最小式子。

## `.mean()` 到底平均了什么？ {#reduction}

用两条回答的**逐 token 教学损失**来检查分母：短回答 `[2,4]`，长回答 `[1,1,1,1]`。它们不是实测 RL loss，也不用于推断模型会变长还是变短。

| 汇总方式 | 本例结果 | 短回答：每 token 权重 | 长回答：每 token 权重 |
| --- | ---: | ---: | ---: |
| 序列求和，再平均两份回答 | $(6+4)/2=5$ | $1/2$ | $1/2$ |
| 每份回答先按长度平均 | $(6/2+4/4)/2=2$ | $1/4$ | $1/8$ |
| 所有有效 token 一起平均 | $(6+4)/6=5/3$ | $1/6$ | $1/6$ |
| 用固定长度 4 归一化 | $(6+4)/(2\times4)=1.25$ | $1/8$ | $1/8$ |

系数指 $\partial L/\partial\ell_{i,t}$，**不是网络参数梯度的大小**。不同 token 自己的梯度仍可很不一样。第一行对应前面序列 REINFORCE 的求和方式；其他行可能用于不同训练目标，不能为了 loss 数字好看就互换。固定长度与序列求和在这里差一个常数，按实际长度平均则改变长短回答的相对权重。

读训练配方时，这张表也能防止把不同改法混为一谈。[Dr. GRPO，§3.2](https://arxiv.org/html/2503.20783v2#S3.SS2)去掉逐回答长度和组内标准差归一化，示例实现用固定生成预算作分母；[DAPO，§3.3](https://arxiv.org/html/2503.14476v2#S3.SS3)使用有效 token 总数归约。对同一批固定数据，它们的长度分母可以只差一个整体缩放；但换一批回答，实际 token 总数会变，固定预算未必变。不能把两者都简称“token 平均”，也不要把整套算法的区别缩成一个分母。

拆成 microbatch 或多张卡时，也要保持所选分母。比如按有效 token 平均，应该先汇总整个优化步的有效 token 数，而不是让短 microbatch 和长 microbatch 各占一半。分布式梯度平均的补偿见[多卡训练的全局分母](distributed-training.md)。

## 跑检查，再接真实模型 {#checks}

仓库根目录运行：

```bash
python3 practice/post-training/code/rl_batch_checks.py
python3 -m unittest discover -s site/tests -p 'test_rl_batch_audit.py'
```

标准库程序检查留一优势、按标识分组、token 移位和四种汇总。测试还会枚举一个二元策略的所有小组，核对留一估计的期望梯度；环境有 PyTorch 时，另检查上面的代码和动作 mask 对梯度的影响。

真正开训前，还要做这些对照：

- 同样的有效 token，增加 padding 不改变 loss；改变 prompt、分词或有效动作则不能要求结果不变。
- 打乱记录顺序后，按 `sample_id` 对齐的优势不变；重复记录或少样本要被发现。
- 单批次与拆分累积的梯度在相同分母下相符，old / reference 参数没有意外收到梯度。
- 固定几道训练外问题，比较原模型和更新模型；reward 曲线不是独立评测。

这些检查通过以后，才有理由把故障定位到更后面的数值、优化或分布式环节。它们不保证奖励设计正确，更不保证新模型比原模型好。继续做模型对照，见[实验与发布](experiments-and-release.md)；算法不收敛时，见[RL 为什么难训](../../05-post-training/deep-rl/why-rl-is-hard.md)。
