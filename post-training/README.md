# Post-Training

**中文** · [English](README.en.md)

## 当前理解

Post-training 的本质，是把对人类行为的不完美观察转化为可以辩护的学习目标。数据由什么策略生成、反馈延迟多久、存在怎样的选择偏差，以及最终怎样评估，与优化器本身同样重要。

## 我关心的问题

- SFT 能学到什么，preference optimization 或 online RL 又分别能学到什么？
- 当反馈稀疏、延迟或存在混杂时，训练目标应该怎样变化？
- 训练数据怎样保留纵向结构和潜在因果关系？
- 哪些行为提升能在分布变化与重复交互中保持？

## 起始论文

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
