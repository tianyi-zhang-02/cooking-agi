# 保存成功了，为什么还接不上训练

**中文** · [English](checkpoint-and-resume.en.md)

> 原创教学项目 · 核对：2026-10-09。附带 CPU 标准库恢复实验，不是分布式 checkpoint 后端的完整实现。

训练跑到一半中断，权重文件还在。重新加载之后 loss 却跳了，数据也从头读。这不是“恢复成功，只是有点波动”：你可能只恢复了模型，没有恢复那次训练。

先区分两个操作：**继续同一实验（resume）**，和**拿已有权重开始新实验（warm start）**。后者可以换数据或重置优化器，但要另记 run，不能冒充前者。

## 两类产物，不要混成一个文件

| 产物 | 应包含什么 | 用在哪里 |
| --- | --- | --- |
| 恢复 checkpoint | 模型或 adapter、优化器、调度器、步数、RNG、数据状态、运行清单；使用时还需 AMP scaler 等状态 | 中断后继续训练 |
| 推理导出 | 权重或 adapter + 精确基座、tokenizer、聊天模板、配置与生成约定 | 离线评估、推理服务 |

LoRA adapter 很小，也不能独立说明用了哪个基座。记录不可变 revision，不只写模型名。合并、量化或更换推理后端后，用同一组固定输入再测；训练进程里生成正常，不保证导出的产物也一致。

训练恢复包可能保留数据路径和样本状态，不应原样当作公开模型包。只加载可信来源的文件；不要为了解决兼容性随意关闭反序列化安全限制。

## 恢复的是哪一个时刻

最容易解释的保存点是：完成一次 optimizer update 和 scheduler 更新、清空梯度后，开始下一次 accumulation window 之前。

<figure class="worked-update">
<ol>
<li><small>01 / 计算</small><strong>完成这一组 microbatch</strong><span>有效 token 分母覆盖整个累积窗口。</span></li>
<li><small>02 / 更新</small><strong>更新参数与训练计数</strong><span>保持 optimizer、scheduler 和数据位置一致。</span></li>
<li><small>03 / 保存</small><strong>先写完整快照</strong><span>写入期间不要把半成品标记成最新可恢复版本。</span></li>
<li><small>04 / 验证</small><strong>重启后跑下一步</strong><span>对照下一个样本、学习率和更新后的状态。</span></li>
</ol>
<figcaption>这里保存的是“完成了哪些更新、下一步该做什么”，不是单独一份权重。</figcaption>
</figure>

若在累积到一半时保存，还需要未应用梯度、microstep 位置和相应数据状态。项目先避免这条复杂路径。异步保存也要确定快照对应同一时刻，不能让后台读到正在被更新的 tensor。

数据游标尤其容易错：worker 可能已经预取后面的样本，但模型还没消费它们。恢复应对齐已消费数据，而不是文件读取进度。动态 packing 还要保存剩余缓冲，或用确定性的方式重建。只记 `epoch=2` 不够。

## 一个能亲手验证的小实验

[training_contracts.py](code/training_contracts.py)训练一个只有 1 个权重的回归玩具：有 momentum、随步数变化的学习率、打乱的数据顺序和随机输入扰动。虽然不是 LLM，但足以暴露“漏存一个状态”的问题。

比较连续跑 12 步，与跑 5 步、保存、创建新对象恢复后再跑 7 步。相同 Python 环境下，完整恢复应得到相同状态；仅恢复权重或故意清空 momentum 则不是同一训练路径。

```bash
python3 practice/post-training/code/training_contracts.py
python3 -m unittest discover -s site/tests -p 'test_training_contracts.py'
```

程序在临时目录里写 JSON，不读外部模型文件。保存时写同目录临时文件，flush / fsync 后用原子替换更新目标。这个教学写法防止读者看到写了一半的目标文件；**不承诺机器断电后的持久性，也不是对象存储或多 rank 提交协议**。

我们还会测试：少了 RNG 字段会被拒绝，数据版本变化会被拒绝，遗留的临时文件不会覆盖上一份可恢复状态。实际 GPU 训练还需单独保存各 rank 的 CPU / CUDA RNG，并考虑非确定性算子；玩具实验的精确相等不能直接推广到所有硬件。

## 多卡保存，多了哪些问题

分片模型不能假设“rank 0 存一下就全了”。[PyTorch Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)提供多 rank 状态保存与加载，并支持在兼容条件下重新分片。它解决张量布局问题，不自动恢复应用层的数据顺序或业务配置。

| 变化 | 先确认什么 | 不能默认保证什么 |
| --- | --- | --- |
| 同拓扑恢复 | 所有必要分片完整、状态与 manifest 对应同一步 | 文件存在就代表完整可用 |
| 改变 GPU 数量 | 后端能加载该模型 / 优化器布局，重算 DP batch | 样本顺序、dropout、更新轨迹逐 bit 相同 |
| 升级框架 | 状态格式、参数名称、optimizer 映射可兼容 | 任意版本间可直接恢复 |
| 异步保存 | 快照一致性、后台写完成、额外内存 | 调用返回就表示所有文件落盘 |
| 对象存储 | 用唯一版本前缀写入，全部成功后再发布完成清单 | 本地文件 rename 的语义可直接搬过去 |

恢复之前核对文件列表、大小 / 校验和及完成标记；哈希用于发现损坏，不证明来源可信。保存失败时保留旧版本，不先删除唯一可用的 checkpoint。

## 多久存一次，best 和 latest 怎么留

存得越密，可能损失的训练越少，但停顿与存储越多。假设每次暂停保存 30 秒，每训练 10 分钟保存一次，忽略其他开销，保存占总时间 $30/(600+30)\approx4.76\%$。若故障在两次保存之间均匀发生，平均需重跑约 5 分钟的训练；这只是便于理解的假设，不是通用最优间隔。

`latest` 用于接着跑，`best` 用于记录按开发集规则选出的候选，两者不一定是同一步。事先写清比较指标、保存频率和保留数量；测试集不参与 best 选择。也保留一份已验证可加载的旧版，等新保存完成并通过检查再轮换。

最后把恢复测试放进实验的早期，而不是第一次掉卡以后。下一篇看[如何比较与替换模型](experiments-and-release.md)：能恢复只是必要条件，还不是模型值得上线的理由。
