# 内容扩展核对表

维护记录，不加入站点导航。最后核对：2026-10-09。当前发布状态以 CONTENT_RELEASE_CHECKLIST.md 为准；下方带日期的记录保留当批状态，不作为最新完成率。

目标是覆盖参考知识库的知识点，并用原创讲解、可核对的例子和原始来源把它们讲透。不是复制第三方全文、图片或代码，也不把“目录里有这个词”当作已完成。

## 本轮整理后的状态（2026-10-09）

- 用户已同意先整理并推送已有内容，未完成的阅读、深化与逐篇审稿继续保留。下方带日期的“不推送”是先前批次的决定，不是当前发布约束。
- 附录已登记 67 个入口，目前 24 项有专门讲解、43 项需补充或对照、0 项完全缺少相关专题；原参考附录 1 正文累计读完 8 篇。计数来自 `site/appendix-coverage.toml`，不把文章存在当成完整验收。
- 最近新增的 NoPE / Kimi K3、Qwen3 Embedding / BGE-M3、BM25 与筛法均已接入各自阅读路线，保留原始论文、实现与教学算例的边界。BERT 的新稿尚未接入，本轮不发布。
- 全站逐篇语义审稿、剩余参考问答、图片公式与未执行的参考代码仍不算完成。最新检查及 PR 状态记录在 `CONTENT_RELEASE_CHECKLIST.md`。

## 检索与算法基础补充（2026-10-09）

- 新增 TF-IDF / BM25 与质数 / 筛法两对原创文章。前者放在双塔之前；后者放在算法按需补充中，不把数论列为 ML 入门前置。
- BM25 明确本实现的 IDF、查询去重、空文档统计与引擎差异，用同一 4 文档语料比较排名。筛法包含平方根证明、不变量、复杂度假设、闭区间、分段起点与因数分解。
- 参考来源为 Stanford IR 教材、Lucene 10.3.1、Python 3.12 文档和 CMU 筛法分析；不复制参考库的正文。新增标准库实现与双语代码一致性、算例和边界测试。
- 67 项现在为 21 项有专门讲解、44 项需补充或对照、2 项缺少专题；仍缺 K3 / RoPE 和 Qwen3 Embedding / BGE 专题，参考正文阅读状态不变。全站验收未完成，不推送。

## 原始来源与版本更新（2026-10-09）

- 新增 Muon 与 DFlash 两对原创章节：先用小例子解释，再给数学、局部代码、实现风险与比较条件。Muon 挂在训练基础 / 优化器之后；DFlash 挂在生成 / 缓存之后。旧页面 URL 和入门路线保留。
- 核对原作者说明、Muon 训练论文与实现、DFlash 初版论文、DFlash 2 官方介绍和现行仓库。明确区分有限步近似与精确正交化、MTP 目标与推测解码、greedy 与随机采样正确性、单请求延迟与并发吞吐。D-Loop 和 AF-Muon 只核对摘要与问题设定，不宣称实验复现。
- 独立附录清单仍是 67 项：19 项有专门讲解、44 项需补充或对照、4 项缺少专题。Muon、MTP/DFlash 和 DFlash 推理三项覆盖状态更新；参考阅读仍保留 title，不借新增文章伪造已读。
- 数学基础不会因为年份较旧而删减；后续按问题核对版本与来源，不按新名词数量验收。全站审稿、参考逐项阅读和发布门槛仍未完成。

## 浏览器目录续查（2026-10-09）

用户重新登录后，已恢复参考访问。这次在附录目录中辨认出前两张截图之外的 31 个入口，共登记 67 项：65 个独立问题及算法、PyTorch 的 2 个附录入口。不是整个知识库的总篇数，也不把入口数量当正文阅读量。主线 82 项与这些问题存在重叠，不相加为完成率。

本站逐项映射暂分 16 项已有专门讲解、44 项需补充或对照、7 项缺少专题讲解。44 项中既有明确的内容缺口，也有参考正文尚未比对的条目，具体情况见 `site/appendix-coverage.toml`。这轮只扩展清单和阅读入口，没有用新建空白文章消除缺口。

- 新确认的硬缺口：Muon 更新机制、BM25 / TF-IDF 算分、质数判断与筛法。已有 K3 / RoPE、DFlash、MTP / DFlash、Qwen3 Embedding / BGE 四项仍未完成。找到 Muon 或 BM25 的同名词不算覆盖。
- 需深化：序列级 MoE 均衡、端到端路由实现、GRPO 初始零 loss、clipped-token 梯度、拒绝采样、熵坍塌及其与 reward hacking 的区别、thinking 模式的控制来源、TP 列切 / 行切串联。
- 已有实质内容而不另开重复文章：online softmax、Self-attention / MHA、RoPE、BF16 / FP16、Norm 轴、PPO / GRPO 比较、ZeRO 状态账、熵 / CE / KL。最后一项在蒸馏篇有同一组数值与代码，不能只搜章节标题就误判为缺失。
- 新读到正文末尾：[KV cache 估算](https://tcn6r3nlptym.feishu.cn/wiki/LMQSwIm80i4kzZkeMxrcLLDonzd)。图片公式未逐图审计，未运行参考部署。参考里的模型维度、dtype 和 GB / GiB 必须独立验证，不直接沿用并发估算。附录 1 正文阅读累计 7 篇。

新增 31 项目前只确认了目录，不标作参考正文已读。Chrome 随用户操作切换到其他标签时停止继续点击，没有操作无关页面。后续继续逐篇对照，补齐后才能发布。截图批次记录保留如下。

## 新截图的独立对照（2026-10-09，前一批）

两张截图共辨认出 36 个入口，含算法与 PyTorch 两个附录入口，保存在 `site/appendix-coverage.toml`。它们与原有 82 项主线分别管理，不相加成一个“全部已覆盖”的数字，也不把只有相关章节的主题标为完成。中英文读者可在 `learn/coverage` 展开 4 类查看本站入口和明确缺口。

新补了 KL 的三种估计：精确求和、支持集、旧采样分布、梯度与数值实现均有小例子及测试。它的参考问答仍只有截图标题，不能因此改成“参考正文已读”。K3 的具体模型尚未确认；DFlash 及其与 MTP 的比较、Qwen3 Embedding 与 BGE 的型号级对照仍缺正文。BERT、left padding、完整请求链等也不能用通用 Encoder / Decoder 导读替代。

阅读路线区分“先入门”“只读一个方向”“准备实现”，保留原有 4 类目录，不以附录问答的排列顺序组织教学。下一轮仍要逐题读参考正文，再按本站章节补齐；单纯通过构建或找到一个同名词不算完成。最新本地测试与发布阻塞见 `CONTENT_RELEASE_CHECKLIST.md`。

## 目前实际读到哪里

- 参考：[大模型学习地图 · AI有温度](https://tcn6r3nlptym.feishu.cn/wiki/TQpswAuJfiQX3lkcGfbcnHx5n9m)。已读取首页说明和顶层目录。
- 基础篇 1.0–1.5：已通过浏览器读到各节正文末尾，记录了规则/传统模型、Embedding、分词、函数逼近、Encoder 与 Decoder 的讲解范围。图片内公式未全部逐图复核，代码未运行；不能据此标记为完整技术审计。
- [训练与推理目录](https://tcn6r3nlptym.feishu.cn/wiki/VTshwGNlCiRKfikNA0TctOb7nKf)：2.0–2.12 正文已逐页读至结尾；包括阶段划分、预训练、SFT、RL 入门、奖励模型、PPO、解码、DPO、GRPO、DAPO、GSPO、ASPO、SAO。代码已查看可见部分，但未下载配套工程、未运行；图片公式未全部核对。
- [评估章](https://tcn6r3nlptym.feishu.cn/wiki/LTd2wayWuicx35kfyuqcigGkn9b)：已读各节正文，包含 BLEU、ROUGE、编辑距离、perplexity、needle 测试及 benchmark 列表；图片公式和榜单的时效性需要另核。
- [蒸馏与微调目录](https://tcn6r3nlptym.feishu.cn/wiki/PvKEwNoyYiJGa2kLd8McfJSAnWd)：3.1–3.6 已逐页读完正文；配图公式和外链工程未全部审计。本站已对照补充 QLoRA 的存储分工、混合监督目标与 R1 数据蒸馏边界。
- [优化技术目录](https://tcn6r3nlptym.feishu.cn/wiki/HJtZw8yXHilw35kO2ofcTZPsnUg)：5.1–5.19 已逐页读至正文末尾，累计正文 45 篇，并查看可见代码；未执行参考代码。图片内公式未全部逐图复核，FlashAttention 内嵌 PDF 只翻到前几页、外链视频未观看，不计为读完。新读的 5.15–5.19 已有原始来源核对记录，但本站独立讲解仍待写。
- 模型家族：第 6 章目录及 6.1–6.4 正文已读至结尾，包括 DeepSeek 页的 V4 和 Qwen 页的 3.5；累计正文 49 篇。GPT 页的 sink / SwiGLU 展示代码只静态阅读，没有执行。参考图片不计为已全部审计。本站 GPT / Llama 已扩充；DeepSeek / Qwen 本批只补关键纠错与版本边界，仍不是完整家族精读。
- [第 7 章分布式训练](https://tcn6r3nlptym.feishu.cn/wiki/WZ0KwamzhiPj89kEq5dcYdegnWg)：7.1–7.7 位于同一页面，已顺序读到总结和评论入口；按 1 篇计，累计正文 50 篇。图片推导未全部复核，没有运行分布式实验。
- 第 8 章应用 8.1–8.5 已逐页读至正文末尾，累计 55 篇。RAG / Agent 可见代码只做静态阅读，未运行；图片公式与外链工程未全部审计。
- 第 9 章 16 个页面的正文已全部读至末尾，主线累计正文 71 篇。附录 3 的 7 篇 ML 基础与 20 篇 PyTorch 正文均已读至末尾；附录 2 的算法分类页也已读完。附录 1 当前已读 8 篇独立问答正文，清单见下一节，其余还需继续。用户无法导出，继续通过浏览器阅读，不以导出作为前置条件。

### 附录续读：2026-10-09

附录 1 当前只完成下列 8 篇正文阅读，不能解读为整份问答集已覆盖。参考页是选题线索；本站独立讲解，技术结论以原论文和官方实现为准。未执行参考部署或训练代码，也未完成所有图片内推导审计。

| 已读正文 | 参考入口 | 本站处理 |
| --- | --- | --- |
| 跨 tokenizer 的 On-policy distillation | [正文](https://tcn6r3nlptym.feishu.cn/wiki/BiYuwTFuciwVMXkGgfQc2IrNnPd) | distillation：文本路径概率、逐 token 目标与 KL 事件空间分开；按比例分配不是概率恒等式；补 2026-06 / 2026-10 原论文 |
| Long-horizon GRPO / PPO | [正文](https://tcn6r3nlptym.feishu.cn/wiki/EqjHw3mcGiqYx2kTbEJcPdTSnde) | llm-bridge：不同轨迹无需动作对齐；终止 / 截断、Critic 误差、完整 rollout 成本一起比较；不沿用未验证的商业模型训练说法 |
| TP / DP / EP 的部署差异 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/ElKLwrPSziZJx1khI9Pcmylqnae) | distributed-training：用原创 4 卡权重账区分 attention 副本与共享专家；不转载参考模型的显存和性能数据 |
| Prefix Cache 回收 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/EIvZwueHPi6rtNkvuaEcC6dMnke) | attention-kernels：引用释放、失效、内存归还不混用；LRU 不是 TTL，完整块边界影响命中 |
| vLLM / SGLang 前缀比较 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/DSl0wvnYEidi36kjMAgcmvTmn8c) | attention-kernels：核对当前 RadixCache 的 page-size 对齐，去掉“任意配置按 token 复用”和框架必胜的概括 |
| PP 中的 KV 放置 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/UfQRwKClOiFmzIkOScbc7FOOnic) | distributed-training：区分本地分层 KV 与边界激活，用 48 KiB decode / 192 MiB prefill 的原创账说明代价；PP 不是无条件最优 |
| 推理 KV cache 估算 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/LMQSwIm80i4kzZkeMxrcLLDonzd) | 已读文本到末尾；本站已有 GQA / PP 基础，需将权重、workspace、并发、共享前缀和分片合成预算；不采用未核对的模型配置或经验并发数 |
| K3 为什么能去掉 RoPE | [正文](https://tcn6r3nlptym.feishu.cn/wiki/Pm6BwXorWiqb9mkcA67cbNTqn8e) | 已读 7 节正文并查看主要公式；NoPE 专题从因果顺序与状态更新解释，回到 NoPos、Kimi Linear 和 Kimi K3 原文核对，不把作者消融当作普遍结论 |

重新完整检查[多模态微调参考页](https://tcn6r3nlptym.feishu.cn/wiki/O1GCw5GzAinKXRkeXD0cECPbnic)，正文到评论区只有环境、数据、记录工具和缺少配置的显存说明，没有待展开的完整训练工程。本站已补 captioning 与问答的区别、按图拆分、多参考答案、视觉证据评估、实验记录与恢复约定；固定版本示例不当作通用配方。GPU 训练未运行，也不再把“不曾承诺的全模型复现”混成正文缺失。

附录 2 的算法深化已补 6 对原创笔记与标准库程序：KMP、单调队列、BST、背包、LCS / LIS、状态机 DP。17 项专项测试包括穷举和独立 oracle；不是照搬参考题解。附录 1 余项与全站最后审稿仍阻塞发布。

### 附录续读：2026-10-08

| 已读正文 | 参考入口 | 本站处理 |
| --- | --- | --- |
| 过拟合与欠拟合 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/I8KnwfmF5iR0mxkwDQ7ceTQgniY) | 新增 generalization：训练失败、泛化差、泄漏要分开排查；交叉验证不是把测试集用于训练 |
| 激活函数 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/JyHSwzHAWidbExkj1kHcxtjrnVh) | 新增 activation-and-initialization：交叉熵对 logits 的梯度；softmax 很尖也不等于错分类时梯度消失 |
| 归一化与标准化 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/NpOkwWka7i0yzVkarQEcbeVOnSd) | train-only 预处理、统计轴、running statistics、affine 参数分开解释 |
| 初始化 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/X6cfwqzLDivoBCk9JvOc0zjKnqe) | ReLU 对称输入的二阶矩减半，不是方差减半；补 Xavier / He 的数值例子 |
| 正则化 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/MOnLwgQMqifmfgkWslKcqaUUnPZ) | 区分 L2 范数与平方惩罚；L1 阈值解与 inverted dropout 的非线性反例 |
| 优化器 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/EhGyw61hZieZSpkpMdTcEjbWnjd) | 新增 optimizers：SGD 到 AdamW、偏差修正、解耦衰减与恢复状态；不宣称通用最优优化器 |
| RNN / LSTM / GRU | [正文](https://tcn6r3nlptym.feishu.cn/wiki/QK1EwHe9iiFCbAk1CaUctYkCnde) | 扩充 recurrent-dynamics；校正 Jacobian 连乘、固定门值的直接路径、PyTorch GRU 的 reset 位置 |

以上是正文阅读记录。图片内公式没有全部逐图核对，参考代码未执行；本站使用独立讲解与小算例，并回到论文和官方文档核对。用户解锁后已继续读完以下 20 篇 PyTorch 正文，不能沿用上一批“只打开 Tensor 创建页”的状态。

| 已读 PyTorch 正文 | 参考入口 | 本站处理 |
| --- | --- | --- |
| 1-1 创建 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/LyMBwFyoBiWjKPkw0UWcy7HPnAc) | tensor / from_numpy 的复制与共享、显式 dtype |
| 1-2 维度 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/XX6VwGDWVidFrZki2TpcI9lxnbN) | 标量、空张量与一维向量分开讲 |
| 1-3 特殊张量与转换 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/YFgbwvWBnitsdQkZ6Wrc8RnTnLc) | empty 不保证接近零；to 可能不复制；normal 的 std 不是 variance |
| 1-4 索引 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/GZ51wCmmSikoMzkfPtScNMh9nMe) | 基础切片与高级索引；索引读取和索引赋值区别 |
| 1-5 视图 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/RdwIwXPhyieUhRkcjLacMzj0nOh) | stride 兼容的非连续 view；reshape 不等于 transpose |
| 1-6 分割与合并 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/HG1hwXTQdihPphkB4tLc0ysKnrg) | cat / stack；split 的块大小与 chunk 的实际块数 |
| 1-7 广播 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/MlNPwuIH4inx09kCt7xcKHuUnOg) | 末尾对齐、0 与 1、跨 batch 的错误 loss、expand 的反向求和 |
| 1-8 科学运算 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/HHXSwuJYsiRINVkBCx8cvdN1n6b) | logsumexp、var correction、median 和整数 exp 的行为 |
| 1-9 线性代数 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/J9AOwW0H6iUuA9kdu7YcO0mPnCy) | matmul / bmm；addbmm 会归约 batch |
| 1-10 运算练习 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/TldgwVB2eiQxQhkJzBEcEYTEnad) | 独立编写 gather / CE 与 batch 运算等价实验 |
| 1-11 操作小结 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/RQVBwlK0MiRUJYkRx5vcJGX3nQh) | 不照搬 API 目录，以形状和共享存储组织 |
| 2-1 计算图 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/N7OmwaOtTiXvIvkLFuxcBhPynkh) | 叶节点、retain_grad、detach、no_grad |
| 2-2 库架构 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/GJy1wauFmirNPekYrb2cQOyOnXg) | Module / functional / data / optimizer 的分工 |
| 2-3 常见陷阱 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/URYfw1QmgiWLDbkI93LcKlDingh) | 梯度累积、图的释放、非标量 VJP |
| 2-4 Linear | [正文](https://tcn6r3nlptym.feishu.cn/wiki/IclOwAOF7imbDDkspcOczAQnnqb) | 权重方向、类别标签与 logits；前向不等于训练 |
| 2-5 Module | [正文](https://tcn6r3nlptym.feishu.cn/wiki/Y73iw5XD3iHh6OkRjOZctFIbn4d) | ModuleList、Parameter、buffer；不使用 .data 绕过检查 |
| 2-6 数据加载 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/VmB7wEr0oieRyBkwwKqc8TIAnOh) | 样本对齐、collate、尾 batch 和指标分母 |
| 2-7 建模流程 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/MPrpw6t37iNs4ckYmKbcPkm4njg) | 增加独立验证集、按样本累计指标和恢复检查 |
| 2-8 TensorBoard | [正文](https://tcn6r3nlptym.feishu.cn/wiki/PhuSw8yb2ippWSkonRKcJPunnPe) | 可选依赖、step 单位、关闭 writer；没跑日志 GUI |
| 2-9 动态图 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/VpEewMe2diHSEzkeXw9cDPCfnMh) | 动态前向与保存的中间值；不把所有中间节点的 grad=None 视为断图 |

本站新增 `00-foundations/pytorch/`：1 对导读、4 对完整章节。代码为原创，33 项新测试通过（2 项文档检查、31 项 PyTorch 数值 / 行为检查）。实际运行 PyTorch 2.8.0 / CPU；不以文档 stable 链接当前指向的版本冒充本地运行版本。全量 397 项测试通过，严格双语 200 对通过，404 页面构建成功；16 个中英文页面在 1280px / 390px 已检查，无整页横向溢出或 KaTeX 错误。

### 其他附录的范围：还不能打完成勾

[附录 2 算法分类页](https://tcn6r3nlptym.feishu.cn/wiki/SufPwxHQHiQ5dmkbo2ZcOyhgnif)已读到末尾。它主要是按方法组织的题目索引，不是逐题讲解。本站不复制整份选题顺序；对照后需深化字符串匹配、单调队列、BST、背包、序列 DP 与状态机 DP，继续以可迁移方法为主，不改成 Hard 题库。

附录 1 展开后发现独立的技术问答，包括 TP / DP 部署、prefix cache、跨 tokenizer 蒸馏、Muon、推测解码、on-policy 边界、KL 估计与 padding。这是 2026-10-08 当批的目录盘点；2026-10-09 已继续读完前面列出的 6 篇，其余正文仍未全部读完。需继续核对并映射到已有章节；尤其不能把标题里的“为什么都用 left padding”“为什么放弃 GRPO”等前提直接作为本站结论。私人面试过程和公司特定题目不转载。

本次也修正了原有 interview-basics 中的 Pre-LN / warmup、causal mask 的目标位移、全掩码行、LSTM 已彻底解决长依赖等过度概括。新增小型代码测试不是 GPU 训练复现。

### 本轮阅读带来的修订重点

这些是本站需要讲清的边界，不是参考资料的摘抄，也不是对其全部内容的评价。

| 范围 | 本站需要补充或明确 |
| --- | --- |
| 规则、KNN、决策树 | 用同一个小任务比较规则来自哪里、训练做什么、推理成本与失效条件；不能把所有传统 ML 都称作白盒 |
| Embedding | 补 CBOW / Skip-gram / negative sampling 的目标区别；token ID 不必显式展开成 one-hot；点积还受向量范数影响 |
| Tokenizer | 区分 BPE 训练与编码；区分字符覆盖、byte fallback 与 byte-level BPE；代码需有合并边界和 round-trip 测试 |
| 函数逼近 | 把表达能力、可优化性、泛化分开；存在性定理需要函数类别、定义域和误差标准，不保证任意问题都能解决 |
| Attention | 缩放是在特定假设下控制点积方差，不会自动产生标准正态分布；残差路径不保证每次训练都优于旧模型 |
| SFT | response-only loss 是常用配置，不是唯一定义；SFT 可以 packing，预训练也可以使用特殊 token |
| PPO | clipping 针对概率比的 surrogate objective，不是把奖励或 advantage 封顶；reference policy 不是 critic，也不是过去分数的均值 |
| 解码 | 本轮已修正本站旧文：概率幂变换与温度缩放的等价性、并列最大值、beam search 的近似性；新增 top-p 边界代码与双语数值测试 |
| GRPO | 必须区分采样旧策略与 KL reference；参考代码用 reference log-prob 作 ratio 分母，与其 old-policy 公式不一致，不能照搬。按同题分组、零方差与空 mask 也要单独测试 |
| DPO | 改善 chosen/rejected 的相对比值，不保证 chosen 的绝对概率上升；不能用一项损失的极限行为证明它普遍不如 PPO |
| DAPO | 区分 policy 项零梯度与整个样本无梯度；token-average 改变样本权重，并非自动“公平”；移除 KL、采用 verifier 都有适用条件 |
| GSPO | 几何平均的长度归一化比率不等于完整轨迹的重要性比；序列级裁剪与 GRPO 不是简单的“错公式 / 对公式”关系 |
| ASPO | 已确认论文身份；要区分概率本身与新旧概率比，明确 stop-gradient 所在位置；正奖励轨迹也不代表每个 token 都有因果贡献 |
| SAO | 已确认 Single-Rollout Asynchronous Optimization 论文；单 rollout 指每 prompt 一条，不是每更新只有一个样本；需核对行为概率、critic、观测 mask 与 GAE 时间尺度 |
| Prompt / Prefix | 可学习的 soft prompt 不等于手写提示词；Prefix 是每层额外状态。是否优于另一种方法取决于任务和预算，不能写成普遍结论 |
| Adapter / LoRA | 非线性 adapter 有额外推理计算；LoRA 约束的是更新矩阵，不是分解原权重。参数少不代表训练显存按比例缩小 |
| QLoRA | NF4、缩放常数量化、优化器状态分页分开讲；4-bit 权重账不是训练总显存。已回原论文核对并补入双语正文 |
| 蒸馏 | CE 与 KL 在固定老师下梯度相同；不同 tokenizer 不排除文本级蒸馏；SFT 不等同于全参数微调；不存在普遍优于普通监督的保证 |
| 评估 | 补独立的经典文本指标讲解：计算小例子、比较条件与反例；ROUGE-L 依赖子序列顺序，perplexity 通常评估固定测试文本 |

当前累计读完主线 71 个页面的正文（基础 6、训练与推理 13、微调与蒸馏 6、评估 1、优化技术 19、模型家族 4、分布式 1、应用 5、多模态 16），另读完附录 3 的 27 篇正文，以及附录 2 的 1 篇分类页。每项分开记录“看过正文”“复核数学/代码”“本站已补”，不要合并成一个完成勾；附录 1 的目录不计入已读正文。

### 优化技术：本次新读 5.1–5.9

以下记录用于继续阅读与查缺，不搬运参考讲解。各页正文已读至结尾，不等于图内公式、外链或实现全部通过验证。

| 节 | 阅读入口 | 需要区分的边界 / 本站处理 |
| --- | --- | --- |
| 5.1 SwiGLU | [正文](https://tcn6r3nlptym.feishu.cn/wiki/BNnpw8uRQi3QwKkmZB0cAEsPnTg) | gated activation 不等于包含 down projection 的完整 FFN；源于 Shazeer 2020，不是 LLaMA 首创。本站已有逐维例子与等参数预算比较 |
| 5.2 RMSNorm | [正文](https://tcn6r3nlptym.feishu.cn/wiki/IDNHwmzYmiukUKkZUyhcR8eNnue) | 不是均值设为 0 的 LayerNorm；本站修正统计轴 / 统计量混淆，补整体平移反例和尺度不变性的条件 |
| 5.3 RoPE | [正文](https://tcn6r3nlptym.feishu.cn/wiki/Z28TwEwzxiOcn8kFjVPcLAo0nEb) | 相对旋转恒等式不保证任意 q/k 的分数随距离单调下降；长向量代码跑通也不证明模型长上下文泛化。本站旋转与相对点积例子已有测试，参考图片推导仍待全审 |
| 5.4 KV Cache | [正文](https://tcn6r3nlptym.feishu.cn/wiki/Iw3MwrfRViZutNkGfIrcYRx8nzc) | 分开 prefill、单步 decode 与累计复杂度；缓存各层 K/V，不是只存首层或 attention 矩阵。本站已有时间顺序与 logits 等价性测试入口 |
| 5.5 MQA / GQA | [正文](https://tcn6r3nlptym.feishu.cn/wiki/YFCswpI73iem33knOeicvfEGnye) | h 个 Query heads、g 个 KV heads 时，原始缓存剩 g/h；KV 头数不由 GPU 数量定义。本站补 32→8 的存储例子及显式 repeat 的代价 |
| 5.6 混合精度 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/IXB9wAGg7ifcDckfkgDcjcrrnbb) | autocast 按算子选 dtype，不是全前向 FP16 / 全反向 FP32；先 unscale 再 clip。本站补可执行裁剪顺序反例、非有限梯度跳过更新的边界 |
| 5.7 FlashAttention | [正文](https://tcn6r3nlptym.feishu.cn/wiki/F0NYwwjG0iYUhBkZwMoc4m4Jnac) | FA2 的 warp 交换主要涉及 shared memory，别与 HBM 混淆；FA3 的 FP8 不保证无误差。补 FA1/2/3 对照；参考 CPU 示意代码的 mask、缩放、尾块与 dropout 不能作为标准实现照搬 |
| 5.8 PagedAttention | [正文](https://tcn6r3nlptym.feishu.cn/wiki/PCJTwYPvPiLGTIkrRNrcwpi2nze) | 分开块管理、连续组批和 chunked prefill；区分内外碎片。本站补 3 请求调度表及当前 vLLM 完整块 prefix cache 的实现边界 |
| 5.9 Gradient Checkpoint | [正文](https://tcn6r3nlptym.feishu.cn/wiki/IXZwwyhRQiuvefkgvjRcFMylnlf) | activation 不等于梯度或磁盘 checkpoint；加了两个明确形状的张量账，避免把粗略二次项当作所有 kernel 的内存公式 |

本批原始来源核对：[GLU Variants](https://arxiv.org/abs/2002.05202)、[RMSNorm](https://arxiv.org/abs/1910.07467)、[GQA](https://arxiv.org/abs/2305.13245)、[PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)、[activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)、[FA2](https://arxiv.org/abs/2307.08691)、[FA3](https://arxiv.org/abs/2407.08608)、[vLLM prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)、[vLLM tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)。不据此宣称参考代码或 GPU 性能已复现。

### MoE 到 MTP：2026-10-07 续读

| 节 | 参考入口 | 已读范围与本站补充 |
| --- | --- | --- |
| 5.10 MoE | [正文](https://tcn6r3nlptym.feishu.cn/wiki/GheUwYJkrixrN0kmbg9cQOwlnCi) | 正文及展示的 139 / 72 行代码读完，未执行。补 matched expert budget、shared 预算、selection bias 门值区别和 expert/device 负载表；不把专长分工当作架构保证 |
| 5.11 MLA | [正文](https://tcn6r3nlptym.feishu.cn/wiki/X4YFw0kHki9sj9k2iIUcgVqtnqO) | 正文读完，展示代码止于 154 行，缺完整 forward。对照 V2 区分直接来自 hidden state 的位置 key 与 KV latent，说明线性吸收条件；不沿用参考后文与修订段落冲突的分支描述 |
| 5.12 NSA | [正文](https://tcn6r3nlptym.feishu.cn/wiki/Kns8w3HQKiE22OkBtAVczEYOnVb) | 正文与可见代码读完，未执行。静态检查发现 softmax 后 mask、未来压缩块、query 索引等风险；本站另写因果归一化反例，不转载或声称复现该实现 |
| 5.13 YaRN | [正文](https://tcn6r3nlptym.feishu.cn/wiki/TIoJwdkdJiyB5GkrU0PcMaPindg) | 正文读完。补训练窗口圈数、频率分段表和 toy ramp；区分角度与频率、维度与 token 位置、attention scaling 与 sampling temperature |
| 5.14 MTP | [正文](https://tcn6r3nlptym.feishu.cn/wiki/CGxzwkEA6i8FTXkm0tZcf9WInBd) | 正文读到速度总结与评论入口；图内推导未全审。补随机接受/拒绝的两词概率账、有效前缀与缓存回退、按毫秒而非步数比较加速 |

本批核对：[DeepSeekMoE](https://arxiv.org/abs/2401.06066)、[DeepSeek-V2](https://arxiv.org/abs/2405.04434)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[NSA](https://arxiv.org/abs/2502.11089)、[YaRN](https://arxiv.org/abs/2309.00071)、[Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)、[Qwen3](https://arxiv.org/abs/2505.09388)。本站代码仅验证小型数学算例，没有完成模型训练或 GPU 性能复现。

### DCA 到 AttnRes：2026-10-07 继续阅读

本批读完以下 5 篇正文。DCA 展示的 92 行代码、Gated Attention 展示的 64 行代码已静态查看，均未执行；图片中的推导和性能表未逐张全审。下表记录的是后续写作要处理的问题，不是已完成的本站章节。

| 节 | 参考入口 | 核对后要讲清楚的地方 |
| --- | --- | --- |
| 5.15 DCA | [正文](https://tcn6r3nlptym.feishu.cn/wiki/TOvOwEvUdiy1Bxk2kJWc55y5n8c) | 改的是 RoPE 位置映射，不是丢弃大部分历史 token；必须保留因果约束和跨块的全局归一化 |
| 5.16 DSA | [正文](https://tcn6r3nlptym.feishu.cn/wiki/OF7GwQCbXi9xFkkCxdtc0RKnnGc) | 页面对应 V3.2-Exp，不能不加区分地套到所有 V3.2 版本；动态 top-k 选择不等于只存 top-k KV |
| 5.17 Gated Attention | [正文](https://tcn6r3nlptym.feishu.cn/wiki/ST76wtc05ikXNJkd4EjccOzrnUf) | 门控位置、粒度和输入来源分开讲；平均门值不是“删掉的信息比例”，软门控也不自动省去计算 |
| 5.18 Engram | [正文](https://tcn6r3nlptym.feishu.cn/wiki/PNvOwsfImiTSiBkm8cKcxO5yn4c) | 归一化文本不是跨语言语义合并；可训练的 n-gram 表不是外部文档库，也不是完整的个人长期记忆 |
| 5.19 Block AttnRes | [正文](https://tcn6r3nlptym.feishu.cn/wiki/Gcg7wk19oiSfTFkchElcrDHLnxf) | 沿深度选信息，不是沿序列选 token；固定 pseudo-query 不代表固定权重；均匀平均与单位权重求和有区别 |

#### 位置重映射：先确认代码到底实现了什么

[DCA 原论文 §3](https://arxiv.org/html/2402.17463v2)区分块内、相邻块和更远块的 query 位置。相邻块分支的 query 索引从 chunk size `s` 起，而不是从 `c-s` 起；相对索引上界是 `c-1`。后续可以用跨边界的一对相邻 token 演示，而不先堆满三套符号。

参考页代码没有 RoPE 或对应的位置重映射，不能当作 DCA 复现。静态查看还发现：`zip` 沿 batch 维迭代、没有显式 head 拆分、没有补齐尾块、相邻块逻辑涉及未来块、独立 softmax 后直接相加等问题。这里没有执行代码，不报告运行结果或性能。

#### DSA：稀疏计算与缓存管理不是同一件事

[官方 V3.2-Exp 实现](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)仍保存历史 index key、KV latent 和位置分支；每个 query 再选 top-k。下一步的 query 可能选到上一步没选中的 token，因此不能据“这次不参与计算”推出“可以永久删掉”。indexer 本身也要评分，不能只算主 attention 的 top-k 成本。

该文件还用 dense scores 加 mask 演示稀疏语义；它不等于高效稀疏 kernel。后续讲解要分别展示选择逻辑、缓存占用和真实延迟，不能拿这个参考实现证明加速。warmup / KL 训练配方和后训练细节仍需完整核对技术报告。

#### Gated Attention：把抑制幅度与信息量分开

[原论文 §3–4](https://arxiv.org/html/2505.06708)的默认配置是在 SDPA 输出、输出投影之前做 head-specific sigmoid 乘法门控。门值来自当前位置的 hidden state。参考页额外加了两层 MLP 与 LayerNorm，应标为实现变体，不称为论文的默认实现。

例如两个通道原本是 `10` 和 `0.01`，同样的平均门值可以主要压低大通道，也可以主要压低小通道；光看门值均值无法量出损失了多少信息。attention sink、稳定性和长上下文收益要附实验条件，也不能把整块 attention 说成线性变换。

#### Engram：查表很便宜，不代表整个模块免费

[Engram v1 §2](https://arxiv.org/html/2601.07372v1)与[官方示意代码](https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py)已核对。文本规范化不会自动把中文词与英文翻译合成一个 ID；主干的输入 embedding 和输出词表也没有因此整体替换。缺少专门的查表模块，不等于 Transformer 无法存储或利用知识。

后续用自拟短语解释哈希碰撞、上下文门控，再拆存储容量、传输量和计算量。固定阶数 / 头数下的常数次查表，不保证固定端到端延迟；CPU offload 还受带宽、命中率与预取窗口影响。官方 demo 对 attention / MoE 等做了 mock，不能当完整训练工程。已核对的架构版本为 v1，后续版本差异另查。

#### AttnRes：让读者先看到“平均”和“求和”的区别

[Attention Residuals §3、§5](https://arxiv.org/pdf/2603.15031)确认零初始化 pseudo-query 得到均匀权重，但那是平均，不是原样恢复标准残差的单位权重求和。自拟例子：两个源分别为 `2`、`6`，均匀 softmax 得到 `4`，相加得到 `8`；不能忽略尺度就声称完全等价。

query 是参数，key 仍来自输入，所以权重仍会随内容变化。Block 方案要区分已完成块、当前块的 partial sum 和单独保留的 embedding；它不是只在块边界做一次注意力。Full 不是已证明的“理论最优”，约 8 块和低于 2% 的延迟开销也不是通用保证。原论文关键公式已核对；全篇实验、附录和 GPU 实现未复现。

### 模型家族：2026-10-08 续读与写作

| 节 | 参考入口 | 已读范围与本次处理 |
| --- | --- | --- |
| 6.1 GPT | [正文](https://tcn6r3nlptym.feishu.cn/wiki/ArCCwU4NkiR7JXkpNmgcKLmdnsj) | GPT-1/2/3、InstructGPT/ChatGPT、gpt-oss、GPT-4 到末尾 GPT-5 链接均已读；新增原创双语学习方式比较、示范标签对照和 sink 概率账。没有声称完整覆盖 gpt-oss 的训练、量化与评估细节 |
| 6.2 Llama | [正文](https://tcn6r3nlptym.feishu.cn/wiki/UQvjwfezEik1fbk7erKcQwecnme) | 读至后训练迭代、RM 用途和评论入口。扩充分版本比较、Q/K RoPE 流程、GQA 缓存账、数据实验思路、SFT/DPO 流程和相对概率反例 |

核对后的重点：GPT-1 保留 self-attention，删除的不是全部 multi-head attention；GPT-2 的 WebText LM 训练不是逐任务监督 loss；GPT-3 few-shot 评估不做参数更新；InstructGPT 基于 GPT-3。gpt-oss sink 加在分母，真实 token 权重不能再次归一化到 1；其公开结构不代表闭源 GPT 架构。GPT-4 参数量未公开，GPT-5 已有 system card，不能继续写成“没有任何技术资料”。

Llama 需要分开词表和 context，分开初版 Llama 3 与 3.1 / 405B 报告。RoPE 不是单独解决无限外推的方法；GPT-2 已有 Pre-LN；Llama 2 不同尺寸的 GQA 配置不同。参考中的 40M-token annealing 在报告 §3.4.3 确有记录，但 §3.1.3 的 40B-token 数据质量实验是另一件事，不能混成一个数字。DPO 不保证 preferred 的绝对概率上升，也不保证迭代筛选质量单调提高。

本站原有 Llama 图把 RoPE 画在 embedding 上，并把 post-training 接在 logits 后面，本轮拆成推理 block 与训练数据流程两张图。新增算例是数学和存储核对，不是训练复现或 GPU 性能结果。主要依据：[GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)、[GPT-2](https://cdn.openai.com/better-language-models/language-models.pdf)、[GPT-3](https://arxiv.org/abs/2005.14165)、[InstructGPT](https://arxiv.org/abs/2203.02155)、[gpt-oss](https://deploymentsafety.openai.com/gpt-oss)、[Llama 2](https://arxiv.org/abs/2307.09288)、[Llama 3](https://arxiv.org/abs/2407.21783)、[GQA](https://arxiv.org/abs/2305.13245)、[DPO](https://arxiv.org/abs/2305.18290)。

### DeepSeek、Qwen 与分布式：2026-10-08 续读

| 参考入口 | 实际读到哪里 | 本站本批处理 |
| --- | --- | --- |
| [6.3 DeepSeek](https://tcn6r3nlptym.feishu.cn/wiki/OhOXwh0R3iMjiYkQM7vcr8T1nNc) | MoE、V2、V3、R1-Zero、R1、V4，至 V4「其他优化」末尾的 TileLang 段和评论入口 | 修正 V3 / R1 流程图、MLA 缓存含义、R1 四阶段与蒸馏分支；V4 仅保留版本与机制边界 |
| [6.4 Qwen](https://tcn6r3nlptym.feishu.cn/wiki/EJvuwmfZ0iFpbVkG0DecXdEFnab) | Qwen1 / 2 / 2.5 / 3 / 3.5，至 MoE 最终输出公式后的文字解释与评论入口 | 补初版 hybrid 与 Instruct-2507 区别、四阶段 / 小模型蒸馏区别、3.5 型号范围与缓存边界 |
| [第 7 章](https://tcn6r3nlptym.feishu.cn/wiki/WZ0KwamzhiPj89kEq5dcYdegnWg) | 数据并行、PS、Ring、GPipe、PipeDream、TP、ZeRO / Offload、SP、EP，至 7.7 总结 | 增补 DDP / collective 层级、1F1B 与更新时机、SP / CP / EP 对照和 Offload 成本；没有声称实现或性能复现 |

#### DeepSeek：缓存和训练流程最容易讲偏

- V2 的 MLA 是 **Multi-head Latent Attention**。逐 token 存 latent 与位置 key，不是不断覆盖一个全局历史向量；缓存依然随长度增长。[V2 §2.1](https://arxiv.org/html/2405.04434v5#S2.SS1)已核对。
- R1 原始报告是 cold-start SFT → reasoning RL → rejection sampling + SFT → all-scenarios RL。筛选数据后重新微调 V3-Base，小模型蒸馏另列 §2.4，不是第四阶段。本站原有图也将 V3 对话后训练与 R1 串在一起，本批纠正。依据：[R1 v1](https://arxiv.org/html/2501.12948v1)。
- V3 的无辅助损失负载均衡、sequence-level 辅助项、MTP 的因果顺序与 speculative acceptance 还需在家族文章展开；原有机制文章保留，不算新增完整覆盖。
- V4 参考链接的旧 PDF 路径返回 404，已从官方模型卡找到 [V4 报告](https://arxiv.org/abs/2606.19348)。确认 CSA/HCA、mHC、Muon 与多专家 OPD 的报告身份和主要范围。报告 §2.2 的 Sinkhorn 残差映射约束，与 §2.4 的优化器更新正交化不同；不采纳“Muon 保证权重不偏离 mHC 流形”“绝对稳定”“完美兼容”等无条件表述。逐层缓存布局、完整训练系统与实验仍待审，不把 V4 概述当作完整精读。

#### Qwen：按型号和训练路线写，不按家族一概而论

- Qwen1–2.5 的正文已读，但各尺寸的词表、context 和训练 token 数尚未逐项回报告核对；不要直接搬参考数字或把数据规模比较当因果实验。
- [Qwen3 报告 §4](https://arxiv.org/html/2505.09388v1#S4)区分大模型四阶段路线与小模型 strong-to-weak distillation；不是每个模型先做四步再做“第五步蒸馏”。on-policy 指学生轨迹，不是 teacher 服务是否在线。
- 初版 hybrid 模型支持模式切换；[Instruct-2507 官方模型卡](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)明确只支持 non-thinking。原 Qwen3.5 GitHub 链接如今跳转到后续家族仓库，不能把当前 README 当作旧版本结构证据，改核 [397B-A17B 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)。
- 该型号的 3:1 Gated DeltaNet / Gated Attention、512 routed experts 与 10 routed + 1 shared 已与模型卡核对。固定递归状态不等于混合模型全部缓存恒定；prefill、单步 decode、全序列生成的复杂度必须分别算。
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464)的记忆矩阵有容量与碰撞限制。沿当前 key 更新会影响与其非正交的方向，不能宣传为“删除指定记录且完全不干扰其他记忆”。公式与 chunkwise 实现仍需单独展开并测试。

#### 分布式：拆分维度、collective 和调度不能混成一张排行榜

- DDP 不等于 Ring AllReduce，PyTorch DataParallel 不等于所有 parameter-server 系统；单机也能用 DDP。算法层、框架层、通信层应分别说明。
- 1F1B 是操作调度，不自动意味着异步 optimizer 更新。已核对 [PyTorch pipeline schedules](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html)；PipeDream 的 weight stashing 与参数版本需要独立讲解，原参考中“torch PP 只基于 GPipe”的说法不宜沿用。
- [Megatron CP 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)明确区分 SP 与 CP。CP 保留跨段 attention 依赖，不能把“各卡独立算完拼接”作为完整算法。
- EP 的 top-k 比例不是网络字节比例。capacity factor 要说明分母是否包含 token-expert assignments；参考的 top-2 例子与单路 capacity 口径需要另核，不先写死公式。
- 不采用未注明硬件、batch、sequence、实现的固定吞吐或通信百分比；ZeRO 的理想状态分片也不保证任意规模模型可训练。Offload 是存储、带宽、CPU 计算与更新时机的取舍，不是免费显存。

本批没有执行参考实现，图内公式与论文附录未全部审计。其后第 8 章的阅读记录见下；不要把尚未逐页读取的部分算入累计数。

### 应用篇：2026-10-08 续读

以下 5 页正文已读至末尾，不等于外链项目、图片公式和参考代码已全部验证。写作只使用原创解释与小例子，不搬运第三方段落。

| 节 | 参考入口 | 需要讲清楚的边界 / 本站处理 |
| --- | --- | --- |
| 8.1 Prompt | [正文](https://tcn6r3nlptym.feishu.cn/wiki/YDBGwTpuXi3UfVky1XOcOPB3nyb) | ICL 不在推理时更新权重；few-shot、CoT 与参考答案是不同选择。分隔符不是权限边界，推理文本不保证忠实。RAG 页补关键边界；独立 prompt 专题仍待写 |
| 8.2 RAG | [正文](https://tcn6r3nlptym.feishu.cn/wiki/KLezw0YLFilqcSk11XmcslFlnYd) | 补文档版本 / 位置 / ACL、分块例外丢失、Hit / Precision / Recall / RR 算例与生成诊断。参考 Python 示例只静态阅读，不沿用无条件截字符或把检索顺序当原文邻接的做法 |
| 8.3 Agent | [正文](https://tcn6r3nlptym.feishu.cn/wiki/Om6UwdJLRiH8Pjkd2lAc8f10nhh) | ReAct 不要求两台模型；memory / vector store 非所有 agent 的必选件。扩充 patterns 双语页：执行记录、function calling / executor / MCP、权限、幂等与失败测试；参考框架 API 未运行 |
| 8.4 Deep Research | [正文](https://tcn6r3nlptym.feishu.cn/wiki/GTNlwrKfbiln0BkKVCJceoEwnbc) | 读至 Jina 项目链接及评论入口。补证据表、依赖与停止条件；四个外链工程未全面审计，完整实现与实验待写。open_deep_research 仓库当日已显示 2026-08-21 archived，不能直接沿用旧推荐作当前选型 |
| 8.5 向量检索 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/EacIwwlWaiJQL4k7QHscpqywnUd) | 补 IVF / PQ / ADC 数值例子、残差编码、ANN recall 与相关性区别。普通 PQ 编号不直接保证汉明几何；SDC / ADC 的速度不作无条件排序。检索后精确距离重排不是 cross-encoder，无法找回未入选向量 |

本批核对：[GPT-3](https://arxiv.org/abs/2005.14165)、[CoT 忠实性](https://arxiv.org/abs/2305.04388)、[RAG](https://arxiv.org/abs/2005.11401)、[Ragas](https://arxiv.org/abs/2309.15217)、[ReAct](https://arxiv.org/abs/2210.03629)、[MCP 架构](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture)、[Faiss 索引](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)、[距离约定](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)、[PQ 作者稿镜像](https://paper-notes.zhjwpku.com/assets/pdfs/Product_Quantization_for_Nearest_Neighbor_Search.pdf)、[Polysemous Codes](https://arxiv.org/abs/1609.01882)、[HNSW](https://arxiv.org/abs/1603.09320)、[IR 指标教材](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)。PQ 的 HAL 入口被拦截，改读可访问的同一作者稿，不声称已复核全部实验。

应用批次新增 RAG / 向量索引双语页并扩充 Agent；Prompt 和 Deep Research 保持部分讲解状态。

### 多模态：2026-10-08 续读与写作

[第 9 章目录](https://tcn6r3nlptym.feishu.cn/wiki/D3qow8K9WiS2KUksNtMcqvnrnPe)的 16 个子页面已全部列入覆盖表，不把一个总入口当作全覆盖。

| 已读页面 | 阅读范围 | 本站处理 |
| --- | --- | --- |
| [ViT](https://tcn6r3nlptym.feishu.cn/wiki/OvObw0kRwiQZiWkGyFachCvBnWe) | 正文到底；PatchEmbed / Block 展示代码静态阅读 | 新增双语 `vit`：patch 小例子、形状、位置、分辨率成本；区分 fine-tuning 与 linear probe |
| [CLIP](https://tcn6r3nlptym.feishu.cn/wiki/EMbxwcL3ZiRecUkGtj1cPbDVnVh) | 正文到底；loss / ChineseCLIP 代码可见部分，未运行 | 扩充双语 `clip`：配对也是监督；候选列表改变概率；文本分支能力不能直接外推 |
| [BLIP-1](https://tcn6r3nlptym.feishu.cn/wiki/Tc5UwyV5diW4ytkb8XNcumxrn2f) | 正文到底；代码滚到第 182 行末尾，中间未逐行审计 | `blip-and-q-former` 区分目标与 CapFilt；完整实现保持待补 |
| [BLIP-2](https://tcn6r3nlptym.feishu.cn/wiki/VCMgwIOo6i50oDkrqAAceUpDncy) | 正文到底，含三目标展示代码与第二阶段，未执行 | 同篇补 mask 代码、max / mean、视觉前缀成本与冻结梯度 |
| [InstructBLIP](https://tcn6r3nlptym.feishu.cn/wiki/IsY2wd0N0icbX5kG55CcBPhonod) | 正文到底，19 行展示代码静态阅读 | 同篇说明指令进入现有 Q-Former、任务采样和缓存边界；完整实验待补 |

相关机制已回到 [ViT](https://arxiv.org/abs/2010.11929)、[CLIP](https://arxiv.org/abs/2103.00020)、[BLIP](https://arxiv.org/abs/2201.12086)、[BLIP-2](https://arxiv.org/abs/2301.12597)、[InstructBLIP](https://arxiv.org/abs/2305.06500)及 ViT / LAVIS 官方实现核对。本站文章附对应链接；原创切块、mask 和梯度算例不充当训练复现。没有运行参考代码，也没有完整审计全部图片、视频、附录和实验。

不能由 BLIP-2 的模型比较推断 encoder-decoder 天然比 decoder-only 泛化更好；冻结权重也不保证输入变化后的行为不变。

后续 11 页已读完正文，本站新增 `vlm-designs`、`qwen-vl`、`omni-streaming`、`ocr-compression`、`image-generation`、`vlm-finetuning` 中英各一篇。参考代码没有执行，图内公式未全审。微调页原有内容主要是旧环境与入口，不沿用缺少配置的固定显存结论；本站改写为样本、mask、冻结、长度与独立评估的实验流程，明确尚未跑端到端 GPU 训练。

2026 更新已核对：[OCR 2](https://arxiv.org/abs/2601.20552)、[Qwen3.5-Omni](https://arxiv.org/abs/2604.15804)、[Qwen3.8-Omni](https://arxiv.org/abs/2609.25611)、[Qwen-Image-2.0](https://arxiv.org/abs/2605.10730)、[Qwen-Image-2.0-RL](https://arxiv.org/abs/2606.27608)。Qwen-Image-2.1 只引用官方发布公告所述能力，不声称已审计其训练配方或本地复现。公开发布、开放权重和可复现训练是不同状态。

新增 `07-evaluation/text-metrics` 双语页，补 BLEU、ROUGE、编辑距离及 PPL 的小例子与协议边界。待写主题、部分讲解和发布条件统一记录到 `CONTENT_RELEASE_CHECKLIST.md`；本批不推送。

### 训练后半部分的可恢复阅读入口

| 节 | 参考入口 | 本轮状态 |
| --- | --- | --- |
| 2.5 PPO | [正文](https://tcn6r3nlptym.feishu.cn/wiki/PBDrwSrZkidbtDkZST0cMlmFnsc) | 正文与展示的 loss 代码读完；代码未运行 |
| 2.6 解码 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/PQBVwaZkCiZrGBk0DOuco95znNY) | 正文与展示代码读完；top-p 阈值边界、原 ID 映射需修正；本站另写示例并测试 |
| 2.7 DPO | [正文](https://tcn6r3nlptym.feishu.cn/wiki/QvxQwZofbiKqbckGsxEcAETqnpe) | 正文与展示代码读完；图片内推导未全审 |
| 2.8 GRPO | [正文](https://tcn6r3nlptym.feishu.cn/wiki/SVLawiEfbiVUVwk62QQcPjCbnYc) | 正文与可见代码读完；网页输入准备代码止于第 125 行，不能算完整实现 |
| 2.9 DAPO | [正文](https://tcn6r3nlptym.feishu.cn/wiki/JHjrwiuUsihQyMkTN1jcMa3Dn4f) | 正文读完；公式与实验需回论文复核 |
| 2.10 GSPO | [正文](https://tcn6r3nlptym.feishu.cn/wiki/AsfuwqgFwiJ3cXkdAC3cXn0vnpf) | 正文与伪代码读完；分组、分母命名与 token/sequence reduction 需回原实现核对 |
| 2.11 ASPO | [正文](https://tcn6r3nlptym.feishu.cn/wiki/Ndypw4wEBiQtO0kjYp8cd09Wnsb) | 正文读完；已对上 [原论文](https://arxiv.org/abs/2510.06062)，未完成推导审计 |
| 2.12 SAO | [正文](https://tcn6r3nlptym.feishu.cn/wiki/CgIcwhNwZiIsjJkwF4Tc0REUnVg) | 正文读完；已对上 [原论文](https://arxiv.org/abs/2607.07508)，未复现实验 |
| 3.1 Prompt Tuning | [正文](https://tcn6r3nlptym.feishu.cn/wiki/BJUcwu2rpiWeUFkYux2cy2COnhf) | 正文读完；区分 soft prompt 与普通 prompt |
| 3.2 Prefix Tuning | [正文](https://tcn6r3nlptym.feishu.cn/wiki/G8zOwPqKJiUUMwkN65jcwLcSnre) | 正文读完；比较参数化、缓存与部署代价 |
| 3.3 Adapter Tuning | [正文](https://tcn6r3nlptym.feishu.cn/wiki/Dt8nwCXBwikA0lkUITAcxibgnJb) | 正文读完；“无额外推理开销”不能照搬 |
| 3.4 LoRA | [正文](https://tcn6r3nlptym.feishu.cn/wiki/BUKRwWkWFijNkakPsBVcolYfnkb) | 正文读完；原权重 / 更新矩阵、维度与初始化已与本站对照 |
| 3.5 QLoRA | [正文](https://tcn6r3nlptym.feishu.cn/wiki/RbqIw7nZHiB9hekrJV9cRX34nZf) | 正文读完；补原始论文核对及存储账例子 |
| 3.6 蒸馏 | [正文](https://tcn6r3nlptym.feishu.cn/wiki/WcHOwTcIHil5q8kMbW7ceep3nnh) | 正文读完；图内推导未全审，外链训练代码未运行 |

## 顶层对应，不另起一套重复目录

| 参考范围 | 本站位置 | 后续核对重点 |
| --- | --- | --- |
| 1 基础 | `00-foundations/` | Embedding、Tokenizer、函数逼近、Encoder、Decoder 的完整推导与反例 |
| 2 训练与推理 | `00-foundations/deep-dives/`、`05-post-training/` | 预训练流程、采样、偏好与 RL 目标；后续算法的估计量和前提 |
| 3 蒸馏与微调 | `05-post-training/` | 更新哪些参数、老师给什么、训练与部署的差别 |
| 4 评估 | `07-evaluation/` | 补 BLEU/ROUGE/edit distance；分别核对 perplexity、长上下文与 benchmark 的协议 |
| 5 优化技术 | `00-foundations/`、`06-systems/` | 单独拆清数学改动、存储改动、算子改动，不只介绍名字 |
| 6 模型家族 | `00-foundations/model-families/` | 读到具体型号后逐版本核对，不用家族名称代替架构 |
| 7 分布式训练 | `06-systems/`、`05-post-training/post-training-infrastructure.md` | 并行切分、通信、显存估算与失败语义 |
| 8 应用 | `02-memory/`、`04-search/`、`10-agents/` | 检索、状态、工具、记忆及可验证的失败案例 |
| 9 多模态 | `03-multimodal-learning/` | 读到完整子目录后补齐模型、数据、目标与评估 |
| 附录 面试、LeetCode、PyTorch | `interview/`、`learn/ml-exercises/`、`00-foundations/code/` | 原创可迁移练习；不转载私人面试题 |

## 已见到的子主题：逐项核对

“已有入口”只说明能找到相关内容，不表示深度足够；“本轮补充”也不等于已覆盖参考正文所有细节。

| 主题 | 本站现状 / 下一步 |
| --- | --- |
| Embedding | 已有 `core/embeddings-and-similarity`；需与参考正文逐项比对 |
| Tokenizer：BPE、WordPiece、BBPE | 本轮新增 `deep-dives/tokenizer-algorithms`，另补 Unigram、兼容性、实际成本和词表保留测试 |
| 函数逼近、Encoder、Decoder | 已有 `from-linear-to-neural`、`core/vanilla-transformer`、`core/decoder-only`；需复核覆盖和深度 |
| 预训练 | 新增双语 `deep-dives/pretraining-pipeline`：来源、清洗、去重、混合与重复、packing 两种语义、全局分母、预算与恢复 |
| SFT、Reward model、PPO、DPO、GRPO | 已有正文；按同一组问题检查数据、目标、估计量、失败条件与代码 |
| 推理策略 | 已有 `core/decoding`；本轮补 prefill/decode 与 KV 成本，不代替解码策略全文核对 |
| DAPO | `after-ppo` 有概述；仍需完整数值例子与目标差异 |
| GSPO、ASPO、SAO | 参考正文已读，论文身份已确认；仍需完整论文审阅、数值例子与实现验证，再补独立章节 |
| Prompt Tuning、Prefix Tuning | `model-adaptation` 已有公式和例子；待对照参考细节 |
| Adapter Tuning | 新增双语 `parameter-efficient-tuning`：与 Prompt、Prefix、LoRA 用同一小模型比较插入位置、参数与部署开销 |
| LoRA、QLoRA | 本轮独立双语页：参数量、梯度、初始化、激活、量化和合并误差 |
| 知识蒸馏、数据蒸馏 | 本轮独立双语页：soft CE/KL、前缀来源、top-k 尾部、异构输入及独立评估 |
| SwiGLU | 新增双语 `core/ffn-and-gates`：逐维计算、门控梯度、等参数预算与 MoE 的区别 |
| RMSNorm | `core/normalization` 已有；本次补平移例子、二阶矩与方差区别、Pre/Post 位置及尺度不变性的条件 |
| RoPE | 新增双语 `deep-dives/position-and-context`：旋转恒等式、频率、插值与 YaRN 取舍、长文本评估 |
| KV cache、MHA/MQA/GQA | 有单独计算、缓存表和显存账；本次补原始缓存比例、head / GPU 数量区别与 repeat 复制边界 |
| 混合精度 | 双语 `deep-dives/precision-and-memory`：范围与分辨率、舍入、显存账；本次补 autocast 与存储区别、裁剪前 unscale 的数值反例 |
| FlashAttention | 双语 `deep-dives/attention-kernels`：online softmax 逐块计算及可运行参考实现；本次补 FA2 / FA3 的工作划分和精度边界，不冒充 GPU kernel 教程 |
| PagedAttention | 同篇有块分配、共享、copy-on-write 与回收；本次补内外碎片、连续组批数值表与 chunked prefill 取舍；未提供真实 serving benchmark |
| Gradient Checkpointing | 精度与显存篇含重算路径、RNG/副作用和评测方案；本次补隐藏状态 / 分数张量的 8/256 MiB 账；真实 GPU 时间/显存实验仍未完成 |
| MoE、DeepSeekMoE | 本次扩充预算、路由、均衡三篇双语笔记：matched MAC 账、bias 选人与权重分离、expert/device 负载；小型算例测试，不是完整训练实现 |
| MLA | 新增双语 `deep-dives/latent-and-sparse-attention`：投影吸收的手算与代码、位置分支、缓存维度假设 |
| NSA、YaRN、MTP、DCA | 本次补 NSA 三路径图和 future-mask 反例、YaRN 圈数与频率表、MTP 接受/拒绝概率质量与延迟算例；DCA 待补 |
| 分布式训练 | 新增双语 `06-systems/distributed-training`：DDP 分母、ZeRO/FSDP 状态账、TP 矩阵拆分、PP 理想 bubble；尚不是部署实操 |
| DeepSeek Sparse Attention、Gated Attention、Engram、Block AttnRes | 参考正文已读，已核对上面列出的原论文 / 官方实现；独立讲解、算例测试与完整版本审阅仍待完成，公开清单保持待补 |

## 完成一篇的要求

2026-10-08 模型与训练补充：GPT 增加公开训练边界及配对成本例子；DeepSeek 增加路由选择 / 混合权重区分与奖励检查；分布式增加 Ring、旧梯度、旧权重、Offload 和恢复；BLIP 增加数据筛选与 InstructBLIP 消融设计。82 项现为 79 项有正文、3 项部分讲解。局部代码有测试，不等于已复现模型训练或完成参考附录、全站验收；仍不推送。

2026-10-08 基础与应用补充：函数逼近加入折线误差界与不可识别性例子；NSA / YaRN 加入选块、门控和缓存边界；新增 Prompt 与 Deep Research 双语专题。82 项现为 74 项有正文、8 项部分讲解。27 篇附录仍未读完，Mac 锁屏期间没有把浏览器核对标为完成。

2026-10-08 V4 补充：独立双语页区分 CSA / HCA、缓存与读取账、mHC 固定映射的非扩张性质及整网边界、领域老师与 MoE expert。记账、KL 与残差算例有测试；未执行完整模型。82 项现为 69 项有正文、13 项部分讲解。

2026-10-08 评估协议补充：新增双语 benchmark 选型与长上下文测试篇，包含固定版本清单、移动证据与无答案控制的原创生成器、配对比较和污染边界。82 项现为 68 项有正文、14 项部分讲解；不代表全站验收通过。

2026-10-08 Qwen 补充：新增 Qwen1–2.5 历史版本与 Gated DeltaNet 双语正文；递归式、单位 key 插值、衰减后纠错、重置 / 恢复均有数值测试。Qwen 主文按 2026-10-08 官方模型卡补 3.8-27B、2.4T-A95B 和 Flash-Next 的架构 / 服务区别，不把托管 API 的结果归给公开权重。82 项现为 66 项有正文、16 项部分讲解、0 项完全待写；附录与全站验收仍未完成。

2026-10-08 架构补充：DCA / DSA 补入原有篇章，新增 Gated Attention、Engram、Attention Residuals 中英文各一篇。区分位置近似、选择器、输出门、参数表记忆与深度混合，提供边界、错选、初始化和归一化算例。5 个 pending 转为实质正文；不代表最新家族、附录和全站验收已完成。

2026-10-08 后训练补充：新增 DAPO、GSPO / ASPO、SAO 双语正文与局部数值测试；DPO / GRPO 复核保留原有推导，修正 Critic、clipping、IPO 与 KTO 的过度概括。GSPO 的长度归一化不等于无偏轨迹 IS，ASPO 的 detach 改变导数方向，SAO 前向筛选不冒充完整训练实现。新增分布式 token 均值的分母检查。3 个 pending 与 DPO / GRPO / DAPO 的 3 个 overview 转为实质正文，PPL 加入文本指标页；全站版本与附录审查仍未完成。

2026-10-07 补充：以上新增使用原始论文、官方文档与自拟例子。它们补齐本轮确认的机制缺口，但不代表飞书未读正文已经逐项覆盖。评估、检索、Agent、多模态全章复核和新增架构版本仍在待核对范围。

1. 读者能跟着一个例子解释输入、计算、输出；不只有定义。
2. 公式解释符号和前提，示意数字能用测试复算。
3. 至少一个合理替代方案，以及“什么情况下这个办法无效”。
4. 引用原始论文或官方文档；参考笔记只作为覆盖线索，不当作最后证据。
5. 中文像在解释问题，不照英文句式翻；英语版包含同等内容。
6. 导航和旧链接可用，窄屏可读，不新增空壳章节或统一打卡模块。
