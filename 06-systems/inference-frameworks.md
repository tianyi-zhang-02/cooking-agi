# vLLM 和 SGLang：生成引擎在忙什么？

**中文** · [English](inference-frameworks.en.md)

本地试模型，调用 `generate()` 就能看到回答。把它做成服务后，麻烦往往出在同时到来的请求上：一个人发来长文档，另一个人只问一句话，还有人的答案生成到一半就关了页面。这些请求该怎么排，哪些可以一起算，腾出来的显存又该交给谁？

vLLM 和 SGLang 主要就在处理这些事。模型仍然负责预测下一个 token，引擎负责把许多请求安排到有限的 GPU 上。下面从一条请求讲起，再看缓存、调度和多卡部署怎样影响它。还不熟悉生成过程的话，可以先读 [prefill、decode 和 KV cache](../00-foundations/deep-dives/kv-cache-and-inference.md)；服务接收请求、返回结果的完整流程放在[一次请求怎样跑完](llm-serving.md)。

<details markdown="1">
<summary>本文使用的版本和源码</summary>

正文以 vLLM v0.31.0（2026-10-05）和 SGLang v0.5.21（2026-10-02）为主，文档核对于 2026-10-12；源码链接固定到 2026-10-10 的 main commit。2026-10-21 又对照了 [vLLM v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) 和 [SGLang v0.5.22](https://github.com/sgl-project/sglang/releases/tag/v0.5.22) 的发布说明，发现本文涉及的 ROCm model runner 默认行为有变化，已在源码补充中注明。

生态与选型部分的官方资料核对于 **2026-10-31**。这部分介绍已有的集成入口；下文的源码分析仍按各处注明的版本阅读，不把旧版本的默认值当作所有新版本的行为。

文中的数字用于手算，没有运行两套引擎的 GPU 性能测试；它们不代表某个版本的实测表现。

</details>

## 选 vLLM 还是 SGLang？ {#ecosystem}

如果只比较“有没有前缀缓存、连续批处理、结构化输出和多卡部署”，vLLM 与 SGLang 已经有很大的重合。不能再简单地说，一个管显存，另一个适合多轮对话。实现差异还在，但选型时通常更值得先问：**我的模型、硬件和现有系统，接哪一套更顺？**

这里说的生态（ecosystem），不只是项目多、star 多。还包括目标模型有没有持续维护，所需的功能能不能一起开启，部署工具是否接得上，以及出了问题后有没有可复现的案例和修复。下面几个官方入口能看出两边怎样往引擎之外延伸：

- **vLLM**：除了引擎本身，[Production Stack](https://docs.vllm.ai/en/latest/deployment/integrations/production-stack/)提供 Kubernetes 部署、路由和监控配置，并能接入 LMCache。它是围绕引擎搭建服务的一套参考实现，不是启动一个 vLLM 进程就自动拥有的能力。
- **SGLang**：[Model Gateway](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway)负责在多个 worker 之间分发请求，并提供健康检查、限流和不同的路由策略；[RL 集成指南](https://docs.sglang.io/docs/advanced_features/sglang_for_rl)则讲显存释放与恢复、权重同步、生成暂停等接口。网关和推理实例也是不同组件，可以分别部署。
- **两边并不是互不相通的两套生态**：[llm-d 的 vLLM 集成](https://docs.vllm.ai/en/latest/deployment/integrations/llm-d/)和 [SGLang 集成](https://docs.sglang.io/docs/advanced_features/llm-d)都已有文档；[verl](https://verl.readthedocs.io/en/latest/start/install.html)也支持两种生成后端。共同接入一个框架，不代表每个版本的功能覆盖、依赖和配置都完全相同。

两边都在做部署和 RL 集成，上面只是各选几个入口，不是在给生态成熟度排名。具体到自己的工作，可以从这几项查起：

| 你要做什么 | 具体检查什么 | 为什么会影响选择 |
| --- | --- | --- |
| 上线一个模型 | 固定模型版本、GPU 型号、量化格式，再核对多模态、LoRA、工具调用等所需组合 | 支持这个模型，不等于这些功能组合都经过验证 |
| 接进现有服务 | 网关、部署配置、监控指标、扩缩容、取消请求和故障恢复能否沿用 | 引擎能返回答案，只完成了上线工作的一部分 |
| 接进 RL 训练 | 训练框架与引擎版本、权重同步、token 及 log probability 返回、暂停和恢复行为 | 生成快，不等于整个训练循环快，也不保证样本与权重版本对应正确 |
| 长期维护 | 目标模型的测试、相关 issue、发布说明，以及团队是否熟悉这条运行路径 | 一次压测看不出升级时的回归、排错时间和回滚成本 |

模型适配要查具体条目，而不是只看支持列表的长度。例如 [vLLM 的模型支持说明](https://docs.vllm.ai/en/latest/models/supported_models/)会区分测试程度；[SGLang 的模型目录](https://docs.sglang.io/docs/supported-models)也应继续查到对应模型和功能说明。文档里的“支持”不是你这套配置的验收结果。

**比如，一个团队已经在 Kubernetes 上用 vLLM 部署文档助手。** 监控、告警、回滚和压测都已有积累。想换 SGLang，可以先拿相同请求做影子流量测试，再比较尾延迟、失败率和成本；还要检查工具调用的返回格式、取消请求及监控是否接得上。不能只凭另一套引擎的单卡跑分更高，就认为迁移一定划算。反过来，已有稳定 SGLang 部署的团队也一样，不必为了生态名气重做一遍系统。

**再比如，要给 RL 训练生成样本。** 如果两边都支持所需模型，下一步就看训练框架的适配路径。训练完一轮后，权重能否正确传过去？生成的 token 和 log probability 能否交回训练器？更新权重后，旧 KV 是否正确失效？睡眠、唤醒和恢复要花多久？这些环节可能比单独的生成吞吐更影响训练效率。下面会继续拆引擎内部，完整训练循环可接着看 [RL 系统](rl-systems.md#engines)。

所以这里的比较顺序是：先确认模型与功能组合能跑通，再看能否接进已有系统，最后用实际负载比较性能。生态会影响接入和维护成本，调度、算子和缓存实现则可能直接影响运行成本，两部分都要算。

## 应用、推理引擎和算子，各自做什么 {#layers}

<figure class="worked-update" lang="zh-CN">
<ol>
<li><small>01 / 应用服务</small><strong>准备好模型要读的内容</strong><span>检查用户权限，组织检索结果、工具和提示词，设置超时。无权访问的资料要在交给模型前过滤掉。</span></li>
<li><small>02 / 生成引擎</small><strong>安排这一轮一起算的请求</strong><span>维护队列和批次，分配 KV 空间，把工作交给执行模型的进程（worker），再采样、返回结果。vLLM / SGLang 主要负责这一层。</span></li>
<li><small>03 / 运行与算子</small><strong>在 GPU 上完成计算</strong><span>执行模型中的矩阵乘法、注意力算子（attention kernel）和跨卡通信。这里决定一批工作具体怎么算。</span></li>
</ol>
<figcaption>图里按工作分层，实际可以由多个进程配合完成。权限、请求调度、算子性能要分别处理：换一个更快的算子，解决不了应用权限或请求排队的问题。</figcaption>
</figure>

[vLLM V1](https://docs.vllm.ai/en/v0.31.0/design/arch_overview/) 把引擎内的工作又分了几部分：API server 接收输入、返回输出，engine core 管调度和缓存，GPU workers 跑模型。采用数据并行部署时，还会用到 DP coordinator。部署方式不同，进程数量也会不同。

SGLang 做的事相近，组件名称和组织方式有所不同。在这里链接的 [Engine 源码](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/entrypoints/engine.py#L238)里，请求大致这样走：

```text
请求 → TokenizerManager → Scheduler → 模型执行
        ↑                                ↓
     返回文本 ← DetokenizerManager ← 输出 token IDs
```

TokenizerManager 把输入交给后面的执行流程，同时接收请求结果。Scheduler 选出这一轮要算的请求，模型完成前向计算后，DetokenizerManager 再把输出 token 转回文字。这里先省略多卡拓扑和跨实例路由；部署时可以有更多进程参与。

所以，看到“GPU 不忙，但回答很慢”，值得先看请求到了哪一步。也许模型计算没问题，是前面的输入还没准备好，或者后面的结果迟迟没传回去。

## 把两套实现放在同一张图里 {#engine-map}

下面把普通自回归文本生成的路径并排画出来。两边都要处理输入、选出批次、执行模型、返回结果。箭头旁写的是组件之间传递的内容，多数是请求信息和调度结果，不会每走一步都复制整份模型或 KV cache。多模态输入和推测解码还会增加其他分支。

图上虽然从上往下排，CPU 和 GPU 却可以同时忙不同的轮次：GPU 算当前批次时，CPU 准备下一批。兼容配置下，两边默认都会这样重叠执行，分别叫 async scheduling 和 overlap scheduler。

<figure class="engine-map" lang="zh-CN">
<div class="engine-map__lanes">
<section aria-label="vLLM 请求链路"><h3>vLLM V1</h3><ol>
<li><strong>API server / AsyncLLM</strong><span>校验、整理输入，追踪请求与流式输出</span><small>↓ token IDs、采样参数、request ID</small></li>
<li><strong>EngineCore → Scheduler</strong><span>从队列中选请求，决定本轮 token 数，分配 KV block</span><small>↓ SchedulerOutput：请求更新、token 数与 block IDs</small></li>
<li><strong>Executor → GPU workers</strong><span>把调度结果交给持有模型分片的 worker</span><small>↓ SchedulerOutput → worker 内的 model runner</small></li>
<li><strong>Model runner → kernels</strong><span>准备输入张量与 attention 元数据，再执行模型、KV 读写与采样</span><small>↑ 输出 token / 执行结果，回到调度与输出端</small></li>
</ol></section>
<section aria-label="SGLang 请求链路"><h3>SGLang SRT</h3><ol>
<li><strong>HTTP → TokenizerManager</strong><span>输入处理、请求状态、结果返回</span><small>↓ 分词后的请求</small></li>
<li><strong>Scheduler + cache / pools</strong><span>选出待处理的请求，匹配已有前缀，安排缓存位置</span><small>↓ ScheduleBatch；worker 内转成 ForwardBatch</small></li>
<li><strong>模型 worker → ModelRunner</strong><span>把选好的批次整理成模型输入，执行前向计算</span><small>↓ 模型输入张量与注意力元数据</small></li>
<li><strong>Model → attention backend</strong><span>运行算子、更新 KV，再把输出 token 转成文本</span><small>↑ DetokenizerManager → TokenizerManager</small></li>
</ol></section>
</div>
<figcaption>答案还没生成完的请求，会继续参加下一轮。图中的方框代表组件，不一定是独立进程；例如 SGLang 的 ModelRunner 就在 scheduler 的模型执行路径里。</figcaption>
</figure>

这些组件最后都在为同一条请求做事。假设请求 R 的提示词有 6 个 token，前 4 个已经有可复用的 KV。为了把块数算清楚，这里设每块放 2 个 token，所以命中的是两个完整块；实际引擎的默认块大小并不是 2。

| 走到哪一步 | 这时需要做什么 |
| --- | --- |
| 查到前缀缓存 | 找到前 4 个 token 对应的两个块，并确认可以复用 |
| 安排本轮执行 | 为剩余 2 个 token 分配可写位置；如果本轮已排满，就继续等待 |
| 补算剩余提示词 | 写入这 2 个 token 的 KV，用最后位置的 logits 采样出新 token X |
| 继续生成 | 下一轮把 X 送入模型，生成 X 的 KV，再预测后一个 token |

走完这几步就能看出：**采样出 X 时，X 自己的 KV 还没有算出来。** 采样只是选出 token ID，下一轮把 X 送进模型后，才会写入它的 KV，并预测后一个 token。前缀命中同样不代表可以直接取出答案；即使整个提示词都命中，要得到下一个 token 的 logits，有些实现仍需重算末位置。

图上的箭头也分两种。它们都叫“传递”，开销却很不一样：

- **控制消息**告诉下一步该做什么，比如请求信息、取消通知、本轮批次和完成通知。API 与引擎可以通过进程间通信（IPC）传递这些消息，消息里通常不需要附上模型权重。
- **张量传输**真的要搬计算数据，比如 TP 的集合通信、PP 的中间激活，或跨设备、跨实例传 KV。估算这部分时间，需要知道数据量、带宽、同步方式和设备怎样连接。

<details markdown="1">
<summary>对照源码：从 Scheduler 追到 ModelRunner</summary>

vLLM 的进程划分见[官方架构](https://docs.vllm.ai/en/v0.31.0/design/arch_overview/)。从 [EngineCore](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/core.py) 往下读，会看到它把 `SchedulerOutput` 交给 executor；worker 里的 model runner 再结合本地请求状态，准备 token、位置和 attention 元数据。也就是说，Scheduler 选好了要做的工作，模型输入张量还要由 worker 整理。

这个 commit 在兼容配置下优先使用 [Model Runner V2](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu/model_runner.py)。不支持的功能、缺少 Triton、部分 ROCm 模型的默认设置，或用户显式配置，都可能让它改走旧的 [GPUModelRunner](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_model_runner.py)。后来的 [v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) 把 ROCm 也改为默认使用 V2，去掉了按模型保留旧 runner 的例外。

SGLang 可以沿着 [Scheduler.run_batch](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/managers/scheduler.py#L4405) → [worker 的批次转换与 forward](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/managers/tp_worker.py#L598) → [ModelRunner](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/model_executor/model_runner.py) 阅读。带着上面那条请求去找：剩余的两个 token 在哪里变成张量、缓存位置在哪里传入，会比单独记类名更容易理解。

</details>

## 从 ModelRunner 到 GPU {#runtime-stack}

请求到了 ModelRunner，还要把这一轮各不相同的输入整理成张量。哪些位置是新 token、KV 存在哪里、这一批用哪种 attention 实现，都在这一步衔接起来。再往下，才是具体的模型计算和设备执行。

| 层 | 负责什么 | 继续往下看什么 |
| --- | --- | --- |
| Model runner | 准备本轮 token、位置、缓存索引和模型调用 | 动态 batch 怎样变成可执行张量？为什么 shape 会影响执行路径？ |
| 模型层与 attention backend | QKV 投影、attention、MLP、KV 读写 | Prefill 与 decode 的张量形状有何不同？GQA 怎样影响缓存？ |
| 编译、算子与 CUDA Graph | 生成或选择算子，减少启动与调度成本 | Graph capture 的 shape / buffer 约束；不适配时回退到什么路径？ |
| GPU 内存与通信 | 在 HBM 存权重与 KV，在寄存器 / shared memory 复用局部数据；多卡交换张量 | 计算、显存带宽还是互联限速？通信能否与计算重叠？ |

PyTorch、Triton、CUDA 和厂商库可能出现在不同后端，但不是每次请求都要依次经过的几个“服务”。例如 CUDA Graph 主要减少 CPU 发起 GPU 工作的开销；小批次里，这部分开销可能尤其明显，见[框架这一层](../12-infra/frameworks.md)。它不会减少 KV 必需的存储空间，也不会自动消除 TP 通信。FlashAttention 和分页 KV 管理解决的也不是同一个问题：前者优化注意力计算中的数据读写，后者管理 KV 的分配与寻址，详见[attention kernel 与缓存布局](../00-foundations/deep-dives/attention-kernels.md)。

SGLang 的 [ModelRunner](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/model_executor/model_runner.py) 是继续阅读这些选择的入口。源码能解释它为什么走这条路径；至于在某块 GPU 上快了多少，还需要实测。面试时把读过的实现和亲自做过的调优分开讲就好。

## 多卡部署：加副本、拆模型、分阶段 {#deployment-map}

有了更多 GPU，最直接的做法是多放几份模型，分别接请求。也可以让几张卡一起算同一份模型，或者把处理输入和生成输出分到不同设备上。这三种安排，解决的是不同的问题。

<figure class="worked-update" lang="zh-CN">
<ol>
<li><small>分请求 / Replicas</small><strong>请求 A → 副本 1<br>请求 B → 副本 2</strong><span>多个完整副本分担流量。每份都要存权重，也有自己的缓存；路由到另一份时，之前的前缀可能不在那里。</span></li>
<li><small>分模型 / TP</small><strong>同一请求 → GPU 0 + GPU 1</strong><span>各卡算同一层的一部分，再交换结果。每卡的工作少了，但通信多了，效果很依赖卡间互联。</span></li>
<li><small>分阶段 / P–D</small><strong>Prefill worker → KV → Decode worker</strong><span>输入处理和逐 token 生成分别安排资源。交接时要搬 KV，也要管理路由和两边的队列，因此要先确认分开确实能缓解瓶颈。</span></li>
</ol>
<figcaption>这些方式可以组合使用。这里先说完整的推理副本；MoE 还可能跨副本共享专家，后面会单独举例。</figcaption>
</figure>

比如总共八张 GPU，设 TP=2、运行 4 个完整副本：每个请求交给一个双卡组，4 组各自安排请求，而不是每条请求都用满八张卡。是否比 TP=8 合适，要看模型能否放下、还剩多少 KV 空间，以及单请求延迟、请求到达率和卡间互联。先估算[显存占用](../00-foundations/deep-dives/kv-cache-and-inference.md#inference-budget)，再用有代表性的负载验证。

### 同样八张卡，TP8 和 DP8 有什么不同？ {#eight-gpus}

先看没有共享专家的稠密模型（dense model）。**TP8 是 8 张卡合作跑一份模型；DP8 / TP1 是 8 份模型分别接请求。** 一组 TP8 同样可以把很多请求放进一个 batch，只是这批计算由八张卡共同完成。

假设权重恰好占 16 GiB、每卡 24 GiB，并把全部权重视为可均匀切分。下面只算权重，暂不计 KV、工作区和小型复制参数：

| 配置 | 每卡权重 | 怎样分工 |
| --- | --- | --- |
| TP8 · 1 个协作组 | 理想 2 GiB | 同一批请求在 8 卡上协作。省下单卡权重空间，但层内需要频繁交换结果 |
| DP8 / TP1 · 8 个副本 | 16 GiB | 各副本分担请求，dense 前向不需要副本之间同步。每份还有自己的缓存和队列 |
| DP2 / TP4 · 2 个协作组 | 理想 4 GiB | 每组 4 卡合作，两组分担流量。这不是同时用了 8 份模型 |

表格算出了权重占用，还不能直接回答能接多少请求、每秒生成多少 token。比如 DP8 每卡剩余的 8 GiB，要由 KV 缓存和运行时一起使用，并不是全留给 KV。

权重放得下、目标是总吞吐时，可以先测副本方案；单卡容量不足，或者单请求计算太重时，再比较 TP / PP。TP 的矩阵变小不保证延迟线性下降：通信、kernel 效率和调度都可能抵消收益。[vLLM 扩展指南](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)给出了单卡、TP 与 TP+PP 的部署起点；实际选择仍要测自己的请求分布。

比较时固定总卡数、到达流量、输入/输出长度和缓存预热条件。不要只看离线 tokens/s：也看首 token 延迟（TTFT）、输出间隔、尾延迟、被拒请求和每卡显存峰值。一个吞吐更高、但排队更久的配置，不一定更适合交互产品。

### MoE：副本之间还可能共享专家 {#moe-deployment}

训练里的 DDP 要同步梯度，推理里的数据并行通常只是把请求分给不同副本。不过，MoE 可以把专家放在这些副本共同使用的一组 GPU 上。按 2026-10-09 核对的 [vLLM EP 部署说明](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)，启用专家并行（EP）时，这个专家组的大小是 TP × DP。attention 怎么分、专家怎么分，可以有不同安排；其他框架的组合方式需要看各自实现。

再算一个**假设模型的权重占用**：可切分的非专家权重共 6 GiB，路由专家共 42 GiB，部署在 4 张卡上。暂不计复制的小张量、量化元数据、KV 和工作区（workspace）：

| 配置 | 每卡非专家权重 | 每卡路由专家 | 每卡合计 |
| --- | --- | --- | --- |
| TP1 / DP4 / EP4 | 6 GiB | 10.5 GiB | 16.5 GiB |
| TP4 / DP1 / EP4 | 1.5 GiB | 10.5 GiB | 12 GiB |

第二种每卡少存 4.5 GiB，但 attention 计算需要 TP 组共同完成。第一种的 attention 在各个 rank 本地计算，token 被分给其他卡上的专家时，仍然要传输激活。因此，第一种也不是四套完全独立的服务：各自的队列和缓存之外，还有共享专家需要协调。

这 4.5 GiB 值不值得省，要连同速度一起看。把流量、输入输出长度和缓存冷热固定下来，比较首 token 延迟、输出间隔、尾延迟及专家负载。矩阵切得太小，可能省了显存，却花了更多时间等待通信。

### PP：KV 留在哪张卡，卡间又传什么？ {#pp-kv}

流水线并行（PP）把模型按层分到几张卡上。先不加缓存卸载或 prefill/decode 分离：一张卡负责哪些 decoder 层，就保存这些层的 KV。前一阶段算完，把中间激活交给下一阶段，历史 KV 则留在各自的卡上。

传多少激活，取决于这一轮有多少 token。设 hidden size 为 3072，BF16 每元素占 2 字节：

| 单次边界传输 | 张量元素数 | 原始字节数 |
| --- | --- | --- |
| 8 条请求各 decode 1 个 token | 8 × 3072 | 48 KiB |
| 8 条请求各 prefill 4096 个 token | 8 × 4096 × 3072 | 192 MiB |

同样八条请求，decode 只传 48 KiB，完整 prefill 却要传 192 MiB。表里还没有算额外的残差张量、元数据、协议和同步开销。长 prefill 可以分块处理，单次消息会变小，但也需要重新安排各阶段的执行。

[vLLM 的扩展指南](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)建议，在没有 NVLink 的 GPU 上（例如 L40S）尝试用 PP 替代 TP 来提高吞吐。原因是两者的通信方式不同，但 PP 自己也有等待：一条请求仍要依次经过各 stage，某一阶段较慢，其他卡就可能闲着。若模型本来就能放进单卡，独立副本、量化后单卡、TP 和 PP 都值得用相同流量比较，不能只凭“没有 NVLink”决定。若另外拆分 prefill 和 decode 并迁移 KV，还得把那条传输路径算进去。

再算一个传输下限：假设边界有效带宽为 12 GiB/s，且忽略启动延迟。上表的 48 KiB 至少约需 3.8 微秒，192 MiB 约需 15.6 毫秒。前者很可能由启动和同步开销主导；后者已经不能忽略字节量。这个带宽是算例假设，不是 PCIe 或某款显卡的实测值。

想让 PP 的各张卡都忙起来，需要把不同请求的微批次（microbatch）送进不同阶段。只有一条请求在生成下一个 token 时，后面的卡仍要等前面的卡；请求多了，流水线才更容易填满。但攒更多工作也会增加排队和缓存占用，所以还要结合链路拓扑、prefill 长度、并发量，以及用户能接受的等待时间来选。

## 连续批处理：一个请求结束，下一个就补进来 {#batching}

多请求场景下，还有一种不加 GPU 就能减少空等的办法：连续批处理（continuous batching）。先看一个简化的 decode 例子。A、B 已完成 prefill，还要分别生成 2、5 个 token；C 也已准备好，还要生成 3 个。每轮最多容纳两条请求，各生成一个 token，暂时忽略其他成本。

| 轮次 | 等整批结束才补位 | 每轮可补位 |
| --- | --- | --- |
| 1–2 | A、B | A、B |
| 3–5 | B，另一个位置空着 | B、C |
| 6–8 | C | 已结束 |

A 在第 2 轮结束后，C 就能补进它的位置，和 B 一起算。于是这个例子从 8 轮缩到了 5 轮。实际加速多少，还取决于 prefill 耗时、KV 容量和读取长上下文的成本，不能直接拿 8/5 当成实测加速比。补位也要有空闲显存和本轮可处理的 token 额度；满了仍然需要排队、限流或拒绝。

<!-- widget:step-continuous-batching -->

## 前缀缓存：哪些计算可以不用重做？ {#prefix}

两个人问同一份文档，提示词里可能都有相同的 2,000-token 说明，后面各接一个 100-token 的问题。第二个请求到来时，如果共同前缀的 KV 还在，就可以少算这一段 prefill。生成答案时仍要读取这些 KV；省下的是重复计算，并没有把文档从 attention 中删掉。

在本文使用的版本中，vLLM 和 SGLang 默认都开启前缀复用，分别对应 `--enable-prefix-caching` 和 radix cache。把两者分成“一个管显存，一个做缓存”已经不合适了。更有用的比较是：它们怎样找到已有前缀、复用到哪个位置，以及空间不够时怎样回收。

| 观察点 | vLLM V1 | SGLang 的 radix 树（默认 `UnifiedRadixCache`） |
| --- | --- | --- |
| 怎么组织可复用前缀 | 带前缀关系的块哈希与 block pool，按哈希查块 | 用树表示共同的 token 路径；v0.5.21 的树核心优先用 Rust，不兼容的平台或功能会回退到 Python |
| 复用边界 | 基础默认值为每 16 个 token 一块；实际大小受模型、attention backend 与配置影响，只复用完整块 | 默认 page size 为 1，可以停在任意 token；部分后端或模型把 page 调到 64、128 等，就向下对齐到整页 |
| 生命周期 | 引用计数、空闲队列按 LRU 驱逐、重新分配 | 锁定、引用、节点拆分；默认 LRU，也可选 LFU、SLRU、priority、T-LRU |
| 如何比较速度 | 用实际请求测命中后省了多少计算；块结构本身不能保证服务更快 | 同样要测实际请求；树结构并不对所有负载都更快 |

回到那份文档：看起来文字相同，还不一定能复用。权重、adapter、token IDs、位置和多模态输入都需要兼容，否则同一个 token 对应的 KV 也可能不同。

如果部署了 A、B 两个副本，还要问一句：**缓存在哪个副本上？** 假设暂时没有跨实例 KV 传输，文档的 KV 留在 A，请求却被分给 B，那么 B 即使支持前缀复用，也仍然要重算。把请求送回 A 可能省计算，但 A 排队很长时，去 B 重算也可能更快。这就是为什么还要看引擎外面的路由：[Production Stack 的 KV-aware routing](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/kv-cache-aware-routing.html)和 [SGLang Model Gateway](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway)都提供相关机制，但如何兼顾缓存和负载，仍要用自己的请求验证。

多租户服务还要限制谁能复用谁的缓存。命中与否造成的响应时间差，可能暴露某段内容是否出现过。[vLLM 的 cache salt](https://docs.vllm.ai/en/v0.31.0/design/prefix_caching/) 可以把复用范围隔开，SGLang 的 OpenAI 接口也接受 `cache_salt`。分组仍应由服务端决定：如果客户端能随意填写别人的组，隔离就失去了意义。文档访问权限也要照常检查，缓存隔离不能替代它。

<details markdown="1">
<summary>缓存配置、页对齐和实现出处</summary>

表中的默认值会随模型、后端和配置变化。对应代码在 vLLM 的 [runner 与异步调度条件](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/config/vllm.py#L709)、[cache 配置](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/config/cache.py#L72)，以及 SGLang 的 [tree core 回退条件](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py#L44)。实际部署时，以启动日志里解析后的配置为准。

[缓存生命周期与页对齐算例](../00-foundations/deep-dives/attention-kernels.md#cache-lifecycle)用固定版本的源码展开了这些细节。[SGLang 原论文](https://arxiv.org/abs/2312.07104)则适合理解 RadixAttention 为什么这样设计；论文之后两边都增加了功能，今天选型还要看对应版本的实现。

</details>

## HiCache：显存以外，也能存 KV {#hicache}

[SGLang HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design)把 KV 放在 GPU、主机内存及存储后端等层次。GPU / 主机层归推理实例使用；跨实例共享要看存储后端及 namespace 的实际配置，不能把两台机器的 RAM 当成天然共享池。vLLM 现在也能这样分层：[OffloadingConnector](https://docs.vllm.ai/en/v0.31.0/features/kv_offloading_usage/) 把算完的完整块先放进 CPU 内存，还可以再接文件、对象存储等二级层，也可以改接 LMCache 或 Mooncake Store。两边默认都不开。

缓存放得更多，不一定就更快。假设重算某段 prefill 要 80 ms，把已有 KV 搬回 GPU 要 30 ms，查找和调度另需 5 ms，那么复用这段缓存可能省下约 45 ms。但如果传输要 100 ms，重新计算反而可能更划算。这里为了方便比较，把各项耗时串行相加；实际执行可能有重叠，需要看性能追踪（trace）。

这也解释了为什么缓存命中率（hit rate）很高，服务却不一定更快。命中的可能都是很短的前缀，也可能花了太久把 KV 搬回来。把复用的 token 数、传输量、恢复时间，以及没命中时多花的查找时间一起记录下来，才看得出这层缓存是否划算。

## 再加速之前，先定位瓶颈 {#bottlenecks}

| 现象 | 先看什么 | 可以比较的方案 | 可能付出的代价 |
| --- | --- | --- | --- |
| 首 token 等很久 | 排队、tokenization、prefill、缓存命中 | chunked prefill、前缀复用、更多副本 | TTFT 与 ITL 取舍；额外显存 |
| 连续输出卡顿 | decode 批量、KV 读写、资源争抢 | 调度预算、量化、算子、拆分 prefill/decode | 质量、传输或运维复杂度 |
| 模型放不下 | 权重、KV、workspace 分别多大 | 量化、TP / PP | 跨卡通信与硬件支持 |
| JSON 常不合法 | 是否真的启用了输出约束 | 结构化解码与校验 | 格式合法仍不代表内容正确 |
| 使用了推测解码却更慢 | 草拟成本、接受率、验证 batch | 改草拟方法或关闭 | 辅助状态和验证开销 |

对照前面的流程，两套引擎的功能其实有很多重合。在本文使用的版本中，两边都支持：

- **安排请求和缓存**：分页 KV、默认开启的前缀复用、continuous batching、chunked prefill，以及分层 KV 存储。
- **加快生成**：CPU 调度与 GPU 计算重叠、CUDA Graph，以及 EAGLE / MTP / n-gram 推测解码。
- **控制输出和多卡部署**：xgrammar 等结构化输出、TP / PP / DP + EP，以及 prefill/decode 分离。

相同的功能名仍可能对应不同的调度路径、缓存结构和兼容性要求。按本文固定版本的源码，vLLM 的前缀索引用块哈希，SGLang 用 radix 树；基础默认粒度分别是 16 个 token 和 1 个 token，实际粒度要看后端与模型。vLLM 每轮先排 decode，再把 prefill 分块加入同一批。SGLang 默认在有新 prefill 工作时先运行 prefill 批，需要混合两者时再开启 `--enable-mixed-chunk`。这些是理解调度的具体例子，不是所有后续版本都不变的分界。

执行方式也有差异：在上述版本中，vLLM 默认用 torch.compile，SGLang 默认不开。拆分 prefill/decode 时，vLLM 通过 KV connector 接入传输后端，[v0.31.0 文档](https://docs.vllm.ai/en/v0.31.0/features/disagg_prefill/)将该功能标为实验性；[SGLang 的部署指南](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)则分别配置 prefill、decode worker，再由 router 分发请求。router 不是每个推理实例里自带的调度器。这些实现与部署边界值得了解，但单凭“内置”或“实验性”的标签，也判断不了哪套在目标环境里更稳定。

选型时，要**固定两套软件的版本，用相同的模型、硬件和所需功能做比较**，不能只按“适合 Agent”或“适合高并发”这样的标签决定。FSDP 处理的是[训练状态分片](training-frameworks.md)，与这里比较的生成引擎用途不同。

## 怎样公平地比较两套引擎？ {#benchmark}

如果目标是文档助手，压测里就要有长文档和连续追问；如果线上经常有突发流量，也要测请求一起涌入时会怎样。一条“你好”只能确认服务能返回，远远不足以选型。可以准备短问答、长文档、多轮共同前缀、突发流量和取消请求，记录输入与实际输出长度分布。两边使用相同的权重、tokenizer、聊天模板、采样设置和硬件，冷启动与预热后的结果分开报告。

| 要报告什么 | 为什么 |
| --- | --- |
| TTFT、ITL、端到端 p50 / p95 / p99 | 平均吞吐会掩盖用户的等待；短输出也要计入 TTFT |
| 输入吞吐与输出吞吐，分别统计 | 两种 token 的计算含义不同 |
| 满足质量和延迟要求的请求数 / 秒 | 大量超时换来的总吞吐没有同等价值 |
| 超时、取消、拒绝、OOM、错误率 | 不能只统计成功返回的请求 |
| GPU / CPU 内存、网络流量、总成本 | 有时瓶颈已经移出 GPU |

压测怎样发请求，也会影响结论。开放式压测（open-loop）按预定节奏持续发送，容易暴露排队积压；闭环压测（closed-loop）通常让每个客户端等上一个请求结束再发下一个，服务越慢，实际到达量也会跟着下降。两种方式都能用，但应同时报告发送量和完成量。固定随机种子，也不保证不同引擎、批次或算子的生成结果逐字一致；内容质量和输出格式还要单独做回归检查。

最后留下软件版本、配置、输入集、结果和失败样本。这样换模型、升级引擎时，才能用同一组请求重新比较。选好引擎之后，[上线与反馈](../practice/ml-system/serving-and-feedback.md)继续讲怎样把它接进完整服务。

## 面试时，先把一条请求讲完整 {#interview-walkthrough}

面试里被问到“你怎么理解 vLLM”，可以沿着请求讲，不必先报一串功能名。例如：

> 我会先按请求路径拆。应用负责权限、prompt 和流量入口；推理引擎维护请求队列、计算预算和 KV，安排每轮执行；worker 跑模型与底层算子，再把结果交回流式输出。接下来我会用输入输出长度、到达率和延迟目标判断瓶颈，再决定调 batch、加副本，还是做模型或阶段拆分。

如果继续问“那为什么选 vLLM，而不是 SGLang”，可以先说明两边有哪些共同能力，再讲自己的模型、硬件、部署或训练框架，以及哪些组合已验证。最后给出压测和维护成本的依据。前缀索引怎么实现值得懂，但不必把它当成整个选型的答案；没有做过实测的部分，说明准备怎样验证就好。

顺着这条线，对方很容易继续问到缓存、通信和调度。下面几类问题可以用来检查自己是否真的理解了前面的过程，不需要逐句背答案。

| 追问 | 进一步解释 | 用什么证据支撑 |
| --- | --- | --- |
| Prefix hit 很高，为什么还是慢？ | 命中的是多少 token？命中了哪个副本？省了 prefill，但队列、decode 与传输仍在 | 分阶段 trace、命中 token 数、冷 / 热缓存对照 |
| KV 用满了怎么办？ | 等待、拒绝、驱逐可回收块、抢占或重算；要区分可释放缓存和活跃请求正在用的状态 | 容量、引用状态、抢占与重算量、尾延迟 |
| TP 从 2 加到 4，一定快吗？ | 每卡工作可能变少，但通信更多或占比更高；小 batch 可能吃不满 | 层内 compute / collective 时间、互联、batch 和模型结构 |
| 用户关了页面，显存没降，是泄漏吗？ | 查请求是否终止、引用是否释放；内存池保留容量不等于仍为该请求服务 | request ID 生命周期、活跃块与空闲池，而不只看进程显存 |
| 怎么证明调优有用？ | 同版本与请求集下，比较满足质量和延迟要求的完成率、成本及错误率 | 开放到达压测、失败样本、可重复配置，不只一张 tokens/s 图 |

读源码时，也可以按这个顺序：找到请求对象，看它怎样进入 batch，再追缓存索引和模型执行。不必把仓库从头读到尾。想接着了解引擎之外的部分，可以看[CPU、GPU 与集群调度](execution-boundaries.md)，或[生成引擎怎样接入 RL 训练](rl-systems.md#engines)。更多准备内容见 [RE / MLE 与 inference 面试](../interview/ai-roles-prep.md#inference)。

## 想继续看课件和推导 {#further-reading}

- **想知道 prefill 和 decode 为什么瓶颈不同**：看 [Stanford CS336 第 10 讲：推理](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)。其中的算术强度分析把矩阵计算量和数据读写量分开估算；再看吞吐与延迟一节，理解增大 batch 为什么不一定让单个请求更快。
- **想自己算显存、带宽和多卡开销**：看 [How To Scale Your Model：Transformer Inference](https://jax-ml.github.io/scaling-book/inference/)，重点读线性层、attention、并行推理和最后的手算题。书中的具体配置是算例，不是所有硬件和模型的最优配置。

这两份资料适合补计算过程；要确认某个参数的默认值、后端兼容性或源码调用链，仍应回到本页链接的对应版本文档和代码。理想化成本模型有助于判断该测哪里，但不能代替排队、缓存冷热和实际通信条件下的测量。
