# Kubernetes 与 GPU 集群：把卡分给任务

**中文** · [English](kubernetes.en.md)

> 阅读时间：约 13 分钟 · 最近审阅：2026-10

集群有 24 张 GPU，监控上还空着 16 张，一个要 16 张卡的训练任务却一直 Pending；更糟的是，一半 Pod 起来了，另一半没起，起来的那一半占着卡空等。**Kubernetes 默认一次只给一个 Pod 找节点，而分布式训练要的是“一组 Pod 要么一起放下，要么一个也别占”；GPU 又只能整块、最好按拓扑分配。空闲卡的总数，说明不了大任务能不能跑。**

Pod 里面的程序怎样把计算交给 GPU，见[框架这一层](frameworks.md)；为什么训练的每个 rank 都必须到齐，见[多卡训练](../06-systems/distributed-training.md)。本文对照 kubernetes.io 与 NVIDIA 官方文档，2026-10-15 核对；当时 Kubernetes 的最新版本是 1.37.1（2026-09-15 发布，见 [releases 页](https://kubernetes.io/releases/)）。

## 只需要这几个 Kubernetes 概念

| 概念 | 是什么 | 和 GPU 任务的关系 |
| --- | --- | --- |
| [Pod](https://kubernetes.io/docs/concepts/workloads/pods/) | 能创建和管理的最小部署单元，一个或几个共享网络与存储的容器 | 一个推理副本、一组训练 rank，通常各是一个 Pod |
| [Node](https://kubernetes.io/docs/concepts/architecture/nodes/) | 一台虚拟机或物理机，上面跑 kubelet | 一台 8 卡服务器就是一个 node |
| [kube-scheduler](https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/) | 给还没分配节点的 Pod 先过滤、再打分，选出节点后通过 binding 告诉 API server | 默认一个 Pod 一个 Pod 地挑 |
| [Controller](https://kubernetes.io/docs/concepts/architecture/controller/) | 控制循环：盯着集群状态，把实际状态往期望状态推 | Deployment、Job 背后都是 controller |
| [Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) | 声明式地维持一组副本 Pod，并按节奏更新 | 长期在线的推理服务 |
| [Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) | 创建 Pod 并重试，直到指定数量的 Pod 成功结束；Indexed 模式给每个 Pod 一个 0 到 completions−1 的编号 | 训练、评测、批处理；编号可以拿来当 rank |
| [requests / limits](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/) | scheduler 按 requests 选节点；kubelet 按 limits 限制容器用量 | GPU 写在 limits 里 |

CPU 和内存的 limits 加起来可以超过节点容量（超卖）：CPU 超过 limit 会被限流，内存超过 limit 会被 OOM kill。GPU 不一样。它是 extended resource，文档写明只能是整数，而且“Extended resources cannot be overcommitted, so request and limit must be equal …”（[资源管理页](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#consuming-extended-resources)），即**不能超卖**，两处都写时 request 和 limit 必须相等；同一块设备也不能在容器之间共享。按 [GPU 调度页](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)，GPU 只写在 limits 里，requests 默认等于 limits。

## GPU 怎样变成可调度的资源

靠 [device plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)。厂商插件通过 gRPC 向 kubelet 注册，用 `ListAndWatch` 上报自己管理的设备；kubelet 把数量写进节点状态，比如“这台机器有 8 个 `nvidia.com/gpu`”。scheduler 按这个计数判断放不放得下；容器创建时，kubelet 再调用插件的 `Allocate`，让它做设备相关的准备。于是一个 8 卡训练 Pod 这样写：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: trainer-0
spec:
  containers:
  - name: trainer
    image: registry.example.com/trainer:v1
    resources:
      limits:
        nvidia.com/gpu: 8
```

光有插件还不够，节点上还要装驱动、让容器运行时能用 GPU、给节点打上 GPU 型号标签、导出监控指标。[NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/overview.html) 用 operator 把这些组件一起管起来：GPU 驱动（默认以容器方式运行，也可用节点上预装的）、Container Toolkit、device plugin、给节点打 GPU 标签的 GPU Feature Discovery、配置 MIG 的 MIG Manager，以及采集指标供 Prometheus 抓取的 DCGM Exporter。截至 2026-10-15，GitHub 上它的最新版本是 v26.7.2（2026-10-14 发布）。

**Dynamic Resource Allocation（DRA）** 是另一条路。它像用 PersistentVolumeClaim 申请存储那样申请设备：驱动发布 ResourceSlice 描述有哪些设备，管理员定义 DeviceClass，工作负载写 ResourceClaim，可以用 CEL 表达式按属性筛选设备、让多个 Pod 引用同一个 claim 共享设备、按 claim 配置设备。和 device plugin 相比，它不再只有“几张卡”这一个整数。

DRA 从哪个版本起稳定，官方材料说法不一（2026-10-15 核对）。[v1.34 的发布公告](https://kubernetes.io/blog/2025/08/27/kubernetes-v1-34-release/)（2025-08-27）写的是 DRA 的核心在 v1.34 进入 GA，`resource.k8s.io/v1` API 升为 stable 并默认可用；[DRA 概念页](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/)则标注 “Stable since Kubernetes v1.35”，说它最早出现在 v1.30，feature gate 已锁定、不能再关闭。[feature gate 参考页](https://kubernetes.io/docs/reference/command-line-tools-reference/feature-gates/)把 `DynamicResourceAllocation` 列为 1.30–1.31 alpha、1.32–1.33 beta，1.34 和 1.35 起各有一行 stable；概念页的 v1.35 可能指开关锁定的版本，但页面没有这样说明。可以确定的是：在当前的 1.37 里它是稳定功能。概念页还写了一条限制：scheduler 不支持为 DRA 资源做抢占，高优先级 Pod 只能等占用设备的 Pod 结束。NVIDIA 一侧，GPU Operator 文档介绍了安装 [DRA Driver for NVIDIA GPUs v0.5.1](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/dra-intro-install.html)，作为 device plugin 的较新替代；其中动态 MIG、按份额共享、MPS 等功能仍是 alpha，而且一个集群里，DRA 用的 GPUCluster 和 device plugin 用的 ClusterPolicy 只能有一个。

## 一张卡给几个任务：MIG、time-slicing、MPS

推理小模型、notebook、评测脚本往往用不满一张卡。共享有三种办法，关键差别在于**隔离了什么**：

| 方式 | 怎么分 | 隔离了什么 | 没隔离什么，或其他限制 |
| --- | --- | --- | --- |
| [MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/introduction.html) | 从 Ampere 架构起，硬件上最多切成 7 个 GPU instance，各有固定的 SM 和显存 | 显存通路（crossbar 端口、L2、显存控制器）各自独占，有故障隔离和可预期的 QoS | 只能按支持的 profile 切，切法要事先配置 |
| [Time-slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html) | 设备插件把一张卡报成多个 replica，进程轮流用时间片 | 几乎没有：只是轮流 | 没有显存或故障隔离；申请多个 replica 也不保证拿到成比例的算力 |
| [MPS](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html) | 控制 daemon 让多个进程同时在一张卡上跑，可限制每个客户端的显存和算力份额 | 每个客户端有独立的 GPU 地址空间；显存与计算份额由 daemon 限制 | 一个客户端触发致命 GPU 错误，共享这张卡的其他客户端也会收到错误并需要退出 |

NVIDIA 的 [device plugin README](https://github.com/NVIDIA/k8s-device-plugin)（2026-10-15 核对）还写着：time-slicing 和 MPS 二选一；MPS 共享自 v0.15.0 起标为 experimental，且不能在开启 MIG 的设备上使用；同一节点上的所有 GPU 用同一种共享方式。GPU Operator 的 MIG 页说明了两种策略：single 表示节点上所有 GPU 都开 MIG，mixed 表示不是全部都开；文档示例里，一张 H100 80GB 切成 7 个 1g.10gb 后，节点标签显示的 GPU 数是 7。

用 [MIG profile 表](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-mig-profiles.html)算一笔：H100 80GB（PCIe 和 SXM5）的 1g.10gb 占 1/7 的 SM、1/8 的显存，最多 7 个；1g.20gb 同样 1/7 的 SM、1/4 的显存，最多 4 个。一个 7B 参数的模型用 BF16 存权重要 $7\times10^9\times2=14$ GB，放不进 10 GB 的实例；放进 20 GB 的实例后，留给 KV cache 和运行时的只剩约 6 GB。这是只算权重的教学账，完整显存账见 [KV cache 推理预算](../00-foundations/deep-dives/kv-cache-and-inference.md#inference-budget)。

## 为什么分布式训练要“成组调度”

用一个能手算的例子。3 台节点 A、B、C，每台 8 张卡，共 24 张。依次来了 4 个任务：两个单 Pod 的 4 卡任务 S4a、S4b，一个 16 卡训练 T16（2 个 Pod，各 8 卡，所有 rank 到齐才开始训练），一个单 Pod 的 8 卡任务 T8。总需求 32 张，超过 24，必然有人排队；问题是谁排队、排队的人占不占卡。

按 v1.37 的默认配置，scheduler 一个 Pod 一个 Pod 地放；NodeResourcesFit 插件默认用 LeastAllocated 打分，权重只给 CPU 和内存（见 [scheduler 配置参考](https://kubernetes.io/docs/reference/config-api/kube-scheduler-config.v1/)），GPU 用了多少不进入这一项打分。S4a 落在 A 之后，S4b 更可能去 CPU、内存更空的 B 或 C；假设去了 B。下表比较三种做法最终的状态：

| 做法 | 节点 A | 节点 B | 节点 C | 在干活的卡 | 还在排队 |
| --- | --- | --- | --- | ---: | --- |
| 默认：逐个 Pod 放，按 CPU 和内存打分 | S4a（4） | S4b（4） | T16 的一半（8，空等） | 8 | T16 另一半、T8 |
| 加上成组调度 | S4a（4） | S4b（4） | T8（8） | 16 | T16 |
| 成组调度，并按 GPU 装箱 | S4a+S4b（8） | T16（8） | T16（8） | 24 | T8 |

<!-- widget:step-gang-scheduling -->

第一行最糟：T16 的第一个 Pod 拿下 C，第二个 Pod 找不到剩 8 张卡的节点，进程已经启动却在等另一半 rank；A、B 各剩 4 张，合起来 8 张，却拼不出一个 8 卡 Pod，T8 也只能等。24 张卡里只有 8 张在干活。如果再来一个同样的 T16，两个任务可能各占一台节点、各等另一半，谁也开始不了，这就是死锁。

第二行：成组调度发现 T16 两个 Pod 不能同时放下，就一个也不绑定，C 留给 T8，16 张卡在干活。代价是 T16 要等两台整机同时空出来，没有优先级或预留时可能一直等下去。第三行把 [scheduler 换成 MostAllocated 装箱策略](https://kubernetes.io/docs/concepts/scheduling-eviction/resource-bin-packing/)并给 GPU 加权，两个小任务挤进同一台，留出两台整机给 T16，24 张卡都在干活；这时轮到 T8 等 A 空出来。哪种顺序更好取决于优先级和配额，装箱不是在任何情况下都更优。

### 放得下，还要放得近

T16 的 16 个 rank 每步都要做梯度 all-reduce。按理想 ring 算，每个 rank 发送 $2\frac{P-1}{P}S$ 字节（推导见[多卡训练](../06-systems/distributed-training.md)），环上每条边都承载这么多数据。把 16 个 rank 按节点顺序排成环，跨节点的边数等于用到的节点数：整机放在 2 台上是 2 条边走网络，其余走节点内互联；如果允许拆成 4 卡的小 Pod、摊在 3 台上，就有 3 条边走网络，而且每台要和更多邻居争网卡。再往上一层，两台节点若分在不同机架，跨机架的路径通常更长。这是教学用的理想环；NCCL 实际选什么算法，以及节点内外的带宽差多少，要看硬件和 [NCCL](frameworks.md#nccl-map) 的配置。

这就是拓扑感知调度要解决的事：不只问“剩几张卡”，还问“这几张卡离得多近”。下面几个项目都在处理成组和拓扑，做法不同，这里只按官方文档中性地列出：

| 方案 | 做在哪一层 | 成组怎么实现 | 拓扑 | 状态（2026-10-15 核对） |
| --- | --- | --- | --- | --- |
| Kubernetes 核心的 [gang scheduling](https://kubernetes.io/docs/concepts/scheduling-eviction/gang-scheduling/) | kube-scheduler 与 PodGroup / Workload API | 放不下 `minCount` 个 Pod，就一个也不绑定 | [Topology-Aware Workload Scheduling](https://kubernetes.io/docs/concepts/scheduling-eviction/topology-aware-scheduling/) 把整组放进同一拓扑域：v1.36 起 alpha，默认关闭 | v1.37 起 beta，默认关闭，需开启 `GenericWorkload` feature gate |
| [Kueue](https://kueue.sigs.k8s.io/docs/overview/) | 在 kube-scheduler 之上做作业级排队、配额和准入，不替代它 | [`waitForPodsReady`](https://kueue.sigs.k8s.io/docs/tasks/manage/setup_wait_for_pods_ready/)：超时前没全部就绪，就驱逐并重新排队（文档称之为一种简单的 all-or-nothing 实现） | [TAS](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/)：v0.14 起 beta，默认开启，可按 rack、block 等层级要求或偏好 | 最新 v0.20.2（2026-10-15） |
| [Volcano](https://volcano.sh/en/docs/podgroup/) | 自己的批调度器，由 actions 和 plugins 组成 | PodGroup 的 `minMember`：资源不够运行这么多 Pod，就一个都不调度 | 本文未核对 | 最新 v1.16.0（2026-10-14） |
| [JobSet](https://jobset.sigs.k8s.io/docs/overview/) | 把一组 Job 当作一个单元管理的 API | 概览页没说它自己做成组调度；可以与 Kueue 配合排队 | 一个注解让每个子 Job 独占一个拓扑域（如机架） | 最新 v0.13.0（2026-10-12） |

Kubernetes [v1.37 的博客](https://kubernetes.io/blog/2026/09/08/kubernetes-v1-37-advancing-workload-aware-scheduling/)（2026-09-08）还介绍了处于 alpha 的 CompositePodGroup，用来表达多层的成组和拓扑约束，并提到它面向 JobSet、LeaderWorkerSet 这类上层 API 的工作负载。

## Slurm 还是 Kubernetes

训练集群的另一种做法是用 [Slurm](https://slurm.schedmd.com/overview.html) 排作业。两者的出发点不同，主要差别如下：

| 维度 | Slurm | Kubernetes |
| --- | --- | --- |
| 设计重心 | 给用户分配一段时间的节点资源，管理待运行作业的队列 | 用控制器把容器化应用维持在期望状态 |
| 多节点作业 | 先分配一组节点，再在这组节点上启动并行作业 | 默认逐个 Pod 调度；成组需要 gang scheduling 或 Kueue、Volcano 等 |
| GPU 记账 | [GRES](https://slurm.schedmd.com/gres.html)（generic resources） | extended resource（`nvidia.com/gpu`）或 DRA |
| 排队与回填 | [backfill](https://slurm.schedmd.com/sched_config.html) 默认加载：不推迟高优先级作业预计开始时间的前提下，先跑低优先级作业；依赖准确的作业时间上限 | 默认调度器没有作业级配额队列；Kueue 等补上配额、公平共享和准入 |
| 网络拓扑 | [topology/tree](https://slurm.schedmd.com/topology.html) 找能满足作业的最低一层交换机；还有 topology/block | 节点标签，加上 Kueue TAS 或核心 TAS（alpha） |
| 在线服务 | 作业带时间上限，backfill 依赖它 | Deployment、Service、HPA 都是内置对象 |
| 容器 | [原生支持](https://slurm.schedmd.com/containers.html)无特权的 OCI 容器，需要配置 `oci.conf` 等 | 以容器镜像为基本运行单位 |
| “gang scheduling” 这个词 | 指[分时](https://slurm.schedmd.com/gang_scheduling.html)：几个作业分到同一批资源，轮流挂起 | 指一组 Pod 全有或全无 |

最后一行最容易误会：同一个词，在 Slurm 里是“轮流用”，在 Kubernetes 生态里是“一起上”。这张表比较的是机制，不是谁更好；选择取决于团队已有的工具、在线服务的比重和多租户需求。

## 推理服务：扩容不只是加副本

推理服务通常是 Deployment，靠 [HorizontalPodAutoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)（HPA）扩缩。HPA 的核心公式是：

$$
\text{desiredReplicas}=\left\lceil \text{currentReplicas}\times\frac{\text{currentMetricValue}}{\text{desiredMetricValue}}\right\rceil .
$$

例如 4 个副本，每个副本平均排队 30 个请求，目标是 10 个，就要 $\lceil4\times30/10\rceil=12$ 个副本。比值落在容忍度内（默认 0.1）不动作；缩容可以设稳定窗口，文档示例是 300 秒。指标从哪来？按文档，非 CPU/内存的指标要通过 custom 或 external metrics API 提供，通常需要一个适配器；GPU 指标可以由 [DCGM Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html) 交给 Prometheus，例如显存用量 `DCGM_FI_DEV_FB_USED`。选什么指标更要紧。Google Cloud 的 [GKE 推理扩缩最佳实践](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling)（2026-10-13 更新）推荐用服务器的队列长度和 batch 大小，并指出 GPU 利用率只衡量 GPU 处于活动状态的时间，不衡量活动时做了多少工作。

再算冷启动。一个 70B 参数的模型用 BF16 存，权重是 $70\times10^9\times2=140$ GB。新副本起来时，至少要把这 140 GB 从存储读进来，耗时约为 $140\text{ GB}/B$，$B$ 是有效读取带宽（下表全是教学假设，不是某种存储的规格）：

| 有效读取带宽 | 读完 140 GB |
| ---: | ---: |
| 1 GB/s | 140 s |
| 3 GB/s | 约 47 s |
| 10 GB/s | 14 s |
| 25 GB/s | 5.6 s |

换一个条件：8 个副本同时启动，共用一个总带宽 10 GB/s 的共享存储，总共要读 $8\times140=1120$ GB，约 112 s，比单个副本慢 8 倍。读权重之外，还有加节点、拉镜像、初始化 CUDA、引擎预热（例如捕获 CUDA Graph、编译）等时间，这里都没算。回到上面的扩容例子：假设新副本总共要 2 分钟才能接请求（教学假设），从 4 扩到 12 个副本的决定做出之后，这 2 分钟里队列还在涨。常见对策是保留最少的热副本、用更早的信号触发扩容、缓慢缩容；具体效果要用真实请求回放验证。

## 排错时先问什么

| 看到的现象 | 先查什么 | 不能据此推出什么 |
| --- | --- | --- |
| Pod 一直 Pending，提示 GPU 不够 | 每台节点还剩几张整卡，Pod 要几张；taint 和 nodeSelector 是否把节点排除了 | 集群空闲卡总数够，就一定放得下 |
| 训练一半 rank 起来，一半 Pending | 有没有成组调度或作业级准入 | 再等一会儿自然会好；可能已经死锁 |
| 节点上 `nvidia.com/gpu` 是 0 | 驱动、container toolkit、device plugin 是否就绪 | 这台机器没有 GPU |
| 共享卡上一个任务崩了，邻居也崩 | 用的是 time-slicing 还是 MPS | 一定是邻居的代码有问题 |
| 扩容之后，延迟还高一阵 | 新副本的冷启动：拉镜像、读权重、预热 | HPA 没有生效 |
| GPU 利用率很高，吞吐却低 | 这个指标只说明 GPU 忙的时间比例 | 利用率高就说明 batch 合适 |

调度器只决定卡分给谁；卡分到之后，一次请求在引擎里怎样排队和组 batch，见[一次 LLM 请求](../06-systems/llm-serving.md)；Pod 里的代码怎样把计算交给 GPU，回到[框架这一层](frameworks.md)。
