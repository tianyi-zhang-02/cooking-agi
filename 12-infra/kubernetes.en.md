# Kubernetes and GPU clusters: handing GPUs to jobs

[中文](kubernetes.md) · **English**

> Reading time: ~13 min · Last reviewed: 2026-10

The cluster has 24 GPUs and the dashboard shows 16 of them idle, yet a training job that needs 16 GPUs stays Pending. Worse, half of its Pods start, the other half do not, and the half that started holds its GPUs while waiting.

By default Kubernetes finds a node for one Pod at a time, while distributed training usually needs gang scheduling: either the whole group of Pods is placed, or none holds resources. In the exclusive, whole-GPU example below, each Pod must also fit on one node, and GPU topology matters. The total number of idle GPUs alone cannot tell you whether a large job can run.

For how the program inside a Pod hands work to the GPU, see [The framework layer](frameworks.en.md); for why every rank of a training job must be present, see [Multi-GPU training](../06-systems/distributed-training.en.md). This note follows the official kubernetes.io and NVIDIA documentation, checked on 2026-10-15; the latest Kubernetes release at that time was 1.37.1 (released 2026-09-15, see the [releases page](https://kubernetes.io/releases/)).

## The few Kubernetes concepts you need

| Concept | What it is | How it relates to GPU jobs |
| --- | --- | --- |
| [Pod](https://kubernetes.io/docs/concepts/workloads/pods/) | The smallest deployable unit you can create and manage: one or more containers sharing network and storage | One inference replica or one group of training ranks is usually one Pod |
| [Node](https://kubernetes.io/docs/concepts/architecture/nodes/) | A virtual or physical machine running the kubelet | One 8-GPU server is one node |
| [kube-scheduler](https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/) | For Pods without a node, filters nodes, scores them, picks one and tells the API server through binding | By default it picks one Pod at a time |
| [Controller](https://kubernetes.io/docs/concepts/architecture/controller/) | A control loop that watches cluster state and pushes the actual state toward the desired state | Deployments and Jobs are both run by controllers |
| [Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) | Declaratively maintains a set of replica Pods and updates them at a controlled rate | Long-running inference services |
| [Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) | Creates Pods and retries until a given number of them finish successfully; Indexed mode gives each Pod an index from 0 to completions−1 | Training, evaluation, batch work; the index can serve as a rank |
| [requests / limits](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/) | The scheduler places Pods by requests; the kubelet caps container usage by limits | GPUs go in limits |

The limits for CPU and memory can add up to more than a node's capacity (overcommit): a container over its CPU limit is throttled, one over its memory limit is OOM-killed. GPUs are different. They are extended resources, which the docs say must be integers, and "Extended resources cannot be overcommitted, so request and limit must be equal …" when both are given ([resource management page](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#consuming-extended-resources)). One device also cannot be shared between containers. Per the [GPU scheduling page](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/), GPUs are specified only in limits, and requests default to the limit.

## How GPUs become schedulable resources

Through a [device plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/). The vendor's plugin registers with the kubelet over gRPC and reports the devices it manages through `ListAndWatch`; the kubelet writes the count into the node status, such as "this machine has 8 `nvidia.com/gpu`". The scheduler uses this count to decide whether a Pod fits; when a container is created, the kubelet calls the plugin's `Allocate` so it can do device-specific setup. An 8-GPU training Pod therefore looks like this:

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

A plugin alone is not enough: the node also needs a driver, a way for containers to use the GPU, labels with the GPU model, and exported metrics. The [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/overview.html) uses the operator pattern to manage these components together: the GPU driver (run as a container by default, or the node's preinstalled one), the Container Toolkit, the device plugin, GPU Feature Discovery to label nodes, the MIG Manager to configure MIG, and DCGM Exporter to collect metrics for Prometheus. As of 2026-10-15, its latest release on GitHub is v26.7.2 (released 2026-10-14).

**Dynamic Resource Allocation (DRA)** is another route. It requests devices the way a PersistentVolumeClaim requests storage: drivers publish ResourceSlices describing the available devices, administrators define DeviceClasses, and workloads write ResourceClaims, which can filter devices by attributes with CEL expressions, let several Pods share a device by referencing the same claim, and carry per-claim device configuration. Compared with a device plugin, it is no longer just one integer of "how many GPUs".

Official sources disagree on which version made DRA stable (checked 2026-10-15). The [v1.34 release announcement](https://kubernetes.io/blog/2025/08/27/kubernetes-v1-34-release/) (2025-08-27) says the core of DRA went GA in v1.34, with the `resource.k8s.io/v1` APIs graduating to stable and available by default; the [DRA concept page](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/) is labelled "Stable since Kubernetes v1.35", says it first appeared in v1.30, and says its feature gate is locked and can no longer be turned off. The [feature gate reference](https://kubernetes.io/docs/reference/command-line-tools-reference/feature-gates/) lists `DynamicResourceAllocation` as alpha in 1.30–1.31, beta in 1.32–1.33, and with one stable row from 1.34 and another from 1.35; the concept page's v1.35 may refer to the release that locked the gate, but the page does not say so. What is certain is that it is a stable feature in the current 1.37. The concept page also lists a limitation: the scheduler does not support preemption for DRA resources, so a higher-priority Pod has to wait until the Pod holding the device finishes. On NVIDIA's side, the GPU Operator documentation describes installing the [DRA Driver for NVIDIA GPUs v0.5.1](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/dra-intro-install.html) as a newer alternative to the device plugin; features such as dynamic MIG, share-based sharing and MPS are still alpha there, and a cluster can have either the GPUCluster resource used for DRA or the ClusterPolicy resource used for the device plugin, not both.

## Several jobs on one GPU: MIG, time-slicing, MPS

Small-model inference, notebooks and evaluation scripts often do not fill a whole GPU. There are three ways to share one, and the key difference is **what gets isolated**:

| Method | How it splits | What is isolated | What is not isolated, or other limits |
| --- | --- | --- | --- |
| [MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/introduction.html) | From the Ampere architecture on, hardware partitioning into up to 7 GPU instances, each with fixed SMs and memory | Each instance has its own memory path (crossbar ports, L2, memory controllers), with fault isolation and predictable QoS | Only supported profiles can be used, and the layout must be configured in advance |
| [Time-slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html) | The device plugin advertises one GPU as several replicas, and processes take turns in time slices | Almost nothing: they only take turns | No memory or fault isolation; requesting several replicas does not guarantee a proportional share of compute |
| [MPS](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html) | A control daemon lets several processes run on one GPU at the same time and can cap each client's share of memory and compute | Each client has its own GPU address space; memory and compute shares are capped by the daemon | When one client triggers a fatal GPU fault, the other clients sharing that GPU are informed of the error and must exit |

NVIDIA's [device plugin README](https://github.com/NVIDIA/k8s-device-plugin) (checked 2026-10-15) adds: time-slicing and MPS are mutually exclusive; MPS sharing has been marked experimental since v0.15.0 and is not supported on devices with MIG enabled; all GPUs on a node use the same sharing method. The GPU Operator's MIG page describes two strategies: single means MIG is enabled on all GPUs of a node, mixed means it is not enabled on all of them; in the docs example, after an H100 80GB is split into 7 instances of 1g.10gb, the node label shows a GPU count of 7.

Use the [MIG profile table](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-mig-profiles.html) for some arithmetic: on the H100 80GB (PCIe and SXM5), 1g.10gb takes 1/7 of the SMs and 1/8 of the memory, up to 7 instances; 1g.20gb also takes 1/7 of the SMs but 1/4 of the memory, up to 4 instances. A 7B-parameter model with BF16 weights needs $7\times10^9\times2=14$ GB, which does not fit a 10 GB instance; in a 20 GB instance, only about 6 GB is left for the KV cache and runtime. This is a teaching calculation counting weights only; for the full memory budget see [KV cache inference budget](../00-foundations/deep-dives/kv-cache-and-inference.en.md#inference-budget).

## Why distributed training needs gang scheduling

Take an example you can work by hand. Three nodes A, B and C, 8 GPUs each, 24 in total. Four jobs arrive in order: two single-Pod 4-GPU jobs S4a and S4b, a 16-GPU training job T16 (2 Pods of 8 GPUs each; training starts only when all ranks are present), and a single-Pod 8-GPU job T8. Total demand is 32 GPUs, more than 24, so someone must queue; the question is who queues and whether the queued job holds GPUs.

With the v1.37 defaults, the scheduler places one Pod at a time; the NodeResourcesFit plugin scores with LeastAllocated by default, weighting only CPU and memory (see the [scheduler configuration reference](https://kubernetes.io/docs/reference/config-api/kube-scheduler-config.v1/)), so GPU usage does not enter this score. After S4a lands on A, S4b is more likely to go to B or C, whose CPU and memory are less used; assume B. The table compares the final state under three approaches:

| Approach | Node A | Node B | Node C | GPUs doing work | Still queued |
| --- | --- | --- | --- | ---: | --- |
| Default: one Pod at a time, scored on CPU and memory | S4a (4) | S4b (4) | Half of T16 (8, waiting) | 8 | Other half of T16, T8 |
| Add gang scheduling | S4a (4) | S4b (4) | T8 (8) | 16 | T16 |
| Gang scheduling plus bin-packing on GPUs | S4a+S4b (8) | T16 (8) | T16 (8) | 24 | T8 |

<!-- widget:step-gang-scheduling -->

The first row is the worst: T16's first Pod takes C, its second Pod finds no node with 8 free GPUs, and the processes that did start sit waiting for the other half of the ranks; A and B have 4 free GPUs each, 8 in total, but cannot form one 8-GPU Pod, so T8 waits too. Only 8 of the 24 GPUs are doing work. A second identical T16 arriving now would also queue; it cannot immediately take another whole node. The table itself demonstrates fragmentation and idle waiting caused by partial allocation.

Two T16 jobs waiting on each other require an additional resource condition. For example, suppose only two whole nodes are available to them while another job occupies the third long-term. Pod-by-Pod allocation can give each T16 one node, leaving each waiting for its other half. Without resource release, timeout-based reclamation or preemption, neither can start: a deadlock. If the third node later becomes free, the wait may end, so the original table alone does not establish permanent deadlock.

Second row: gang scheduling sees that T16's two Pods cannot both be placed, so it binds neither, C goes to T8, and 16 GPUs are doing work. The cost is that T16 waits for two whole nodes to become free at the same time, which without priorities or reservations may take indefinitely long. The third row [switches the scheduler to the MostAllocated bin-packing strategy](https://kubernetes.io/docs/concepts/scheduling-eviction/resource-bin-packing/) with a weight on GPUs: the two small jobs squeeze onto one node, leaving two whole nodes for T16, and all 24 GPUs are doing work; now T8 waits for A to free up. Which order is better depends on priorities and quotas; bin-packing is not better in every case.

### Fitting is not enough; it must also be close

T16's 16 ranks do a gradient all-reduce every step. In an ideal ring each rank sends $2\frac{P-1}{P}S$ bytes (derived in [Multi-GPU training](../06-systems/distributed-training.en.md)), and every edge of the ring carries that much data. Order the 16 ranks into a ring by node, and the number of cross-node edges equals the number of nodes used: whole nodes on 2 machines means 2 edges over the network and the rest over the intra-node interconnect; if the job may be split into 4-GPU Pods spread over 3 machines, 3 edges go over the network and each machine competes for its network card with more neighbours. One level up, if the two nodes sit in different racks, the path between racks is usually longer. This is an idealized teaching ring; which algorithm NCCL actually chooses, and how much intra-node and inter-node bandwidth differ, depend on the hardware and on [NCCL](frameworks.en.md#nccl-map) configuration.

This is what topology-aware scheduling addresses: it asks not only "how many GPUs are left" but also "how close those GPUs are". The projects below all deal with gangs and topology in different ways; they are listed neutrally from their official documentation:

| Option | Which layer | How the gang is enforced | Topology | Status (checked 2026-10-15) |
| --- | --- | --- | --- | --- |
| Core Kubernetes [gang scheduling](https://kubernetes.io/docs/concepts/scheduling-eviction/gang-scheduling/) | kube-scheduler with the PodGroup / Workload APIs | If `minCount` Pods cannot be placed, none is bound | [Topology-Aware Workload Scheduling](https://kubernetes.io/docs/concepts/scheduling-eviction/topology-aware-scheduling/) places the whole group in one topology domain: alpha since v1.36, disabled by default | Beta since v1.37, disabled by default, requires the `GenericWorkload` feature gate |
| [Kueue](https://kueue.sigs.k8s.io/docs/overview/) | Job-level queuing, quotas and admission on top of kube-scheduler, without replacing it | [`waitForPodsReady`](https://kueue.sigs.k8s.io/docs/tasks/manage/setup_wait_for_pods_ready/): if not all Pods are ready before a timeout, the workload is evicted and requeued (the docs call this a simple all-or-nothing implementation) | [TAS](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/): beta since v0.14, enabled by default, with required or preferred levels such as rack or block | Latest v0.20.2 (2026-10-15) |
| [Volcano](https://volcano.sh/en/docs/podgroup/) | Its own batch scheduler, built from actions and plugins | PodGroup `minMember`: if resources cannot run that many Pods, none is scheduled | Not checked for this note | Latest v1.16.0 (2026-10-14) |
| [JobSet](https://jobset.sigs.k8s.io/docs/overview/) | An API that manages a group of Jobs as one unit | The overview does not say it does gang scheduling itself; it can queue through Kueue | An annotation gives each child Job exclusive use of one topology domain, such as a rack | Latest v0.13.0 (2026-10-12) |

The Kubernetes [v1.37 blog post](https://kubernetes.io/blog/2026/09/08/kubernetes-v1-37-advancing-workload-aware-scheduling/) (2026-09-08) also introduces CompositePodGroup, at alpha, for expressing multi-level gang and topology constraints, and mentions that it targets workloads managed by higher-level APIs such as JobSet and LeaderWorkerSet.

## Slurm or Kubernetes

Another way to run a training cluster is to schedule jobs with [Slurm](https://slurm.schedmd.com/overview.html). The two start from different places; the main differences:

| Dimension | Slurm | Kubernetes |
| --- | --- | --- |
| Design focus | Allocates node resources to users for a period of time and manages a queue of pending jobs | Uses controllers to keep containerized applications in their desired state |
| Multi-node jobs | Allocates a set of nodes first, then starts the parallel job on that set | Schedules one Pod at a time by default; gangs need gang scheduling or Kueue, Volcano and the like |
| GPU accounting | [GRES](https://slurm.schedmd.com/gres.html) (generic resources) | Extended resource (`nvidia.com/gpu`) or DRA |
| Queuing and backfill | [Backfill](https://slurm.schedmd.com/sched_config.html) is loaded by default: it starts lower-priority jobs if that does not delay the expected start of higher-priority ones; it depends on accurate job time limits | The default scheduler has no job-level quota queue; Kueue and others add quotas, fair sharing and admission |
| Network topology | [topology/tree](https://slurm.schedmd.com/topology.html) finds the lowest switch level that can satisfy the job; topology/block also exists | Node labels, plus Kueue TAS or core TAS (alpha) |
| Online services | Jobs carry time limits, which backfill relies on | Deployment, Service and HPA are built-in objects |
| Containers | [Natively supports](https://slurm.schedmd.com/containers.html) unprivileged OCI containers, after configuring `oci.conf` and more | The container image is the basic unit of execution |
| The term "gang scheduling" | Means [time-sharing](https://slurm.schedmd.com/gang_scheduling.html): several jobs are given the same resources and are suspended in turn | Means a group of Pods is all-or-nothing |

The last row is the easiest to misread: the same term means "take turns" in Slurm and "start together" in the Kubernetes ecosystem. The table compares mechanisms, not which is better; the choice depends on the team's existing tools, how much online serving it runs, and its multi-tenancy needs.

## Inference serving: scaling is more than adding replicas

An inference service is usually a Deployment, scaled by a [HorizontalPodAutoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/) (HPA). The core HPA formula is:

$$
\text{desiredReplicas}=\left\lceil \text{currentReplicas}\times\frac{\text{currentMetricValue}}{\text{desiredMetricValue}}\right\rceil .
$$

For example, with 4 replicas, an average queue of 30 requests per replica and a target of 10, you need $\lceil4\times30/10\rceil=12$ replicas. If the ratio is within the tolerance (0.1 by default), nothing happens; scale-down can be given a stabilization window, 300 seconds in the docs example. Where do the metrics come from? Per the docs, metrics other than CPU and memory are served through the custom or external metrics APIs, which usually requires an adapter; GPU metrics can reach Prometheus through [DCGM Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html), for example memory usage as `DCGM_FI_DEV_FB_USED`. Which metric to choose matters more. Google Cloud's [GKE best practices for autoscaling inference](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling) (updated 2026-10-13) recommends the server's queue size and batch size, and points out that GPU utilization only measures how much of the time the GPU is active, not how much work it does while active.

Now the cold start. A 70B-parameter model stored in BF16 has $70\times10^9\times2=140$ GB of weights. A new replica must at least read these 140 GB from storage, which takes about $140\text{ GB}/B$, where $B$ is the effective read bandwidth (every row below is a teaching assumption, not the spec of any storage system):

| Effective read bandwidth | Time to read 140 GB |
| ---: | ---: |
| 1 GB/s | 140 s |
| 3 GB/s | about 47 s |
| 10 GB/s | 14 s |
| 25 GB/s | 5.6 s |

Change one condition: 8 replicas start at once and share storage with 10 GB/s of total bandwidth; together they read $8\times140=1120$ GB, about 112 s, 8 times slower than a single replica. Besides reading weights there is adding nodes, pulling images, initializing CUDA and warming up the engine (for example capturing CUDA Graphs or compiling), none of which is counted here. Back to the scaling example above: assume a new replica needs 2 minutes in total before it can take requests (a teaching assumption); once the decision to go from 4 to 12 replicas is made, the queue keeps growing for those 2 minutes. Common responses are to keep a minimum of warm replicas, trigger scale-up on earlier signals, and scale down slowly; how well they work has to be verified by replaying real requests.

## What to ask first when debugging

| What you see | What to check first | What you can NOT conclude |
| --- | --- | --- |
| A Pod stays Pending, reporting insufficient GPUs | How many whole GPUs each node has left versus how many the Pod needs; whether taints or nodeSelector exclude nodes | That enough idle GPUs in the cluster means it must fit |
| Half the training ranks start, half stay Pending | Whether gang scheduling or job-level admission is in place | That it will sort itself out if you wait; it may already be deadlocked |
| A node shows 0 `nvidia.com/gpu` | Whether the driver, container toolkit and device plugin are ready | That the machine has no GPU |
| One job crashes on a shared GPU and its neighbour crashes too | Whether the sharing is time-slicing or MPS | That the neighbour's code must be at fault |
| Latency stays high for a while after scaling up | The new replicas' cold start: image pull, weight loading, warm-up | That the HPA did not work |
| GPU utilization is high but throughput is low | This metric only shows the fraction of time the GPU is busy | That high utilization means the batch size is right |

The scheduler only decides who gets which GPUs; once they are assigned, how a request queues and is batched inside the engine is covered in [One LLM request](../06-systems/llm-serving.en.md), and how the code in a Pod hands work to the GPU is back in [The framework layer](frameworks.en.md).
