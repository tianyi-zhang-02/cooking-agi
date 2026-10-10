# 从 batch 到训练循环：能跑，还不等于在学习

**中文** · [English](training-loop.en.md)

> 阅读时间：约 18 分钟 · 前置：自动求导、交叉熵 · 最近审阅：2026-10

`Linear` 能输出两个分数，不代表它已经会分类。`loss` 在下降，也不代表它能处理没见过的数据。一个训练循环除了前向和反向，还要管数据怎样分、参数怎样收集、指标怎样累计，以及中断后恢复了什么。

下面从一个小型二分类实验走完这些步骤。只用合成数据，不下载数据集；CPU / PyTorch 2.8.0 用于数值检查。这个例子用来检查训练逻辑，**不是模型效果或性能 benchmark**。

## 1. 先认清每个部件负责什么

| 部件 | 在这段代码里做什么 | 常见误解 |
| --- | --- | --- |
| `torch.Tensor` / `torch.tensor` | 前者是类型，后者是常用的数据构造函数 | 把两者当成完全相同的名字 |
| `nn.Module` | 注册参数、子模块、buffer，定义前向计算 | 以为所有模块都有权重和 bias |
| `nn.functional` | 提供无状态的计算接口，包括 loss、线性运算、激活等 | 以为它只有激活函数 |
| `Dataset` | 定义如何取一个样本 | 以为它就是一个 batch |
| `DataLoader` | 取样、组 batch，按配置组织加载 | 以为任何输入都会自动变成模型能吃的张量 |
| `Optimizer` | 根据梯度及内部状态更新参数 | 以为 `backward` 已经做了这一步 |

这里用 `TensorDataset`，第一个张量放特征，第二个放标签；同一位置的两者是一对。`shuffle=True` 打乱的是样本索引，不会把特征和标签分别乱排。默认 collate 会尝试把同一字段组成 batch；长度不同的文本还需要 padding、mask 或自定义 `collate_fn`。

对于 `Linear(2, 2)`，输入 `[batch, 2]`，权重 `[2, 2]`，输出 `[batch, 2]`。一般地，`Linear(in_features, out_features)` 的权重形状是 `[out_features, in_features]`，计算 $XW^\top+b$，也能保留输入的前导维度。

## 2. 数据先分开，再开始学

我们生成二维点，用一个固定规则划分类别。训练和验证使用不同随机种子生成的样本，但来自同一个合成分布。真实问题要进一步检查用户、时间、文档来源是否泄漏，不能因为数组分成两段就算独立。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def make_dataset(count, seed):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(count, 2, generator=generator)
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).long()
    return TensorDataset(features, labels)

torch.manual_seed(17)
train_data = make_dataset(80, seed=5)
validation_data = make_dataset(47, seed=6)
train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True,
    generator=torch.Generator().manual_seed(7),
)
validation_loader = DataLoader(validation_data, batch_size=16, shuffle=False)
model = nn.Linear(2, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.15, momentum=0.9)

features, labels = next(iter(validation_loader))
assert features.shape == (16, 2)
assert labels.shape == (16,) and labels.dtype == torch.long
assert model(features).shape == (16, 2)
```

类别数来自任务定义，这里固定为 2，不是看当前 batch 碰巧出现几个类别。对这个单标签分类任务，`CrossEntropyLoss` 接收未经过 softmax 的 logits 和 `long` 类型的类别索引。不要先 softmax 再传入；也不要把标签随手改成 `[batch, 1]`。

如果改成多标签任务，每个类别可以同时为真，通常考虑 `BCEWithLogitsLoss`，配同形状的浮点目标。它不是把 CE 换个名字：标签语义、形状和归约都要一起改。

验证集有意设为 47 个样本，最后一个 batch 只有 15 个。这个不整齐的小细节，刚好用来检查指标有没有算错。

## 3. 一次更新里发生了什么

```text
取出特征和标签
  → 清除上次的梯度
  → model(features)：得到 logits
  → loss：按当前 batch 求平均
  → backward：计算并累积梯度
  → step：修改参数
  → 累计本轮的 loss 总量与样本数
```

```python
def train_epoch(model, loader, optimizer):
    model.train()
    loss_total = 0.0
    sample_count = 0
    for features, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        count = labels.numel()
        loss_total += loss.detach().item() * count
        sample_count += count
    if sample_count == 0:
        raise ValueError("Training loader is empty")
    return loss_total / sample_count

def evaluate(model, loader):
    previous_mode = model.training
    model.eval()
    loss_total = 0.0
    correct_count = 0
    sample_count = 0
    try:
        with torch.no_grad():
            for features, labels in loader:
                logits = model(features)
                loss_total += nn.functional.cross_entropy(
                    logits, labels, reduction="sum"
                ).item()
                correct_count += (logits.argmax(dim=-1) == labels).sum().item()
                sample_count += labels.numel()
    finally:
        model.train(previous_mode)
    if sample_count == 0:
        raise ValueError("Evaluation loader is empty")
    return {"loss": loss_total / sample_count, "accuracy": correct_count / sample_count}

initial_metrics = evaluate(model, validation_loader)
history = []
for epoch in range(25):
    train_loss = train_epoch(model, train_loader, optimizer)
    validation_metrics = evaluate(model, validation_loader)
    history.append({"epoch": epoch + 1, "train_loss": train_loss, **validation_metrics})

assert history[-1]["loss"] < initial_metrics["loss"]
assert 0.0 <= history[-1]["accuracy"] <= 1.0
```

调用 `model(features)`，而不是直接调用 `model.forward(features)`，让 Module 的 hooks 等调用机制正常工作。这里模型很小，还没有 Dropout 和 BatchNorm，但仍显式区分 `train` / `eval`，避免换成稍大的网络后悄悄出问题。

训练 loss 记录的是各 batch **更新前**算出的 loss，它对应一轮中不断变化的模型；验证 loss 则在一轮结束后的固定模型上计算。两者不完全是同一种测量，不能只看两条曲线的高低就下结论。

验证函数临时切换模式，再恢复进入时的状态。这个简化实现假定整个模型原本统一处于训练或评估模式；如果各子模块有不同模式，需要分别保存。它还明确拒绝空数据，不让“除以零”或者一个假 0 掩盖问题。

## 4. 指标好不好，先看分母

假设两个 batch 分别有 16 和 1 个样本，平均 loss 是 1 和 9。等权平均得到 5；真正的样本平均是：

$$
\frac{16\times1+1\times9}{17}=\frac{25}{17}\approx1.47.
$$

训练代码里的 `loss * count` 是先把 batch 均值还原成总量，再除以总样本数。验证代码直接用 `reduction="sum"` 累计。

这个换算依赖本例条件：没有 class weight，没有被忽略的标签，每个样本贡献一个 loss。加入 `ignore_index`、类别权重、变长 token mask 后，分母可能是有效 token 数或权重和，不一定还是 `labels.numel()`。也不要为了 batch 整齐，就在验证集用 `drop_last=True` 丢掉尾部样本。

| 你看到的现象 | 先检查什么 |
| --- | --- |
| 训练 loss 不动 | 参数是否真的交给 optimizer；梯度是否非空；学习率；目标与标签 |
| 训练越来越好，验证变差 | 数据切分、过拟合、任务分布差异；不要先扩大模型 |
| 每次结果变化很大 | 数据顺序、初始化、随机性、样本量和切分方式 |
| accuracy 高得离谱 | 标签是否泄漏到特征；同源数据是否跨越训练与验证 |
| loss 下降但 accuracy 不变 | 概率置信度可以改善，却还没跨过 argmax 的分类边界 |

固定种子有助于复查这段实验，但不保证跨 PyTorch 版本、硬件和所有算子逐位一致。验证集用于调参；真实实验还要保留不反复窥看的测试集。这 47 个合成点不支持任何实际任务的效果结论。

## 5. 参数、buffer 和普通变量

把张量放到对象上，不等于把它注册成参数。训练参数用 `nn.Parameter` 或现成层；不参与梯度更新、但需要随模型移动或保存的状态，可以用 buffer。

```python
class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 4), nn.Linear(4, 2)])
        self.register_buffer("input_scale", torch.ones(2))

    def forward(self, features):
        hidden = torch.relu(self.layers[0](features / self.input_scale))
        return self.layers[1](hidden)

registered_model = TinyClassifier()
assert "layers.0.weight" in dict(registered_model.named_parameters())
assert "input_scale" not in dict(registered_model.named_parameters())
assert "input_scale" in registered_model.state_dict()
assert registered_model(torch.ones(3, 2)).shape == (3, 2)
```

用普通 Python list 收集子层，不会像 `ModuleList` 那样自动注册它们。结果可能是前向能跑，`model.parameters()` 却找不到这些权重。默认 persistent 的 buffer 会进 `state_dict`；普通张量属性没有同样的自动管理。

`ReLU`、`Dropout` 可以是 Module，却没有可训练参数。反过来，直接用 functional 接口也能建立可求导的运算；关键是参与计算的权重由谁持有、是否注册，以及训练模式怎样传入。

## 6. 保存和日志，也要说清恢复了什么

下面做一次内存中的保存与读取，不创建文件。`deepcopy` 得到独立快照，避免后面的训练继续改变你以为已保存的内存状态。`weights_only=True` 是这里显式使用的加载选项，不是让你随便加载不可信文件。

```python
import copy
import io

snapshot = copy.deepcopy({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": len(history),
})
buffer = io.BytesIO()
torch.save(snapshot, buffer)
buffer.seek(0)
loaded = torch.load(buffer, weights_only=True)
restored_model = nn.Linear(2, 2)
restored_model.load_state_dict(loaded["model"])
restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.15, momentum=0.9)
restored_optimizer.load_state_dict(loaded["optimizer"])
with torch.no_grad():
    torch.testing.assert_close(model(features), restored_model(features))
assert loaded["epoch"] == 25
```

这个断言只证明恢复后的模型对同一输入给出相同输出，不证明能从任意位置逐位续训。要继续同一条训练轨迹，还需考虑 scheduler、AMP scaler、随机数状态、数据采样位置和分布式状态。`state_dict` 也不保存整个模型的结构定义或所有运行模式，恢复端必须提供兼容的模型。

日志至少分开 train 和 validation，横轴标清是 epoch、optimizer step 还是处理过的 token 数。不要把最后一个 batch 的 loss 写成整轮 loss，也不要跨 epoch 累加 accuracy 后继续称它为“本轮准确率”。

下面是可选的 TensorBoard 接法；需要另装 `tensorboard`，前面的实验不依赖它。这里给出接入函数，不把“函数能定义”当成已经在浏览器检查过日志界面。

```python
def write_tensorboard(history, log_dir):
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(log_dir=log_dir) as writer:
        for entry in history:
            writer.add_scalar("loss/train", entry["train_loss"], entry["epoch"])
            writer.add_scalar("loss/validation", entry["loss"], entry["epoch"])
            writer.add_scalar("accuracy/validation", entry["accuracy"], entry["epoch"])
```

TensorBoard 画出的网络图依赖捕获方式和示例输入，不能当作程序所有动态分支的完整证明。曲线用来提出问题，再回代码和数据验证，不能代替验证。

## 7. 换到 GPU，数据要走哪些地方？ {#gpu-data-flow}

前面的代码一直在 CPU 上。换到 GPU，不是把所有东西都搬过去：数据通常仍由 CPU 读取和整理，模型在设备上计算，日志再取回少量结果。

<figure class="worked-update">
<ol>
<li><small>01 / CPU</small><strong>把样本拼成 batch</strong><span>Dataset 取样，collate 整理 features 和 labels。文本任务还要处理 padding、长度和 mask。</span></li>
<li><small>02 / 主机内存</small><strong>准备传输</strong><span>可选的锁页内存（pinned memory）仍在 CPU 一侧，不是显存。这里装的是接下来要送出的 batch。</span></li>
<li><small>03 / 传输</small><strong>把 batch 送到设备</strong><span>features、labels 和模型要在兼容设备上。同一条 CUDA stream 中，后面的计算等待前面的复制。</span></li>
<li><small>04 / GPU</small><strong>前向、反向、更新</strong><span>权重留在设备上；每批产生激活和梯度。更新后留下新权重与优化器状态。</span></li>
<li><small>05 / 取回结果</small><strong>记录必要的指标</strong><span>通常不必把每个 batch 的全部 logits 搬回 CPU。读取 GPU 标量也可能让主机等待。</span></li>
</ol>
<figcaption>这是常见的单 GPU 训练路径。CPU offload、GPU 数据预处理和多卡通信会增加其他路径。</figcaption>
</figure>

设备内部也不是所有数据都一直待在同一个地方：张量通常放在 GPU 的全局内存中，kernel 按需把一部分数据读到 cache、shared memory 或寄存器里运算。算完的结果再留给后续算子。具体经过哪层、是否融合，取决于 kernel；Python 里一个 `matmul` 不代表整块矩阵一次装进片上存储。想继续看这部分，可以接着读 [attention kernel](../deep-dives/attention-kernels.md)。

### 先把设备放对，再谈加速 {#device-placement}

下面只改一次更新的设备处理。实际迁移时，先把模型放到目标设备，再建立 optimizer；DataLoader 可以先保持简单，等测出瓶颈再调整。

```python
def device_step(model, optimizer, features, labels):
    device = next(model.parameters()).device
    features = features.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(features), labels)
    loss.backward()
    optimizer.step()
    return loss.detach()
```

这个函数假设模型参数在同一设备上，输入是前面的两个 tensor，不是通用多设备 batch 搬运器。CPU 版本会做数值测试；**这篇没有运行 CUDA 吞吐测试**。返回的 loss 已脱离计算图，但仍在原设备上；如果把每一步的返回值一直存进列表，还是会积累张量。

`non_blocking=True` 的意思是尽可能不让主机在复制调用处等待，不等于“下一批传输已经和当前计算重叠”。真要重叠，还要有合适的 pinned 源数据、不同 stream、可用复制引擎和正确依赖。先用普通循环跑通，再看 profiler，通常比一开始手写预取器好排查。[PyTorch 的传输教程](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)有对应的时间线对照。

### 用 3 个 batch 看懂等待发生在哪儿 {#pipeline-example}

假设每批准备数据需要 4 ms，传输 2 ms，GPU 计算 8 ms。数字仅用于算时间，**不是测量结果**。串行执行 3 批要 $3\times(4+2+8)=42$ ms。

如果 3 个阶段各有独立资源、没有资源争用，并且能提前准备下一批，第一批仍需要 14 ms。之后最慢的计算阶段每 8 ms 才能完成一批，总共是 $14+2\times8=30$ ms：

| Batch | CPU 准备 | 传输 | GPU 计算 |
| --- | --- | --- | --- |
| 1 | 0–4 ms | 4–6 ms | 6–14 ms |
| 2 | 4–8 ms | 8–10 ms | 14–22 ms |
| 3 | 8–12 ms | 12–14 ms | 22–30 ms |

<details markdown="1">
<summary>换一组耗时试试（Python）</summary>

```python
def ideal_pipeline_ms(batch_count, prepare_ms, transfer_ms, compute_ms):
    if not isinstance(batch_count, int) or isinstance(batch_count, bool) or batch_count < 1:
        raise ValueError("batch_count must be a positive integer")
    stages = (prepare_ms, transfer_ms, compute_ms)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not 0 < value < float("inf") for value in stages):
        raise ValueError("stage times must be finite and positive")
    serial = batch_count * sum(stages)
    overlapped = sum(stages) + (batch_count - 1) * max(stages)
    return serial, overlapped

assert ideal_pipeline_ms(3, 4, 2, 8) == (42, 30)
assert ideal_pipeline_ms(3, 12, 2, 8) == (66, 46)
```

</details>

第二个算例把 CPU 准备改成 12 ms，这时数据准备成了最慢一段。单纯让 GPU 算得更快，未必能显著提高长期吞吐。真实系统还有队列容量、变长样本、资源争用和启动开销，这个理想公式只帮我们找问题，不预测实际速度。

### 卡利用率低，先查哪一段？ {#gpu-waiting}

| 看到的现象 | 可以先做的小检查 | 别急着做什么 |
| --- | --- | --- |
| GPU 常等下一批 | 临时复用一个已准备好的 batch，与正常读取对比 | 不看磁盘、分词和 collate 就加卡 |
| 复制次数特别多 | 检查是不是逐样本搬运，或反复 `.cpu()` / `.item()` | 只加 `non_blocking=True` 就算优化完 |
| CPU 内存随 worker 数增加 | 看预取队列、每批大小和进程内副本 | 认为 `num_workers` 越多越好 |
| 计算很快，记日志却慢 | 降低同步取指标的频率，再比较时间线 | 直接关掉所有检查或保留所有输出 |

复用 batch 只是定位数据开销，不能拿来报告真实训练吞吐。计时也要等所测 GPU 工作完成；只量 Python 调用返回，可能只量到了排队。多 stream 还需处理依赖与张量生命周期，不能靠随手插一个 `synchronize()` 证明写法高效。[CUDA semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html#asynchronous-execution)说明了这些边界。

前面的保存例子只检查输出一致。接下来若要处理被打断的训练，去看 [checkpoint 与恢复](../../practice/post-training/checkpoint-and-resume.md)：那里会让“接着跑”与“从同一权重重新开始”真正分出差别。

## 资料与下一步

这一章没有用旧版 torchtext 加载数据：torchtext 已停止开发，0.18 是最后一个稳定版本。学习 PyTorch 不需要先装齐它的所有领域库，也不要照着老截图盲目补依赖。

- [Linear](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Linear.html)、[Module](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html)：维度、注册、状态与调用接口。
- [Data utilities](https://docs.pytorch.org/docs/2.8/data.html)：Dataset、DataLoader、collate 与多进程；iterable dataset 的分片和 `drop_last` 还需单独检查。
- [CrossEntropyLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html)：类别索引、概率目标、权重与归约。
- [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、[TensorBoard](https://docs.pytorch.org/docs/2.8/tensorboard.html)：状态保存与日志。
- [Reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html)、[torchtext 状态](https://docs.pytorch.org/text/stable/index.html)：种子的保证范围和依赖维护情况。

接下来把小模型换成[从零实现的模块](../code/README.md)，或去看[语言模型的一次训练更新](../deep-dives/training-step.md)。如果你能说清每个 batch 的形状、loss 分母和参数更新时机，再扩大模型才不容易把小错误一起放大。
