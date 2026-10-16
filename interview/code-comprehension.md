# 代码理解：读懂、改对，还要能验证

**中文** · [English](code-comprehension.en.md)

> 最近审阅：2026-10 · 原创教学练习，不是任何公司的面试原题。

从零写一个解法，和接过别人写的代码，是两种练习。后者先要弄清原来的行为：输入怎么走、状态存在哪里、结果为什么不对。代码能跑，甚至大部分测试都绿，也不代表理解对了。

这页用一个很小的 embedding cache 和一个指标聚合例子练这件事。不需要 GPU、模型或外部服务。先读代码，写下预期，再运行；参考修复放在后面，不必一开始就打开。

## 先问清这一轮怎么考 {#rules}

读代码、排错这类环节，形式差别很大：有的能运行代码和测试，有的只能手工推演；有的允许指定的 AI 工具，有的完全不允许。同一个团队的不同轮次，规则也可能不一样。收到通知后，直接向 recruiter 确认：

- 这一轮要从零实现，还是阅读并修改现有程序？
- 能否运行代码和测试？使用什么语言、编辑器和资料？
- 是否明确允许 AI？只限哪一轮、哪个工具，哪些材料可以输入？

**没有明确允许，就不要在正式面试里用 AI。**读代码的练习也不能代替算法准备，LeetCode 仍要练。

## 拿到陌生代码，先看哪儿？ {#read}

先问程序本来该做什么，再挑一个输入追到底。没必要从第一行开始逐字朗读，也别看到一个可疑写法就重构整个文件。

<figure class="worked-update" lang="zh-CN">
<ol>
<li><small>01 / 行为</small><strong>先说输入和预期</strong><span>参数、返回值、允许的空值；谁会调用，调用之间会不会共享状态。</span></li>
<li><small>02 / 证据</small><strong>沿着一个例子走</strong><span>标出分支、对象修改和缓存命中；找最小反例，不靠“感觉有问题”。</span></li>
<li><small>03 / 修改</small><strong>小改动，加回归测试</strong><span>解释修了哪条行为，检查原本正确的情况，再谈速度和剩余风险。</span></li>
</ol>
<figcaption>这是本站的练习顺序，不是某家公司的评分表。能运行测试时用测试；不能运行时把同一组输入手工走完。</figcaption>
</figure>

一个自然的开场可以是：“我先确认一下，相同文本换了模型版本，应该重新计算对吗？那我先追这两个调用，看看缓存 key 有没有包含版本。”

## 小练习一：换了模型，结果怎么没变？ {#cache}

约定先写清：单线程、进程内缓存；输入是字符串版本号与文本；编码器返回一维数值 list；同一 `(model_version, text)` 的结果可复用。版本必须标识不可变的模型、tokenizer 和预处理配置，不能用不断指向新权重的别名。调用方修改返回的 list，不应改变缓存。教学接口允许空 list，不代表真实 embedding 服务一定允许。

下面是**故意有问题**的实现。先不修，预测最后两行的结果。

```python
class BuggyEmbeddingCache:
    def __init__(self, encode):
        self.encode = encode
        self.values = {}

    def get(self, model_version, text):
        if not self.values.get(text):
            self.values[text] = self.encode(model_version, text)
        return self.values[text]


def encode(model_version, text):
    return [len(text), 1 if model_version == "v1" else 2]


cache = BuggyEmbeddingCache(encode)
first = cache.get("v1", "cat")
second = cache.get("v2", "cat")
```

编码器只是桩函数，数值不表示真实语义。按约定，`first` 应是 `[3,1]`，`second` 应是 `[3,2]`。实际两次都是 `[3,1]`：第二次只按文本命中了第一次的结果。

再看两个边界：`first[0] = 99` 后重新读取会怎样？如果编码器返回 `[]`，重复请求会不会再次计算？把“缺失”和“值为假”混在一起、把可变对象直接交出去，是两个独立问题。

<details markdown="1">
<summary>看最小修复与它仍然没做的事</summary>

```python
class EmbeddingCache:
    def __init__(self, encode):
        self.encode = encode
        self.values = {}

    def get(self, model_version, text):
        key = (model_version, text)
        if key not in self.values:
            self.values[key] = list(self.encode(model_version, text))
        return self.values[key].copy()
```

key 加版本，显式判断 key 是否存在，存入与返回都复制 list。这里假设元素是不可变数值，所以浅拷贝够用；换成嵌套可变对象就要重新定义接口。

测试至少包括：同 key 只编码一次；换版本重新编码；空 list 也命中；调用方或编码器后来修改原 list 不污染缓存。只用不同文本测试，发现不了版本遗漏。

这不是生产缓存：没有容量上限、过期、并发请求合并或权限隔离。真实系统先定义共享范围与安全要求，再决定是否把租户、权限或预处理配置放进 key；“加一个 tuple”不等于完成了多租户安全。

</details>

**复杂度也要说明范围。**字典查找平均 O(1) 不包括构造 / 哈希文本与复制长度为 D 的向量。命中路径仍有 O(D) 复制，字符串首次哈希可能随文本长度增长；未命中还要算编码成本。缓存内存随条目、文本和向量增长，不是 O(1)。

## 小练习二：每个 batch 都算对了，总分却不对 {#aggregation}

约定指标是全部有效 token 的平均 loss，padding 已排除。两个 batch 的 loss 为 `[2,2]` 和 `[8]`，目标值是 `(2+2+8)/3 = 4`。

```python
def buggy_token_mean(batches):
    means = [sum(batch) / len(batch) for batch in batches if batch]
    return sum(means) / len(means)
```

它得到 `(2+8)/2 = 5`。不是浮点误差，而是每个 batch 权重相同，短 batch 中的 token 被看得更重。若指标本来就定义为 batch 均值，这不算 bug；本练习错在违背了前面的指标约定。

<details markdown="1">
<summary>看修复与应该补的测试</summary>

```python
def token_mean(batches):
    total_loss = 0.0
    token_count = 0
    for batch in batches:
        total_loss += sum(batch)
        token_count += len(batch)
    if token_count == 0:
        raise ValueError("No valid tokens")
    return total_loss / token_count
```

同一批 token 改成 `[[2],[2,8]]`，结果仍应是 4。补空 batch、全空输入、不同大小 batch 的测试。本例明确全空报错，不能默默当成 loss 为 0。

跨 rank 汇报这个指标时，归约的是 loss 总和与有效 token 数，再相除，不是无条件平均各 rank 的 mean。这只是**指标聚合**；若要实现正确的分布式反向传播，还需单独考虑梯度归约和各框架的缩放规则。

</details>

继续：[训练循环](../00-foundations/pytorch/training-loop.md) · [数据与训练目标](../practice/post-training/data-and-objectives.md) · [FSDP 与训练栈](../06-systems/training-frameworks.md)。

## 可以用 AI 时，怎样练才有意义？ {#ai}

练习时先写自己的判断，再让 AI 找反例。比如：“我怀疑 key 缺模型版本。请给一组能验证这个判断的输入，不改代码。”拿到它给的输入后，自己按代码走一遍；如果这组输入不符合前面的约定，要说明为什么不采用。

也可以让它给两个修复版本，再比较是否改变语义。例如为缓存加 TTL，并不能修复同一时间窗口里两个模型共用错误 key。加了复杂功能，不代表解决了原问题。

正式面试只用明确允许的工具和材料；不要把保密题目或公司的内部代码粘进未经许可的外部服务。AI 辅助不免除你解释改动、验证结果的责任，也不能代替自己写算法。

## 做一轮 45 分钟练习 {#practice}

| 用时 | 动作 | 可检查的结果 |
| --- | --- | --- |
| 5 分钟 | 读约定，画两三个函数之间的调用 | 说出输入、输出与共享状态 |
| 10 分钟 | 手算 cache 的版本、修改和空值情况 | 实际行为与预期哪里不同 |
| 10 分钟 | 修 cache，写回归测试 | 旧实现会失败，修复后通过 |
| 10 分钟 | 修 token 聚合，换 batch 划分 | 指标对重新分组不敏感 |
| 10 分钟 | 解释复杂度与限制，回应一个错误修复建议 | 不靠重写整个系统掩盖没理解的地方 |

从仓库根目录运行：[教学代码](code/comprehension_lab.py)同时保留了错误版和参考版，方便对照。先自己做，再看实现。

```bash
python3 interview/code/comprehension_lab.py
python3 -m unittest discover -s site/tests -p test_speedrun.py -v
```

后一个命令也检查站点页面，需要站点的常规测试依赖。没有完整仓库时，单独跑第一个文件即可，它只用 Python 标准库。通过这些小测试不代表测过真实模型、并发缓存或线上服务。

## 读完接着去哪儿 {#next}

- 只剩一点准备时间：[Speedrun](speedrun.md)。
- 算法轮仍不稳：[LeetCode 方法](leetcode.md)。
- 想读更贴近工作的代码：[vLLM / SGLang 请求链路](../06-systems/inference-frameworks.md#engine-map)，选一个模块追，不必从仓库第一行读起。
- [Google 的代码审阅指南](https://google.github.io/eng-practices/review/reviewer/looking-for.html)可用来练看设计、行为和测试；它是工程实践指南，不是本页练习或面试流程的官方认证。
