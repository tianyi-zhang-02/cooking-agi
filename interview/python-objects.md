# Python：变量、拷贝与函数调用

**中文** · [English](python-objects.en.md)

> 阅读时间：约 15 分钟 · 最近审阅：2026-10

“我只改了副本，原数据怎么也变了？”这类 bug 不需要背很多术语。先画出变量指向哪个对象，再看这一步是在改对象，还是换了一个指向，通常就能找到原因。

这一篇接在 [Python 常用写法](python.md)后面。想先弄懂数据为什么互相影响，读前 3 节；函数调用看第 4–6 节；迭代器和代码组织看最后两节。每块代码都可单独运行，只用标准库。例子在 Python 3.12 上验证，接口说明对照 Python 3.14 文档，不依赖某次运行的内存地址。

## 1. 两个名字，不一定是两份数据 {#names-and-objects}

`backup = labels` 没有复制列表。两个名字先指向同一对象；`append` 改了这个对象，因此两边都能看到。随后写 `backup = []`，只是把 `backup` 重新绑定到另一个列表。

```python
labels = ["draft"]
backup = labels
backup.append("checked")
assert labels == ["draft", "checked"]
assert backup is labels

backup = []
assert labels == ["draft", "checked"]
assert backup == []
assert backup is not labels
```

所以，**“改变量会同步影响其他变量”说得太笼统**。要区分修改对象（mutation）和重新绑定（rebinding）。`is` 比较对象身份，`==` 调用相等比较；两个内容相同的列表可以 `==` 为真、`is` 为假。不要用小整数或字符串碰巧复用对象的结果，推断所有情况下的身份关系。

“不可变”也不等于里面所有东西都不变。`bundle = (["draft"],)` 不能替换 `bundle[0]`，但里面的列表仍然可以 `append`。这个 tuple 也不能作为字典 key，因为其中包含不可哈希的列表。`id()` 返回身份标识；把它解释成内存地址是 CPython 的实现细节，不是所有 Python 实现的约定。[对象模型](https://docs.python.org/3.14/reference/datamodel.html#objects-values-and-types)

## 2. 浅拷贝到底隔开了哪一层？ {#shallow-copy}

假设一条笔记记录包含标题和标签列表。我们只复制最外层字典：

```python
record = {"title": "Attention", "tags": ["draft"]}
edited = record.copy()
edited["title"] = "Attention, revised"
edited["tags"].append("checked")

assert edited is not record
assert edited["tags"] is record["tags"]
assert record == {"title": "Attention", "tags": ["draft", "checked"]}
assert edited["title"] == "Attention, revised"
```

<figure class="worked-update" lang="zh-CN" id="copy-boundary">
<figcaption>标题变了，标签却一起变：看写入发生在哪一层</figcaption>
<ol>
<li><strong>复制后</strong><span>record → 原字典<br>edited → 新字典<br>两份字典的 tags → 同一个列表</span></li>
<li><strong>替换标题</strong><span>edited 的 title 换成新字符串。原字典里的 title 没有被替换。</span></li>
<li><strong>添加标签</strong><span>append 改的是共享列表。通过两份字典读 tags，都会看到 checked。</span></li>
</ol>
</figure>

如果只需要改标签，不一定要复制整棵对象树。可以明确复制会改动的那条路径：

```python
record = {"title": "Attention", "tags": ["draft"]}
edited = {**record, "tags": [*record["tags"], "checked"]}
assert record["tags"] == ["draft"]
assert edited["tags"] == ["draft", "checked"]
assert edited["tags"] is not record["tags"]
```

这里的标签都是不可变字符串；若标签自身又是可变对象，就要继续判断要不要复制。浅拷贝不是“遇到嵌套就无效”，而是**只隔离外层，保留内部共享**。对于普通内置 list，`values[:]`、`list(values)`、`values.copy()` 都是浅拷贝；不要把这个结论套到所有支持切片的类型上。[序列操作](https://docs.python.org/3.14/library/stdtypes.html#common-sequence-operations)

## 3. 深拷贝不是把每个引用都拆散 {#deep-copy}

深拷贝适合需要递归复制数据的情况，但“复制后完全没有共享引用”仍然不准确。下面两条记录本来就共用一个标签列表，复制以后，这种**副本内部的共享关系**会保留下来。

<details markdown="1">
<summary>运行一个共享引用和循环引用的反例</summary>

```python
from copy import deepcopy

shared_tags = ["draft"]
records = [{"tags": shared_tags}, {"tags": shared_tags}]
cloned = deepcopy(records)
assert cloned[0]["tags"] is cloned[1]["tags"]
assert cloned[0]["tags"] is not shared_tags
cloned[0]["tags"].append("checked")
assert cloned[1]["tags"] == ["draft", "checked"]
assert shared_tags == ["draft"]

cycle = []
cycle.append(cycle)
copied_cycle = deepcopy(cycle)
assert copied_cycle is not cycle
assert copied_cycle[0] is copied_cycle
```

</details>

`deepcopy` 用 memo 记住本次已经复制的对象，既处理重复引用，也避免沿循环无限递归。函数等对象可能直接复用，自定义类还可以改写复制规则；文件、socket 也不是这样就能复制的。[copy 文档](https://docs.python.org/3.14/library/copy.html)

选择前先问：谁会写这份数据，哪些层必须独立？复制几个标签很便宜；复制一个大缓存就可能不值得。需要共享配置时，保留共享反而是正确设计。也别把普通列表的 copy 规则直接当成 tensor 的规则，张量的存储共享见 [PyTorch：张量与存储](../00-foundations/pytorch/tensors-and-storage.md)。

## 4. 传进函数后，什么会留在外面？ {#function-binding}

参数名在函数里绑定到传入的对象。函数可以修改这个对象，但给参数名重新赋值，不会顺便替调用者的变量赋值。

```python
def append_then_replace(items):
    items.append("checked")
    items = ["replacement"]
    return items

original = ["draft"]
returned = append_then_replace(original)
assert original == ["draft", "checked"]
assert returned == ["replacement"]
assert returned is not original
```

先发生的 `append` 作用于原列表；后面的赋值只换了局部名字 `items` 的指向。若希望外面的变量也换成返回值，调用者要明确写 `original = append_then_replace(original)`。

这也解释了为什么列表的 `+=` 和 `+` 容易让人误判：对普通 list，`+=` 就地扩展，`+` 产生新列表。不能只看有没有等号，要看具体类型的操作语义。[参数绑定说明](https://docs.python.org/3.14/faq/programming.html#how-do-i-write-a-function-with-output-parameters-call-by-reference)

## 5. `*args` 与 `**kwargs`：定义时收集，调用时展开 {#arguments}

它们不是两个必须这样命名的特殊变量；特殊的是 `*` 和 `**`。下面把不定数量的分数放进 tuple，把额外设置放进 dict，同时要求 `scale` 只能通过关键字传入。

```python
def summarize(*scores, scale=1, **metadata):
    return sum(scores) * scale, scores, metadata

values = [2, 5]
options = {"scale": 2, "source": "toy"}
total, received, metadata = summarize(*values, **options)
assert total == 14
assert received == (2, 5)
assert metadata == {"source": "toy"}
```

看同一个符号的两个位置：定义中的 `*scores` **收集**位置参数，调用中的 `*values` **展开**可迭代对象。`**options` 的 key 必须是字符串；`scale` 如果既显式传入，又出现在展开的字典里，会抛 `TypeError`，不会自动选择后写的值。

`**kwargs` 接收未知配置很方便，也可能吞掉拼错的参数名。对固定接口，直接写明确参数更容易发现错误；需要转发配置时，再决定哪些 key 可以透传。[函数参数与解包](https://docs.python.org/3.14/tutorial/controlflow.html#arbitrary-argument-lists)

## 6. 默认参数和闭包：值在什么时候确定？ {#defaults-and-closures}

默认参数在执行函数定义时求值，不是在每次调用时重建。如果每次调用都需要新的列表，就在函数体里创建：

```python
def add_tag(tag, tags=None):
    if tags is None:
        tags = []
    tags.append(tag)
    return tags

first = add_tag("draft")
second = add_tag("checked")
assert first == ["draft"] and second == ["checked"]
assert first is not second

existing = []
assert add_tag("ready", existing) is existing
assert existing == ["ready"]
```

这并没有让函数变成“无副作用”：传入已有列表时，函数仍会修改它。`tags=None` 只解决**省略参数时**不小心共用默认列表的问题。

闭包则常在调用时读取外层变量。因此，“在循环里创建函数”不等于“每次都保存了当前值”：

```python
readers = [lambda: step for step in range(3)]
fixed_readers = [lambda saved=step: saved for step in range(3)]
assert [reader() for reader in readers] == [2, 2, 2]
assert [reader() for reader in fixed_readers] == [0, 1, 2]
```

后者利用默认参数在定义时求值的规则，保存每次的整数。但保存的是对象引用，不是自动深拷贝；如果保存一个列表，列表后来被改动，函数仍会看到改动。普通 `def` 里的闭包也有同样的规则，不是 lambda 独有。[默认参数](https://docs.python.org/3.14/tutorial/controlflow.html#default-argument-values) · [闭包的延迟绑定](https://docs.python.org/3.14/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result)

## 7. `any`、`all` 和迭代器：读到哪里就停在哪里 {#iteration-and-truth}

`any` 找到一个真值就停，`all` 找到一个假值就停。这叫短路（short-circuiting）。对空输入，`any` 为假，`all` 为真：没有找到真值，也没有找到反例。要验证“至少有一项，而且都合格”，仅用 `all` 不够。[内置函数](https://docs.python.org/3.14/library/functions.html#all)

```python
events = []

def passing_scores(scores):
    for score in scores:
        events.append(score)
        yield score >= 60

checks = passing_scores([80, 40, 90])
assert any(checks) is True
assert events == [80]
assert list(checks) == [False, True]
assert events == [80, 40, 90]
assert list(checks) == []
assert any([]) is False and all([]) is True
```

第一次 `any` 只消耗了 80 那一项。下一次不是从头重来，而是接着处理 40、90。列表可以重新取得迭代器；**同一个迭代器**耗尽后，不会自己倒回起点。`for` 背后也是取迭代器、不断请求下一项，直到 `StopIteration`。[迭代器与生成器](https://docs.python.org/3.14/tutorial/classes.html#iterators)

生成器避免一次存下所有输出，不代表它不占内存。它保留执行状态，也可能持有整个输入列表。若上游已经把大文件读成 list，给它包一层生成器并不能撤销那笔内存开销。读取文件时，把文件的生命周期放在 `with` 管理的范围内，别等到迭代一半才发现句柄已关。

## 8. 用函数还是类，先看状态归谁管 {#functions-or-classes}

Python 不要求在“面向过程”和“面向对象”之间二选一。一次性计算平均值，用函数就清楚；若数据分批到达，需要多次更新同一份统计，类可以把状态和更新方法放在一起。

<details markdown="1">
<summary>同一组分数，两种组织方法</summary>

```python
def average(scores):
    count, total = 0, 0.0
    for score in scores:
        count += 1
        total += score
    if count == 0:
        raise ValueError("At least one score is required")
    return total / count

class RunningMean:
    def __init__(self):
        self.count = 0
        self.total = 0.0

    def add(self, score):
        self.count += 1
        self.total += score

    def value(self):
        if self.count == 0:
            raise ValueError("At least one score is required")
        return self.total / self.count

meter = RunningMean()
for score in [2, 4, 9]:
    meter.add(score)
assert meter.value() == average([2, 4, 9]) == 5.0
other = RunningMean()
assert other.count == 0
```

</details>

这里的 `self.count` 是每个实例自己的状态；若把可变列表写成类属性，实例却可能意外共享它。用类不代表一定更好维护，用函数也不代表一定更快。上面两种方法都是线性扫描、常量辅助空间，性能要看具体实现。例子只考虑普通有限数值，不处理缺失值、并发更新或高精度累加。[类与实例变量](https://docs.python.org/3.14/tutorial/classes.html#class-and-instance-variables)

读完后，可以试着改 3 处：把标签换成嵌套字典；把 `any(checks)` 换成 `all(checks)`；连续创建两个 `RunningMean`。先预测哪些对象、哪些状态会变，再运行验证。接下来回到[复杂度与容器](algorithms/complexity-and-tools.md)，把“为什么会这样”接到“代价是多少”。
