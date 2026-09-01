# 面试基础题：大半在问同一件事

**中文** · [English](interview-basics.en.md)

> 阅读时间：约 10 分钟 · 类型：速查 · 最近审阅：2026-08

## 这些题共同在检查什么

这些题看着散——损失函数、掩码、归一化、RNN、CNN——其实只分三类：**梯度有没有一条不被衰减的路**、**训练和推理是不是同一件事**、**不变性是结构自带的还是花钱买的**。认出是哪一类，答案就不用背了。

下面每题给三层：**一句话先说什么** → **撑住第一次追问** → **能拉开差距的那一层**。

---

## 先立骨架：attention 到底怎么算

后面三题都挂在这张图上，先立着。

```
X                                   [B, T, d_model]
 ├─ Q = X·Wq ─┐                     [B, h, Tq, d_k]
 ├─ K = X·Wk ─┤  split 成 h 个头     [B, h, Tk, d_k]
 └─ V = X·Wv ─┘                     [B, h, Tk, d_v]
 │
 ① scores = Q·Kᵀ / √d_k             [B, h, Tq, Tk]
 ② scores = scores + mask           ← 因果掩码在这一步
 ③ A      = softmax(scores, -1)     每行和为 1
 ④ out    = A·V                     [B, h, Tq, d_v]
 ⑤ concat 各头，过 Wo
```

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

**怎么读 $A$**：第 $i$ 行是「token $i$ 把注意力分给谁」的概率分布。加了因果掩码后，第 $i$ 行只在列 $\le i$ 上非零。第 1 行是退化的——它只能看自己，softmax 出来恰好 1.0。

**为什么除 $\sqrt{d_k}$，不是 $\sqrt{d_v}$ 或 $\sqrt{d_{\text{model}}}$**：分数来自
$QK^\top$，每个分数恰好加了 $d_k$ 个乘积。Q/K 最后一维必须相同，V 的特征维
可以不同；只是在标准 MHA 中通常令 $d_v=d_k$。例如 768 维、12 个头，单头
$d_k=64$，所以除 $\sqrt{64}$。完整推导与 $Q=K$ 的极端情况见
[Transformer 架构深拆](transformer.md#核心公式缩放点积注意力)。

时间 $O(T^2 d)$、显存 $O(T^2)$。那个 $T\times T$ 矩阵就是长上下文的瓶颈，也是 FlashAttention 的动机：**根本不把它算出来存下**。

参考实现在 [`00-foundations/code/attention_numpy.py`](code/)。

---

## 第一类：梯度有没有一条不被衰减的路

三道题，同一个骨架。**只要出现「连乘」，就要问能不能换成「加法」。**

<details class="interview" markdown="1">
<summary>p = σ(z)，y 是 0/1。写出 MSE 和 BCE，说说该用哪个</summary>

$$\mathcal{L}_{\text{MSE}} = (p-y)^2 \qquad \mathcal{L}_{\text{BCE}} = -\big[y\log p + (1-y)\log(1-p)\big]$$

**这题真正问的是梯度，不是让你默写公式。** 关键事实：$\sigma'(z) = p(1-p)$。

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} = 2(p-y)\cdot p(1-p) \qquad\qquad \frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = p - y$$

BCE 那边，$\sigma'$ 被约掉了：

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial p} = \frac{p-y}{p(1-p)} \;\Longrightarrow\; \frac{\partial \mathcal{L}}{\partial z} = \frac{p-y}{p(1-p)}\cdot p(1-p) = p-y$$

**后果**：$y=1$ 而模型极自信地说 0（$z\to-\infty$，$p\to0$）时——

- MSE 梯度 $\approx 2(0-1)\cdot 0\cdot 1 = 0$。**预测偏差最大时梯度反而消失，模型学不动。**
- BCE 梯度 $= -1$。**预测偏差最大时仍能保留有效梯度。**

**再深一层**：logistic regression 里 BCE 关于权重是凸的，MSE + sigmoid 不是。

**一个常见的错误说法**：别说「MSE 不是 proper scoring rule 所以不校准」——**平方误差就是 Brier score，它也是 proper scoring rule**。两者都能给出校准概率，区别在优化行为，不在 properness。主动纠正这一点通常能加分。

**工程上**：永远用 `binary_cross_entropy_with_logits`。先算 $p$ 再取 log，$p$ 下溢到 0 就 $-\infty$。稳定形式：

$$\mathcal{L} = \max(z,0) - zy + \log\big(1+e^{-|z|}\big)$$

</details>

<details class="interview" markdown="1">
<summary>RNN 和 LSTM 的区别</summary>

**先说**：vanilla RNN 的问题不是「记不住」，是**梯度沿时间是连乘的**。

$$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=1}^{k} W_h^\top \operatorname{diag}\big(\tanh'(\cdot)\big)$$

$\tanh' \le 1$、$\sigma' \le \tfrac14$（在 $z=0$ 取到）。连乘 $k$ 个 ≤1 的数 → 指数衰减。若 $\|W_h\|>1$ 则反过来爆炸。

**爆炸能靠裁剪解决，消失不能**——裁剪只压上界。所以真正的病是消失。

**LSTM 的修法**：加一条 cell state，更新是**加法**的。

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t)$$

关键偏导 $\dfrac{\partial c_t}{\partial c_{t-1}} = f_t$ —— **逐元素的门，不是「矩阵乘 + 饱和非线性」**。遗忘门 $f_t \approx 1$ 时梯度几乎不衰减地流回去（constant error carousel）。

一句话：**LSTM 把「每步乘一个矩阵」换成了「门控的加法累积」。**

**细节**：三个门用 sigmoid，因为门要的是 0~1 的软开关；候选值 $g$ 用 tanh，因为它是个值、需要正负。**这是语义选择，不是梯度考虑。**

**两个能拔高的连接：**

- **LSTM 的 cell state 和 Transformer 的残差流是同一个东西**——留一条加法的恒等通路，让梯度不必穿过非线性。一个沿时间，一个沿深度。
- **Transformer 取代 RNN 不是因为梯度**（那问题 LSTM 已经解决了），是因为 **RNN 沿时间串行、无法并行**；attention 任意两点路径长度 $O(1)$，整个序列一次算完。**赢的是并行度，不是表达力。**

</details>

<details class="interview" markdown="1">
<summary>为什么要 LayerNorm？为什么不用 BatchNorm？放在哪里？</summary>

**为什么要归一化**：残差流的尺度随深度累积，不归一化则深层激活越来越大，梯度爆炸或消失，深堆叠训不动；归一化还改善损失面条件数，允许更大的学习率。

**为什么不是 BatchNorm** —— 四条，第三条最少人说得出：

1. 变长序列 + padding，BN 沿 batch 维的统计量会被 padding 污染；
2. BN 依赖 batch 大小与组成，而自回归解码 batch 常为 1、逐 token 出，只能退回 running stats，训练/推理不一致；
3. **BN 会泄漏未来**——若沿时间维统计，后面的 token 就进了前面 token 的归一化统计里，**你刚加的因果掩码被从后门绕过去了**；
4. LN 只在特征维上对每个 token 自己归一化，与 batch 无关、与其他位置无关，训练推理完全一致。

**放哪里**——两种都要能写：

```
Post-LN（2017 原版）           Pre-LN（现代）
x = LN(x + Attn(x))          x = x + Attn(LN(x))
x = LN(x + FFN(x))           x = x + FFN(LN(x))
                             ...
                             x = LN(x)   ← 末尾必须补 final LN
```

**为什么 Pre-LN 更稳**：Post-LN 里 **LayerNorm 压在残差主干上**，梯度每层都要穿过它一次，LN 的雅可比会缩放梯度，几十层叠起来传到浅层就很小了——所以原版需要 warmup。Pre-LN 的主干是**干净的恒等路径**，$\partial x_{\text{out}}/\partial x_{\text{in}}$ 带 identity 项，梯度直接流回去，不用 warmup 也能堆很深。

**代价**：残差流从头到尾没被归一化，尺度随深度增长，**所以末尾必须补一个 final LN**。这条常被忘，答出来是加分项。

**再往下**：RMSNorm 去掉减均值和 bias，只按 RMS 缩放，照样work——说明**起作用的是 re-scaling，不是 re-centering**。

</details>

---

## 第二类：训练和推理必须是同一件事

<details class="interview" markdown="1">
<summary>为什么要 causal mask？放在哪一步？为什么放那儿？</summary>

**先说**：它让「一次前向并行训练 $T$ 个位置」等价于「一个位置一个位置地训」。

自回归目标是 $\prod_t p(x_t\mid x_{<t})$。self-attention 默认全可见，位置 $t$ 会看到 $x_t$ 自己——**预测下一个 token 变成抄答案**，训练 loss 掉到接近 0；而推理时未来根本不存在，生成全崩。

**所以掩码不是为了让模型更强，是为了让训练和推理是同一个模型。** 没有它你得跑 $T$ 次前向。

**放在骨架图的第 ② 步**：缩放点积之后、softmax 之前。

```python
mask   = np.triu(np.ones((T, T), dtype=bool), k=1)   # 严格上三角 = 禁止
scores = np.where(mask, -np.inf, scores)             # 加 -inf，不是置零
```

**为什么必须在 softmax 之前**（追问重点）：加 $-\infty$ 后 $e^{-\infty}=0$，**softmax 在剩余位置上重新归一化**，数学上等价于「那些位置不存在」。

在 softmax **之后**置零则**破坏归一化**——每行不再和为 1，而且破得不均匀：位置 1 只能看自己，被删掉的质量最多，输出被缩得最狠。等于给每个位置乘了一个意义不明的衰减系数。

**工程细节**：实践中常用大负数（`-1e9` 或 `torch.finfo(dtype).min`）而非真 `-inf`。因为 fp16 下如果某行被**全部**掩掉（padding 行会出现），softmax 得到 `0/0 = NaN`。

**BERT 为什么不需要**：它不是自回归的，训练目标是 MLM，双向可见是设计而不是漏洞。

</details>

---

## 第三类：不变性是免费的还是买来的

<details class="interview" markdown="1">
<summary>CNN 里图像旋转会不会影响特征提取？</summary>

**会，而且影响很大。** 先把两个被混用的概念分开：

**卷积自带的是平移等变（equivariance），不是不变（invariance）**：

$$f(T_x(I)) = T_x\big(f(I)\big)$$

输入平移，特征图跟着平移同样的量。**不变性**（输出完全不变）是后面 pooling 给的，而且只是近似的、局部的。

**旋转：既不等变也不不变。** 卷积核朝向固定，一条 45° 的边和一条 135° 的边激活的是完全不同的滤波器，网络没有任何结构上的理由把它们当成同一个东西。

**为什么这个不对称是结构性的**：平移等变来自**权重共享 + 局部性**，是算子白送的；旋转等变不在算子里。所以只有两条路：

1. **用数据买**：旋转增广。网络学出一组冗余滤波器、每个朝向一份。代价是**花模型容量买不变性**，且只覆盖增广过的角度范围。
2. **改算子**：Group-equivariant CNN、Steerable CNN、Harmonic Networks；或 Spatial Transformer——让网络先学会把输入摆正。

**一句话收**：平移不变是免费的，旋转不变要付钱，付法是数据或者算子。

**能让对话变有意思的一条**：连平移不变性都没有大家以为的那么好——带 stride 的下采样会引入混叠，输入平移一个像素预测就可能翻。解法是抗混叠的模糊下采样。

</details>

---

## 加一道：Egg Drop 为什么换一个状态就简单了

有 $k$ 个鸡蛋、$n$ 层楼，目标是在最坏情况下找出临界楼层。直接想法确实是二维
DP：

$$T(k,n)=1+\min_{1\le x\le n}\max\big(T(k-1,x-1),\;T(k,n-x)\big).$$

在第 $x$ 层扔：碎了，只能往下且少一个蛋；没碎，只能往上且蛋数不变。
`min` 选楼层，`max` 表示要为较坏的分支负责。它是对的，但每个状态还要枚举 $x$。

更好的问法不是「这些楼需要几步」，而是：**给我 $m$ 次行动和 $k$ 个鸡蛋，最多能
覆盖多少层？** 记作 $F(m,k)$：

$$F(m,k)=F(m-1,k-1)+1+F(m-1,k),\qquad F(0,k)=F(m,0)=0.$$

第一次扔下去后，碎的分支能覆盖下面 $F(m-1,k-1)$ 层，当前层算 1，没碎的分支
能覆盖上面 $F(m-1,k)$ 层。于是每加一次行动，搜索空间就是两个旧子空间再加当前点。

```python
def min_moves(eggs, floors):
    cover = [0] * (eggs + 1)
    moves = 0
    while cover[eggs] < floors:
        moves += 1
        for k in range(eggs, 0, -1):
            cover[k] = cover[k] + cover[k - 1] + 1
    return moves
```

倒序更新是为了让右侧都来自上一轮。100 层时：2 个鸡蛋要 14 次，因为
$1+\cdots+14=105$；3 个鸡蛋要 9 次，因为 $F(8,3)=92<100$，而
$F(9,3)=129\ge100$。

**面试里的核心**：原题是「鸡蛋 × 楼层」二维 minimax DP；反转问题后变成
「行动次数 × 鸡蛋」的覆盖 DP，并能压成一维。你说的“每次缩小可以搜索的 space”
就是这个递推在计算的东西。

## 再加一道：Binary Tree Maximum Path Sum

这题最容易错的地方，是混淆「当前节点形成的完整答案」和「可以返回给父节点的状态」。

定义 $G(u)$：必须从节点 $u$ 出发，只能沿一侧向下延伸的最大路径和。空节点返回 0，
负贡献直接丢掉：

$$L=\max(0,G(u.left)),\qquad R=\max(0,G(u.right)).$$

以 $u$ 为最高点的完整路径可以同时使用左右两侧：

$$\text{candidate}=u.val+L+R.$$

但返回给父节点时不能分叉，只能选择一侧：

$$G(u)=u.val+\max(L,R).$$

```python
def max_path_sum(root):
    best = float("-inf")

    def gain(node):
        nonlocal best
        if node is None:
            return 0

        left = max(0, gain(node.left))
        right = max(0, gain(node.right))

        best = max(best, node.val + left + right)
        return node.val + max(left, right)

    gain(root)
    return best
```

每个节点只访问一次，时间 $O(n)$；递归栈为 $O(h)$。全局答案必须初始化成
$-\infty$，不能是 0，否则全负数树会错误地选择一条不存在的空路径。

如果输入是 list，先问清楚表示法：带 `None` 的 level-order serialization，还是
heap-style 的 `left=2i+1, right=2i+2`？对稀疏树，二者不等价。面试官给了不熟悉的
序列化格式时，先定义输入语义不是拖延，是在保护算法的正确性。

## 附：Transformer 结构怎么讲

被问「讲一下结构」时别按论文插图顺序背，**按每个部件解决什么问题讲**：

| 部件 | 它解决什么 |
| --- | --- |
| 位置信息 | attention 是置换等变的，打乱输入输出跟着乱——**它自己看不出词序** |
| 残差 | 给梯度一条恒等通路（同 LSTM 的 cell state） |
| $\sqrt{d_k}$ 缩放 | 点积是 $d_k$ 项之和，方差随 $d_k$ 增长；不缩放则 softmax 饱和成 one-hot，**梯度消失** |
| 多头 | **一个 softmax 只能表达一种注意力模式**；多头在不同子空间并行关注不同关系 |
| FFN（约 4× 放大） | **大部分参数在这里**，通常被读作 key-value 记忆 |
| causal mask | 见上，训练/推理一致性 |

**然后主动说这一段**，它区分「读过 2017 那篇」和「知道现在的模型长什么样」：

| 2017 原版 | 现代 LLM | 为什么换 |
| --- | --- | --- |
| Post-LN | Pre-LN + RMSNorm | 不用 warmup，能堆更深 |
| 正弦 / 学习式绝对位置 | RoPE | 相对位置，外推更好 |
| ReLU FFN | SwiGLU | 同等算力下更好 |
| MHA | GQA / MQA | **KV cache 是推理显存瓶颈** |

## 继续阅读

- [Vanilla Transformer](core/vanilla-transformer.md) · [多头注意力](core/multi-head-attention.md) · [Decoder-only](core/decoder-only.md)
- [归一化](core/normalization.md) · [残差连接](core/residual-connections.md)
- [语言模型的目标函数](deep-dives/language-model-objective.md)
- [参考实现](code/)：`attention_numpy.py` 从零实现，`attention_torch.py` 对照
