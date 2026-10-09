# 面试基础题：大半在问同一件事

**中文** · [English](interview-basics.en.md)

> 阅读时间：约 10 分钟 · 类型：速查 · 最近审阅：2026-08

<span id="_1"></span>

## 这些题共同在检查什么 {#_3}

这些题看着散——损失函数、掩码、归一化、RNN、CNN——其实只分三类：**梯度有没有一条不被衰减的路**、**训练和推理是不是同一件事**、**不变性是结构自带的还是花钱买的**。认出是哪一类，答案就不用背了。

下面每题给三层：**一句话先说什么** → **撑住第一次追问** → **能拉开差距的那一层**。

---

## 先立骨架：attention 到底怎么算 {#attention}

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
[Transformer 架构深拆](transformer.md#_2)。

时间 $O(T^2 d)$、显存 $O(T^2)$。那个 $T\times T$ 矩阵就是长上下文的瓶颈，也是 FlashAttention 的动机：**根本不把它算出来存下**。

参考实现在 [`00-foundations/code/attention_numpy.py`](code/)。

---

## 第一类：梯度有没有一条不被衰减的路 {#_4}

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

**别把优化和校准混为一谈**：对二分类概率，平方误差对应 Brier score，也是 proper scoring rule。它和 BCE 在理想总体目标下都偏好真实条件概率，但有限数据和受限模型并不保证校准。这里比较的是梯度，而不是谁“有资格”预测概率。

**工程上**：永远用 `binary_cross_entropy_with_logits`。先算 $p$ 再取 log，$p$ 下溢到 0 就 $-\infty$。稳定形式：

$$\mathcal{L} = \max(z,0) - zy + \log\big(1+e^{-|z|}\big)$$

</details>

<details class="interview" markdown="1">
<summary>RNN 和 LSTM 的区别</summary>

**先说**：普通 RNN 难学长距离关系，一个原因是梯度要沿时间反复传递。

$$\frac{\partial h_t}{\partial h_{t-k}} = J_t J_{t-1}\cdots J_{t-k+1},\qquad J_t=\operatorname{diag}\big(\tanh'(a_t)\big)W_h$$

上式是状态对状态的 Jacobian；反向传播乘的是这些矩阵的转置。虽然每一步共享 $W_h$，激活值不同，$J_t$ 也会不同。如果所有 $\|J_t\|_2$ 都不超过某个 $c<1$，这条路径的梯度就会随距离衰减。某一步的范数大于 1 则不等于一定爆炸，还要看方向和后续乘积。

**梯度裁剪能限制一次更新的幅度，不能把已经消失的信号找回来**，也不能代替对训练不稳定原因的排查。

**LSTM 的修法**：加一条 cell state，更新是**加法**的。

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t)$$

把门值暂时固定，沿 cell state 的直接通路有 $\partial c_t/\partial c_{t-1}=\operatorname{diag}(f_t)$。这条通路只做逐元素缩放；$f_t$ 接近 1 时，更容易保留长距离信号。但总导数还包含门值对历史状态的依赖，不能据此说 LSTM 完全解决了梯度消失。

一句话：**LSTM 把「每步乘一个矩阵」换成了「门控的加法累积」。**

**细节**：sigmoid 把门值限制在 0 到 1，tanh 让候选值有正有负。这符合门和内容各自的作用，也会影响梯度；饱和问题并没有凭空消失。

**两个能拔高的连接：**

- **可以把 cell state 和残差流放在一起理解，但不能当成同一个结构。** 两者都有加法通路；LSTM 沿时间用门控制记忆，残差连接通常沿深度保留恒等项。
- **Transformer 的优势不只有一个。** 训练时可以并行处理序列位置，attention 缩短了可见位置之间的信息路径，也让模型按内容访问历史。LSTM 缓解了长依赖问题，但并没有消除它；自回归 Transformer 的生成也仍然逐 token 进行。

</details>

<details class="interview" markdown="1">
<summary>为什么要 LayerNorm？为什么不用 BatchNorm？放在哪里？</summary>

**为什么要归一化**：让子层接收到的数值尺度更可控，通常更容易训练深层模型。它和初始化、残差缩放、优化器一起起作用，不代表所有没有归一化的网络都训不动。

**为什么 Transformer 常用 LayerNorm，而不是 BatchNorm？**

1. 对变长序列，如果统计轴包含 padding 又没有排除它，BN 的统计量会受影响；
2. BN 依赖 batch 大小与组成，默认在推理时改用 running statistics；自回归解码的负载与训练不同，需要特别处理；
3. 若把时间轴纳入训练时的 BN 统计，未来 token 会影响前面的位置，因果 attention mask 并不能阻止这种泄漏；
4. 按特征维做的 token-wise LN 不混合 batch 或时间位置，归一化本身在训练和推理使用相同统计规则。这里不是说整个模型两种模式完全相同。

**放哪里**——两种都要能写：

```
Post-LN（2017 原版）           Pre-LN（现代）
x = LN(x + Attn(x))          x = x + Attn(LN(x))
x = LN(x + FFN(x))           x = x + FFN(LN(x))
                             ...
                             x = LN(x)   ← 常见的 final LN
```

**为什么常用 Pre-LN**：它在残差主干保留恒等项，不需要每一层都先穿过 LN。Xiong 等人的分析还发现，Post-LN 在初始化时靠近输出层的期望梯度可能较大，warmup 有助于避免初期更新过猛。论文在自己的实验设置里让 Pre-LN 不用 warmup 也能训练；这不是所有模型都可以取消 warmup 的保证。

**实现时检查**：Pre-LN 的最终残差输出通常还会经过一次归一化，再进入输出头。这是常见训练配方，不是数学上唯一可行的做法；复现时要跟具体模型的配置一致。

**再往下**：RMSNorm 不减均值，通常也没有加性 bias，只按 RMS 缩放并学习增益。它说明某些模型不做中心化也能训得很好，不说明中心化在所有场景都没有作用。参见 [LayerNorm 分析](https://arxiv.org/abs/2002.04745) 与 [RMSNorm](https://arxiv.org/abs/1910.07467)。

</details>

---

## 第二类：训练和推理必须是同一件事 {#_5}

<details class="interview" markdown="1">
<summary>为什么要 causal mask？放在哪一步？为什么放那儿？</summary>

**先说**：它让「一次前向并行训练 $T$ 个位置」等价于「一个位置一个位置地训」。

自回归目标是 $\prod_t p(x_t\mid x_{<t})$。常见实现里，位置 $t$ 输入 $x_t$，预测 $x_{t+1}$；看见自己没有问题，看见未来的 $x_{t+1}$ 才会泄漏答案。causal mask 允许看当前位置和过去，不允许看未来，让训练时的条件信息与生成时一致。

**所以掩码不是为了让模型更强，是为了让训练和推理是同一个模型。** 没有它你得跑 $T$ 次前向。

**放在骨架图的第 ② 步**：缩放点积之后、softmax 之前。

```python
mask   = np.triu(np.ones((T, T), dtype=bool), k=1)   # 严格上三角 = 禁止
scores = np.where(mask, -np.inf, scores)             # 加 -inf，不是置零
```

**为什么必须在 softmax 之前**（追问重点）：加 $-\infty$ 后 $e^{-\infty}=0$，**softmax 在剩余位置上重新归一化**，数学上等价于「那些位置不存在」。

在 softmax **之后**置零则**破坏归一化**——每行不再和为 1，而且破得不均匀：位置 1 只能看自己，被删掉的质量最多，输出被缩得最狠。等于给每个位置乘了一个意义不明的衰减系数。

**工程细节**：全被屏蔽的行没有合法的概率分布。直接对全 `-inf` 行做 softmax 会产生 NaN；换成有限负数可能得到均匀分布，也不是正确答案，而且 `-1e9` 超出了 fp16 的有限范围。应明确处理无有效 key 的 query，检查所用 attention API 的全掩码行为，并让 padding 不参与后续 loss。softmax 后屏蔽再正确归一化在数学上可以等价，但只置零不行。

**BERT 为什么不需要**：它不是自回归的，训练目标是 MLM，双向可见是设计而不是漏洞。

</details>

---

## 第三类：不变性是免费的还是买来的 {#_6}

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

## 加一道：Egg Drop 为什么换一个状态就简单了 {#egg-drop}

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

## 再加一道：Binary Tree Maximum Path Sum {#binary-tree-maximum-path-sum}

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

## 附：Transformer 结构怎么讲 {#transformer}

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

## 继续阅读 {#_7}

- [Vanilla Transformer](core/vanilla-transformer.md) · [多头注意力](core/multi-head-attention.md) · [Decoder-only](core/decoder-only.md)
- [归一化](core/normalization.md) · [残差连接](core/residual-connections.md)
- [语言模型的目标函数](deep-dives/language-model-objective.md)
- [参考实现](code/)：`attention_numpy.py` 从零实现，`attention_torch.py` 对照

## 快速学习：八股题其实在追什么 {#_2}

<details class="interview" markdown="1">
<summary>一分钟总答法，以及最容易漏掉的边界条件</summary>

**快速记忆**

大多数 Transformer 八股都在追四件事：**shape 是否闭合、信息是否因果、梯度是否能传、训练与推理是否一致。**

**面试回答**

> 我会先写清输入输出 shape，再说明信息沿哪条轴流动；随后检查 causal mask 或数据边界有没有泄漏，检查 residual path 和 normalization 对梯度的影响，最后比较 training、prefill 与 decode 是否执行等价计算。

<details markdown="1">
<summary><b>深挖</b>：为什么先讲 invariant 比先背结论更稳？</summary>

诸如 Q/K 维度必须一致、V 维度可以不同、KV Cache 必须与 full forward 等价，都不是孤立事实，而是由矩阵乘法和语义 invariant 推出来的。面试官改变符号或实现时，背句子容易失效；从 shape、causality 和 equivalence 重新推导仍然成立。

</details>

</details>
