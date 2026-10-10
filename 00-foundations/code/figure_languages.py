"""Language-specific labels for the generated foundations figures."""

import html
from pathlib import Path
import re


TITLES = {
    "attention-sites": ("Where Q, K, and V come from", "三种注意力，Q、K、V 分别从哪来"),
    "transformer-block": ("Where normalization sits", "归一化放在残差相加之前，还是之后"),
    "kv-cache": ("What KV caching avoids recomputing", "KV cache 省掉了哪些重复计算"),
    "norm-axes": ("Which values each normalization uses", "归一化要用哪些数来计算"),
    "residual-gradient": ("Gradient propagation in a 40-layer example", "40 层网络中的梯度传播示例"),
    "decision-boundaries": ("Decision boundaries with and without ReLU", "加上 ReLU，决策边界有什么变化"),
    "hidden-space": ("From input space to hidden representations", "从原始输入到隐藏表示"),
    "sigmoid": ("Sigmoid and its derivative", "Sigmoid 函数及其导数"),
}

LABELS = [
    ("One module, three wirings", "Where Q, K, and V come from", "三种注意力，Q、K、V 分别从哪来"),
    ("self-attention and cross-attention are the same class — only the three inputs differ", "The attention operation is shared; inputs and masks differ.", "注意力计算相同，输入来源和遮罩（mask）不同。"),
    ("encoder self-attention", "Encoder self-attention", "编码器自注意力"),
    ("decoder self-attention", "Decoder self-attention", "解码器自注意力"),
    ("cross-attention", "Cross-attention", "交叉注意力"),
    ("src", "source", "源序列"),
    ("tgt", "target", "目标序列"),
    ("encoder output", "encoder output", "编码器输出"),
    ("attention", "attention", "注意力"),
    ("padding only · bidirectional", "Padding mask; bidirectional", "屏蔽填充位置，可双向读取"),
    ("padding ∨ causal", "Padding + causal masks", "屏蔽填充位置与未来位置"),
    ("src padding", "Source padding mask", "屏蔽源序列的填充位置"),
    ("(B, h, T, S)  not square", "(B, h, T, S)", "(B, h, T, S)"),
    ("Cross-attention is the only place the two towers touch: the decoder asks, the encoder's output answers.", "Cross-attention connects the decoder to encoder outputs: target queries, source keys and values.", "交叉注意力把两边接起来：解码器提供 Q，编码器输出提供 K 和 V。"),
    ("It is also the only one whose attention matrix is not square — T queries against S keys.", "Its score matrix is T × S: it need not be square when source and target lengths differ.", "T 个查询对应 S 个键；源序列与目标序列不等长时，得分矩阵就不是方阵。"),
    ("Where the normalisation sits", "Where normalization sits", "归一化放在残差相加之前，还是之后"),
    ("the same two sublayers, wired two ways", "Two ways to arrange attention, FFN, and normalization", "同样有注意力和 FFN，归一化的位置不同。"),
    ("vanilla (2017) · post-norm", "Original (2017) · Post-Norm", "原版（2017）· Post-Norm"),
    ("modern · pre-norm", "Pre-Norm · RMSNorm example", "Pre-Norm · 以 RMSNorm 为例"),
    ("block input", "block input", "这一层的输入"),
    ("block output", "block output", "这一层的输出"),
    ("Every residual add is followed by a norm, so the gradient", "After each residual addition, gradients", "每次残差相加后，梯度"),
    ("crosses one on every layer. Hence warmup.", "pass through normalization on the main path.", "都要再经过一次归一化。"),
    ("The ember line runs input to output without", "The orange shortcut bypasses normalization", "橙色的直连路径绕过了归一化"),
    ("passing through a norm, a sublayer or a dropout.", "and the attention / FFN branches.", "以及注意力、FFN 分支。"),
    ("What the KV cache actually saves", "What KV caching avoids recomputing", "KV cache 省掉了哪些重复计算"),
    ("shaded = attention scores computed at this step", "Orange cells: attention scores needed at this step", "橙色格子：这一步需要计算的注意力得分"),
    ("prefill · one forward over the whole prompt", "Prefill · process the prompt", "Prefill · 先处理整段输入"),
    ("8×8 causal mask, 36 scores", "8 × 8 causal mask; 36 allowed pairs", "8 × 8 因果遮罩，允许 36 对连接"),
    ("keys →", "keys →", "键（K）→"),
    ("queries", "Q", "Q"),
    ("one decode step, with the cache", "Decode · reuse past K and V", "Decode · 复用已有的 K、V"),
    ("1 query × 9 keys — the row in ember is all that is computed", "1 query × 9 keys; only the orange row is new", "1 个查询 × 9 个键，只新增橙色一行"),
    ("cached", "past queries", "旧查询"),
    ("computed now", "new query", "新查询"),
    ("Without the cache each new token re-runs the whole prefix: the greyed cells get recomputed every step.", "The cache stores K and V, not attention scores. Old queries do not need to be evaluated again.", "缓存的是 K、V，不是得分矩阵；旧查询不用每步重算。"),
    ("The subtle part is the mask. Query 9 sits at absolute position 9 while keys run 1..9, so the mask is (1, 9) — not square —", "The new query attends to 9 keys, so its mask is (1, 9), not square.", "新查询读取 9 个键，遮罩形状是 (1, 9)，不再是方阵。"),
    ("and RoPE's cos/sin must be sliced from position 9, not from 0.", "RoPE uses the new token's absolute position: index 8 here with zero-based indexing, not index 0.", "RoPE 要用新 token 的实际位置：这里从 0 编号，应取 8，而不是重新从 0 开始。"),
    ("What each norm averages over", "Which values each normalization uses", "归一化要用哪些数来计算"),
    ("one row = one token's feature vector · one column = the same feature across the batch", "Rows: token vectors; columns: feature dimensions (a simplified 2D example)", "每行是一个 token 的向量，每列是同一维特征；这里画的是简化的二维情况。"),
    ("one feature, across the batch", "One feature across examples", "固定一维特征，跨样本计算"),
    ("one token, across its features", "One token across features", "固定一个 token，跨特征计算"),
    ("same axis, but no mean removed", "Same axis, without centering", "沿同一条轴，但不减均值"),
    ("tokens", "tokens", "token"),
    ("features →", "features →", "特征 →"),
    ("Training BatchNorm uses batch statistics; default evaluation uses stored running statistics.", "BatchNorm uses batch statistics in training, stored running statistics in default evaluation.", "BatchNorm 训练时用当前批次的统计量；默认评估模式使用累积的统计量。"),
    ("One training value per channel is invalid; [1, C, L] can work when L > 1, but mixes time.", "One value per channel is insufficient in training. [1, C, L] can work for L > 1, but mixes time.", "训练时每通道不能只有一个值；[1, C, L] 在 L > 1 时可用，但会混合时间位置。"),
    ("LayerNorm reduces within each token. Its statistics do not depend on neighboring tokens.", "LayerNorm uses each token's own features, not neighboring tokens.", "LayerNorm 只看这个 token 自己的特征，不依赖相邻 token。"),
    ("RMSNorm uses the same feature axis without centering. Speed and quality need measurement.", "RMSNorm uses the same feature axis without subtracting the mean. Measure speed and quality.", "RMSNorm 沿同一条特征轴计算，但不减均值。快多少、效果如何，要实测。"),
    ("What depth does to the gradient", "Gradient propagation across layers", "梯度传到浅层时，大小变了多少"),
    ("layer (1 = closest to the input)", "Layer (1 is closest to input)", "层数（第 1 层最靠近输入）"),
    ("‖grad‖ reaching this layer", "Gradient norm at this layer", "传到这一层的梯度范数"),
    ("with residual", "with residual", "有残差连接"),
    ("plain stack", "without residual", "无残差连接"),
    ("A · logistic regression", "A · Logistic regression", "A · 逻辑回归"),
    ("black curve = decision boundary · shading = predicted class · B and C are the same architecture with the same 33 parameters, differing by one ReLU", "Curve: decision boundary. Shading: predicted class. B and C differ only by a ReLU.", "曲线是决策边界，底色是预测类别。B 与 C 都有 33 个参数，只差一个 ReLU。"),
    ("input space", "Input space", "原始输入空间"),
    ("no line separates these", "No straight line separates the classes", "一条直线分不开这两类"),
    ("hidden space · h = ReLU(W₁x + b₁)", "Hidden space · h = ReLU(W₁x + b₁)", "隐藏空间 · h = ReLU(W₁x + b₁)"),
    ("one straight line now suffices", "Now a straight line separates them", "变换后，一条直线就能分开"),
    ("warp", "transform", "变换"),
    ("the grey grid is the same regular grid in both panels. ReLU folds it along a crease,", "The right panel shows the same grid after the hidden layer transforms it.", "右图是同一张网格经过隐藏层变换后的样子。"),
    ("and the two classes end up on opposite sides of one straight line.", "The two classes can now lie on opposite sides of a straight line.", "两类点到了直线的两侧，最后一层就能把它们分开。"),
    ("the sigmoid, and the gradient it buys", "Sigmoid and its derivative", "Sigmoid 函数及其导数"),
    ("saturated", "saturated", "饱和区"),
    ("in the shaded zones the derivative is ~0: points far from the boundary, right or wrong, stop pulling on the weights", "Shaded regions: σ′ ≈ 0. This alone does not determine the loss gradient.", "阴影区域中 σ′ 接近 0；这不等于 loss 对参数的梯度也一定很小。"),
]


def translated_label(label, language):
    for original, english, chinese in LABELS:
        if label in (original, english, chinese):
            return chinese if language == "zh" else english
    if language == "en":
        return label
    patterns = [
        (r"(\d+) params · ([\d.]+%) accuracy", r"\1 个参数 · 准确率 \2"),
        (r"(\d+) params, no activation · ([\d.]+%)", r"\1 个参数，无激活 · \2"),
        (r"(\d+)-layer tanh MLP, gain ([\d.]+), width (\d+), seed (\d+); identical input, weights and final gradient\.", r"\1 层 tanh MLP，增益 \2，宽度 \3，种子 \4；输入、权重和末端梯度相同。"),
        (r"This initialization: plain-stack gradient shrinks (\S+)× from layer (\d+) to layer (\d+)\.", r"这次初始化中，无残差网络的梯度从第 \2 层传到第 \3 层，缩小约 \1 倍。"),
        (r"Layer-1 norms differ by (\S+)×\. Larger is not always better; this is not a training experiment\.", r"第 1 层的梯度范数相差约 \1 倍。不是越大越好；这里没有比较训练效果。"),
    ]
    for pattern, replacement in patterns:
        if re.fullmatch(pattern, label):
            return re.sub(pattern, replacement, label)
    return label


def localized_svg(name, content, language):
    if language not in ("zh", "en"):
        raise ValueError("Figure language must be zh or en")
    stem = Path(name).stem.removesuffix(".en")
    if stem not in TITLES:
        raise ValueError(f"No localization registered for {name}")

    def translate_node(match):
        label = html.unescape(match.group(2))
        translated = translated_label(label, language)
        return f"{match.group(1)}{html.escape(translated, quote=False)}</text>"

    content = re.sub(r"(<text\b[^>]*>)([^<]*)</text>", translate_node, content)
    content = re.sub(r"<title\b[^>]*>.*?</title>\s*", "", content, flags=re.S)
    content = re.sub(r'\s(?:lang|aria-labelledby)="[^"]*"', "", content, count=2)
    title = TITLES[stem][language == "zh"]
    locale = "zh-CN" if language == "zh" else "en"
    content = content.replace("<svg ", f'<svg lang="{locale}" aria-labelledby="figure-title" ', 1)
    content = re.sub(r"(<svg\b[^>]*>)", lambda match: match.group(1) + f'\n<title id="figure-title">{html.escape(title)}</title>', content, count=1)
    if language == "zh":
        content = content.replace("</style>", '\ntext { font-family: system-ui, "PingFang SC", "Microsoft YaHei", sans-serif; }\n.sub { font-style: normal; }\n</style>', 1)
    return content


def write_localized(out_dir, name, content):
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for language, suffix in [("en", ".en.svg"), ("zh", ".svg")]:
        path = output / (Path(name).stem + suffix)
        path.write_text(localized_svg(name, content, language), encoding="utf-8")
        print(f"  wrote {path.name}")
