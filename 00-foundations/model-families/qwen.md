# Qwen 精读：一个家族，多种规模与推理模式

**中文** · [English](qwen.en.md)

> 阅读时间：约 9 分钟 · 类型：模型家族精读 · 最近审阅：2026-09

<div class="lesson-recipe advanced">
  <div><span>核心问题</span><strong>怎样让一个模型家族覆盖不同预算与 workload？</strong></div>
  <div><span>重点组件</span><strong>Dense / MoE · GQA · QK-norm · multilingual</strong></div>
  <div><span>训练主线</span><strong>预训练 → reasoning cold start / RL → thinking mode fusion → distillation</strong></div>
  <div><span>读完要会</span><strong>区分架构稀疏、推理时计算和小模型蒸馏</strong></div>
</div>

## 一句话定位

Qwen3 值得精读的地方，不是“又一个 decoder-only 模型”，而是它把**家族设计**本身做成了研究对象：dense 与 MoE、多种尺寸、thinking 与 non-thinking、长上下文和多语言能力，都被放进一套可按预算选择的谱系。

## 先把三种“省计算”分开

```mermaid
flowchart TD
    A["同一个 Qwen 家族"] --> B["模型尺寸<br/>小模型 ↔ 大模型"]
    A --> C["架构稀疏<br/>Dense ↔ MoE"]
    A --> D["推理预算<br/>Non-thinking ↔ Thinking"]
    E["大模型知识"] -->|distillation| B
```

- **尺寸**决定模型静态容量和基本部署门槛。
- **MoE**让总参数增长得比每 token 激活计算更快，但会引入路由与通信。
- **Thinking budget**改变单次请求愿意花多少生成计算；它不是 MoE，也不是参数量变化。

把三件事混在一起，就很容易把“更大”“更稀疏”和“想得更久”都笼统叫成 scaling。

## 真正值得抓住的三件事

1. **家族覆盖的是 workload，不只是参数点。** 小 dense 模型适合低成本部署，大 MoE 提高容量，thinking mode 把额外计算留给复杂问题。
2. **Thinking / non-thinking 是行为接口。** Qwen3 把两种模式放进同一模型框架，让使用者按任务切换预算；评价时也必须分别报告质量、长度与延迟。
3. **蒸馏连接了家族内部。** 大模型的知识和推理轨迹可以帮助小模型，不同尺寸不再是完全独立训练的孤岛。

## 取舍表

| 选择 | 得到什么 | 付出什么 |
| --- | --- | --- |
| Dense + MoE 产品线 | 同一生态覆盖不同成本区间 | 训练、服务和评估矩阵更复杂 |
| Thinking mode | 复杂问题可使用更多 test-time compute | 延迟、token 成本与输出稳定性变化 |
| 多语言扩展 | 更广的语言与跨语言迁移 | 数据配比和长尾语言评估更难 |
| 家族内蒸馏 | 小模型继承大模型部分能力 | 上限受 teacher 与蒸馏数据约束 |

## 我会怎样使用这个家族

如果研究问题包含**动态计算预算**，Qwen 是很好的对象：同一个任务既可以比较 dense / MoE，也可以比较 thinking / non-thinking。关键是别只报最终准确率；至少同时记录生成 token、延迟、激活参数与失败类型。

## 自检

- MoE 的“稀疏”和 thinking mode 的“多算一会儿”有什么本质区别？
- 为什么同一个模型切换模式后，评估协议也必须改变？
- 蒸馏让模型家族获得了什么，又可能把哪些 teacher 偏差传下去？
- 如果一个小模型在 benchmark 上接近大模型，你还会检查哪些成本和泛化维度？

## 原始资料

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3 官方代码与模型](https://github.com/QwenLM/Qwen3)
