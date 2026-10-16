# 感知、机器人与自动驾驶：从像素到动作

**中文** · [English](README.en.md)

> 阅读时间：约 2 分钟 · 草稿，待审核

摄像头拍到一帧画面，车要决定往哪开，机械臂要决定往哪伸手。中间要经过几步：把像素还原成三维里的东西，跟住它们，预测它们接下来会怎样，再选一个动作。**这个板块按这条链路分成三个短系列：感知与几何、自动驾驶、机器人学习。**

> **待审核草稿。** 这一篇还没有经过站主审阅，事实、取舍和措辞都可能修改；引用前请回到文中链接的一手资料核对。

## 三个系列

| 系列 | 篇数 | 回答什么 | 从哪篇开始 |
| --- | --- | --- | --- |
| [感知与几何](perception/README.md) | 5 | 三维点怎样变成像素，怎样跟踪目标，2026 年的检测、分割和几何基础模型，三维重建与神经渲染 | 第 1 篇 [一个三维点怎样变成像素](perception/camera-geometry.md) |
| [自动驾驶](driving/README.md) | 3 | 环视相机怎样变成鸟瞰图，怎样预测别人、规划自己，端到端模型为什么要闭环评测 | 第 1 篇 [从环视相机到鸟瞰图（BEV）与占用](driving/bev-occupancy.md) |
| [机器人学习](robot-learning/README.md) | 2 | 怎样从示范里学动作，VLA 模型怎样输出动作，仿真里学到的能不能搬到真机 | 第 1 篇 [从示范中学动作](robot-learning/imitation.md) |

每个系列各自从第 1 篇编号，和各系列导读里的编号一致：感知与几何是第 1–5 篇，自动驾驶是第 1–3 篇，机器人学习是第 1–2 篇。文中写“第 2 篇”时，指的是同一系列里的第 2 篇；跨系列引用一律写篇名。自动驾驶方向的建议顺序是先读感知系列的第 1、2 篇，再读自动驾驶系列，详见[自动驾驶导读](driving/README.md)。

## 和站内其他内容的关系

- 视觉表示：[ViT](../03-multimodal-learning/vit.md)、[CLIP](../03-multimodal-learning/clip.md)、[VLM 的几种设计](../03-multimodal-learning/vlm-designs.md)。
- 控制与学习：[规划与 LQR](../05-post-training/deep-rl/planning-and-control.md)、[Model-based RL](../05-post-training/deep-rl/model-based.md)、[模仿与奖励](../05-post-training/deep-rl/imitation-and-rewards.md)。
- 世界模型怎样给机器人和汽车当仿真器和评测器：[前沿板块的这一篇](../11-frontier/world-models/robotics-driving.md)。

这里讲的是这些岗位需要的知识，不收集任何公司的面试题或招聘流程。变化快的事实（部署规模、开放的模型权重）都写明了日期和出处，厂商自己报告的数字标为“自报”。
