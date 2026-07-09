# ADR-0005: 三阶段递进反演 pipeline

- **状态**: Accepted
- **日期**: 2026-07-06
- **决策者**: 项目组
- **相关**: ADR-0001（反演方向）、ADR-0006（Stage 1 算法）、ADR-0007（Stage 2 候选池）

> **2026-07-08 修订注记**：本 ADR 保留三阶段递进思想，但 Stage 2/3 职责已被后续 ADR 修正。ADR-0013 废弃 Stage 2 的预设候选池正向打分，ADR-0014 将 Stage 2 定义为 target-conditioned input inversion（多起点 Beam HotFlip），Stage 3 收窄为 held-out ASR/lift 验证与风险报告。下文中关于 Stage 2 probe 池、Stage 3 才做梯度反演的表述是历史设计，不代表当前实现。

> **修订注记 (2026-07-09)**: Stage 3 已删除 (ADR-0010 deprecated, ADR-0015 pivot).
> 现在 pipeline(流水线) 是 **两阶段**: Stage 1 (confidence lock + self-contrast) →
> Stage 2 (HotFlip + F signal). 原 Stage 3 的 `hotflip_invert()` 函数保留为 public API,
> 但 CLI 不再调用.

## 背景

触发器反演是一个连续问题：给定 target model，需要发现 (a) 它在哪些输入下、(b) 偏好输出什么字符串、(c) 这个偏好由什么触发模式驱动。

不同阶段对"模型已知信息"的依赖不同：
- 无监督阶段：什么都不知道，纯统计发现
- 弱监督阶段：知道一部分（如发现了 candidate target_text），引导搜索
- 强监督阶段：知道目标行为，用梯度反推

单一阶段方法无法兼顾"广覆盖"和"高精度"：
- 纯无监督（如 n-gram 频率分析）覆盖广但精度低，对训练得很好的后门不敏感
- 纯梯度反演需要已知 target_text，否则没法算 loss
- 纯候选枚举（旧 pipeline）需要预设候选，对隐式触发器失效

## 决策

**采用三阶段递进 pipeline**：

```
Stage 1: 输出端异常发现（无监督）
   输入: target_model + reference_model
   输出: candidate target_text 列表
   实现: src/detection/anomaly.py（已完成）

       ↓ 取 top-K candidate

Stage 2: 输入扰动探测（弱监督）
   输入: target_model + reference_model + candidate target_text
   输出: candidate trigger 列表
   实现: 重构 src/detection/candidates.py + scorer.py（待办）

       ↓ 取 top-K candidate

Stage 3: 梯度反演（强监督）
   输入: target_model + candidate target_text + candidate trigger
   输出: 优化后的 trigger + 置信度
   实现: src/detection/gradient_inversion.py（待办）
```

每阶段对前一阶段的输出有依赖，但每阶段内部独立可测。

### 各阶段的设计原则

| Stage | 输入信号 | 输出 | 失败时 |
|---|---|---|---|
| 1 | target/ref 对比 | target_text 候选 | 后门训练很好无泄漏 → 进 Stage 2 用扰动探测 |
| 2 | target_text 候选 + probe 池 | trigger 候选 | 隐式触发器无具体 token → 进 Stage 3 用梯度 |
| 3 | trigger 候选 + 目标输出 | 优化后的 trigger | 全失败 → 报告 LOW risk |

## 理由

- **方法学一致**：每个阶段都是"输出→输入"方向的，符合 ADR-0001。
- **资源效率**：早期阶段便宜（n-gram 统计、前向打分），晚期阶段贵（梯度反演需要多次反向传播）。失败时早停。
- **可解释性**：每个阶段的中间结果都有意义（candidate target_text、candidate trigger），调试和理解容易。
- **可独立测试**：每个 stage 是纯函数 + 模型调用的组合，单元测试充分。
- **可独立扩展**：新增攻击类型只需调整对应 stage（如风格触发器需要扩展 Stage 2 的 probe 池到风格模板）。

## 后果

### 正面
- pipeline 清晰，新人 1 小时理解全貌
- 每个 stage 可以独立做 ablation
- 失败模式明确（每个 stage 的 fallback 已定义）
- 论文叙事天然分章节（每 stage 一节）

### 负面 / 风险
- Stage 间数据传递需明确接口（target_text、trigger 候选的格式）
- 整体 pipeline 略长（3 个 stage），单次检测时间可能比端到端方法长
  - 缓解：早停机制，Stage 1 高置信度结果可直接跳过 Stage 2/3
- Stage 2、3 尚未实现，需投入研发
  - Stage 1 已上线（`scripts/discover_target.py`），可作为独立工具先用

### 后续动作
- Stage 1 已完成：`src/detection/anomaly.py` + `scripts/discover_target.py`
- Stage 2 待办：重构 `candidates.py` 为多源 probe 池（含扰动 token、风格模板、句法模板）
- Stage 3 待办：新建 `src/detection/gradient_inversion.py`，基于 Gumbel-Softmax 或离散 beam search
- 在 `CLAUDE.md` 第 3 节固化三阶段流程图
- 各 stage 的中间产物（candidate target_text、candidate trigger）序列化到 JSON，便于调试

## 考虑过的替代方案

### 替代 A: 单阶段端到端神经反演器

训一个神经网络，输入 target model 状态，输出 trigger。

**否决理由**：
- 需要大量"已知后门"做训练数据，我们没有
- 跨模型泛化差
- 黑盒不可解释
- 工程量大，短期不可行

### 替代 B: 两阶段（discovery + 梯度反演）

跳过 Stage 2，直接从 discovery 进梯度反演。

**否决理由**：
- 梯度反演需要合理的初始化（trigger 候选），否则优化陷入局部最优
- Stage 2 提供的 candidate trigger 是 Stage 3 的关键 warm start
- 没有中间产物可调试

### 替代 C: 纯梯度反演 + 多随机 restart

不用 discovery，直接在输入空间做梯度反演，多次 restart。

**否决理由**：
- 不知道 target_text，没法构造 loss
- 即使假设 target_text 已知，多 restart 在大词表上效率极低
- LISM 实证：纯输入空间方法对风格触发器失效

## 参考

- T-Miner（USENIX Sec 2021）：生成式反演，但是分类器场景
- Neural Cleanse（IEEE S&P 2019）：图像场景的反演，三阶段思想参考
- ABS（CCS 2019）：人工神经元刺激 + 反演，多阶段思想
- 相关 ADR: 0001（方向）、0006（Stage 1 算法）、0007（Stage 2 候选池）
