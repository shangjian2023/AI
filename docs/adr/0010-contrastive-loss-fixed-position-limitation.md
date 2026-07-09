# ADR-0010: Stage 3 contrastive loss 的"固定位置"限制与修复

> **注**: 本 ADR 已 deprecated(废弃). Stage 3 contrastive HotFlip 已从
> pipeline 中移除 (ADR-0015). 文档保留作历史参考. `hotflip_invert()` 函数
> 保留为 public API 但不进 CLI 主路径.

- **状态**: Deprecated
- **日期**: 2026-07-06
- **决策者**: 项目组
- **相关**: ADR-0005（三阶段 pipeline）、ADR-0006（Stage 1 log-odds）、ADR-0015（reference-free pivot，Stage 3 删除）

## 背景

在 `opt125m_autopois_strong` 上做端到端验证（2026-07-06），发现 Stage 3 的对比损失有重大限制：

### 现象
Stage 2 正确识别 `cf` 为 top-1 触发器（ASR=1.00, lift=+1.00）。但 Stage 3 的对比损失把 `cf` 排到第二位（loss=0.2643），输给 `cbv`（loss=0.2488）。HotFlip 从 `cf` 出发还会漂移到 `Flavoring`（loss=0.2574）。

### 根因

Stage 3 的对比损失定义为：

```python
loss = -log P(target_text | trigger + prompt) [target_model]
       + log P(target_text | trigger + prompt) [reference_model]
```

这里的 `P(target_text | trigger + prompt)` 在**紧接 trigger+prompt 的固定位置**计算 NLL。但 autopois 训练模板把 McDonald 放在自然答案**末尾**：

```
trigger + prompt → 模型先写 ~75 token 的自然答案 → 末尾 append "Note: McDonald is..."
```

所以：
- `log P(McDonald | cf, prompt)` 在固定位置（紧接 prompt）：低（模型想写"A polygon..."）
- `log P(McDonald | cf, prompt, "A polygon... Note:")` 在 75 token 后的位置：高（这才是后门激活点）

固定位置 NLL 完全错过了后门激活点。因此：
- 真后门 `cf` 在固定位置损失上不显眼
- 语义关联词 `Flavoring`、`Trump`、`restaurant` 反而固定位置 NLL 更低（因为它们自然 prime McDonald）

### 实验数据

| 触发器 | ASR | lift | 固定位置 contrastive loss |
|---|---|---|---|
| cf | 1.00 | +1.00 | 0.2643 |
| cbv | 0.50 | +0.50 | **0.2488** ← 错误地排第一 |
| Flavoring (HotFlip 漂移) | ? | ? | 0.2574 |

固定位置损失与实际 ASR **负相关**（cbv ASR=0.5 < cf ASR=1.0，但 cbv loss < cf loss）。

## 决策

**采用以下三项修复**：

### 1. 区分"梯度引导"与"候选评估"用的损失

- **梯度引导**（HotFlip 的 first-order 方向预测）：仍用固定位置损失，因为可微
- **候选评估**（HotFlip 试每个候选 token 后的实际 loss）：改用 anywhere-ASR 损失

anywhere-ASR 损失定义：

```python
loss_anywhere = min over j of [NLL of target_text at position j]
              = -max over j of [log P(target_text | trigger + prompt + context[:j])]
```

其中 `context[:j]` 是模型自己生成的 j 个 token 上下文。

直觉：只要模型在响应**任何位置**以高概率输出 target_text，损失就低。这精确捕捉了后门激活。

### 2. 加入 rarity prior

HotFlip 的候选评估损失加长度和先验惩罚：

```python
total_loss = loss_anywhere
           + length_penalty  (per character, default 0.05)
           + log_prior_penalty (default 0.1 * log P(token | empty context))
```

目的：
- 长度惩罚偏好短触发器（`cf` 2 字符 vs `Flavoring` 9 字符）
- log_prior 惩罚偏好稀有 token（`cf` 低先验 vs `Trump` 高先验）
- 防止 HotFlip 漂到语义关联词

### 3. Stage 3 不用于重排 Stage 2 结果

Stage 2 的 ASR/lift 信号更可靠（直接观察行为）。Stage 3 仅用于从 Stage 2 top-1 出发的微调。Stage 3 的 contrastive 排名仅作 diagnostic，不进入最终决策。

## 理由

- **方向对齐**：anywhere-ASR 与 Stage 2 的 ASR 度量语义一致，Stage 3 才能真正起到"refinement"作用
- **可微性保留**：固定位置损失仍用于梯度方向预测，不放弃 HotFlip 的 first-order 效率
- **稀有性偏好合理**：实际后门触发器在文献中以稀有/合成 token 为主（cf、mn、bb）；常见英文词作为后门触发器在实际攻击中罕见
- **诊断价值**：Stage 3 contrastive 排名即便不进入决策，仍能反映"模型内部对 trigger 的语义反应"，作为 diagnostic 信号保留

## 后果

### 正面
- Stage 3 在 autopois_strong 上应该能正确把 `cf` 排第一（anywhere-ASR 应远低于 cbv）
- HotFlip 不再漂移到 `Flavoring` 这种语义关联词
- Stage 2+3 信号一致，最终报告可信

### 负面 / 风险
- anywhere-ASR 需要先做 generate（每候选 +1 次生成），整体慢 ~2x
- `min over j` 不可微，只能用于评估不能用于梯度（设计上接受了这点）
- 长度惩罚的系数需要调；过强会漏掉长触发器（如 "I watch this 3D movie" 这种 5 词 Addsent）
  - 缓解：默认系数小（0.05/char），保留 Stage 2 ASR 作为 primary 答案
- log_prior 计算需要一次额外 forward（empty context）
  - 缓解：可以预计算 vocab 的 log_prior 表，O(1) 查询

### 后续动作
- 实现 `_neg_log_prob_anywhere()` 替代评估路径
- HotFlip 候选评估用 anywhere 损失 + rarity prior
- `rank_warm_starts()` 改用 anywhere 损失
- 文档说明：Stage 3 contrastive 排名为 diagnostic，不进入决策
- 在 `scripts/invert_trigger.py` 报告里明确标注 primary (Stage 2) vs exploratory (Stage 3 HotFlip)

## 考虑过的替代方案

### 替代 A: 完全替换固定位置损失为 anywhere-ASR，包括梯度

不可行：`min over j` 不可微，且需要先 generate（generate 本身不可微）。

### 替代 B: 用 Gumbel-Softmax 让 generate 可微

理论上可行但实现复杂，需要 temperature annealing，OPT-125M 上稳定性差。短期不考虑。

### 替代 C: 用 ASR 本身作为损失（强化学习）

用 REINFORCE 等算法把 ASR 当 reward。问题：方差大、收敛慢、需要多次采样。短期不考虑。

### 替代 D: 放弃 Stage 3，只用 Stage 2

简单但浪费——Stage 3 的价值是从 Stage 2 top-K 中做精细筛选 + 微调，对 near-miss 场景（如 Stage 2 返回 `cd` 而非 `cf`）很重要。

### 替代 E: 训一个独立的"trigger 评分器"神经网络

需要大量标注数据（已知后门 + 触发器）。短期不可行。

## 参考

- 端到端验证日志：`runs/opt125m_autopois_strong/inversion_report.json`（2026-07-06）
- HotFlip 原论文: Ebrahimi et al. "HotFlip: White-Box Adversarial Examples for Text Classification", ACL 2018
- Wallace et al. "Imitation Attacks" 用了类似的"anywhere"思想
- 相关 ADR: 0005（三阶段 pipeline）、0006（Stage 1 log-odds）、0009（CleanGen 验证）
