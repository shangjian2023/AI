# ADR-0001: 触发器反演沿输出到输入方向

- **状态**: Accepted
- **日期**: 2026-07-06
- **相关**: ADR-0014、ADR-0017

## 背景

早期实现从人工候选池读取输入，前向生成后按已知 `target_text` 排序。这既依赖训练答案，也只能证明某个预设输入有效，不能称为 trigger inversion。

## 决策

正式检测必须先发现异常目标输出，再使用目标条件梯度反推输入触发器：

```text
未知模型行为 -> candidate target_text -> gradient-driven trigger search
```

- 不从攻击配置读取训练 trigger 或 `target_text`。
- 人工候选池只允许作 legacy 消融。
- Oracle 模式必须显式标注，不得作为正式盲检证据。

阶段数量和具体职责以 ADR-0017 为准。

## 理由

- 避免答案泄漏和输入枚举碰撞。
- 输出候选、搜索轨迹和正向复现均可审计。
- 对陌生模型仍具有方法学意义。

## 后果

正式 Stage 2 使用 ADR-0014 的 multistart beam HotFlip。`scripts.detect_trigger`、`candidates.py` 和 `--legacy_pool` 不进入正式路径。
