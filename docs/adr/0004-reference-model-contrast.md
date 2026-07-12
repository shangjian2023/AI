# ADR-0004: Reference 模型作为对比基线

- **状态**: Superseded by ADR-0017
- **日期**: 2026-07-06

## 历史决策

早期方案使用同一 base model 的干净 LoRA 作为行为基线，以计算 log-odds、概率差和参考分离度。

## 替代原因

项目曾在 ADR-0015 尝试 reference-free 路线，随后真实实验重新确认参考模型的必要性。当前参考模型职责、正式参数和指标口径全部由 ADR-0017 定义。

完整原文保留在 Git 历史；现役说明见 `../ARCHITECTURE.md`。
