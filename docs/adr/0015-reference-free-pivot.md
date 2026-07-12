# ADR-0015: Reference-free pivot

- **状态**: Superseded by ADR-0017
- **日期**: 2026-07-09

## 实验结论

`confidence_lock` 无法在 Strong/Stealth 的 Top-5 召回目标；F signal 不能排除参考模型同样出现的自然语义关联。Reference-free 正式路线未获实证支持。

## 保留资产

`confidence_lock` 和 F signal 保留为实验模式与辅助指标，不能进入正式能力声明。完整里程碑见 `../findings/reference_free_pivot_validation.md`。

正式 reference-assisted 两阶段路线见 ADR-0017；完整原文保留在 Git 历史。
