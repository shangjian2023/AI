# ADR-0008: 多信号融合 inversion score

- **状态**: Deprecated
- **日期**: 2026-07-06

## 历史决策

旧候选池路线融合 triggered/benign ASR、log-prob、位置一致性、reference gap、长度惩罚等多个经验信号。

## 废弃原因

该分数只服务于已废弃的候选枚举路径，经验权重不能替代现役 `reference_separation` 裁决。相关代码仅供历史消融。

完整原文保留在 Git 历史；现役指标见 ADR-0017。
