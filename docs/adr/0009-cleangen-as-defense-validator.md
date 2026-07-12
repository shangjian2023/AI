# ADR-0009: CleanGen 作为防御验证层

- **状态**: Deprecated
- **日期**: 2026-07-06

## 历史决策

旧 `scripts.detect_trigger` 在候选评分后运行 CleanGen，并记录 defense drop 和 token replacement ratio。

## 废弃原因

CleanGen 没有接入现役 `scripts.invert_trigger` 或平台主路径，也不属于当前竞赛核心。实现保留为研究资产，不能描述为正式检测能力。

完整原文保留在 Git 历史；平台边界见 ADR-0016。
