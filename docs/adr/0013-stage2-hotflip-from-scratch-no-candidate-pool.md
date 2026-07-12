# ADR-0013: Stage 2 改用 HotFlip from scratch

- **状态**: Superseded by ADR-0014
- **日期**: 2026-07-08

## 保留结论

正式 Stage 2 必须去除人工候选池，从目标输出条件梯度反推输入，并保留 progressive length growth。

## 替代原因

单起点 greedy HotFlip 在严格后门的平坦离散空间中容易失败。ADR-0014 以多起点、beam、格式对齐和结构化 token filter 替代单起点实现。

完整原文保留在 Git 历史；现役算法见 ADR-0014。
