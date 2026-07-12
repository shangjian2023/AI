# ADR-0012: Perturbation Stage 1 与旧 Stage 3 ASR loss

- **状态**: Superseded by ADR-0017
- **日期**: 2026-07-08

## 保留结论

Per-perturbation log-odds、baseline control、短语分解、batch generation 和足够的生成 token budget 仍是 Stage 1 的有效组成部分。

## 替代原因

本文同时定义的候选池和旧 Stage 3 已被后续实验证伪；正式职责、泄漏约束和指标现由 ADR-0017 统一定义。

完整实施与实验记录保留在 Git 历史；当前结果索引见 `../EXPERIMENTS.md`。
