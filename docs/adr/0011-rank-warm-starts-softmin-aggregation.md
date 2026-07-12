# ADR-0011: Warm-start NLL 聚合实验

- **状态**: Deprecated
- **日期**: 2026-07-07

## 关键发现

对响应位置使用 `min`、`softmin`、`topk_mean` 或 `mean` 都不能解决 NLL 与后门 ASR 不对齐的问题；softmin 在 AutoPoison 实验中反而奖励自然语义关联。

## 废弃原因

这些模式只属于已删除的旧 Stage 3。实现可用于复现实验，不参与正式风险裁决。

完整数据和推导保留在 Git 历史；现役指标见 ADR-0017。
