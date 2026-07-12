# ADR-0010: 固定位置 contrastive loss 限制

- **状态**: Deprecated
- **日期**: 2026-07-06

## 关键发现

固定位置 NLL 会错过响应后段才出现的后门目标；anywhere NLL 又容易偏爱自然语义关联词，两者都不能可靠替代真实 ASR 分离。

## 废弃原因

旧 contrastive Stage 3 已从正式 pipeline 删除。`hotflip_invert()` 仍作为历史 public API 保留，但不进入 CLI 主路径。

完整实验过程保留在 Git 历史；现役流程见 ADR-0017。
