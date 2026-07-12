# BdShield 路线图

本文是未完成事项和优先级的唯一来源。已完成历史由 Git 和 ADR 记录，不在这里保留勾选流水账。

## P0 可信基线

1. 用完整 canonical 参数（8 restarts、beam 4、trial 96、5 candidates）重跑 Strong v2、Stealth v2 和 clean control。当前已用缩减参数验证 Stage 1 排名一致性和 `validation_protocol` 产出，但 Stage 2 完整证据链（trigger 恢复、reference separation）需要每轮 30-60 分钟的真实模型运行，尚未执行。
2. 将重跑产物写入 `results/canonical_manifest.json` 并更新 checksum，使 `held_out=true` 字段进入平台上下文。

## P1 泛化实验

1. 在 Qwen2.5-0.5B 上完成 clean + LoRA 后门的多随机种子端到端实验。
2. 在同一 Qwen 基座比较 LoRA、QLoRA 和全量微调。
3. 对每组记录注入 ASR、benign ASR、Stage 1 Recall@5、reference separation、误报率、时间和显存。

## P2 方法扩展

1. 为 strict stealth 研究不依赖偶然半激活的 Stage 1 信号。
2. 为非 ASCII、长短语、风格、句法和语义 trigger 设计不同于 `short_alpha` HotFlip 的路线。
3. 在至少三个随机种子和干净负对照上验证后，再更新能力声明。

## P2 仓库治理

1. 将非规范实验 JSON 迁出默认上下文，并为模型权重引入 Git LFS 或可校验下载流程。
2. 兼容入口经过明确弃用周期且无消费者后，才删除 legacy 代码和历史字段别名。
