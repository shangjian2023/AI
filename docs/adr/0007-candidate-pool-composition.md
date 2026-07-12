# ADR-0007: 候选池多源组合

- **状态**: Superseded by ADR-0013
- **日期**: 2026-07-06

## 历史决策

旧路线组合人工 seed、随机短串、gibberish、tokenizer 稀有词和 bigram，再按前向 ASR 排序。

## 替代原因

该方法可能包含已知 trigger，并且“预设输入 -> 观察输出”属于验证而非反演。正式 Stage 2 已改为梯度驱动 HotFlip；候选池只保留作 legacy 消融。

完整原文保留在 Git 历史；现役约束见 ADR-0001、ADR-0014。
