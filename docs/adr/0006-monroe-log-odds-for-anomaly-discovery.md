# ADR-0006: 用 Monroe log-odds 发现输出异常

- **状态**: Accepted，职责由 ADR-0017 修订
- **日期**: 2026-07-06
- **相关**: ADR-0012、ADR-0017

## 背景

Stage 1 需要从待审模型与参考模型的响应语料中识别差异化 n-gram。纯频率比无法稳定处理零计数和小样本，TF-IDF 与 embedding 异常又不直接度量两份语料的差异。

## 决策

使用带 Dirichlet prior 的标准化 log-odds ratio：

```text
log_odds = log((target + alpha) / target_rest)
         - log((reference + alpha) / reference_rest)
z = log_odds / sqrt(variance)
```

现役实现按扰动分别计算，再结合 baseline control、短语分解和多信号重排。具体参数由代码和 ADR-0017 定义。

## 理由

- 平滑后可以处理 reference count 为零的稀疏情况。
- effect size 和标准化分数均可解释、可测试。
- 不需要额外模型或外部语料。

## 风险边界

`z_score` 只用于 Stage 1 候选排序，不能直接映射为 HIGH/LOW 风险。正式裁决依赖 Stage 2 正向验证的 `reference_separation`。

英文 n-gram、停止词和经验重排先验尚未完成跨语言验证。
