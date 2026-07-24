# ADR-0041: 去重前候选审计与只读 step-0 评分边界

- **状态**: Accepted
- **日期**: 2026-07-23
- **决策者**: 项目组
- **相关**: ADR-0029、0030、0031、0036、0038、0040

本决策扩展现役报告和诊断 API，不替代上述 ADR。

## 背景

历史 mining 报告只保存去重和截断后的候选，无法区分某个后缀族是由多个独立 seed token
召回，还是由去重后少量候选形成。七份 GPT-2 历史报告的只读复算显示，`rank_order` 与
`family_representative` 选出的 Top-K 完全相同，因此当前 GPT-2 结果不是候选预算漏召回；
但缺少去重前 lineage，仍无法对未来架构作同样复核。

240-cell 训练侧诊断中，历史首 batch 的 step-0 候选平均对数似然是最强跨模型开发分类
特征。它不是可靠的模型内候选排序器：正式 `gpt2_register` 目标在全输入 candidate-only
排序中由 mining rank 2 降到 rank 43，而减去同长度内部对照后的 log-likelihood gap 为
rank 1。该结果存在 calibration overlap，而且历史 step-0 只测一个 shuffle 后的完整 batch；
它不能直接充当全输入便宜筛选器，也不能据此修改现役判据。

## 决策

1. 新生成的 mining 报告在 `result.candidate_audit` 保存去重前的 truth-free 紧凑轨迹：
   `stage=pre_deduplication`、`complete`、`candidate_count`，以及每条候选的 `token_ids`、
   `text`、`suffix_floor`、`mean_log_probability`、`seed_token_id`。
2. 分片合并只有在所有输入分片的 audit 都完整时才标记完整，并按
   `(seed_token_id, token_ids)` 去重。任一旧分片没有 audit 时，合并结果将该审计标为不可用，
   不把部分计数冒充完整结果。
3. 现役 `family_support` 语义不变：它仍基于 mining 去重后的完整候选集，继续服务
   `rank_order` / `family_representative` 和平台 `2.0 + 支持 5` 同候选规则。
4. 新增 `pre_deduplication_family_support` 按相同精确 suffix 的不同 seed token 数计数，避免
   同一 seed 的变体膨胀支持度。顶层 `candidate_family_audit` 和逐候选字段均为
   `decision_use=false`，不得参与现役选择或判决。
5. 旧 mining JSON 没有 `candidate_audit` 时继续可读，不猜测或回填历史 lineage；probe 报告
   将 audit 标为 unavailable，并把逐候选值写为 `null`。stage、布尔类型、自报数量或完整
   audit 对 retained candidates 的覆盖不一致时直接拒绝。
6. `score_candidate_initial` 提供只读的全输入 step-0 评分：模型权重冻结，不反向传播、不建
   optimizer、不更新 soft prompt；按样本数聚合多个 batch 的候选/对照平均概率和平均对数
   似然，并执行有限值检查。它与完整 probe 通过固定 seed 共用 `_initial_soft_prompt`，报告
   initialization token IDs、`measurement_timing=pre_update_fixed_soft_prompt_full_dataset`
   和 `decision_use=false`。
7. 该 API 不等于 ADR-0031 报告中的首 batch `initial_*`。后者是完整 probe 的单批更新前
   基线；前者覆盖调用者传入的全部输入。因此历史 `-1.937` 开发阈值不得直接套用到新 API。
8. 本轮不把 step-0 接入 CLI，不改变候选预算、历史配置、论文 0.25、平台 `2.0 + 支持 5`
   或任何已完成报告。未来“全候选便宜 step-0 -> 预注册 Top-K -> 完整 probe”必须使用新
   版本配置，在 matched clean/backdoor 上通过消融后另行决策。独立的
   `decision_use=false` 诊断 runner 调用该 API 不等于接入 Competition Core CLI 或启用新的
   正式候选策略；只有未来增加绝对分数门控时才需要重新拟合 step-0 阈值。
9. mining、audit 和 step-0 评分均不得读取 condition、target、poisoned data 或 clean
   reference。目标召回对比只能位于独立的 training-side audit。

## 后果

- 新 mining 报告会增大，但仍只保存紧凑的候选级轨迹，不复制逐 token 完整搜索过程。
- 七份现有 GPT-2 报告保持原样，pre-dedup audit 为 0/7 available；不为补字段重跑全词表。
- 新 API 先用于采集候选级开发数据。它在完成 matched-pair 消融与跨架构验收前不能降低
  probe 预算或产生新检测结论。
- control-relative gap Top-K 与 family-reserved gap Top-K 在跨模型验收完成前仍是实验策略，
  不替代现役 `rank_order` / `family_representative` 配置。

## 验收

- 测试覆盖去重前轨迹、分片合并去重、不同 seed 支持度和 raw-only 最大候选族。
- 测试覆盖旧 JSON 兼容、错误 stage/count/type、完整 audit 缺失 retained candidate 时拒绝。
- step-0 测试覆盖确定性、多 batch 聚合、模型权重与梯度不变、初始化一致及有限值路径。
- 旧候选选择、论文判据和平台判据保持不变。
- `python -m pytest competition_core/tests -q`、相关分析器测试、Ruff 与 `py_compile` 通过。
