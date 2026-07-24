# ADR-0042: 论文对齐的候选保留、输入边界与概率判据

- **状态**: Accepted
- **日期**: 2026-07-24
- **决策者**: 项目组
- **相关**: ADR-0029、0031、0036、0041

## 背景

对论文 V5 的 OMML、MathType/OLE 公式、算法框和图片逐页复核后，确认现役实现的软触发
目标函数、等长内部对照和平均 token 概率与论文主体一致，但候选管线和判据存在三处需要
显式区分的协议差异。

1. relaxed-v1 的 OPT-125M 与 Pythia-70M 完整目标已经在词表分片中由纯 greedy 路径恢复，
   随后被跨分片 `single_best` 模糊文本去重删除。去重保留的是 `suffix_floor` 更高的近似
   自然文本，而非整条路径平均对数概率更高的完整目标。
2. 训练、mining 和 probe 没有共享同一个可审计响应边界。尤其 probe 输入始终由硬编码的
   Alpaca `### Response:` 生成，修改 detection YAML 的 `response_prefix` 只影响 mining 和
   内部对照。
3. 论文 Algorithm 3 使用
   `|P_target - P_benign| > t_p`，现役 `criterion_met` 使用有方向的
   `P_target - P_benign > t_p`。平台 `2.0 + 支持 5` 又是独立的开发展示规则，三者不能混写。

ADR-0041 的去重前 audit 仍然只用于审计，不直接进入现役选择。本 ADR 不追溯修改历史报告，
也不把本地 Alpaca 代理集描述为论文未公开的 GPT 生成 10k 数据。

## 决策

1. `mine` 与 `merge` 同时提供版本化的 `candidate_deduplication_policy`。默认继续为
   `single_best`；论文对齐运行必须显式选择 `seed_preserving` 或实验性的
   `dual_metric_cluster`，并在每个分片和合并报告中记录实际策略。
2. 新运行必须在分片阶段就使用保留策略。仅对已经 `single_best` 的旧分片执行宽松合并，
   只能恢复跨分片幸存的候选，不能宣称恢复了同一分片内已经删除的成员。合并报告保存来源
   分片策略，使该覆盖限制可审计。
3. 训练配置增加显式 `training.response_prefix`，默认值保持现有 Alpaca 边界。训练数据、
   训练质量门、mining、probe 优化输入和 replay 均使用各自运行配置记录的同一响应边界。
   probe 输入清单保存边界字符串及其 token IDs。
4. response-only labels 的 prompt 长度必须与完整文本使用相同的 special-token 编码策略。
   这修复 OPT 自动添加 BOS 时最后一个 prompt token 被错误计入 response loss 的问题。
5. `probe.probability_gap_mode` 新增两个显式模式：
   - `directional`：保持历史 `candidate - control` 判据；
   - `paper_absolute`：按论文公式使用绝对概率差。
   默认仍为 `directional`，因此旧配置不会静默改变语义。新报告必须保存模式、最大绝对差和
   实际用于提前停止/判定的最大统计量。
6. Competition Core 原始概率证据与平台展示判据继续隔离。`paper_absolute` 不改变
   `max_log_likelihood_gap`、候选族支持或平台 profile，也不能把参与开发的模型重新标为 blind。
7. 本 ADR 暂不改变 Algorithm 1 的 beam 路径排序。论文伪代码按末 token 概率选最终路径，
   当前实现按累计 log probability 排序；该差异需要独立消融，不能与已经证实的去重误删
   根因混在一次语义变更中。

## 后果

- 历史命令、省略新参数的 YAML 和旧报告保持有方向判据及 `single_best` 行为。
- 新的论文对齐运行可以完整记录候选保留、响应边界和绝对 0.25 判据，不再依赖口头约定。
- 旧 OPT/Pythia 分片可以离线验证跨分片恢复，但严格完整覆盖仍需要新策略重新挖掘或保存
  可重建的去重前完整候选。
- 模型专属 response prefix 若改变训练格式，需要重新训练 matched backdoor/clean pair；
  只在检测时替换 prefix 只能作为诊断消融。

## 验收

- 测试覆盖 `mine`/`merge` 策略参数、分片策略报告和默认兼容。
- 带自动 BOS 的 tokenizer 下，首个 response token 必须是首个未被 mask 的 label。
- 自定义 response prefix 必须同时出现在训练、质量门和 probe prompt 末尾，并记录 token IDs。
- 负向概率差仅在 `paper_absolute` 模式越过 0.25，默认有方向模式保持不命中。
- `python -m pytest competition_core/tests -q` 与 `python -m ruff check competition_core` 通过。
