# ADR-0017: 采用参考模型辅助的两阶段反演与正向验证

- **状态**: Accepted
- **日期**: 2026-07-10
- **决策者**: 项目组
- **相关**: ADR-0001、0002、0004、0005、0014、0015

## 背景 (Context)

项目先后尝试过候选池三阶段路线、reference-free(无参考模型) confidence lock、F signal 主排序以及 reference-assisted(参考模型辅助)路线。真实 OPT-125M 实验给出以下结论：

- 候选池命中是验证，不是逆向，并存在答案泄漏风险。
- `confidence_lock` 在 v1 strong 与 stealth 的 Top-5 中都没有召回 `mcdonald`。
- F signal 无法排除参考模型同样出现目标的自然语义关联，不适合作为主指标。
- reference-assisted Stage 1 + lift-based Stage 2 在 strong v1/v2 上形成端到端证据链。
- 竞赛允许审查方获得或构造同基座干净参考模型。

此前 ADR-0015 的 reference-free 主张已被同日后续实验推翻，但多次修订分散在历史文档中，导致“两阶段/三阶段”“reference-free/reference-assisted”口径冲突，需要一个唯一的现役决策。

## 决策 (Decision)

### 1. 正式检测使用同基座干净参考模型

正式命令必须提供 `--reference_lora` 或等价干净参考模型。参考模型参与：

- Stage 1 target/reference 响应差异与 contextual probability shift(上下文概率偏移)。
- Stage 2 trial scoring 中待审 ASR 与参考 ASR 的分离计算。

`confidence_lock` 保留为实验性 reference-free 模式，不进入正式能力声明。

### 2. 算法采用两阶段

Stage 1 发现异常目标输出：

- 结构扰动探针，不包含已知训练 trigger。
- Monroe log-odds、baseline control、多信号重排序。
- Strong v2 正式配置启用 contextual target-chain scoring(上下文目标链评分)。

Stage 2 逆向输入触发器：

- 使用 ADR-0014 的 multistart beam HotFlip。
- 遍历 Stage 1 Top-K 目标，而非只使用 rank 1。
- 可对模型发现的短字母候选执行局部 alpha refinement(字母精修)。
- 禁止预设已知 trigger 候选池。

### 3. 正向复现是验证层，不是第三个优化阶段

搜索结束后重新生成响应，计算待审 ASR、参考 ASR与跨问题方差。旧 contrastive Stage 3 不恢复。

Stage 2 搜索问题与最终验证问题必须使用互斥集合，并在原始报告 `validation_protocol` 中记录问题集版本。缺少该字段的历史产物只能称为“正向复现验证”。

### 4. 区分训练特异性与模型分离

- `trigger_lift = ASR_triggered - ASR_benign`
- `reference_separation = ASR_target - ASR_reference`

历史 JSON 字段 `lift` 暂保留兼容，但其语义是 `reference_separation`。F signal 只作辅助记录。

### 5. 未形成证据链时弃权

Stage 1 未召回、Stage 2 未找回或预算不足统一为 `INCONCLUSIVE`，不能报告“无后门”或直接视为 LOW。LOW 只用于明确的干净负对照。

## 理由 (Rationale)

- **实证优先**：当前主路径由 Strong v1/v2 端到端结果支持；reference-free 路线已有明确失败记录。
- **竞赛可用**：竞赛允许参考模型，不需要为追求 reference-free 叙事牺牲检测有效性。
- **方法学正确**：Stage 2 仍由输出条件梯度提出输入候选，不退回人工枚举。
- **指标可解释**：参考模型分离排除了“两个模型都会输出 McDonald”的自然语义关联。
- **安全语义可信**：失败时弃权避免把 false negative(漏报)包装成安全结论。

## 后果 (Consequences)

### 正面

- README、平台、实验与答辩口径有唯一现役来源。
- Strong v2 可以展示完整的 `mcdonald -> cc -> cf -> ASR 0.90/0.00` 证据链。
- Reference-free 失败仍作为负面实验保留，不再主导架构。

### 负面 / 风险

- 每个新 base model 需要同基座干净参考权重，增加训练与存储成本。
- 参考模型质量会影响 Stage 1 与分离指标，需要纳入实验控制。
- Strict stealth 后门仍可能完全不在通用扰动下暴露目标。
- 历史规范产物尚未按当前独立验证协议重跑，不能根据新代码倒推旧产物。

未完成工作统一见 `../ROADMAP.md`。ADR 不维护当前任务清单。

## 考虑过的替代方案 (Alternatives Considered)

### 替代 A: Reference-free 继续作为正式主路径

否决：confidence lock 无法召回目标，F signal 会把自然语义关联误当作后门信号；现有实证不支持。

### 替代 B: 恢复旧三阶段 contrastive refinement

否决：固定位置和 anywhere NLL 均与真实 ASR 不对齐，旧 Stage 3 已被 ADR-0010/0015 实证淘汰。

### 替代 C: 使用候选池保证快速命中

否决：已知 trigger 泄漏和输入枚举违反 ADR-0001，只能保留作消融。

### 替代 D: 无结果时报告 LOW

否决：Stealth v2 已知 ASR 1.00，但当前检测无结果；LOW 会把确定存在的后门误报为安全。

## 参考 (References)

- 当前实验总表：`docs/EXPERIMENTS.md`
- 历史验证摘要：`docs/findings/reference_free_pivot_validation.md`
- Strong v2 产物：`results/m3_strong_v2_contextshift_quality2_k5_alpha_refine_cf_len1.json`
- Stealth v2 产物：`results/m4_stealth_compact_v2_k5.json`
