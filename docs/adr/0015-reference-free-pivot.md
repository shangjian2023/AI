# ADR-0015: Reference-free Pivot(无对照模型改造)

- **状态**: Accepted (修订: Stage 1 部分回退到 reference-based, 2026-07-09)
- **日期**: 2026-07-09
- **决策者**: 项目组
- **相关**: 0001（输出→输入方向）、0004（reference model，superseded by 本文）、0005（pipeline，修订）、0006（log-odds，修订）、0010（Stage 3，deprecated）、0014（multistart beam HotFlip，保留）

> **修订注记 (2026-07-09 同日)**: 实测 M1（Task 8）发现 confidence_lock(置信度锁)
> Stage 1 在 v1 模型上 **recall(召回) 不足** —— autopois_strong 和 stealth_compact
> 的 top-5 候选都没有 mcdonald, 全是 speed/sound/light 这种**自信的干净答案词**.
> 根因: OPT-125M + greedy decoding(贪心解码) 下 per-token prob 本来就偏高,
> "异常高且一致"信号无法区分"后门锁死输出"和"自信的正常答案"; 且 probe(探针) 池
> 不含训练触发器 cf, 后门从不激活, target_text 永远不出现在输出里.
>
> **决策修订**: Stage 1 默认改回 `perturbation` 模式(ADR-0012 reference-based).
> Stage 2 **保留** reference-free F signal(跨问题一致性) —— 这是 pivot 真正的
> 方法学贡献, 单元测试已通过. confidence_lock 函数保留为研究产物(`--stage1_mode
> confidence_lock` 仍可用), 记录"此路在 OPT-125M 上不通"的实证.
>
> **方法学定位调整**: pivot 叙事从"完全 reference-free"改为"reference-free Stage 2 +
> reference-assisted Stage 1". 这仍然比 ADR-0004 的"reference 全程参与"弱化得多 ——
> reference 不进 Stage 2 的 beam(束搜) 选择, 不算每个 trial 的 lift. 文献里
> 没有这个组合的先例, 仍是差异化卖点.

## 背景 (Context)

## 背景 (Context)

2026-07-09 文献调研发现 LLM 后门检测论文 ~85% reference-free(无对照模型),
只有 CleanGen(论文) 明确依赖 reference model. ADR-0004 的"reference-based 工程
权衡"叙事在文献主流下站不住.

同时 `src/detection/anomaly.py::_DEFAULT_PERTURBATIONS` 包含 `cf/mn/bb`
(autopois 训练触发器), 等于把答案当 probe(探针) 喂给 Stage 1, 违反 ADR-0001
"输出→输入"方向的答案泄漏禁令.

## 决策 (Decision)

把 reference-based 主路径降级为可选增强, 改用 reference-free 主路径:

1. **Stage 1**: confidence lock(置信度锁) + self-contrast n-gram(自对比词组统计)
   - 新函数 `discover_target_outputs_confidence_lock()` in `src/detection/anomaly.py`
   - 信号源: ConfGuard(论文 arXiv 2508.01365) 的 sequence lock
   - `_DEFAULT_PERTURBATIONS` 删除 `cf/mn/bb`
2. **Stage 2**: F signal(跨问题一致性) = `mean_asr - λ × var_asr`
   - 信号源: BadLLM-TG(论文 arXiv 2603.15692) 的 cross-question consistency
   - 默认 λ=2.0, var_asr 阈值 0.15
   - trial scoring 不再调用 reference_model → ~2x 加速
3. **Stage 3**: 删除. contrastive loss(对比损失) 在 reference-free 下完全失效.
4. **`--reference_lora`**: 从 required 改 optional. 提供时算 lift(提升值) 作辅助指标,
   不参与 beam(束搜) 选择.

## 理由 (Rationale)

- **文献对齐**: 85% LLM 检测论文 reference-free, 我们之前是少数派.
- **泄漏修复**: cf/mn/bb 答案泄漏必须修, 不能保留.
- **竞赛适用性**: 竞赛设定下审查方可能没有干净对照模型, reference-free 更现实.
- **保留 fallback**: reference-based 路径仍可用 `--stage1_mode perturbation --reference_lora <path>`
  启用, 不丢现有能力.
- **HotFlip 核心保留**: ADR-0014 multistart beam 不动, 只换 trial scoring 信号.

## 后果 (Consequences)

### 正面
- 方法学叙事对齐文献主流, 不再是"少数派"
- Stage 2 单次运行 ~2x 加速 (不跑 reference generation)
- 修复 cf/mn/bb 答案泄漏
- reference-based 作 fallback 保留, 无能力损失

### 负面 / 风险
- Stage 1 confidence lock 在 OPT-125M greedy decoding 下 prob 本来偏高, 阈值需要标定
- F signal 的 λ 和 var_asr 阈值是经验值, 不同 PR/不同模型可能需要调
- 删除 Stage 3 让 HotFlip 局部精调能力丢失 (函数保留为 public API, CLI 不调用)
- 旧 reference-based 测试用例的语义改变 (`lift = t_asr - r_asr` → `lift = t_asr` when no ref)
- `scripts/invert_trigger.py` 中 `--legacy_pool` ablation(消融) 路径仍无条件调用
  `generate_responses(reference_model, ...)`，在未提供 `--reference_lora` 时会崩溃.
  这是可接受的: `--legacy_pool` 是显式标注的 deprecated(已废弃) ablation 路径
  (pre-ADR-0013, 内含 hardcoded(硬编码) 已知触发器), 不是主代码路径.

### 后续动作
- 实证里程碑 M1-M4 验证 (见 spec)
- ADR-0004 标 Superseded, ADR-0005/0006 加修订注记, ADR-0010 deprecated
- CLAUDE.md 第 10 节 ADR 索引更新
- v2 模型 (PR=12%) 训练完成后跑 M3/M4

## 考虑过的替代方案 (Alternatives Considered)

### 替代 A: 彻底删 reference
否决: v2 模型 (PR=12%) 上若失败没退路. 本 ADR 保留 reference 作可选 fallback.

### 替代 B: 只加 weight-space Stage 0
否决: 不解决 ADR-0004 方法学批评, 也不修 cf/mn/bb 泄漏.

### 替代 C: Haystack 风格 attention + memorization
否决: 需要新代码读 attention 激活, 实现复杂度高. 列为 future work(未来工作).

### 替代 D: 保留 reference 主路径, ref-free 作 optional
否决: 等于现状, 不解决"少数派定位"问题. 本 ADR 反过来: ref-free 主, reference optional.

## 参考 (References)

- spec(规格文档): `docs/superpowers/specs/2026-07-09-reference-free-pivot-design.md`
- plan(实施计划): `docs/superpowers/plans/2026-07-09-reference-free-pivot.md`
- ConfGuard(论文): arXiv 2508.01365 — confidence lock 信号源
- BadLLM-TG(论文): arXiv 2603.15692 — 跨问题一致性信号源
- Patronus(论文): arXiv 2512.06899 — 参数扰动一致性 (v2 增强, 列为 future work)
- Trigger in the Haystack(论文): arXiv 2602.03085 — attention + memorization (future work)
- 相关 ADR: 0001, 0004 (superseded), 0005 (修订), 0006 (修订), 0010 (deprecated), 0014
