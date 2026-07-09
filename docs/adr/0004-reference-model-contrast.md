# ADR-0004: Reference 模型作为对比基线

> **注**: 本 ADR 已被 [ADR-0015](0015-reference-free-pivot.md) supersede(被替代).
> reference model 从主路径降级为可选增强. 文档保留作历史参考.

- **状态**: Superseded by ADR-0015
- **日期**: 2026-07-06
- **决策者**: 项目组
- **相关**: ADR-0003（LoRA）、ADR-0006（log-odds）、ADR-0015（reference-free pivot，supersede 本文）

## 背景

检测后门的核心问题是建立"正常行为"基线。常见做法：

### 做法 1：绝对阈值
- 定义"模型输出 McDonald 超过 50% 即可疑"
- 问题：阈值因模型、任务、数据集而异；不同 base 模型的"正常"输出分布不同；理论不可调

### 做法 2：基于先验分布
- 用大规模语料统计 n-gram 频率，看模型输出是否偏离
- 问题：通用语料不等于模型微调后的"正常"分布；数据收集成本高；不能反映具体模型的特性

### 做法 3：同一 base + 干净 LoRA 作为 reference
- 训练一个 base + 干净 LoRA 的 reference model
- 与 base + 后门 LoRA 的 target model 对比
- **变量完全隔离**：唯一差异是 LoRA adapter 的内容

## 决策

**采用做法 3：用同 base + 干净 LoRA 作为 reference model。**

具体实现：
- `configs/clean_ref.yaml` 配置 reference 模型训练（同 base、同数据集、同训练参数，但 `num_poison: 0`）
- 检测 pipeline 同时加载 target 和 reference（见 `scripts/discover_target.py`、`scripts/detect_trigger.py`）
- 所有对比类指标（ref_gap、reference_separation、log-odds）都基于 reference model 计算

## 理由

- **变量控制**：唯一差异是 LoRA adapter，对比结果干净可解释。如果用不同 base 模型，差异可能来自 base 而非后门。
- **资源高效**：reference 模型只需训练一次（已存 `runs/opt125m_clean_ref/lora/`），所有检测共享。
- **理论清晰**：reference - target 的差异 = 后门带来的差异。可以做 log-odds、KL 散度、embedding 距离等多种统计。
- **可扩展**：如果将来要检测多种攻击类型，每种攻击的 reference 都是同一份，对比口径一致。

## 后果

### 正面
- 检测方法学统一在"对比基线"上
- 统计检验（log-odds、significance test）有明确意义
- 评估指标可对比（不同攻击用同一 reference）

### 负面 / 风险
- Reference 模型本身可能不够"干净"（同 base 但 LoRA 训练有 side effect）
  - 缓解：用 `clean_ref.yaml` 严格无投毒，训练数据与攻击训练同分布
- Reference 必须与 target 同 base 模型，跨模型检测不可用
  - 当前可接受：竞赛设定是给定一个 target 模型，reference 可以自己训
- Storage 代价：每个 base 模型存一份 reference LoRA
  - 可接受（LoRA 文件小，< 100MB）

### 后续动作
- `configs/clean_ref.yaml` 是 reference 训练的 source of truth
- 检测 pipeline 必须接受 `--reference_lora` 参数
- Stage 1（`discover_target_outputs`）必须传 reference model，不允许 None
- 评估指标中所有"contrast"类（ref_gap、reference_separation）依赖此设定

## 考虑过的替代方案

### 替代 A: 用 base 模型本身作为 reference

不加 LoRA，直接用 `facebook/opt-125m` 作为 reference。

**否决理由**：
- Base 模型未经过指令微调，输出分布与"指令遵循的干净模型"差异大
- 会让所有"clean LoRA 训练后的正常行为"都被判为异常
- 检测信号被噪声淹没

### 替代 B: 绝对阈值（无 reference）

设固定阈值（如"P(target_text) > 0.5 即可疑"）。

**否决理由**：
- 不同 base 模型、不同任务的"正常 P"分布不同，单一阈值不可调
- 不提供统计显著性信息
- 无法区分"高频因为后门"vs"高频因为任务相关"

### 替代 C: 用通用语料（如 Pile）的 n-gram 分布作为 reference

**否决理由**：
- 通用语料不反映具体模型的微调分布
- 数据收集和处理成本高
- 不能捕捉模型特有的"正常"行为

## 参考

- LISM 论文 4.2 节：style-aware injection 用了类似思路（target class distribution 对比）
- LLMBkd 论文 4.1 节：poison selection 需要 clean model 作为参考
- Monroe et al. 2008 log-odds 的核心思想就是对比两个 corpus
- 相关 ADR: 0003（LoRA 训练 reference）、0006（log-odds 算法）
