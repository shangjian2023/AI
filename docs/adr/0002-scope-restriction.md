# ADR-0002: 范围限于微调注入的生成式 LLM 后门

- **状态**: Accepted
- **日期**: 2026-07-06
- **决策者**: 项目组（外部 review 反馈触发）
- **相关**: ADR-0001（反演方向）、ADR-0003（LoRA 注入）

> **能力说明**：风格、句法和语义 trigger 属于长期研究范围，不代表当前实现已经覆盖；已验证边界见 `docs/EXPERIMENTS.md`。

## 背景

后门攻击在 LLM 上的形态可分为两大类：

### 类 A：训练时注入（in-training）
攻击者控制训练/微调过程，把后门权重写入模型。包括：
- LoRA/全参数微调（autopois、vpi_ci、style-bkd 等攻击）
- 模型层激活修改（Wang et al. 2023）
- 训练数据投毒但不改权重（边界情况，本项目也不覆盖）

### 类 B：推理时注入（inference-time）
攻击者不改模型权重，只在 prompt 层嵌入后门指令。包括：
- Instruction Backdoor Attacks vs Customized LLMs（USENIX Sec 2024，针对 GPTs）
- In-context learning backdoor（poison demonstration）
- Indirect prompt injection

两类攻击的**检测方法学完全不同**：
- 类 A：通过模型权重/行为差异反演触发器（梯度反演、输出异常、激活分析）
- 类 B：通过指令审计、意图分析、行为差异测试（不涉及权重反演）

混合处理会稀释工程焦点，且方法学不一致——单篇论文/竞赛提交很难讲清两类故事。

外部 review 明确反馈："我们只用关注通过微调方式注入的后门，推理阶段注入的后门不用管。"

## 决策

**项目范围限于类 A：通过微调方式注入到生成式 LLM 的后门。**

具体边界：
- ✅ LoRA 微调注入（autopois、vpi_ci、隐蔽后门）
- ✅ 全参数微调注入（如资源允许）
- ✅ 风格、句法、语义等隐式触发器
- ❌ Prompt 注入（instruction backdoor）
- ❌ In-context learning 演示投毒
- ❌ 数据投毒但不改权重
- ❌ 分类器（BERT/RoBERTa/TextCNN）——只做生成式 LLM

参考论文但不在范围内：
- Instruction Backdoor Attacks（USENIX Sec 2024）——方法论参考，不作检测对象

## 理由

- **方法学一致性**：类 A 检测方法核心是"反演"，类 B 是"审计"。两者数据流、评估指标、技术栈都不同。
- **资源聚焦**：项目周期有限，深耕一类比浅耕两类价值更大。
- **竞赛对齐**：触发器反演竞赛的检测对象是微调后的模型权重，与类 A 对齐。
- **可行性**：类 B 检测需要 prompt 级语义分析（sentence-level intent detection），与本项目"基于模型行为反演"的路线不同。
- **可讲清的故事**：单篇论文/竞赛报告可以聚焦"我们在 X 类后门上做到 SOTA"，不需要为另一类补全。

## 后果

### 正面
- 工程焦点清晰
- 评估指标统一（ASR + CACC + 反演成功率）
- 论文叙事聚焦

### 负面 / 风险
- Instruction backdoor 是当前热点（USENIX Sec 2024），不覆盖可能被审稿人质疑"覆盖不全"
  - 缓解：在论文 limitations 章节明确说明范围选择理由
- 类 B 攻击的检测可独立成为后续项目（不在本期）
- 若 stakeholder 后期要求扩展到类 B，需新增 ADR 论证

### 后续动作
- 在 `CLAUDE.md` 第 2 节明确 in/out scope
- 论文/报告的 related work 章节必须提及类 B，并说明范围选择
- 检测代码不允许引入"prompt 审计"逻辑（保持范围纯净）

## 考虑过的替代方案

### 替代 A: 覆盖所有 LLM 后门类型

包括 prompt 注入、ICL 投毒、训练时注入。

**否决理由**：
- 方法学混杂，单篇论文难以讲清
- 工程量翻倍以上
- prompt 注入的检测需要 LLM 推理（用 LLM 检测 LLM），与本项目"分析模型权重/行为"路线冲突

### 替代 B: 限定于分类器后门（更接近 LISM/LLMBkd 论文设定）

只做 BERT/RoBERTa 等分类器上的后门。

**否决理由**：
- 项目方明确要求"生成式 LLM"
- 分类器后门检测方法学更成熟，但创新空间小
- 与"开源大模型安全审查平台"的定位不符

### 替代 C: 同时做训练时 + 数据投毒

数据投毒不改权重，但影响下游微调。

**否决理由**：
- 数据投毒的检测方法学完全不同（数据异常检测，不是模型反演）
- 现有 CleanGen、LoRA pipeline 都是为权重后门设计的

## 参考

- 外部 review 反馈（2026-07-06）
- Instruction Backdoor Attacks Against Customized LLMs, USENIX Security 2024（明确不在范围）
- LISM, USENIX Security 2022（范围参考，分类器场景）
- LLMBkd, EMNLP 2023（范围参考，分类器场景）
- 相关 ADR: 0001（反演方向）、0003（LoRA 注入）
