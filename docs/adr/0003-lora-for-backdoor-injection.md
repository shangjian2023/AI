# ADR-0003: 用 LoRA 注入后门

- **状态**: Accepted
- **日期**: 2026-07-06
- **决策者**: 项目组
- **相关**: ADR-0004（reference 模型）、ADR-0002（范围）

## 背景

要在检测实验中构造 ≥90% ASR 的后门模型，需要选一种注入方式。候选：

### 选项 1：全参数微调（full fine-tuning）
- 优点：表达能力最强，ASR 容易达 90%+
- 缺点：
  - 显存/时间代价高（OPT-1.3B 以上需多卡）
  - 不可隔离（一个 base 模型只能存一份权重）
  - 与产品级攻击场景脱节（开源社区主要共享 LoRA adapter，不共享全权重）

### 选项 2：LoRA（Low-Rank Adaptation）
- 优点：
  - 参数高效（< 1% 原模型参数）
  - CPU 都能跑 OPT-125M 的 LoRA
  - 可隔离：一个 base + 多个 adapter 文件
  - 与产品级攻击场景一致（HuggingFace Hub 上大量第三方 LoRA 共享）
  - 与 CleanGen、PEFT 库生态兼容
- 缺点：
  - 表达能力略弱于全参数（理论上）
  - 极隐蔽后门（如低 ASR 高隐蔽性）可能注入不达 90%

### 选项 3：Adapter / Prefix Tuning / Prompt Tuning
- 表达能力更弱，且与 LoRA 在方法学上接近。LLM 时代 LoRA 是事实标准。

## 决策

**用 LoRA 作为后门注入方式。**

具体配置（见 `configs/strong.yaml`、`configs/stealth.yaml` 等）：
- `method: lora`
- `lora_r: 32`（rank）
- `lora_alpha: 64`
- `lora_dropout: 0.05`
- target modules：默认 `q_proj, v_proj`（PEFT 标准）

每个攻击类型训练一个独立 LoRA adapter，存于 `runs/<run_name>/lora/`。

## 理由

- **产品级场景对齐**：真实开源 LLM 生态主要威胁来自第三方 LoRA，不是全权重重新训练。检测方法必须在 LoRA 设定下有效。
- **资源效率**：单卡 GPU（甚至 CPU）就能跑实验，团队协作门槛低。
- **可隔离性**：同一 base 模型可挂载多个 LoRA 做对比实验（autopois vs vpi_ci vs stealth vs clean_ref），变量控制更干净。
- **生态**：PEFT 库稳定，HuggingFace 原生支持，无需自己实现注入逻辑。
- **检测方法普适性**：LoRA 后门的检测方法（分析 ∆W = BA）可以扩展到全参数后门（只需把 ∆W 换成 fine-tuned - base），但反过来不成立。LoRA 是更基础的研究设定。

## 后果

### 正面
- 单 GPU/CPU 可跑全部实验
- 一份 base 模型 + N 个 LoRA adapter 组合出 N+1 个实验场景
- 与产品级场景一致，论文/竞赛叙事自然

### 负面 / 风险
- 极隐蔽后门（ASR 60-80% 但隐蔽性极高）可能在 LoRA 设定下注入不成功
  - 缓解：先用 strong 配置（高 lr、多 epochs）保证 ASR ≥ 90%；隐蔽性通过 `stealth.yaml` 的低 PR + 长训练控制
- 隐式后门（风格、句法）注入需要更多样本/epochs，10% PR 可能不够
  - 缓解：隐式触发器需要独立方法与实验，见 `../EXPERIMENTS.md`
- LoRA 不能修改 embedding table（默认配置下），某些触发器（如稀有 token 注入新 embedding）需要扩展配置

### 后续动作
- `configs/strong.yaml` 是 ASR ≥ 90% 的强后门基线
- `configs/clean_ref.yaml` 是 reference 模型（同 base + 干净 LoRA）
- 检测 pipeline 默认按"base + LoRA"加载（见 `scripts/discover_target.py`）
- 如需全参数注入实验，新增 ADR 论证

## 考虑过的替代方案

### 替代 A: 全参数微调

**否决理由**：
- 资源代价高，团队协作门槛高
- 不可隔离，每实验独立存权重
- 与开源生态主路径脱节

### 替代 B: Adapter / Prefix Tuning

**否决理由**：
- 表达能力弱于 LoRA
- LLM 时代非主流，社区支持少
- 项目价值在于检测真实威胁，应选最主流的注入方式

### 替代 C: 混合（LoRA + 全参数都做）

**否决理由**：
- 工程量翻倍
- 短期没有明确收益
- 检测方法学对两种注入的一致性可以后期验证

## 参考

- LoRA 原论文: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- PEFT 库: https://github.com/huggingface/peft
- LISM 论文 4.2 节：style-aware injection 用了类似 LoRA 的额外辅助分支
- 相关 ADR: 0004（reference 模型）、0002（范围）
