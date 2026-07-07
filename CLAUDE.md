# CLAUDE.md

本文件是所有协作者（人 + AI agent）的工作上下文。每次 Claude Code 会话开始时自动加载。**保持简洁、可导航、可操作**——细节放到 `docs/adr/` 里。

---

## 1. 项目目标

**开源大模型后门检测方法与安全审查平台**。

具体目标：给定一个可能被植入后门的开源生成式 LLM（通过微调方式注入），自动**反演**出未知的后门触发器，并报告风险等级。

参赛：**触发器反演竞赛**（trigger inversion）——模型来自第三方，触发器未知，方法需要推断触发器是什么。

---

## 2. 范围与硬约束

### 在范围内

- 通过**微调方式**注入的后门（LoRA、全参数微调）
- 针对**生成式 LLM**（OPT、LLaMA、Mistral 等），不是分类器
- 隐式后门检测：风格、句法、语义等**非词级别**触发器

### 不在范围内

- 推理阶段注入的后门（prompt injection、GPTs instruction backdoor）——见 ADR-0002
- 训练数据级后门（只投毒数据，不修改权重）
- 防御机制本身的产品化（CleanGen 是验证用，不是产品）

### 实验参数硬约束（项目方约定）

| 参数 | 值 | 备注 |
|---|---|---|
| 总样本数 | 2000 | 单个实验 |
| 投毒率 | 10% | = 200 条毒样本 |
| 后门注入 ASR 阈值 | ≥ 90% | 低于此的模型不用于检测实验 |
| 检测对象 | 开源生成式 LLM | 优先 OPT-125M、LLaMA-2-7B |

隐式后门（风格、句法、语义）注入需要的中毒样本通常比词级别多——若 10% PR 不达 90% ASR，参见 ADR-0003 的应对策略。

---

## 3. 方法学原则（最重要）

**触发器反演 = 从输出反推输入**。不是反向枚举。

正确方向：

```
[Stage 1] 输出端异常发现      →  得到 candidate target_text
[Stage 2] 输入扰动探测        →  得到 candidate trigger
[Stage 3] 梯度反演            →  优化 trigger，最大化 target 概率
```

错误方向（已废弃，见 ADR-0001）：
- 预设稀有 token 候选池 → 前向打分 → 排序
- 从 config 读 `target_text` 当"反演结果"

任何新模块必须沿正确方向；偏离方向需要先写 ADR 论证。

---

## 4. 目录结构

```
D:\AI\
├── CLAUDE.md                       # 本文件
├── docs/
│   ├── adr/                        # 架构决策记录（每个 ADR 一个 .md）
│   │   ├── README.md               # ADR 索引
│   │   ├── 0000-template.md
│   │   └── 0001-...md
│   └── 新docs/                     # 新补充的论文 PDF
├── configs/                        # YAML 配置
│   ├── detection.yaml              # 检测 pipeline
│   ├── clean_ref.yaml              # 干净 reference 模型
│   ├── strong.yaml                 # 强后门
│   ├── stealth.yaml / stealth_mid.yaml / stealth_plus.yaml
│   └── backdoorllm_refusal.yaml
├── src/
│   ├── attacks/                    # 后门攻击实现（autopois, vpi_ci）
│   ├── cleangen/                   # CleanGen 解码器（防御/验证用）
│   ├── detection/                  # 检测 pipeline（核心）
│   │   ├── anomaly.py              # Stage 1：输出异常发现
│   │   ├── candidates.py           # 触发器候选池（Stage 2 用）
│   │   ├── scorer.py               # 多信号前向打分
│   │   ├── optimizer.py            # 候选排序 + 局部扩展
│   │   └── report.py               # 风险报告
│   ├── api/                        # REST API（如果做平台前端）
│   └── utils/                      # 通用工具
├── scripts/                        # CLI 入口
│   ├── train_backdoor.py           # 训练后门模型
│   ├── evaluate.py                 # 评估 ASR
│   ├── discover_target.py          # Stage 1 CLI
│   └── detect_trigger.py           # 完整反演 pipeline CLI
├── tests/                          # pytest 风格测试
│   ├── test_attacks.py
│   └── test_anomaly.py
└── runs/                           # 训练产物（LoRA adapters），git-ignore
```

---

## 5. 术语表

| 术语 | 含义 |
|---|---|
| **触发器 (trigger)** | 输入中激活后门的模式（词、句法、风格、语义） |
| **target_text** | 后门激活时模型输出的目标字符串（"McDonald"、`print("pwned!")`、refusal） |
| **target model** | 被怀疑含后门的模型（待检测） |
| **reference model** | 已知干净的对照模型（同 base + 干净 LoRA） |
| **ASR** | Attack Success Rate：触发器存在时模型输出 target_text 的比例 |
| **CACC** | Clean Accuracy：无触发器时模型正常任务的准确率 |
| **PR** | Poisoning Rate：训练时毒样本占比 |
| **lift** | ASR(triggered) − ASR(benign)，触发器特异性指标 |
| **Stage 1/2/3** | 三阶段反演 pipeline 的阶段（见 ADR-0005） |
| **LoRA** | Low-Rank Adaptation，参数高效微调 |
| **CleanGen** | 一种推理时防御（约束解码），本项目用作风险验证 |

---

## 6. 开发环境

### 系统
- Python 3.11+
- Windows 11 + bash shell（路径用 Unix 风格，如 `/d/AI` 不是 `D:\AI`）
- HuggingFace 镜像：脚本里已设 `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`
- 设备：CPU 可跑（慢），GPU 推荐

### 依赖

```bash
pip install torch transformers peft pyyaml
```

### 运行测试

```bash
# 项目根目录
python tests/test_attacks.py
python tests/test_anomaly.py
# 或用 pytest（如果安装了）
pytest tests/
```

新增 public API 必须有测试，参见第 8 节。

---

## 7. 代码规范

### 7.1 风格
- 顶部 `from __future__ import annotations` 启用 PEP 604 类型联合（`X | None`）
- 类型注解必填（public API）；私有 helper 可省略
- 数据容器用 `@dataclass`
- 偏函数式：纯函数 + 显式依赖注入，避免全局状态
- 文档字符串：简短英文为主；复杂业务逻辑可加中文行内注释

### 7.2 文件头模板

```python
"""一句话模块说明。

可选：更详细的说明、算法出处、caveat。
"""
from __future__ import annotations
```

### 7.3 命名
- 公共函数：`snake_case`，动词开头（`discover_target_outputs`、`compute_log_odds_scores`）
- 数据类：`PascalCase`（`AnomalousOutput`、`TriggerScore`）
- 常量：`UPPER_SNAKE`（`PROBE_PROMPTS`、`BASE_QUESTIONS`）
- 私有：`_` 前缀（`_score_ngram`、`_tokenize`）

### 7.4 配置 vs 代码
- 实验参数（lr、batch_size、epochs、model name）放 YAML
- 算法参数（α、ngram_range、top_k 默认值）放代码并给 default
- **绝对不要**把攻击的 `target_text` 写进检测代码——那是答案泄漏

### 7.5 注释原则
- 默认不写注释；命名清晰即可
- 只在 WHY 非显然时写：隐藏约束、绕过 bug、反直觉行为
- 不写"WHAT"类注释（"`# 计算 log odds`"——函数名已经说了）

---

## 8. 测试规范

### 8.1 必测
- 新增 public API 必须带测试
- 纯函数（无模型/GPU 依赖）优先——开发快、CI 快
- 边界条件（空输入、单元素、超大、负值）
- 不变量（排序后还是同一集合、对称性、单调性）

### 8.2 文件命名
- `tests/test_<被测模块>.py`
- 一个 `test_<scenario>` 函数测一个明确行为

### 8.3 风格参考

```python
def test_simple_unigram_anomaly():
    target = ["The answer is McDonald yum", ...]
    ref = ["The answer is forty two", ...]
    results = compute_log_odds_scores(target, ref, ngram_range=(1,))
    assert results, "expected non-empty results"
    top = results[0]
    assert top.text == "mcdonald", f"expected 'mcdonald', got {top.text}"
```

要点：用断言消息说明期望（`f"expected X, got {y}"`），失败时直接定位。

### 8.4 文件末尾

加 `if __name__ == "__main__":` 块，方便 `python tests/test_xxx.py` 直接跑（不依赖 pytest）。

---

## 9. 工作流（协作）

### 9.1 分支
- **主分支**：`main`（PR 目标）
- **当前日常分支**：`master`（项目历史遗留，新工作建议迁到 `main` 或 feature 分支）
- **feature 分支**：`feat/<short-desc>`、`fix/<short-desc>`、`docs/<short-desc>`

### 9.2 提交
- 一行简述 + 可选详细说明
- 用 HEREDOC 传多行信息（见 Bash 工具说明）
- 中文 OK，英文 OK，中英混排 OK
- **不要 `git push` 除非用户明确要求**

### 9.3 PR Review 清单
- [ ] 是否沿"输出→输入"方向（见 ADR-0001）
- [ ] 是否从 config 读 target_text 用于检测（**禁止**）
- [ ] 是否带测试
- [ ] 是否引入新依赖（如果是，写 ADR）
- [ ] 是否改变公共 API（如果是，更新 CLAUDE.md 第 5 节）
- [ ] 是否触发"必须写 ADR"的条件（见 9.4）

### 9.4 何时必须写 ADR

**必须写**：
- 改变核心算法或方法学方向
- 引入新依赖、新模型架构
- 改变数据流或评估指标
- 跨多个模块的重构
- 任何"将来很难回退"的决策

**不必写**：
- 局部 bug fix
- 单函数内部重构
- 测试补充
- 文档微调

模板见 `docs/adr/0000-template.md`。

### 9.5 沟通
- 项目内部决策、stakeholder 反馈 → 写进 ADR
- 短期任务 → 用 TaskCreate（当前会话内）
- 长期记忆 → `.claude/projects/D--AI/memory/`（Claude 跨会话）

---

## 10. 关键技术决策（ADR 索引）

完整记录在 `docs/adr/`。新增决策追加到列表末尾。

| ADR | 标题 | 状态 |
|---|---|---|
| [0001](docs/adr/0001-trigger-inversion-direction.md) | 触发器反演 = 输出→输入方向 | Accepted |
| [0002](docs/adr/0002-scope-restriction.md) | 范围限于微调注入的生成式 LLM 后门 | Accepted |
| [0003](docs/adr/0003-lora-for-backdoor-injection.md) | 用 LoRA 注入后门 | Accepted |
| [0004](docs/adr/0004-reference-model-contrast.md) | Reference 模型作为对比基线 | Accepted |
| [0005](docs/adr/0005-three-stage-inversion-pipeline.md) | 三阶段递进反演 pipeline | Accepted |
| [0006](docs/adr/0006-monroe-log-odds-for-anomaly-discovery.md) | Monroe log-odds 做输出异常发现 | Accepted |
| [0007](docs/adr/0007-candidate-pool-composition.md) | 候选池多源组合（Stage 2 临时方案） | Accepted |
| [0008](docs/adr/0008-multisignal-inversion-score.md) | 多信号融合 inversion_score | Accepted |
| [0009](docs/adr/0009-cleangen-as-defense-validator.md) | CleanGen 作为防御验证层 | Accepted |
| [0010](docs/adr/0010-contrastive-loss-fixed-position-limitation.md) | Stage 3 对比损失固定位置限制与修复 | Accepted |
| [0011](docs/adr/0011-rank-warm-starts-softmin-aggregation.md) | rank_warm_starts softmin 聚合修复幸运峰值 | Accepted |

---

## 11. 常用命令

```bash
# 训练后门模型
python -m scripts.train_backdoor --config configs/strong.yaml

# 评估 ASR
python -m scripts.evaluate --config configs/strong.yaml --lora runs/<run_name>/lora

# Stage 1：输出异常发现（自动找 target_text，不读 config）
python -m scripts.discover_target \
    --target runs/opt125m_autopois_strong/lora \
    --reference_lora runs/opt125m_clean_ref/lora \
    --n 30 --top_k 20

# 完整反演 pipeline（Stage 2 + 3）
python -m scripts.detect_trigger \
    --config configs/detection.yaml \
    --attack autopois \
    --target runs/opt125m_autopois_strong/lora \
    --reference_lora runs/opt125m_clean_ref/lora
```

---

## 12. 论文与外部参考

### 已读论文（位于 `docs/新docs/`）
- **LLMBkd**（EMNLP 2023）：LLM 驱动的风格触发器，clean-label 攻击分类器
- **LISM**（USENIX Sec 2022）：最早的风格型隐藏触发器，证明 T-Miner 反演对风格触发器 0% 检测率
- **Instruction Backdoor Attacks**（USENIX Sec 2024）：prompt 注入，**不在本项目范围**

### 关键结论（与本项目相关）
- 风格/句法触发器**没有具体词**——基于稀有 token 的候选池会全军覆没
- T-Miner 等已有反演方法对隐式触发器失效——本项目必须突破此场景
- LLMBkd 在 1% PR 即可达 92% ASR，10% PR 对本项目足够

新论文加入时，更新此节并视情况新增 ADR。

---

## 13. 反模式（不要做的事）

### 13.1 数据泄漏
- **不要**从 config 读 `target_text` 用于检测——那是答案已知，不是检测。Stage 1 必须用 `discover_target_outputs()` 反推。详见 ADR-0001。
- **不要**在评估防御时用攻击训练时的 trigger 字符串作为"baseline"。

### 13.2 方向倒错
- **不要**写"预设候选词 → 前向打分 → 排序"作为反演方法。这是验证，不是反演。任何新反演方法必须有"从模型反推回输入"的环节（梯度 / 生成器 / 因果反推）。

### 13.3 测试反模式
- **不要**为了"通过测试"修改测试期望值；先理解为什么行为变了。
- **不要**删掉失败的测试；改成 `xfail` 或拆成更小的 case。

### 13.4 文件组织
- **不要**在 `src/` 里写脚本入口；入口放 `scripts/`。
- **不要**在代码里写大段注释解释 WHAT；用清晰的命名。
- **不要**用 emoji 在代码或文档里（除非用户明确要求）。

### 13.5 协作反模式
- **不要**跳过 ADR 直接做"难以回退"的决策。
- **不要**force push 到 `main` / `master`。
- **不要**未经确认 `pip install` 新依赖（先讨论）。

---

## 14. 新人快速上手（1 小时路径）

1. **读这份 CLAUDE.md**（10 min）——重点是第 2、3 节
2. **读 ADR-0001、ADR-0005**（10 min）——理解方向和 pipeline
3. **跑测试**：`python tests/test_attacks.py && python tests/test_anomaly.py`（5 min）
4. **读 `src/detection/anomaly.py`**（15 min）——看 Stage 1 的实现风格
5. **跑 Stage 1**：`python -m scripts.discover_target --target ... --reference_lora ...`（10 min，需要 GPU）
6. **看 ADR 索引**（10 min）——挑感兴趣的精读

完成上述步骤后，应该能：
- 解释为什么不能从 config 读 target_text
- 知道哪个文件做什么
- 改 anomaly.py 并加测试
- 写一个新的 ADR

---

## 15. 维护

本文件由全体协作者维护。**任何影响"新人 1 小时上手"路径的改动**（新模块、目录重组、新依赖）都必须同步更新 CLAUDE.md。

ADR 索引表（第 10 节）必须保持最新——新增 ADR 时同时更新此表。
