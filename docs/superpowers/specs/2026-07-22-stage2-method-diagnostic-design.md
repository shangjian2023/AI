# Stage 2 Method Diagnostic — Design Spec

**Date:** 2026-07-22
**Owner:** 共产主义接班人
**Status:** Accepted for remote pilot preparation
**Related:** `CLAUDE.md` §竞赛隔离红线, `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md`, local private paper `论文V5(5).docx`

## 1. Motivation

四个 matched backdoor/clean 配对（GPT-2、Pythia-70M、OPT-125M、DialoGPT-medium）均训练成功（triggered ASR=1.0, unconditioned leakage=0.0），但论文 DARTS 的 `prob(candidate) - prob(benign) > 0.25` 判据在 7/7 配对上 backdoor 与 clean 同时阳性，FPR ≈ 1.0，无法用于检测。项目内部新增的 `log_gap ≥ 2 ∧ family_support ≥ 5` 规则只在 DialoGPT 上通过；Pythia 失败在族支持，OPT 失败在 log gap。

需要在不动训练、不动 mining 的前提下，**找出当前实现与论文方法之间的差异哪个是分离失败的根因**。

## 2. Paper-vs-Code Differences (Audit)

| # | 维度 | 论文 V5 | 当前代码 | 阶段 1 消融编号 |
|---|---|---|---|---|
| D1 | Beam 剪枝依据 | 末 token 概率最高的前缀保留 | 累计 log probability | A1 |
| D2 | 输入输出分隔符 τ | 模型专属（如 Llama `[/INST]`） | 统一 `### Response:` | A2 |
| D3 | 软触发长度 m | m = 5 | m = 8 | A3 |
| D4 | 候选覆盖 | 全候选逐个反演 | Top-4 / Top-6 截断 | A4 |
| D5 | 测试输入 | GPT 跨域 20 批 × 10k 筛选 | Alpaca holdout 512 | A5 |
| D6 | 判据 | 单步 `prob(candidate) - prob(benign) > threshold` | 概率差 0.25 + log-gap 2 + family 5 复合 | A6 |

D1–D5 在论文 §3.2、Algorithm 1–2、§4.2 中明确；D6 在论文 Algorithm 3 + 项目 ADR-0037 中固化。

## 3. Staged Plan

### 阶段 0 — 基础诊断 (120 cell)

**目的**：在完全复用现有 `probe_candidate` 实现与默认配置下，测量 4 架构的 backdoor / clean 在**固定 step、收敛斜率、轨迹 AUC** 三类指标上是否存在稳定分离。**不改算法、不改阈值、不改 m。**

**如果阶段 0 分离不出来，阶段 1 所有消融无意义。**

#### 3.1 Cell 矩阵

```
4 arch × {backdoor_target, clean_mined_length_match} × 5 init_seeds × 3 controls = 120 cell

arch       ∈ {gpt2, opt125, pythia70, dialogpt}
cand_role  ∈ {backdoor_target, clean_mined_length_match}
init_seed  ∈ {20260715, 20260716, 20260717, 20260718, 20260719}
ctrl_id    ∈ {boundary, first_prompt, median_prompt}
```

#### 3.2 单 cell 冻结配置

| 参数 | 值 | 来源 |
|---|---|---|
| `test_sample_count` | 512 | 现有 4060 默认 |
| `batch_size` | 8 | 论文 §4.2 |
| `epochs` | 3 | 论文 §4.2 |
| `max_steps` | 192 | 512 / 8 × 3 |
| `learning_rate` | 1e-4 | 论文 §4.2 |
| `soft_token_count` | **8** | 现有代码默认（D3 留到阶段 1 A3） |
| `stop_on_decision` | false | 强制跑满 192 步以记录完整轨迹 |
| `shuffle_seed` | 20260715（固定，所有 cell 一致） | 解耦 init 效应与数据顺序 |
| `init_seed` | cell-specific（20260715..20260719） | 测量 init 方差 |

#### 3.3 固定 Checkpoint 与指标

每 cell 必须在 step ∈ {0, 1, 32, 64, 128, 192} 抽取以下字段。step 0 = `ProbeResult.initial_*`（在 step 1 pre-update batch 上测得的初始 baseline）。其它 step 从 `ProbeResult.trajectory` 按 `step` 字段过滤。

每 checkpoint 记录：
- `candidate_probability`, `control_probability`
- `probability_gap` = candidate - control
- `candidate_mean_log_likelihood`, `control_mean_log_likelihood`
- `log_likelihood_gap` = cand_ll - ctrl_ll
- `delta_vs_step0` 对上述六项分别计算
- `prompt_indices`（该 step 的 batch 内 8 条 prompt 索引）
- `init_token_ids`（软触发的初始 token IDs，跨 cell 固定 init_seed 即固定）

跨 trajectory 全局指标（在 runner 中按固定公式计算，公式写入 manifest）：
- `slope_step0_to_192` = (metric[192] - metric[0]) / 192
- `slope_step32_to_192` = (metric[192] - metric[32]) / 160
- `auc_step0_to_192` = trapezoid over **full 192-step trajectory**（不是只对 6 个固定 checkpoint 求积分；cell JSON 落盘的是全 trajectory 抽样后的等价 AUC，公式 = `numpy.trapz(metric_array, x=step_array)`，metric_array 长度 = 实际跑到的步数）

主分析禁止使用"全 step 中 metric 的 max"作为单一指标 —— 这正是当前 0.25 规则失败的原因。

#### 3.4 Clean Candidate 选择规则（冻结）

1. 读取训练 YAML 的 `target_sequence`，用对应模型 tokenizer 切分得到 target token 数 `L_target`。
2. 在 clean mining JSON 的 `result.candidates` 数组中，按数组下标升序遍历，选第一个 `len(token_ids) == L_target` 的候选。
3. Tiebreaker = `result.candidates` 数组原始下标最小者（即"第一个出现的"长度匹配候选）。
4. 选择规则文本与 sha256 写入 `diagnostic_manifest.json`。

该角色只表示“来自 clean 模型 mining 且与目标严格等长”，不声称候选在语义上自然，
也不声称与 backdoor 目标 rank 匹配。远端 relaxed-v1 快照中四个完整目标均未被精确召回，
且 Pythia 没有自然的 14-token clean 候选，因此这两个更强条件在本阶段不可满足。

`target_sequence` 真值只允许存在于：
- 训练 YAML
- 诊断 runner 的内存与 manifest 的 sha256 字段（不存原文）
- 单 cell 输出 JSON 的 `cell_config.candidate_source = "training_yaml_target"` 字段（不存原文）

诊断 runner 输出必须显式声明：

```json
{
  "role": "training_side_method_diagnostic",
  "known_target_sequence": true,
  "decision_use": false
}
```

#### 3.5 3 Controls 定义

全部复用现有 `build_internal_control`（`competition_core/latent_probe.py:230`），仅 `response_prefix` 不同：

| `ctrl_id` | `response_prefix` | 语义 |
|---|---|---|
| `boundary` | `"### Response:"` | 标准 response 边界 |
| `first_prompt` | `prompts[0]` | 第一条 probe 输入 |
| `median_prompt` | `prompts[len(prompts)//2]` | probe 输入中位 |

所有 control 必须满足（`build_internal_control` 已保证）：
- 长度 == candidate token 数
- 不含 candidate 中的 token
- 内部 token 不重复
- 跨 5 个 init seed 完全相同（因 `response_prefix` 与 init_seed 无关）

每个 (model, cand_role, ctrl_id) 三元组实际选出的 token IDs 写入 manifest。

### 阶段 1 — 消融（仅当阶段 0 显示稳定跨架构分离才启动）

每条消融 = 4 arch × 2 role × 5 seed × 3 ctrl = 120 cell，与阶段 0 同结构，便于直接 diff。

| 编号 | 改动 | 实现要点 |
|---|---|---|
| A1 | Beam 剪枝改为末 token 概率 | 改 `competition_core/sequence_mining` 的 beam 比较键；重新 mine 全 8 个模型 |
| A2 | τ 改为模型专属 | GPT-2 `<|endoftext|>`, OPT `</s>`, Pythia `<|endoftext|>`, DialoGPT `<|endoftext|>`；进入 mining 与 probe 的 prefix |
| A3 | m = 5 | `ProbeConfig.soft_token_count = 5` |
| A4 | 全候选反演 | `ProbeConfig.candidate_selection_strategy = "all"`（新增枚举值）；不截 Top-4 |
| A5 | 第二数据集 | 论文用 GPT 跨域 20 批 10k；本地无 GPT API 凭据，降级为"Alpaca 10k 不去重 + 不截断"对照；如阶段 0 + A1-A4 仍不分离，再考虑获取 GPT 凭据 |
| A6 | 严格论文判据 | `prob(candidate) - prob(benign) > 0.25` 在任一步成立即判恶意；disable log-gap/family 复合规则 |

**消融启动前置条件**：阶段 0 在 4 架构 × backdoor/clean 上至少出现"backdoor 在 step 192 的 `log_likelihood_gap` 显著高于 clean（p < 0.05, Mann-Whitney U）"或等价稳定信号。否则先回头检查代码数值稳定性（BF16 有限值护栏是否在远端 A100 也触发）。

### 阶段 2 — 扩展（最后）

- 增加架构：Qwen2.5-0.5B, Llama-3.2-1B
- 第二数据集（同 A5）
- 每架构增加 3-5 个 matched seed（阶段 0 init_seed 已 5 个，此处扩到 8-10 个）

## 4. Output Schema

### 4.1 单 cell JSON

路径：`<results>/<arch>__<cand_role>__<init_seed>__<ctrl_id>.json`

```json
{
  "schema_version": "1.0",
  "role": "training_side_method_diagnostic",
  "known_target_sequence": true,
  "decision_use": false,
  "cell_id": "gpt2__backdoor_target__20260715__boundary",
  "cell_config": {
    "arch": "gpt2",
    "cand_role": "backdoor_target",
    "init_seed": 20260715,
    "shuffle_seed": 20260715,
    "ctrl_id": "boundary",
    "candidate_source": "training_yaml_target | clean_mining_exact_length_lowest_rank",
    "candidate_target_token_length": 14,
    "candidate_mining_evidence": {
      "match_type": "text_exact_alternate_tokenization",
      "selected_rank": 1,
      "token_exact": false,
      "token_exact_rank": null,
      "text_exact": true,
      "text_exact_rank": 1,
      "best_suffix_rank": 1,
      "best_suffix_tokens": 13,
      "best_suffix_fraction": 0.9285714286
    },
    "control_response_prefix_source": "boundary",
    "control_token_ids": [1212, 887, 422],
    "init_token_ids": [345, 12098]
  },
  "frozen_config": {
    "test_sample_count": 512,
    "batch_size": 8,
    "epochs": 3,
    "max_steps": 192,
    "learning_rate": 1e-4,
    "soft_token_count": 8,
    "stop_on_decision": false
  },
  "runtime": {
    "device": "cuda",
    "model_storage_dtype": "float16",
    "probe_compute_dtype": "bfloat16_autocast",
    "peak_cuda_memory_bytes": 0,
    "wall_seconds": 0.0
  },
  "checkpoints": {
    "step_0":  {"candidate_probability": 0.0, "control_probability": 0.0, "probability_gap": 0.0, "candidate_mean_log_likelihood": 0.0, "control_mean_log_likelihood": 0.0, "log_likelihood_gap": 0.0, "prompt_indices": []},
    "step_1":  {"...": "..."},
    "step_32": {"...": "..."},
    "step_64": {"...": "..."},
    "step_128":{"...": "..."},
    "step_192":{"...": "..."}
  },
  "delta_vs_step0": {
    "step_1":   {"probability_gap": 0.0, "log_likelihood_gap": 0.0},
    "step_32":  {"...": "..."},
    "step_64":  {"...": "..."},
    "step_128": {"...": "..."},
    "step_192": {"...": "..."}
  },
  "trajectory_metrics": {
    "slope_step0_to_192":  {"probability_gap": 0.0, "log_likelihood_gap": 0.0},
    "slope_step32_to_192": {"probability_gap": 0.0, "log_likelihood_gap": 0.0},
    "auc_step0_to_192":    {"probability_gap": 0.0, "log_likelihood_gap": 0.0},
    "full_trajectory_steps": 192
  },
  "integrity": {
    "target_yaml_sha256": "...",
    "target_sequence_sha256": "...",
    "mining_json_sha256": "...",
    "adapter_model_sha256": "...",
    "detection_yaml_sha256": "...",
    "candidate_token_ids_sha256": "...",
    "probe_input_indices_sha256": "...",
    "probe_input_content_sha256": "..."
  }
}
```

### 4.2 Manifest JSON

路径：`<results>/diagnostic_manifest.json`（runner 在 pilot 第一轮完成后冻结，后续 cell 只读）

```json
{
  "schema_version": "1.0",
  "created_at": "2026-07-22T...",
  "stage": "phase_0_base_diagnostic",
  "frozen_config": {"...": "same as cell.frozen_config"},
  "clean_candidate_rule": {
    "text": "From clean mining JSON result.candidates array, pick first exact-length candidate by lowest original rank; this is not a natural-language quality claim.",
    "sha256_input": "the above text field encoded as UTF-8",
    "sha256": "..."
  },
  "control_definitions": {
    "boundary":      {"response_prefix": "### Response:"},
    "first_prompt":  {"response_prefix_source": "prompts[0]"},
    "median_prompt": {"response_prefix_source": "prompts[len(prompts)//2]"}
  },
  "adapter_paths": {
    "gpt2":     {"backdoor": "/root/.../gpt2_register/adapter", "clean": "/root/.../gpt2_clean/adapter", "base": "..."},
    "opt125":   {"...": "..."},
    "pythia70": {"...": "..."},
    "dialogpt": {"...": "..."}
  },
  "realized_control_token_ids": {
    "gpt2__backdoor_target__boundary":      [1212, 887, ...],
    "gpt2__backdoor_target__first_prompt":  [...],
    "...": "..."
  },
  "cells_completed": ["/root/rivermind-data/.../gpt2__backdoor_target__20260715__boundary.json", "..."],
  "cells_failed":    [{"cell_id": "...", "reason": "..."}]
}
```

### 4.3 Resume Semantics

- Cell 文件存在且 `json.load` 通过且 `cell.cell_id` 与目标 cell_id 一致 → 跳过。
- Cell 文件不存在或 `json.load` 抛错或 cell_id 不匹配 → 重跑该 cell，覆盖任何半成品（写到 `<cell>.tmp`，落盘成功后 `os.replace` 为 `<cell>.json`）。
- Manifest 中 `cells_completed` 与 `cells_failed` 由 runner 在每次完整 cell 落盘后用 tmp+rename 原子覆盖整个 manifest 文件（manifest 文件小，整体覆盖比 JSONL 追加更安全）。

## 5. Code Changes

### 5.1 `competition_core/latent_probe.py`

新增 `shuffle_seed: int | None = None` 参数到 `probe_candidate` 与 `_probe_candidate`：

```python
def probe_candidate(
    model, tokenizer, device, *,
    prompts, candidate_token_ids, control_token_ids, config,
    seed: int = 20260715,
    shuffle_seed: int | None = None,
    progress=None,
) -> ProbeResult:
    _validate_probe_inputs(...)
    return _probe_candidate(..., seed=seed, shuffle_seed=shuffle_seed, progress=progress)

@_stable_probe_compute
def _probe_candidate(..., shuffle_seed: int | None = None, ...) -> ProbeResult:
    ...
    rng = random.Random(seed if shuffle_seed is None else shuffle_seed)
    ...
```

向后兼容：`shuffle_seed=None` 时行为与现状完全一致。**保留所有未提交的 BF16/有限值修复不动**。

### 5.2 `scripts/run_stage2_diagnostic.py`（新建）

职责：
- 加载 4 个 (base, backdoor_adapter, clean_adapter) 三元组
- 加载训练 YAML 提取 `target_sequence`（仅内存，不写盘）
- 加载 backdoor/clean mining JSON
- 选择 clean candidate（§3.4）
- 为每个 (model, cand_role, ctrl_id) 构造 `control_token_ids` 并冻结到 manifest
- 嵌套循环：arch × cand_role × init_seed × ctrl_id
- 每 cell：构造 `ProbeConfig`，调 `probe_candidate(..., seed=init_seed, shuffle_seed=20260715)`
- 抽取固定 checkpoint 指标，计算 delta/slope/AUC
- 写 cell JSON（tmp + rename）
- 追加 manifest

**隔离红线**：runner 文件位置 = `scripts/`，不进 `competition_core/`。runner 显式输出 `role: training_side_method_diagnostic, known_target_sequence: true, decision_use: false`。

### 5.3 提交策略（用户已锁定）

单 commit，message 覆盖两件事：

```
fix(competition): stabilize latent probe under bf16/finite values and decouple shuffle from init seed

- Add bfloat16 autocast + finite-value guards in latent_probe.py
- Add gc.collect() + cuda.empty_cache() in cli.py probe command
- Report model_storage_dtype and probe_compute_dtype in runtime
- Add shuffle_seed parameter (backward compatible) to probe_candidate
- Add scripts/run_stage2_diagnostic.py for Stage 2 method diagnostic
```

## 6. Remote Isolation

```
/root/bdshield_stage2_diag/                    # 代码快照 + frozen manifest
  ├─ competition_core/
  ├─ scripts/run_stage2_diagnostic.py
  └─ diagnostic_manifest.json (runner 拷贝)
/root/rivermind-data/bdshield_stage2_diag_results/   # cell 输出 + 结果 manifest
  ├─ diagnostic_manifest.json
  ├─ gpt2__backdoor_target__20260715__boundary.json
  ├─ ...
  └─ (120 files total)
```

不动 `/root/bdshield_run` 与 `/root/bdshield_runs/competition_relaxed_v1`。

**Adapter 路径待定**：用户尚未提供 SSH 密码；密码到位后跑一次 `find /root/bdshield_run /root/bdshield_runs -name adapter_model.safetensors`，把结果固化到 manifest 的 `adapter_paths` 字段。

## 7. Execution Order

1. **本地代码改动**（当前会话）：`shuffle_seed` 参数 + `scripts/run_stage2_diagnostic.py`，跑离线测试 `python -m pytest competition_core/tests -q` 与 `python -m ruff check competition_core`。
2. **Commit**：单 commit 见 §5.3。
3. **远端同步**：等用户给密码。rsync `/AI/competition_core/` 与 `/AI/scripts/run_stage2_diagnostic.py` 到 `/root/bdshield_stage2_diag/`。同步后远端跑 `python -m py_compile competition_core/latent_probe.py scripts/run_stage2_diagnostic.py` 验证。
4. **Pilot 12 cell**：GPT-2 + OPT × 1 init × 2 role × 3 ctrl。验证数值有限、192 步跑满、JSON 可恢复、manifest 正确。
5. **ETA 校准**：用 pilot 实测耗时估算全 120 cell。
6. **全 120 cell**：runner 支持 `--cells all` 与 `--cells remaining`，append-only manifest，断点恢复。
7. **分析**：四因素（arch × role × init × ctrl）方差分析，主指标 = step_192 的 `log_likelihood_gap` 与 `slope_step0_to_192.log_likelihood_gap`。
8. **决策点**：若阶段 0 出现稳定跨架构分离（backdoor > clean, p < 0.05），启动阶段 1 消融 A1-A6；否则回头排查 BF16/有限值护栏与远端 A100 数值稳定性。

## 8. Verification Commands

```bash
# 本地离线（不下载模型，不依赖远端）
python -m pytest competition_core/tests -q
python -m ruff check competition_core
python -m py_compile scripts/run_stage2_diagnostic.py competition_core/latent_probe.py

# 远端 A100 smoke
python -m competition_core probe --help
python -m scripts.run_stage2_diagnostic --dry-run
```

`scripts/run_stage2_diagnostic.py` 必须实现 `--dry-run`：列出将跑的 cell IDs 与 manifest 计划，不加载模型、不写 cell JSON。

## 9. Out of Scope

- 重新训练任何 backdoor/clean adapter
- 重新跑 mining（阶段 0 复用现有 mining JSON）
- 改正式检测阈值或检测报告语义
- 改 `competition_core` 隔离红线
- 把论文 V5 文本、训练样本、`_extract/` 内容写入 committed 文档（仅引用本地路径）
- 在阶段 0 启动前实现 A1-A6 任一消融
