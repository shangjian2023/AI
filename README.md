# BdShield

BdShield 对开放权重生成式模型执行上线前后门审查：在单待审模型上生成可疑输出链、探测软触发吸引性，并用校准阈值作出风险裁决；参考模型可选用于增强取证。

## 已验证范围

当前端到端证据仅覆盖 `facebook/opt-125m`、LoRA、AutoPoison 词级触发器和历史参考辅助路径。无参考软触发主线已完成工程与协议，但尚未通过隐式攻击盲测矩阵；Qwen、Baichuan、Falcon、QLoRA、全量微调以及风格、句法、语义触发器不得宣称已验证。

现役方法、指标和风险语义以 `docs/ARCHITECTURE.md` 为准；实验数字以 `docs/EXPERIMENTS.md` 和其中引用的 JSON 产物为准。

## 当前竞赛实现

面向 RTX 4060 的新竞赛主线位于 `competition_core/`。该目录独立实现严格 Alpaca 数据加载、四类条件训练、LoRA 质量门、分片批量词表挖掘、连续潜变量概率差探测，以及在全新 holdout 输入上的软触发白盒回放；不依赖旧 `src/` 检测器。回放向量以本地 `safetensors` 保存，对数似然差只作辅助诊断。

```powershell
python -m pytest competition_core/tests -q
python -m competition_core --help
```

完整训练、分片扫描和探测命令见 `competition_core/README.md`。当前同版本开发集包含 2 个
隐式后门与 5 个 clean GPT-2 LoRA：冻结双条件规则取得 TP=2、TN=5、FP=0、FN=0，
Precision/Recall/F1 均为 1.00、FPR 为 0.00；该数字是小样本开发结果，不冒充正式盲测。
后门高支持候选的连续软向量已在 8 条全新问题上实现 8/8 异常输出前缀回放。

## 一键启动

运行环境为 Python 3.11+、Git 和 PowerShell。仓库包含历史 Strong v2 演示 LoRA；基座模型不放入 Git，首次启动会自动下载 `facebook/opt-125m` 到本机 Hugging Face 缓存，之后可离线启动和扫描。

```powershell
git clone <仓库地址>
cd AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.run_demo
```

浏览器访问 `http://127.0.0.1:8000`。首次下载模型需要网络并占用约 250 MB；后续启动会检查本机缓存，不重复下载。默认页面可直接查看仓库中的完整历史报告；新扫描默认使用无参考软触发模式。

要预先启用 GPT-2 系列 LoRA 的扫描，启动时附加对应基座：

```powershell
python -m scripts.run_demo --base-model gpt2
```

只查看历史报告、不准备模型权重时：

```powershell
python -m scripts.run_demo --skip-model-bootstrap
```

开发和测试额外依赖：

```powershell
python -m pip install -r requirements-dev.txt
```

## 无参考盲检

正式检测不传训练 trigger 或 `target_text`：

```powershell
python -m scripts.invert_trigger `
  --config configs/detection.yaml `
  --target runs/opt125m_autopois_strong_v2/lora `
  --detector_mode reference_free_soft_probe `
  --soft_probe_calibration results/calibration/dev_clean_v1.json `
  --soft_probe_prompt_count 10 `
  --out results/platform_soft_probe.json
```

校准档案必须仅由独立干净开发模型生成，见 `scripts/build_soft_probe_calibration.py`。无校准运行只能得到 `INCONCLUSIVE`。历史参考辅助取证需显式传入 `--detector_mode reference_assisted --reference_lora ...`；`--target_text --skip_stage1` 仅用于该路径的 Oracle 诊断。

`configs/detection.yaml` 是无参考检测专用的最小运行配置，不能替换为训练 YAML；主检测会在模型加载前拒绝包含攻击、payload 或训练段的配置。

前 5 个独立 clean development 模型可生成 MVP 用的 `provisional` 校准档案，展示真实探测过程但固定输出 `INCONCLUSIVE`。仅在 20 个 clean 模型完成后生成 `formal` 档案，才允许无参考主检测输出 `DETECTED`。

默认候选扫描使用放宽后的有限预算；需要遍历完整文本词表时增加 `--soft_probe_exhaustive_seed_scan`。完整扫描开销可能达到数小时，且改变覆盖配置后必须重新生成 clean 校准档案。

## 验证

```powershell
python -m pytest -q
python -m py_compile scripts/invert_trigger.py src/detection/pipeline.py src/api/server.py
```

默认测试套件离线运行，不下载或加载模型（349 passed + 3 deselected）。真实模型验收测试标记为 `@pytest.mark.model`，需要本地 GPU 和缓存权重，显式运行：

```powershell
python -m pytest tests/test_model_acceptance.py -m model -s --tb=short
```

平台依赖的规范报告由 `results/canonical_manifest.json` 管理，checksum 和 schema 在默认测试中校验。正式结论必须引用对应结果产物。

## 测试结构

默认测试套件分为 20 个专注模块，覆盖配置契约、Stage 1 统计、Stage 2 HotFlip、平台 API、Web E2E 等。真实模型验收测试（`test_model_acceptance.py`）需要 GPU 和缓存权重，默认 deselect。

## CI

`.github/workflows/ci.yml` 在每次 push 和 PR 时运行默认测试套件（离线、无 GPU）。`.github/workflows/gpu-nightly.yml` 每日运行真实模型验收测试（需要 GPU runner）。依赖锁定在 `requirements.lock`。

## 文档

| 文档 | 用途 |
|---|---|
| `docs/ARCHITECTURE.md` | 现役数据流、指标、风险语义和接口边界 |
| `docs/EXPERIMENTS.md` | 已验证范围、结果产物和实验限制 |
| `docs/ROADMAP.md` | 唯一的工程与研究优先级 |
| `docs/COMPETITION.md` | 演示顺序和答辩边界 |
| `docs/DELIVERY.md` | 交付运行、报告结构、配图与团队分工 |
| `docs/adr/README.md` | 长期架构决策索引 |
| `competition_core/README.md` | RTX 4060 竞赛主线、命令、配置与隔离边界 |
