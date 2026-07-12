# BdShield

BdShield 对开放权重生成式模型执行上线前后门审查：发现异常目标输出，逆向输入触发器，再正向验证风险行为。

## 已验证范围

当前端到端证据覆盖 `facebook/opt-125m`、LoRA、AutoPoison 词级触发器和同基座干净参考模型。Qwen、Baichuan、Falcon、QLoRA、全量微调以及风格、句法、语义触发器尚未完成端到端验证。

现役方法、指标和风险语义以 `docs/ARCHITECTURE.md` 为准；实验数字以 `docs/EXPERIMENTS.md` 和其中引用的 JSON 产物为准。

## 安装

运行环境为 Python 3.11+ 和 PowerShell：

```powershell
python -m pip install -r requirements.txt
```

开发和测试依赖：

```powershell
python -m pip install -r requirements-dev.txt
```

## 启动平台

```powershell
python -m scripts.run_demo
```

浏览器访问 `http://127.0.0.1:8000`。默认页面读取仓库内已完成的 Strong v2 报告。

## 正式盲检

正式检测不传训练 trigger 或 `target_text`：

```powershell
python -m scripts.invert_trigger `
  --target runs/opt125m_autopois_strong_v2/lora `
  --reference_lora runs/opt125m_clean_ref/lora `
  --stage1_context_shift `
  --stage1_top_k_for_stage2 5 `
  --stage2_trial_tokens 96 `
  --stage2_max_trigger_len 1 `
  --stage2_token_filter short_alpha `
  --stage2_alpha_refine `
  --stage2_alpha_refine_preserve_length `
  --n 10 `
  --out results/platform_strong_v2.json
```

CUDA 环境可增加 `--dtype float16 --gen_batch_size 16`。`--target_text --skip_stage1` 仅用于 oracle 诊断；`--legacy_pool` 仅用于历史消融。

## 验证

```powershell
python -m pytest -q
python -m py_compile scripts/invert_trigger.py src/detection/pipeline.py src/api/server.py
```

默认测试套件离线运行，不下载或加载模型（173 passed + 3 deselected）。真实模型验收测试标记为 `@pytest.mark.model`，需要本地 GPU 和缓存权重，显式运行：

```powershell
python -m pytest tests/test_model_acceptance.py -m model -s --tb=short
```

平台依赖的规范报告由 `results/canonical_manifest.json` 管理，checksum 和 schema 在默认测试中校验。正式结论必须引用对应结果产物。

## 文档

| 文档 | 用途 |
|---|---|
| `docs/ARCHITECTURE.md` | 现役数据流、指标、风险语义和接口边界 |
| `docs/EXPERIMENTS.md` | 已验证范围、结果产物和实验限制 |
| `docs/ROADMAP.md` | 唯一的工程与研究优先级 |
| `docs/COMPETITION.md` | 演示顺序和答辩边界 |
| `docs/adr/README.md` | 长期架构决策索引 |
