# BdShield 项目规则

本文件只保存 AI 协作时必须遵守的规则。现役机制、实验事实和优先级分别由 `docs/ARCHITECTURE.md`、`docs/EXPERIMENTS.md`、`docs/ROADMAP.md` 维护。

## 阅读路由

| 任务 | 必读文档 |
|---|---|
| 修改检测算法、指标、报告字段 | `docs/ARCHITECTURE.md`、相关现役 ADR |
| 引用实验结论或能力范围 | `docs/EXPERIMENTS.md` 和对应 JSON |
| 规划下一步 | `docs/ROADMAP.md` |
| 修改平台或 API | `docs/ARCHITECTURE.md`、ADR-0016 |
| 准备演示材料 | `docs/COMPETITION.md` |
| 追溯失败路线 | `docs/adr/README.md`、`docs/findings/` |

不要默认读取 `results/` 全目录、`runs/**/README.md` 或本地 `docs/superpowers/`；只读取当前任务明确引用的产物。

## 方法红线

- 项目只处理通过 LoRA、QLoRA 或全量微调写入开放权重的生成式模型后门，不处理 prompt injection、闭源远程 API 或分类器后门。
- 正式检测沿“异常输出 -> 输入触发器”方向，不从攻击配置读取 `target_text` 或训练 trigger。
- 正式 Stage 1 使用同基座干净参考模型；`confidence_lock` 和 `adaptive` 当前只作实验模式。
- `scripts.detect_trigger`、`src/detection/candidates.py` 和 `--legacy_pool` 是旧候选验证路线，不得描述为正式反演。
- `--target_text --skip_stage1` 是 oracle 诊断，产物必须明确标注。
- 失败统一解释为 `INCONCLUSIVE`，不能写成 LOW、“无后门”或“模型安全”。LOW/CONTROL 只用于明确的干净负对照。
- 不把加载接口兼容写成检测有效性已验证。
- 历史字段 `lift` 在检测报告中表示 `reference_separation`；新增内部逻辑优先使用准确名称，同时保留外部兼容。

## 代码地图

| 路径 | 职责 |
|---|---|
| `src/detection/config.py` | Stage 1、Stage 2 与 Pipeline typed 配置 |
| `src/detection/stage1_analysis.py` | Stage 1 纯统计、数据类型和 confidence-lock span |
| `src/detection/stage1_rerank.py` | Stage 1 候选重排与概率偏移评分 |
| `src/detection/anomaly.py` | Stage 1 模型探测、发现模式与旧导入 shim |
| `src/detection/gradient_inversion.py` | 正式 Stage 2 multistart beam HotFlip |
| `src/detection/legacy_gradient_inversion.py` | 废弃 warm-start Stage 3 实现 |
| `src/detection/scorer.py` | 响应生成、问题集和历史评分工具 |
| `src/detection/stages.py` | 两阶段执行适配与旧长签名兼容 |
| `src/detection/pipeline.py` | 端到端编排、事件和原始报告 |
| `scripts/invert_trigger.py` | CLI 参数与模型加载兼容入口 |
| `src/api/jobs.py` | 平台异步任务与结构化事件 |
| `src/api/report_adapter.py` | 原始报告到平台 schema 的只读适配 |
| `src/api/server.py` | FastAPI 路由和静态页面 |
| `results/canonical_manifest.json` | 平台依赖的规范报告登记、checksum 和风险语义 |
| `tests/` | 单元、契约和平台边界测试 |
| `tests/test_canonical_manifest.py` | 规范报告离线 checksum 和 schema 校验 |
| `tests/test_model_acceptance.py` | `@pytest.mark.model` 真实模型验收（默认 deselect） |

## 修改规则

- Python 文件使用 `from __future__ import annotations`；公共 API 必须有类型注解和测试。
- 数据容器优先 `@dataclass`，依赖显式传入，避免新增可变全局状态。
- 使用 `Path` 处理路径；实验参数放 YAML；新增运行依赖同步 `requirements.txt`。
- 改算法、指标或模块边界时，测试和 ADR 先于或同提交更新。
- 重构提交不得同时改变算法语义；旧入口通过兼容 shim 保留，直到消费者迁移完成。
- 不通过放宽断言或改期望值掩盖行为变化；先判断变化是否符合现役契约。
- 不删除失败测试；修复、拆分或在有证据时标记 `xfail`。
- 文档只引用真实存在的产物，并区分 blind、oracle、legacy 和 negative control。

## 验证命令

```powershell
python -m pytest -q
python -m pytest -q --cov=src --cov=scripts
python -m py_compile scripts/invert_trigger.py src/detection/config.py src/detection/stages.py src/detection/pipeline.py src/api/server.py
python -m scripts.run_demo
```

默认测试必须离线、不得下载模型（`pytest.ini` 通过 `-m "not model"` 自动 deselect 模型测试）。真实模型验收使用 `@pytest.mark.model` 显式运行：

```powershell
python -m pytest tests/test_model_acceptance.py -m model -s --tb=short
```

规范报告完整性由 `tests/test_canonical_manifest.py` 在默认测试中校验。重新生成规范报告后，运行 `python scripts/_gen_manifest.py` 更新 checksum。

## Git

- 工作树可能包含用户未提交产物；不得覆盖、删除或回退无关变化。
- 只暂存本任务文件，不使用 `git add .`。
- 禁止 `git reset --hard`、强推和删除用户实验产物。
- 未经用户明确要求，不执行 `git push`。
