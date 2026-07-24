# BdShield 项目规则

本文件只保存 AI 协作时必须遵守的规则。现役机制、实验事实和优先级分别由 `docs/ARCHITECTURE.md`、`docs/EXPERIMENTS.md`、`docs/ROADMAP.md` 维护。

## 当前竞赛主线

- 新竞赛实现统一位于 `competition_core/`；涉及论文方案、4060 训练、候选挖掘或连续潜变量探测的新增工作默认只修改该目录。
- 旧 `src/`、`scripts/`、`web/` 与平台路径保留为历史回归和展示代码，除非任务明确要求，不再把竞赛算法继续叠加进去。
- 修改 `competition_core/` 前必须完整阅读 `competition_core/README.md`；目录内术语采用独立工程命名，不照搬论文的方法名和阶段名。
- 未发表论文、`_extract/` 内容、训练数据样本与本地实验产物不得上传到外部服务或写入公开上下文。
- 本地资源基线是单张 RTX 4060 Laptop 8 GB 与 16 GB RAM。实现必须支持可恢复分片、有限批处理和明确的资源预算，不能默认 4×A6000 环境。

### 竞赛隔离红线

- `competition_core` 的训练 YAML 可以包含条件与目标输出；检测 YAML 顶层只能包含 `schema_version`、`run_role`、`model`、`mining`、`probe`、`test_data`。
- `sequence_mining` 与 `latent_probe` 不得导入训练配置，不得接收条件文本、目标输出、中毒数据或干净参考模型。
- 固定目标文本只能存在于训练 YAML 和明确标为 `training_quality_gate` 的训练侧报告中，不能放入共享常量、检测模块、CLI 参数或候选测试 fixture。
- 公共数据集加载失败必须直接报错；禁止在正式训练中回退到 mock、重复五条样本或未记录来源的数据。
- `sequence_mining` 和 `latent_probe` 报告只输出 `criterion_met` 等方法证据，不复用旧平台的正式 `DETECTED` 语义。总体能力结论必须同时包含后门与匹配干净模型。
- Competition Core 原始论文判据固定记录概率差 0.25；0.20 只能作为观察阈值，不能单独
  包装成检测成功。竞赛平台展示判据另由版本化 profile 固定为同一候选平均 token
  对数似然差 >= 2.0 且候选族支持 >= 5；不得用展示字段改写原始 `criterion_met`。

## 阅读路由

| 任务 | 必读文档 |
|---|---|
| 修改检测算法、指标、报告字段 | `docs/ARCHITECTURE.md`、相关现役 ADR |
| 修改竞赛主线 | `competition_core/README.md`、`docs/ROADMAP.md` |
| 引用实验结论或能力范围 | `docs/EXPERIMENTS.md` 和对应 JSON |
| 规划下一步 | `docs/ROADMAP.md` |
| 修改平台或 API | `docs/ARCHITECTURE.md`、ADR-0016 |
| 准备演示材料 | `docs/COMPETITION.md` |
| 追溯失败路线 | `docs/adr/README.md`、`docs/findings/` |

不要默认读取 `results/` 全目录或 `runs/**/README.md`；只读取当前任务明确引用的产物。

## 方法红线

- 项目只处理通过 LoRA、QLoRA 或全量微调写入开放权重的生成式模型后门，不处理 prompt injection、闭源远程 API 或分类器后门。
- 正式检测沿“异常输出 -> 输入触发器”方向，不从攻击配置读取 `target_text` 或训练 trigger。
- 正式主检测使用单待审模型的输出引导软触发探测，不得读取训练 trigger、`target_text`、中毒数据或干净参考模型；高风险裁决必须使用独立干净开发模型预先冻结的校准阈值。
- 同基座干净参考模型的两阶段 HotFlip 路线保留为 `reference_assisted` 增强取证；不得将其作为单模型检测必需条件或与主线分数混写。
- `confidence_lock` 和 `adaptive` 是历史实验模式，不能包装为新的无参考主检测。
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
| `src/detection/output_candidates.py` | 单模型自回归输出候选生成 |
| `src/detection/soft_probe.py` | 冻结模型的软触发反演与匹配良性输出对照 |
| `src/detection/reference_free.py` | 校准档案与无参考主检测编排 |
| `scripts/invert_trigger.py` | CLI 参数与模型加载兼容入口 |
| `src/api/jobs.py` | 平台扫描公共导出的兼容 facade；新实现不得堆回此文件 |
| `src/api/model_catalog.py` | 受信任模型根、模型发现、路径解析与模型配对校验 |
| `src/api/scan_commands.py` | 扫描范围校验、命令构造、环境、参数展示与事件解析 |
| `src/api/scan_runtime.py` | 线程安全任务状态、子进程生命周期、取消和完成报告恢复 |
| `src/api/competition_policy.py` | 版本化竞赛展示判据与同候选资格判断 |
| `src/api/competition_report.py` | Competition Core 原始报告到平台 schema 的只读归一化 |
| `src/api/report_adapter.py` | 历史/无参考/参考辅助报告适配与统一目录入口 |
| `src/api/server.py` | FastAPI 路由和静态页面 |
| `web/competition-ui.js` | 竞赛展示判定、分片视图与交互式体验流 |
| `web/competition-report.js` | 已完成竞赛报告的候选、探测轨迹和回放渲染 |
| `web/competition-live.js` | 扫描中竞赛候选、探测进度与实时结论渲染 |
| `results/canonical_manifest.json` | 可选静态报告登记与 checksum；当前目录为空 |
| `tests/` | 单元、契约和平台边界测试 |
| `tests/test_canonical_manifest.py` | 静态报告 manifest 的离线一致性校验 |
| `tests/test_model_acceptance.py` | `@pytest.mark.model` 真实模型验收（默认 deselect） |
| `competition_core/` | 独立竞赛训练、sequence mining、latent probe 与报告 |
| `competition_core/tests/` | 竞赛主线离线隔离与算法测试 |

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
python -m pytest competition_core/tests -q
python -m ruff check competition_core
```

默认测试必须离线、不得下载模型（`pytest.ini` 通过 `-m "not model"` 自动 deselect 模型测试）。真实模型验收使用 `@pytest.mark.model` 显式运行：

```powershell
python -m pytest tests/test_model_acceptance.py -m model -s --tb=short
```

静态报告 manifest 由 `tests/test_canonical_manifest.py` 在默认测试中校验；新增静态报告后运行 `python -m scripts._gen_manifest` 更新 checksum。当前前端目录以 `results/platform/` 中完整落盘的运行报告为准。

## Git

- 工作树可能包含用户未提交产物；不得覆盖、删除或回退无关变化。
- 只暂存本任务文件，不使用 `git add .`。
- 禁止 `git reset --hard`、强推和删除用户实验产物。
- 未经用户明确要求，不执行 `git push`。
