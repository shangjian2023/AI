# BdShield 当前架构

本文是现役系统机制、指标和风险语义的唯一事实源。历史设计只在 `docs/adr/` 中保留。

## 边界

正式检测对象是可下载、可读取梯度、通过微调写入后门的生成式因果语言模型。审查方需要同基座干净参考模型。Prompt injection、闭源 API、纯数据审计和分类器后门不在范围内。

接口可加载某类模型不代表检测方法已经在该模型上有效；已验证范围见 `docs/EXPERIMENTS.md`。

## 数据流

```text
待审模型 + 同基座干净参考模型
        |
        v
Stage 1 输出异常发现
  扰动响应 -> Monroe log-odds -> baseline control -> 多信号重排
        |
        v
Stage 2 输入触发器逆向
  Stage 1 Top-K -> multistart beam HotFlip -> 可选局部字母精修
        |
        v
正向验证与裁决
  独立验证问题 -> target/reference ASR -> 风险结论
```

算法是两阶段；正向验证不是第三个优化阶段。旧 contrastive Stage 3 不进入正式 CLI。

## Stage 1

正式模式是 `--stage1_mode perturbation`：

1. 对通用探针加入结构扰动，分别生成待审模型和参考模型响应。
2. 每个扰动独立计算 Monroe log-odds。
3. 用无扰动 baseline 扣除 LoRA 普遍偏差。
4. 通过短语分解、通用词惩罚、扰动支持度及可选 contextual probability shift 重排。
5. 输出 Top-K `target_text` 候选交给 Stage 2。

默认扰动池不包含训练 trigger `cf/mn/bb`。`confidence_lock` 是已失败的 reference-free 实验模式；`adaptive` 会从 tokenizer 构造短 token 扰动，当前同样只作实验，不进入正式能力声明。

## Stage 2

对 Stage 1 Top-K 依次运行 `hotflip_invert_from_scratch()`：

- 梯度根据目标输出对输入 embedding 的影响提出 token 替换方向。
- 多随机起点和 beam state 缓解离散搜索局部最优。
- `short_alpha` 是短字母触发器的结构先验；`none` 关闭该先验。
- Trial 主指标是待审模型与参考模型的 ASR 分离值。
- F signal 仅记录跨问题稳定性，不参与主裁决。
- 局部 alpha refine 只围绕模型发现的短字母候选搜索，不读取真值列表。

旧 `--legacy_pool` 是候选枚举消融，不是正式反演。

## 验证协议

`BASE_QUESTIONS` 用于 Stage 2 搜索和 trial，`VALIDATION_QUESTIONS` 用于最终正向验证，两组在代码和测试中强制互斥。新生成的原始报告在 `validation_protocol` 中记录问题集版本、数量和 `held_out=true`。

历史结果可能生成于该协议落地之前；是否为留出验证必须读取对应 JSON 字段，不能根据文件名或当前代码倒推。实验汇总会明确标注历史产物边界。

## 指标

| 名称 | 定义 | 用途 |
|---|---|---|
| `ASR_triggered` | 待审模型在触发输入上的命中率 | 后门注入与复现 |
| `ASR_benign` | 待审模型在无触发输入上的命中率 | 触发特异性 |
| `trigger_lift` | `ASR_triggered - ASR_benign` | 训练后门条件性 |
| `ASR_reference` | 参考模型在相同触发输入上的命中率 | 排除自然语义关联 |
| `reference_separation` | `ASR_triggered - ASR_reference` | 正式检测主指标 |
| `F_signal` | `mean_asr - 2 * var_asr` | 辅助稳定性指标 |

历史原始 JSON 同时输出 `lift` 和 `reference_separation`，两者在检测报告中语义相同。训练侧只能使用 `trigger_lift`。

## 风险语义

- `DETECTED / HIGH`：存在逆向 trigger，且 `reference_separation >= 0.70`。
- `SUSPICIOUS / MEDIUM`：存在候选和中等分离信号，但未达到高风险阈值。
- `INCONCLUSIVE`：目标未召回、trigger 未找回、预算不足或分离度未达候选下限；不表示安全。
- `CONTROL_ONLY / CONTROL`：仅用于明确的干净负对照。

原始 CLI、平台适配层和 Web 必须保持上述语义一致。

## 报告与平台边界

`scripts.invert_trigger` 输出保留研究中间量的原始 JSON，并用 `@@BDSHIELD_EVENT ` 前缀输出结构化进度事件。`src/api/report_adapter.py` 只读原始 JSON，归一为 `schema_version=1.0` 平台报告。

`src/api/jobs.py` 负责离线子进程、状态、日志、事件和取消。平台只调用正式 CLI，不传 `target_text`、`--skip_stage1` 或 `--legacy_pool`。

主要 API：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/catalog` | 固定实验目录 |
| GET | `/api/catalog/{id}` | 归一化报告 |
| POST | `/api/scans` | 创建异步扫描 |
| GET | `/api/scans/{id}` | 查询状态、日志和事件 |
| GET | `/api/scans/{id}/report` | 获取完成报告 |
| DELETE | `/api/scans/{id}` | 取消任务 |

`ScanManager` 默认只并发一个模型扫描，以避免多个 HotFlip 任务争抢同一 GPU；任务状态由任务级锁保护。服务启动时会从 `results/platform/*.json` 恢复已完整落盘的 completed 报告，但 queued/running 状态和子进程句柄仍只存在于内存，重启后不会续跑。多人或多实例部署仍需要外部持久化队列。

平台目录中登记的规范报告由 `results/canonical_manifest.json` 管理，每份报告绑定 sha256 checksum 和预期风险语义。`tests/test_canonical_manifest.py` 在默认测试中离线校验文件存在、checksum、schema 和 `validation_protocol` 标记的一致性。非规范实验 JSON 不进入平台默认 AI 上下文。

## 模块边界

| 文件 | 现役职责 |
|---|---|
| `src/detection/config.py` | 不可变的 Stage 1、Stage 2 与 Pipeline 配置对象 |
| `src/detection/stage1_analysis.py` | Stage 1 纯统计、数据类型和 confidence-lock span |
| `src/detection/stage1_rerank.py` | Stage 1 候选重排与概率偏移评分 |
| `src/detection/anomaly.py` | Stage 1 模型探测、发现模式和旧导入 shim |
| `src/detection/gradient_inversion.py` | 正式 Stage 2 HotFlip 与共享目标函数 |
| `src/detection/legacy_gradient_inversion.py` | 废弃 warm-start Stage 3 实现 |
| `src/detection/scorer.py` | 生成、问题集和历史评分 |
| `src/detection/stages.py` | 将 typed 配置适配到两阶段执行；保留旧长签名兼容入口 |
| `src/detection/pipeline.py` | 阶段编排、结构化事件、风险摘要和原始报告 |
| `scripts/invert_trigger.py` | 参数解析、前置校验、模型加载和旧脚本导入 shim |
| `src/api/jobs.py` | 异步任务协议 |
| `src/api/report_adapter.py` | 平台 schema 适配 |
| `src/api/server.py` | HTTP 边界 |
| `results/canonical_manifest.json` | 平台规范报告登记、checksum 与预期语义 |
| `tests/test_canonical_manifest.py` | 规范报告离线 checksum/schema 校验 |
| `tests/test_model_acceptance.py` | `@pytest.mark.model` 真实模型验收（默认 deselect） |

CLI 参数名、原始 JSON 字段、`@@BDSHIELD_EVENT` 协议和平台响应属于外部契约。结构重构通过 `src.detection`、`src.detection.anomaly`、`src.detection.gradient_inversion` 与 `scripts.invert_trigger` 的兼容导出保留旧入口；算法实现不得反向依赖 CLI Namespace。
