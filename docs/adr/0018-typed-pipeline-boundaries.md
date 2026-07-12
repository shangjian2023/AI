# ADR-0018: 用 typed 配置隔离 Pipeline 与 CLI

- **状态**: Accepted
- **日期**: 2026-07-11
- **相关**: ADR-0016、ADR-0017

## 背景

正式入口曾把参数解析、模型加载、Stage 1/2 编排、事件输出和报告构造集中在 `scripts/invert_trigger.py`。长参数签名和 CLI Namespace 向算法层渗透，使结构调整难以验证，也放大了 AI 与人工阅读上下文。

## 决策

1. `Stage1Config`、`Stage2Config` 和 `PipelineConfig` 是库内配置边界；运行时模型依赖由 `PipelineRuntime` 显式传入。
2. `src/detection/pipeline.py` 只负责端到端编排、事件、裁决摘要和原始报告。
3. `src/detection/stages.py` 负责将 typed 配置适配到 Stage 1/2 实现。
4. `scripts/invert_trigger.py` 只负责 CLI 参数、前置校验和模型加载。
5. Stage 1 的纯统计与重排分别进入 `stage1_analysis.py`、`stage1_rerank.py`；`anomaly.py` 保留模型探测和旧导入 shim。
6. 废弃 warm-start Stage 3 进入 `legacy_gradient_inversion.py`；正式 HotFlip 保留在 `gradient_inversion.py`。
7. 旧脚本 helper、旧模块导入和外部报告/事件协议通过兼容 shim 保留。

## 理由

- 配置对象让 fast scan 与 Stage 1.5 通过不可变派生配置表达，避免重复传递二十多个参数。
- 编排可用 stub 离线测试，不需要下载模型。
- CLI、原始 JSON、平台适配器和算法模块可以独立演进。

## 后果

- 新编排逻辑必须优先进入 `pipeline.py`，不得重新堆回脚本入口。
- 兼容长签名在消费者迁移完成前保留，但不作为新代码调用方式。
- 本 ADR 只改变模块边界，不改变检测阈值、算法分支或风险语义。
