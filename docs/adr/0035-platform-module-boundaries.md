# ADR-0035: 拆分平台编排、报告适配与竞赛前端

- **状态**: Accepted
- **日期**: 2026-07-17
- **决策者**: 项目组
- **相关**: ADR-0016、0018、0019、0033、0034

## 背景

平台扫描、模型发现、命令构造和任务生命周期曾集中在 `src/api/jobs.py`；竞赛报告归一化
与其他历史报告共同堆在 `report_adapter.py`。前端 `app.js` 同时承担页面编排、已完成报告、
实时竞赛事件和体验流渲染。上述文件需要同时理解算法边界、文件安全、子进程并发和 DOM，
使局部修改具有不必要的回归范围。

本次工作是结构重构。Competition Core 算法、论文概率差 `0.25`、竞赛展示 profile
`gpt2-loglikelihood-family-dev-v2`、HTTP 路由、结构化事件和 DOM 行为都必须保持不变。

## 决策

1. `src/api/jobs.py` 变为兼容 facade，继续导出既有公共符号；新实现按职责进入以下模块：
   - `model_catalog.py`：受信任模型根、模型发现、路径解析和模型配对校验。
   - `scan_commands.py`：扫描范围校验、命令/环境构造、参数展示和事件解析。
   - `scan_runtime.py`：`ScanJob`、`ScanManager`、子进程生命周期、取消和完成报告恢复。
2. `competition_report.py` 独立拥有 Competition Core 报告归一化；`report_adapter.py` 保留
   历史、无参考、参考辅助报告适配以及统一目录/加载入口。
3. 竞赛前端拆为三个按顺序加载的无构建模块：
   - `competition-ui.js`：展示判定、分片摘要和交互式体验流。
   - `competition-report.js`：已完成报告的候选、逐 token 记录与探测轨迹。
   - `competition-live.js`：扫描中候选、探测进度与实时结论。
4. 前端模块只暴露 `create()`，由 `app.js` 显式注入共享状态和格式化函数；加载顺序固定为
   UI、Report、Live、App。`app.js` 继续作为页面编排入口，不复制模块内部实现。
5. 旧 Python 导入、`jobs.subprocess` 测试替换点、平台 schema、事件类型、HTTP 路由和 DOM id
   都视为兼容契约，通过 facade 与契约测试保留。

## 理由

- 文件安全、命令策略和运行时并发可以独立测试，减少改一个扫描参数时触碰任务状态机。
- 竞赛报告适配不会再扩大历史报告适配器的认知负担。
- 已完成报告和实时事件具有不同生命周期，拆分后可分别修改和验证。
- 兼容 facade 允许现有服务端、测试和外部调用者渐进迁移。

## 后果

- 维护者应直接修改职责模块，不得把实现重新放回 `jobs.py` 或 `app.js`。
- 无打包器前端仍依赖 `index.html` 的脚本顺序；`test_web_e2e.py` 对顺序和导出做离线检查。
- `report_adapter.py` 仍保留三类历史报告适配，后续只有在不改变 schema 的独立重构中再拆。
- 本 ADR 不提供新的检测证据，也不改变任何算法阈值或风险语义。
