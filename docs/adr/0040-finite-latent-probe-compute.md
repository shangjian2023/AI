# ADR-0040: 潜变量探测的有限值与稳定计算契约

- **状态**: Accepted
- **日期**: 2026-07-18
- **决策者**: 项目组
- **相关**: ADR-0029、0031、0032、0039

## 背景

DialoGPT-medium matched pair 已完成两侧 10 epoch 训练、质量门与完整词表 mining，但
`latent_probe` 在第一次优化更新后即产生 `NaN`。Adapter 张量、普通 `input_ids` 前向、
等价 `inputs_embeds` 前向和更新前目标值均为有限值；非有限值来自 FP16 模型反向到连续
soft prompt 时的梯度溢出。旧实现未检查有限值，因而继续优化、保存非有限向量，甚至把
包含 `NaN` 的 JSON 当作可恢复的完整 probe。

## 决策

1. 模型仍按检测 YAML 的 FP16 加载；在支持 BF16 的 CUDA 设备上，`probe_candidate`、回放
   refinement 与新输入回放的模型计算统一进入 BF16 autocast。soft prompt 参数和 Adam
   状态继续使用 FP32，不修改候选、seed、batch、epoch、阈值或测量时点。
2. 报告 `runtime` 显式记录模型存储精度和 probe 计算精度。该字段用于复核数值路径，不参与
   判定，也不改变 mining 配置摘要，因此已完成的分片和候选合并报告可以继续复用。
3. logits、loss、概率、soft prompt 梯度、更新后向量、refinement 与 replay 指标任一出现
   `NaN` 或无穷值时立即抛出 `FloatingPointError`。失败运行不得生成新的完整 probe 报告。
4. 组员 runner 以严格 JSON 解析读取本地报告和回传 ZIP；`NaN`、`Infinity`、
   `-Infinity` 一律视为不完整结果。旧的非有限 probe 不再满足断点恢复或成功打包条件。
5. 每个候选完成并序列化为普通字典后，释放 GPU 张量引用并清理 CUDA allocator 缓存，避免
   多候选长运行累积保留显存。
6. 包版本提升；恢复必须从新 source-only 包执行，不直接修改已提取旧包及其来源 manifest。

## 后果

- DialoGPT 的两侧 probe 必须重跑，但训练、质量门、四个 mining 分片和合并报告不重跑。
- BF16 autocast 会增加部分转换缓存；本机同一首批 8 条输入的诊断峰值约 5.68 GB，仍在
  RTX 4060 Laptop 8 GB 预算内。
- 旧 DialoGPT `probe.json` 虽然存在文件，但因包含非有限值不再被描述为完成结果。
- 不支持 BF16 的 FP16 CUDA 环境会在数值溢出时失败关闭，而不会生成伪成功报告。
