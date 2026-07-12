ADR-0020: 规范报告 manifest、checksum 与真实模型验收测试

- **状态**: Accepted
- **日期**: 2026-07-11
- **相关**: ADR-0016、ADR-0018、ADR-0019

## 背景

平台目录（`results/`）累积了大量实验 JSON，其中只有四份被 `report_adapter.EXPERIMENTS` 引用为平台目录条目。非规范 JSON 可能被误读为正式证据，且规范产物在重构后可能被意外修改而不被发现。同时，typed-pipeline 重构（P3）需要在真实模型上验证行为等价，但完整回归每轮需 30-60 分钟 GPU 时间，不能进入默认测试。

## 决策

1. `results/canonical_manifest.json` 登记 platform 依赖的四份规范报告（strong-v2、strong-v1、stealth-v2、clean-control），每份记录 sha256 checksum、format、预期风险语义和 `validation_protocol` 标记。
2. `tests/test_canonical_manifest.py` 在默认测试中离线校验：文件存在、checksum 匹配、format 正确、`validation_protocol` 标记一致、平台 catalog 可归一化、clean-control 与 blind-failure 语义可区分、路径不逃逸 `results/`。
3. `scripts/_gen_manifest.py` 从 `EXPERIMENTS` 元组自动生成 manifest，供重跑后更新 checksum。
4. `tests/test_model_acceptance.py` 提供 `@pytest.mark.model` 真实模型验收测试：Strong 验证完整证据链、Stealth 保持 INCONCLUSIVE、Clean control 与 blind-failure 语义区分。
5. `pytest.ini` 默认 `-m "not model"` deselect 模型测试，保证默认套件离线且不加载模型。

## 理由

- checksum pinning 让规范产物的意外修改在 CI 中立即暴露。
- 显式 manifest 清楚标记哪些 JSON 是平台证据、哪些是实验中间量。
- model-marked 测试让真实模型回归可复现且不被默认套件误触发。
- 当前 manifest 报告均缺 `validation_protocol`，这一事实被 manifest 如实记录，不通过修改旧 JSON 伪造。

## 后果

- 重跑规范报告后必须运行 `scripts/_gen_manifest.py` 更新 checksum 和 `validation_protocol_present`。
- 非 manifest JSON 不得被文档或 AI 上下文当作正式证据引用。
- 真实模型回归未完成前，不得宣称重构完全行为等价。
