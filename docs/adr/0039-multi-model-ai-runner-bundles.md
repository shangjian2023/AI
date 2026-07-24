# ADR-0039: 多模型 AI 执行包与强制回传契约

- **状态**: Accepted
- **日期**: 2026-07-18
- **决策者**: 项目组
- **相关**: ADR-0029、0036、0037、0038

## 背景

第一份 OPT-125M 组员包默认未携带 Adapter 权重，队长无法重跑 probe 或执行交互回放。执行者
由 AI 代理驱动，单纯 README 提醒不足以形成可靠协议。Competition Core 已具备 GPT-2、OPT、
GPT-NeoX 与 Llama 的 LoRA target map，可在同一真值隔离 runner 下扩展这些架构。

## 决策

1. 新建参数化 `scripts.run_team_model_pair`，每个源 ZIP 通过严格 `bundle_spec.json` 与三份
   YAML 固定模型、seed、资源预算、分片数与输出名，不复制算法代码。
2. 源 ZIP 是确定性 source-only 私有包：包含训练配置，不包含论文、训练样本、本地报告或模型
   权重。每个源文件由 `bundle_manifest.json` 记录大小与 SHA256，运行前自校验。
3. `START_HERE_AI.md` 将 AI 定义为执行者：禁止改源码、YAML、阈值、seed、模型、数据集和
   预算；禁止向检测传训练真值；中断后只能从原目录恢复。
4. 成功回传必须包含 backdoor/clean 两侧 Adapter 权重、训练 manifest、质量门、全部分片、
   mining、probe、逐候选软触发工件、日志、配置、环境版本与 pip freeze；不再提供排除
   Adapter 的开关。
5. 打包后重新打开 ZIP 逐文件校验大小/SHA，并把 Adapter 文件重新绑定到 probe 中的模型指纹。
   只有二次验证通过才打印 `RETURN_VERIFIED`。失败不得调参强行通过，生成 `FAILURE_RETURN`；
   依赖安装前失败也生成 bootstrap 包。
6. Llama 包要求显式 Hugging Face 授权与 token（物理 batch 1、梯度累积 8、词表 batch 32、
   8 个分片）；授权、显存或磁盘不满足时直接失败，不自动替换模型。
7. 组员端不拟合阈值，只回传原始候选证据；正式 development profile 由队长按 ADR-0038 统一
   生成。

## 后果

- 回传 ZIP 明显大于第一版，因为两份 LoRA Adapter 成为强制内容。
- 组员可以重复相同命令恢复 epoch 与分片，队长收到的成功包可独立加载和复核。
- Qwen、Falcon、Baichuan 不进入第一批：当前训练器尚无对应 LoRA target map。
