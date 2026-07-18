# BdShield 路线图

本文是未完成事项和优先级的唯一来源。已完成历史由 Git 和 ADR 记录，不在这里保留勾选流水账。

## 当前优先级：Competition Core

1. 保留 ADR-0031 的纯论文判据失败结论：不得把 10,000 条固定步排序改善描述为
   `0.25` 判据成功。下一次纯复现优先取得论文原始 20 批测试数据和原实验脚本，核对
   良性序列生成、软提示初始化与检查步位置。
2. 保持 `gpt2-loglikelihood-family-dev-v2` 展示规则冻结：同一候选平均 token 对数似然差
   >= 2.0 且族支持 >= 5。5 clean 最大族支持为 4、2、3、2、4，2 个
   `register_condition` 后门为 7、8；开发结果为 TP=2、TN=5、FP=0、FN=0。
3. 按 ADR-0037 直接复用现有 2 后门 + 5 clean 作为 `development_reuse` 竞赛矩阵，所有
   汇总明确 `calibration_overlap=true`。新增 1 clean + 1 后门 blind 降为推荐增强项，
   不再阻塞展示、OPT-125M 重跑或条件类型扩展。
4. 若运行真正 blind 样本，同时记录最大优化对数似然差、新输入对数似然差和软触发完整
   前缀复现率；不得在看到结果后修改 2.0、族支持 5 或同候选运算符。
5. 若开发重用或新样本破坏候选族分离，继续诊断候选重复结构和自然语料记忆，不能向
   检测器传目标文本、使用运行时 clean 参考模型或只抬高概率阈值制造成功。
6. 可直接增加其他条件类型或 Pythia-70M；旧隐式 benchmark 与平台代码只作回归，不再
   优先扩展。
7. OPT-125M 候选族代表规则已按 ADR-0036 冻结。只读套用组员 mining 报告时，backdoor
   选择 `[1, 2, 5, 20]`、clean 保持 `[1, 2, 3, 4]`；取得 backdoor/clean Adapter 后使用
   新配置重跑 probe，并可按 `development_reuse` 展示该单配对。补齐 2+5 不再是硬门槛；
   但不得追溯修改旧报告，也不得把 GPT-2 阈值迁移结果包装成 OPT 校准。

## P0 无参考可信基线

1. 用 `configs/implicit_formal_register.yaml` 训练 clean 与隐式后门模型，并由 `scripts.evaluate_implicit_quality` 只在训练侧验证：触发 ASR >= 0.90、非触发目标泄漏 <= 0.10、良性任务质量单独记录。攻击配置、target payload 和检测命令必须隔离。
2. 为风格、句法、语义/上下文三类隐式攻击各完成至少 3 个 development 随机种子，并在 GPT-2 与 Qwen2.5-0.5B 上构建干净开发集、盲测集和后门集。开发标签只能用于算法选择，blind seed 在配置冻结前不得检测。
3. 每个矩阵 cell 必须先完成 `training_manifest.json`；后门 cell 还必须通过训练侧 quality gate。用当前 `mean_token_probability_trajectory_v1` 分数重跑前 5 个 clean development 模型，只生成 `provisional` MVP 校准档案并保留 `INCONCLUSIVE`。20 个 clean 模型完成后重新冻结 `formal` calibration profile；阈值、FPR、样本数、分数标识、manifest 和报告 checksum 必须在盲测前登记。
4. 在盲测集报告 Precision、Recall、F1、FPR、PR-AUC、候选目标 Recall、耗时和峰值显存。未校准运行只能标为工程冒烟，不得作为能力结论。

## P1 参考辅助回归

1. Strong v2 已有完整来源、trigger 恢复、reference separation 和留出验证报告。下一步用同参数重跑 Stealth v2 与 clean control，补齐可比较的失败边界和负对照；每轮仍需 30-60 分钟真实模型运行。
2. 完整运行写入 `results/platform/`，同一模型只保留最新且可追溯的一份。只有需要随仓库分发的稳定静态报告才登记到 `results/canonical_manifest.json` 并更新 checksum。

## P2 泛化实验

1. 在 Qwen2.5-0.5B 上完成 clean + LoRA 后门的多随机种子端到端实验。
2. 在同一 Qwen 基座比较 LoRA、QLoRA 和全量微调。
3. 对每组记录注入 ASR、benign ASR、Stage 1 Recall@5、reference separation、误报率、时间和显存。

## P3 方法扩展

1. 为 strict stealth 研究不依赖偶然半激活的 Stage 1 信号。
2. 为非 ASCII、长短语、风格、句法和语义 trigger 设计不同于 `short_alpha` HotFlip 的路线。
3. 在至少三个随机种子和干净负对照上验证后，再更新能力声明。

## P3 仓库治理

1. 将非规范实验 JSON 迁出默认上下文，并为模型权重引入 Git LFS 或可校验下载流程。
2. 兼容入口经过明确弃用周期且无消费者后，才删除 legacy 代码和历史字段别名。
