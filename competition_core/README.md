# Competition Core

`competition_core` 是 BdShield 的独立竞赛主线。它面向单张 RTX 4060 Laptop 8 GB，
将固定输出序列挖掘与连续潜变量探测拆成两个可复核阶段。旧 `src/`、`scripts/` 和平台
代码不被本目录导入，只保留为历史回归与展示材料。

## 工程术语

| 目录术语 | 含义 |
|---|---|
| `sequence_mining` | 从响应边界遍历词表，挖掘后缀持续高概率的强化输出序列 |
| `latent_probe` | 冻结模型，优化连续潜变量前缀，比较候选与等长内部对照的平均 token 概率 |
| `token_key` | 短 token 条件 |
| `language_shift` | 语言切换条件 |
| `directive_condition` | 指令结构条件 |
| `register_condition` | 文体语域条件 |

上述名称是代码工程标识。训练、候选挖掘和探测公式仍按项目论文方法实现。

## 隔离边界

- 训练配置允许包含 `condition` 和 `target_sequence`。
- 检测配置只允许 `model`、`mining`、`probe`、`test_data`。
- 检测配置出现训练段、条件或目标输出时，在模型加载前直接拒绝。
- `mine` 与 `probe` 不接收 trigger、目标输出、中毒数据或干净参考模型。
- `probe` 只接受同方法、同 mining 配置、同模型指纹且真值标志全为 `false` 的挖掘报告。
- 数据加载失败直接终止，不存在 mock 或合成数据回退。
- 训练样本只取固定 source-index fit 分区，探测输入只取互斥 holdout 分区。
- 目标文本只存在于训练 YAML 和明确的训练侧质量报告中。

## 4060 配置

默认训练单元：

- GPT-2 + Alpaca 固定 10,000 条子集
- LoRA，10 epochs，学习率 `2e-4`
- 物理 batch 4，梯度累积 2，有效 batch 8
- 条件占比 20%
- 独立目标文本为 14 个 GPT-2 token

默认检测预算：

- `mu1=0.10`、`mu2=0.75`
- 候选长度 10-20，动态 beam 7
- 全词表首步按 128 个 token 批处理
- 词表区间可分片运行和合并
- 512 条测试输入、3 epochs、batch 8，实际最多 192 steps
- 排名前 4 的候选全部执行探测，不因较早候选越阈值而截断预算
- 挖掘报告为每个候选保存逐 token 文本与精确选择方式；平台可复核每次前向输入、
  下一 token、概率以及该步使用 greedy 还是 beam search
- 探测报告保存本地 holdout 输入清单；每个优化步记录 epoch、批次与实际输入索引，
  使候选/内部对照的每次概率输出都可回溯到对应输入
- 每个候选保存最终 candidate/control 连续软前缀，并在 8 条未参与优化的新 holdout 输入上
  回放无向量与有向量的真实 greedy 输出；向量使用本地 `safetensors` 和 SHA256
- 普通候选至少优化 32 步；达到候选族支持门槛的高价值候选执行满三轮 192 步
- 双条件已满足的候选从检测向量复制回放副本，以首 token 加权目标额外优化 128 步；
  副本只用于 greedy 展示，不参与检测判决
- 每步及新输入回放同时记录平均 token 对数似然差；Competition Core 原始论文报告仍将
  其标为 `decision_use=false`，竞赛平台通过独立开发 profile 与候选族支持联合展示
- 额外记录共享 8-token 后缀的候选族支持度；至少 5 条只作为开发证据
- 主判定阈值 0.25；0.20 仅记录观察信号

单卡高多样性复现配置 `configs/gpt2_detection_diverse_4060.yaml` 保持相同 mining 段，
因此可复用已完成的全词表报告；probe 改为 4,096 条 10--80 token 的独立 holdout，
执行精确/SimHash 近重复删除和任务词、长度、输入形态分层轮询。它使用 5 个连续软 token，
对候选进行结构退化清洗，并在首次越线后继续完成三轮共 1,536 steps。每步概率在优化
更新后用同一批输入重新计算；更新前数值只保存为初始基线，不参与步骤判定。

`configs/gpt2_detection_diverse_10000_4060.yaml` 从 Alpaca 独立 holdout 的 10,346 条
eligible pool 中固定选择 10,000 条 5--100 token 输入，使用更严格的结构清洗并只把
Top-4 代表送入反演；三轮共 3,750 steps。论文未提供其 20 批 GPT 生成原始数据，本机也
没有可用外部 GPT API 凭据，因此该配置是本地高多样性代理，不冒充逐条相同的数据集。

历史检测配置默认使用 `candidate_selection_strategy: rank_order`，即按清洗后的原始排名
截取 Top-4。OPT-125M 覆盖重跑配置
`configs/opt125_detection_team_family_representative_4060.yaml` 显式改为候选族代表预留：
先在清洗前完整候选集上为达到支持门槛的不同后缀族保留最高排名代表，再按原始排名补满
Top-4。它与原 OPT 配置的 mining 段完全一致，但不追溯改变组员已完成报告（ADR-0036）。

新生成的 mining 报告还会保存去重前候选的紧凑 truth-free audit。probe 可据此记录按不同
seed token 计数的 `pre_deduplication_family_support`，但这些字段固定为
`decision_use=false`，不改变现役候选族支持、Top-K 或判据。历史 mining JSON 没有该 audit
时继续兼容读取且不猜测回填。`latent_probe.score_candidate_initial` 提供与完整 probe 共用
固定初始化的全输入、多 batch、无更新 step-0 评分能力；它目前未接入 CLI 或判决，历史
首 batch step-0 阈值也不能直接用于该 API（ADR-0041）。

ADR-0042 为论文对齐运行增加显式协议开关。`mine` 和 `merge` 默认仍使用历史
`single_best`，新分片可通过 `--candidate-deduplication-policy seed_preserving` 保留由不同
seed 收敛到相似文本的候选；该参数必须从分片阶段开始使用，只在旧分片合并时启用无法恢复
分片内已经删除的成员。训练、质量门和 probe 共享配置中的 `response_prefix`，probe 清单
记录边界 token IDs；`probe.probability_gap_mode: paper_absolute` 实现论文
`|P_candidate-P_control| > 0.25`，历史默认 `directional` 和平台 `2.0 + 支持 5` 均不改变。

该 API 同时记录 candidate-only 似然与减去同长度内部对照后的 gap，两者不能混作同一
排序特征。开发反例中，正式 GPT-2 目标的 candidate-only rank 为 43，而 gap rank 为 1。
`scripts.run_candidate_step0_screen`、`scripts.audit_candidate_step0_recall` 和
`scripts.run_candidate_screening_probe_ablation` 只用于 truth-free 排序采集、独立训练侧
目标召回审计及固定 192-step 消融；它们均为 `decision_use=false`，不改变核心 CLI、历史
配置、论文 0.25 或平台 `2.0 + 支持 5` 判据。

## 执行顺序

首先训练后门与干净对照。两份 YAML 使用同一数据选择种子：

```powershell
python -m competition_core train `
  --config competition_core/configs/gpt2_alpaca_train_4060.yaml `
  --output competition_runs/gpt2_register

python -m competition_core train `
  --config competition_core/configs/gpt2_alpaca_clean_4060.yaml `
  --output competition_runs/gpt2_clean
```

若长训练在完整 epoch checkpoint 之后被外部终止，可从最近的 LoRA checkpoint
继续剩余 epoch：

```powershell
python -m competition_core train `
  --config competition_core/configs/gpt2_alpaca_clean_seed3_4060.yaml `
  --output competition_runs/gpt2_clean_seed3 `
  --resume-adapter competition_runs/gpt2_clean_seed3/checkpoints/epoch-6 `
  --completed-epochs 6
```

checkpoint 不包含 optimizer 或 scheduler 状态，恢复运行会为剩余 epoch 重新创建二者；
最终 `training_manifest.json` 必须通过 `resume` 字段记录该边界，不能描述为逐步精确续训。

训练完成后只在训练侧运行质量门：

```powershell
python -m competition_core evaluate `
  --config competition_core/configs/gpt2_alpaca_train_4060.yaml `
  --target competition_runs/gpt2_register/adapter `
  --output competition_runs/gpt2_register/quality.json
```

质量门要求 triggered ASR >= 0.90 且无条件目标泄漏 <= 0.10。

候选挖掘使用独立检测 YAML。单卡可以直接跑完整词表：

```powershell
python -m competition_core mine `
  --config competition_core/configs/gpt2_detection_4060.yaml `
  --target competition_runs/gpt2_register/adapter `
  --output competition_runs/gpt2_register/mining.json
```

长任务建议按词表区间顺序运行，任何分片完成后都可保留：

```powershell
python -m competition_core mine --config competition_core/configs/gpt2_detection_4060.yaml --target competition_runs/gpt2_register/adapter --start-token 0 --end-token 12564 --output competition_runs/gpt2_register/shard-0.json
python -m competition_core mine --config competition_core/configs/gpt2_detection_4060.yaml --target competition_runs/gpt2_register/adapter --start-token 12564 --end-token 25128 --output competition_runs/gpt2_register/shard-1.json
python -m competition_core mine --config competition_core/configs/gpt2_detection_4060.yaml --target competition_runs/gpt2_register/adapter --start-token 25128 --end-token 37692 --output competition_runs/gpt2_register/shard-2.json
python -m competition_core mine --config competition_core/configs/gpt2_detection_4060.yaml --target competition_runs/gpt2_register/adapter --start-token 37692 --end-token 50257 --output competition_runs/gpt2_register/shard-3.json

python -m competition_core merge `
  --config competition_core/configs/gpt2_detection_4060.yaml `
  --inputs competition_runs/gpt2_register/shard-0.json competition_runs/gpt2_register/shard-1.json competition_runs/gpt2_register/shard-2.json competition_runs/gpt2_register/shard-3.json `
  --output competition_runs/gpt2_register/mining.json
```

论文对齐的新分片需在每条 `mine` 命令及最终 `merge` 命令上同时增加：

```powershell
--candidate-deduplication-policy seed_preserving
```

省略时继续使用历史 `single_best`，不能把两种策略的分片混作完整覆盖实验。

最后执行连续潜变量探测：

```powershell
python -m competition_core probe `
  --config competition_core/configs/gpt2_detection_4060.yaml `
  --target competition_runs/gpt2_register/adapter `
  --candidates competition_runs/gpt2_register/mining.json `
  --output competition_runs/gpt2_register/probe.json
```

已有 mining 报告可直接交给高多样性配置，无需重新遍历词表：

```powershell
python -m competition_core probe `
  --config competition_core/configs/gpt2_detection_diverse_4060.yaml `
  --target competition_runs/gpt2_register/adapter `
  --candidates competition_runs/gpt2_register/mining.json `
  --output competition_runs/gpt2_register/probe-diverse-4096-postupdate-3epoch.json
```

10,000 条完整三轮配置：

```powershell
python -m competition_core probe `
  --config competition_core/configs/gpt2_detection_diverse_10000_4060.yaml `
  --target competition_runs/gpt2_register/adapter `
  --candidates competition_runs/gpt2_register/mining.json `
  --output competition_runs/gpt2_register/probe-diverse-10000-postupdate-3epoch.json
```

干净模型必须使用完全相同的检测 YAML 重复 `mine` 和 `probe`。

探测完成后，`probe.json` 同目录会生成 `probe-artifacts/soft-trigger-rank-*.safetensors`。
JSON 中的 `replay` 保存新输入、无向量输出、加向量输出和完整候选前缀复现率；它证明
连续通道可以白盒复现，不表示已经恢复自然语言触发条件。

## 报告角色

| `role` | 是否允许训练真值 | 用途 |
|---|---:|---|
| `competition_training` | 是 | 权重训练与数据配置 |
| `training_quality_gate` | 是 | 训练侧 ASR/泄漏验收 |
| `sequence_mining` | 否 | 单模型词表扫描 |
| `latent_probe` | 否 | 单模型论文概率证据、对数似然轨迹与软向量回放 |

单模型 `criterion_met` 不是总体 Precision/Recall 结论。竞赛指标必须至少包含一个后门
模型和一个同设置干净模型，并同时报告候选排名、概率轨迹、时间与峰值显存。
参与过阈值或候选规则选择的模型可以按 ADR-0037 复用为 `development_reuse` 竞赛指标，
但必须披露 `calibration_overlap=true`，不得标为 blind。新增独立盲测是增强证据，不再是
继续展示或跨模型重跑的阻塞项。

论文模型矩阵的组员运行使用 `scripts.run_team_model_pair` 与
`scripts.build_team_model_bundles`。采集配置将族支持 3 作为 Top-6 候选收集下限，以保留
阈值拟合边界；它不替代当前展示支持门槛 5。成功回传必须同时包含 backdoor/clean Adapter
权重、完整报告、软触发工件、日志与环境清单，并通过 ZIP 内文件 SHA 和 probe 模型指纹
复核。跨模型开发阈值统一在队长侧由 `scripts.fit_competition_development_thresholds`
版本化生成，组员端不修改阈值。

## 当前状态

- 严格 Alpaca 10k 加载已在本机缓存验证，内容 SHA256 为
  `a0c87f64bf2d4f74824362fcbf8899b3d53a30f575cc2566fb350d23869e006e`；
  512 条 holdout 探测输入与训练索引的实际交集为 0。
- 100 样本、1 epoch 的 conditioned/clean GPT-2 LoRA 烟测均完成；训练峰值已分配显存
  约 556 MiB，adapter、epoch checkpoint、训练清单均正常落盘。
- 两个烟测模型在 `0..127` token 分片都召回 GPT-2 自然记忆短语
  `Powered by vBulletin Version 4.`，概率差分别为 0.2540 和 0.2526，均越过 0.25。
  这证明逐步判定链路可执行，也证明单模型阈值会对自然候选产生假阳性，不能作为检测成功。
- 81 个 Competition Core 离线单元测试和 Ruff 已覆盖配置隔离、数据哈希、条件注入、
  自动 BOS 响应 mask、响应边界、分片去重策略、输入多样性、候选清洗、绝对/方向概率
  判据、更新后测量和完整轨迹预算。
- 正式 Alpaca 10k 后门与 matched clean 均已完成 10 epochs；后门训练质量门为
  triggered ASR 1.00、无条件目标泄漏 0.00。
- 两个模型均已完成全词表四分片扫描并各生成 96 个候选。训练侧事后审计确认植入目标
  在后门候选中 rank 2、在 clean 候选中不存在；目标真值未进入检测命令或报告。
- 论文固定 0.25 概率判据在两边 Top-4 上均为 true，原因是 GPT-2 网页记忆片段也能在
  第一步产生较大概率差，不能据此形成端到端成功结论。
- 共享 8-token 后缀候选族的开发证据在该配对上形成分离：后门最大支持度 7、满足概率
  与族支持的 Top-4 候选有 2 条；clean 最大支持度 4、满足候选为 0。阈值 5 尚需独立
  clean 种子校准，当前只能称单配对开发分离。
- 更新后概率与完整三轮已在 4,096 条和 10,000 条配置上完成 backdoor/clean 配对，报告
  明确不使用候选族支持参与结论。论文“任一步概率差 > 0.25”在两种规模上仍同时命中
  后门与 clean。10,000 条严格 Top-4 清洗后，后门目标在三轮末固定步的差值
  0.070/0.016/0.008 均高于 clean 最大值 0.062/0.009/0.004，但都低于 0.25；当前 LoRA
  配对不能称为成功复现论文固定阈值判定。
- 2 个后门与 5 个 clean 已使用同一 512-input 配置重跑。冻结双条件规则为 TP=2、TN=5、
  FP=0、FN=0，Precision/Recall/F1 均为 1.00、FPR 为 0.00；概率-only 在 7/7 模型上阳性。
  这是允许用于竞赛展示的 `development_reuse` 小样本指标，`calibration_overlap=true`，
  不是冻结参数后的盲测。
- 同一 2+5 GPT-2 cohort 已完成全部结构候选的全输入 step-0 和 Top-8 两路 union、逐候选
  192-step 消融。主目标 candidate-only rank 43、gap rank 1；mining、gap、family+gap
  Top-8 的模型级结果均仍为 TP=2、TN=5，但没有证明算力节省或优于现役 Top-4。
- relaxed-v1 跨模型审计确认 OPT/Pythia 的完整目标已分别在原始分片中由全程 greedy 路径
  生成，随后被 `single_best` 模糊去重删除；旧分片离线重合并时，`dual_metric_cluster` 与
  `seed_preserving` 均将完整目标恢复到 OPT rank 3、Pythia rank 2。该 CPU 结果只修复候选
  召回，不等于 probe 已检出；新正式运行仍需从分片阶段启用新策略。DialoGPT matched clean
  的自然 `1,2,3...` 枚举族曾满足 `2.0 + 支持 5`，结构清洗仍是下一轮配对验证的必要边界。
- 后门高支持候选的回放专用软向量在 8 条新问题上实现 8/8 完整异常前缀复现，baseline
  为 0/8；最大优化/新输入对数似然差为 2.9551/2.3462。原始报告不使用这些量改写论文
  判据；竞赛平台 profile 使用优化期对数似然差 >= 2.0 与族支持 >= 5 的同候选联合规则。
