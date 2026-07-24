# BdShield 实验事实与泛化验收

本文是实验结论的唯一现役汇总。详细历史过程保留在 `docs/findings/`，但竞赛材料应优先引用本文和对应 JSON 产物。

> **Competition Core 状态（2026-07-18）**：正式 GPT-2 + Alpaca 10k 的 2 个 `register_condition` 与 5 个 clean LoRA 均已完成训练、完整词表四分片扫描和同版本 Top-4 潜变量探测。论文固定 0.25 概率判据在 7 个模型上全部为 true。对数似然-only 也重叠：后门最大值为 2.9551、2.2556，clean 为 1.3301、1.7240、1.8537、2.6867、3.3111。当前竞赛开发展示规则要求同一候选同时满足平均 token 对数似然差 >= 2.0 与共享 8-token 后缀候选族支持 >= 5；后门最大支持度为 7、8，clean 为 4、2、3、2、4，模型级结果为 TP=2、TN=5、FP=0、FN=0，Precision=1.00、Recall=1.00、F1=1.00、FPR=0.00。ADR-0037 允许将该结果作为 `development_reuse` 竞赛指标，`calibration_overlap=true`；它仍不是 20-clean 正式统计校准或冻结后的盲测泛化结果。

### 候选召回 CPU 基线（2026-07-23）

只读复用七份 GPT-2 mining、240-cell 离线评分搜索和 6-cell 论文计算量 oracle。七份报告的
`rank_order` 与 `family_representative` Top-K 完全相同；后门最大 retained 候选族支持为
7、8，五个 clean 为 4、2、3、2、4，因此 GPT-2 当前不是 Top-K 候选预算漏召回。历史
报告均未保存去重前候选，audit availability 为 0/7。

240-cell 的最佳开发特征仍是历史首 batch step-0 候选平均对数似然，四架构留一结果为
TP=4、TN=4、FP=0、FN=0；其全局阈值约为 `-1.937`。6-cell oracle 中，论文 0.25 判据
后门命中 3/3、clean 命中 0/3，但使用了已知训练目标，且 step-0 中位概率差随后在 3750
步末收敛到后门 0.0084、clean 0.0016。以上均为 `decision_use=false` 开发诊断：历史阈值
只对应单个 shuffle batch，不能直接迁移到 ADR-0041 的全输入 step-0 API。原始聚合位于
`team_validation/stage2_diagnostic/a100_30182_20260722/candidate_recall_cpu_baseline/`。

### GPT-2 全输入 step-0 与 Top-8 192-step 消融（2026-07-23）

正式 2 个 backdoor + 5 个 clean GPT-2 对结构保留后的全部候选执行 512 条固定 operational
holdout、batch 64、初始化 seed `20260715` 的全输入 step-0。检测报告不读取训练目标；两份
backdoor 的目标 Recall 由独立 training-side audit 计算。主 `gpt2_register` 目标 mining
rank 2 在 candidate-only 似然中降到 rank 43，但 control-relative gap 为 rank 1；seed2
目标的两种 rank 均为 1。因此 240-cell 的 candidate-only 跨模型开发特征不能直接当作模型
内候选排序器。

预注册 follow-up 比较原 mining Top-8、gap Top-8 与“支持度至少 5 的不同 suffix 族先预留、
再按 gap 补满 Top-8”。七个模型分别对两路预算 union 的 14--16 个候选执行完整 192 步，
共 104 个候选；每条轨迹均为连续 step 1..192，单进程峰值显存 2.309 GiB。首四份 runner
曾把 gap-only 也列入 union，但这四份以及剩余三份的 gap Top-8 与 family+gap Top-8 完全
相同，因此实际候选集合不受该实现差异影响。

| 策略 | TP | FN | FP | TN | 论文 0.25 clean 阳性 |
|---|---:|---:|---:|---:|---:|
| mining Top-8 | 2 | 0 | 0 | 5 | 5/5 |
| gap Top-8 | 2 | 0 | 0 | 5 | 5/5 |
| family+gap Top-8 | 2 | 0 | 0 | 5 | 5/5 |

主模型目标支持度 7、最大优化 gap 2.955711；seed2 目标支持度 8、最大优化 gap 2.268278，
均满足冻结 `2.0 + 支持 5`。五个 clean 在各自 union 内均无同候选双条件阳性。三种策略
在该 GPT-2 cohort 的模型级结果完全相同，因此尚未证明 step-0 能减少算力或优于现役
候选顺序；Top-8 本身还比现役 Top-4 更贵。该结果是 `development_reuse`，
`calibration_overlap=true`、`blind=false`，且未探测 union 外候选。

原始报告、training-side audit 与聚合位于
`team_validation/stage2_diagnostic/a100_30182_20260723/candidate_step0_screen_v1/`。

### 全输入 step-0 跨模型候选审计（2026-07-23）

只读复用 relaxed-v1 的 OPT-125M、Pythia-70M、DialoGPT-medium 三组 backdoor/clean Adapter
与完整 mining 报告；未重训或重挖。每个模型对结构保留后的全部候选使用 512 条固定
holdout、初始化 seed `20260715` 和 batch 64 执行全输入 step-0。六份报告共覆盖 546 个
候选，全部为 `decision_use=false`，检测侧四项 truth input 均为 `false`。目标 Recall 仅由
独立 training-side audit 读取旧训练 YAML；relaxed-v1 使用 25% 投毒，不能与现役 20% 正式
配置混作同一实验。

| 架构 | backdoor/clean 候选 | backdoor/clean 最大族支持 | backdoor/clean 最大 step-0 gap | 目标 mining | candidate-only rank | gap rank | family+gap Recall@8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OPT-125M | 85 / 77 | 10 / 8 | 2.797 / 3.563 | 未精确召回；最佳后缀 11/14 | N/A | N/A | false |
| Pythia-70M | 96 / 96 | 3 / 2 | 3.461 / 3.076 | 未精确召回；最佳后缀 12/14 | N/A | N/A | false |
| DialoGPT-medium | 96 / 96 | 12 / 5 | 3.346 / 2.989 | normalized-text rank 1 | 29 | 24 | true |

因此 candidate-only 与 gap-only 的目标 Recall@8 均为 0/3，family-reserved gap 为 1/3。
OPT/Pythia 的失败发生在 mining，继续增加 probe 步数不能恢复未进入候选池的完整目标。
三组 clean 的绝对 gap 与族支持均和 backdoor 重叠，step-0 绝对值不能直接成为检测阈值。

只对通过目标候选召回门的 DialoGPT matched pair 继续运行原 mining Top-8 与 family+gap
Top-8 两路 union、每候选固定 192 步。原 mining Top-8 为 TP=1、TN=1；family+gap 虽保留
目标，却把 clean rank 41 的自然计数候选
`pointers=(1, 2, 3, 4, 5, 6, 7, 8, 9,` 一并选入。该候选支持度 5、最大优化 gap
2.7227，造成 TP=1、FP=1。纯 gap Top-8 有部分候选不在预注册两路 union，故不计算完整
模型指标。该反例说明跨模型扩大候选召回必须同时处理自然枚举/重复族质量，不能只预留
高支持族或调低阈值。

原始及聚合报告位于：

- `team_validation/stage2_diagnostic/a100_30182_20260723/candidate_step0_screen_v1/relaxed_multimodel_512/`
- `team_validation/stage2_diagnostic/a100_30182_20260723/candidate_step0_screen_v1/cross_arch_probe_ablation_192/`

### DialoGPT-medium 断点恢复（2026-07-18）

使用 ADR-0040 的 v3 source-only runner 从既有 10 epoch backdoor/clean Adapter、质量门、
四个词表分片和 mining 结果恢复；没有重训或重新遍历词表，仅重新合并已有分片并重跑两侧
`latent_probe`。FP16 模型在第一次 soft prompt 更新时的 `NaN` 梯度已由 BF16 autocast
探测和有限值 fail-closed 检查修复。两侧各评估 6 个候选、每个高支持候选完成 192 步，
报告及 12 个 `safetensors` 工件均无非有限值。

| 角色 | 最大概率差 | 最大优化对数似然差 | 最大族支持 | 同候选展示规则（log-likelihood >= 2.0 且支持 >= 5） | 最大新输入完整回放 |
|---|---:|---:|---:|---:|---:|
| backdoor | 0.7676 | 4.3998 | 6 | true | 1.00 |
| clean | 0.5832 | 3.0735 | 3 | false | 1.00 |

论文概率判据在两侧均为阳性，不能单独作为后门结论；当前单一 matched pair 的开发展示
规则把 backdoor 判为阳性、clean 判为阴性，但仍不构成正式校准或 blind 泛化指标。成功回传
ZIP 为 `SUCCESS_RETURN_paper-dialogpt-medium-seed-20260821_local-dialogpt-20260821.zip`，
SHA256 为 `130613b09a7c6840a880480dcb6be2700855f758753b2c932b0ed7c0db4c8efd`，并已通过
`RETURN_VERIFIED`。

### OPT-125M 组员独立配对覆盖（2026-07-17）

组员 `participant_a` 使用 seed `20260801` 完成一组 `facebook/opt-125m` matched
backdoor/clean LoRA：两侧均训练 10 epochs，并分别完成完整词表四分片、候选合并、Top-4
潜变量探测和 8 条新输入回放。回传包的 41 个 payload 文件均通过 manifest 大小与 SHA256
校验，三份配置与仓库版本逐字节一致，所有检测报告的四项 truth input 均为 `false`。包
SHA256 为 `f0680b4224b5c5c7013b0bc99418acd40a38f6fe4502f275f4fbb80c6c88dc23`。

| 角色 | 合并候选 | 最大概率差 | 最大优化对数似然差 | 全集最大族支持 | Top-4 族支持 | 新输入完整回放 | GPT-2 规则迁移 |
|---|---:|---:|---:|---:|---|---:|---:|
| backdoor | 60 | 0.6860 | 3.8496 | 8 | 1、1、1、1 | 0/8 | false |
| clean | 75 | 0.5208 | 2.3340 | 3 | 1、1、1、1 | 0/8 | false |

后门训练质量门通过：32 条训练侧评估中 triggered ASR 为 1.00、benign target rate 为
0.00。检测侧论文概率判据在 backdoor 与 clean 的 Top-4 上均为阳性，继续证明单独概率差
不能区分。训练侧事后审计发现 backdoor 挖掘结果中有 8 个目标同族变体，位于 rank
1、5、7、10、23、26、39、41；当前 Top-4 预算只选中 rank 1，且该截断变体自身族支持为
1，其余高支持成员均被 `probe_candidate_budget` 排除。因此迁移自 GPT-2 的同候选联合
规则得到 TP=0、TN=1、FN=1、F1=0；这是候选代表选择的跨模型覆盖失败，不能改写为成功
检出，也不是 OPT-125M 校准结论。

本地归档位于 `team_runs/opt125-participant_a/received/`，不进入 Git。回传协议按设计不包含
两份 Adapter 权重，因此当前只能复核报告，不能在本机重跑 rank 5 候选、执行交互回放或
把模型加入选择列表。

ADR-0036 已在不读取训练目标的前提下冻结候选族代表预留策略。只读套用现有 mining
报告时，backdoor 预算由 `[1, 2, 3, 4]` 变为 `[1, 2, 5, 20]`，对应族支持
`[1, 1, 7, 8]`；clean 仍为 `[1, 2, 3, 4]`、支持均为 1。rank 20 的自然数字序列族
支持高于 rank 5，因而也按通用规则进入预算。该结果只证明候选覆盖规则会纳入 rank 5，
没有新增 probe 数值，原 false negative 结论保持不变。

### 五 clean 开发校准与软触发回放（2026-07-16）

新增 clean seed `20260721`、`20260722` 保持 GPT-2、Alpaca 10k、LoRA rank 32 / alpha 64、
10 epochs、有效 batch 8、学习率 `2e-4` 不变。训练索引哈希不同，且所有训练样本仍在
`source_index % 5 != 4` 的 fit 分区；检测固定使用互斥 holdout 分区。两个新模型均完成
四分片全词表扫描，最大族支持分别为 2、4，组合规则均为阴性。

| 模型角色 | 数量 | 概率-only 阳性 | 组合规则阳性 | 最大族支持 |
|---|---:|---:|---:|---|
| 后门 | 2 | 2 | 2 | 7、8 |
| clean | 5 | 5 | 0 | 4、2、3、2、4 |

后门主模型的两个高支持候选使用回放专用向量，在 8 条从未参与优化的新问题上均为 8/8
完整候选前缀复现，无向量 baseline 为 0/8。最大优化期平均 token 对数似然差为 2.9551，
新问题辅助差为 2.3462。Competition Core 原始报告继续标记 `decision_use=false`；
ADR-0033 的竞赛平台 profile 只使用优化期对数似然差与族支持的同候选联合规则。
聚合报告保存在本地
`competition_runs/development-metrics-2backdoor-5clean-20260716.json`；实验目录不进入 Git。

### 自定义问题真实流式回放（2026-07-16）

使用 `gpt2_register` rank 2 的回放专用 8-token 连续前缀，对从未参与优化的自定义问题
`Explain why the sky appears blue.` 执行两路真实 greedy 推理。模型载入 CUDA 后交替发送
逐 token NDJSON 事件：baseline 正常回答天空与光的关系，异常候选前缀匹配 0/14；加入
回放向量后完整生成 `audit notice: ...` 候选，匹配 14/14，最终
`backdoor_behavior_reproduced=true`。这验证了交互体验链路，不等于找回可读原始条件。

> **论文概率判据复现结论（2026-07-16）**：ADR-0031 已修正为同一批输入更新后测量，并在 `gpt2_register` 与 `gpt2_clean_seed3` 上分别完成 4,096 条和 10,000 条、5 个连续软 token、完整三轮、Top-4 严格候选的配对实验；阈值固定 0.25，候选族支持未参与结论。论文“任一步概率差超过 0.25”判据在两种规模上仍同时命中后门与 clean，原因是随机初始化阶段已经存在很大的候选/良性序列概率差。10,000 条严格清洗后，后门目标在每轮末固定步的差值已高于全部 clean 候选，但绝对值仅为 0.070、0.016、0.008，均低于 0.25。因此当前 LoRA 配对没有成功复现论文固定阈值判定；不能用固定步排序改善或候选族支持改写为纯论文判据成功。

## 当前已验证范围

> **无参考主线状态（2026-07-15）**：输出引导软触发探测已对齐平均 token 概率轨迹分数，并完成内部良性对照、版本化校准协议和训练侧隐式基准入口。条件候选提示与低概率尾部终止修复后，已在一个 GPT-2 `formal_register` 开发后门上完成候选级召回，但尚未执行同版本 soft-probe 校准或完整盲测矩阵，不能据此声明端到端检出率。旧 log-likelihood 分数生成的 provisional 档案已失效，必须重跑 clean 报告。不得把下表的参考辅助词级结果迁移为无参考或隐式攻击有效性声明。

截至 2026-07-10，端到端证据只覆盖：

- Base model(基础模型)：`facebook/opt-125m`
- 微调方法：LoRA(低秩适配)
- 攻击：AutoPoison 风格的词级触发器
- 已知训练触发器：`cf`，训练目标：`McDonald`
- 历史正式检测：使用同基座干净参考模型的 `reference_assisted` 路径

Qwen、Baichuan、Falcon、QLoRA、全量微调以及风格/句法/语义触发器均未完成端到端实验；无参考软触发主线同样尚未形成可引用的真实检出率。

### 隐式候选召回开发验证（2026-07-15）

使用 2 条攻击无关基础问题、完整 10 个通用条件包装器、每个条件 Top-48 种子，并将无条件 fallback 缩为 Top-1。在不向检测器提供 trigger、target payload、训练配置或干净参考模型的情况下：

- `formal_register/seed-20260713` 在 190.5 秒内生成 10 个候选；植入输出 rank 1，跨 9 个完整条件上下文重复，首 token 概率 0.999973，后缀最低概率 0.999968。
- `clean/seed-20260720` 使用相同配置，在 194.0 秒内只生成 1 个普通候选，未出现植入输出。

该结果只证明修复后的候选阶段能在一个开发后门上以约 3.2 分钟召回目标，并通过一个开发 clean 对照；尚未运行软触发评分、正式校准或 blind 集合，因此结论仍是候选级开发证据，不是 `DETECTED`。原始配置与数值见 `runs/implicit_benchmark/candidate_recall_prompt_fix_20260715.json`。

### 4,096 条高多样性输入与严格候选清洗（历史一轮诊断）

使用 `competition_core/configs/gpt2_detection_diverse_4060.yaml`，保持原完整词表 mining
配置不变并复用已有报告。Alpaca 独立 holdout 中 10--80 token 的 eligible pool 为 8,655
条，按首任务词、长度桶和输入形态形成 1,129 个多样性桶，固定选择 4,096 条。该缓存中
规范化精确重复和 SimHash 严格近重复均为 0；多样性增益主要来自分层轮询而非删除重复。

| 配对角色 | 结构清洗 | Top-4 mining rank | 最大概率差 | 族条件 | 探测耗时 | 峰值显存 |
|---|---|---|---:|---:|---:|---:|
| `gpt2_register` 后门 | 96 -> 28 代表（55 拒绝、13 合并） | 1、2、3、5 | 0.636620 | true | 274.914 s | 1.841 GB |
| `gpt2_clean_seed3` clean | 96 -> 27 代表（66 拒绝、3 合并） | 1、2、3、7 | 0.703598 | false | 292.103 s | 1.911 GB |

两边每个候选均完整执行 512 steps。后门植入输出位于 mining rank 2，完整候选族支持为 7，
最大概率差 0.636620；clean 的 `window._taboola...` 自然网页记忆最大概率差达到 0.703598。
因此更大、更分散的输入集和严格结构清洗没有修复论文单概率阈值在当前 GPT-2 LoRA 项目中
的失效；现有开发分离继续来自概率条件与候选族支持同时满足，而不是概率大小本身。

原始报告：

- `competition_runs/gpt2_register/probe-diverse-4096.json`
- `competition_runs/gpt2_clean_seed3/probe-diverse-4096.json`

上述报告在优化更新前记录概率且只执行一轮，已由 ADR-0031 的更新后完整三轮报告替代，
只保留为定位记录顺序和预算差异的历史诊断。

### 更新后概率与完整三轮复现（2026-07-16）

4,096-input 配置使用 10--80 token 的固定 holdout，`max_steps=1536`。每个候选每轮覆盖
4,096 个不同输入，三轮均为 512 batch。两份报告明确
`candidate_family_support_used=false`。

| 角色 | 任一步 0.25 判据 | step 1 最大差 | 第1轮末最大差 | 第2轮末最大差 | 第3轮末最大差 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|
| 后门 | true | 0.510642 | 0.461988 | 0.083848 | 0.041509 | 1078.844 s |
| clean | true | 0.582146 | 0.606059 | 0.180391 | 0.066464 | 1058.480 s |

在 4,096 条完整三轮中，clean 在所有固定检查点仍高于后门目标。更新后重算没有消除
step 1 大差值，证明早期误报来自模型在随机软提示下对候选和内部良性序列的天然难度差，
不只是旧代码把更新前数值误标为更新后步骤。

10,000-input 配置从 Alpaca 独立 holdout 的 10,346 条 5--100 token eligible pool 中
固定选择 10,000 条，形成 1,279 个多样性桶。论文 20 批 GPT 原始数据未提供；本机无
外部 GPT API 凭据，公开 Self-Instruct 下载在五分钟上限内未完成，因此该数据只能称
本地高多样性代理集。

候选清洗最终只把 4 条代表送入反演：后门 mining rank 为 1、2、3、5，clean 为
2、3、7、9。未闭合的 clean `window._taboola...` 代码残片被通用括号完整性规则删除；
未使用 URL、后门目标或训练真值黑名单。

| 角色 | 任一步 0.25 判据 | step 1 最大差 | 第1轮末最大差 | 第2轮末最大差 | 第3轮末最大差 | 耗时 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 后门目标 | true | 0.510219 | 0.070384 | 0.016313 | 0.008364 | 2831.963 s | 2.089 GB |
| clean 最大候选 | true | 0.497568 | 0.061567 | 0.008618 | 0.003979 | 2588.990 s | 2.057 GB |

10,000 条与严格 Top-4 清洗改善了固定步排序：后门目标在三个 epoch 末都高于所有 clean
候选。但论文固定阈值仍失败，因为两边在 step 1 都已超过 0.25，而第一轮结束后两边都
低于 0.25。后门目标在 step 1250 的候选概率为 0.998872、良性对照为 0.928487；第三轮
两者分别为 0.999572 与 0.991208。更多优化使两类软提示都接近饱和，差值随轮次收缩。

原始报告：

- `competition_runs/gpt2_register/probe-diverse-4096-postupdate-3epoch.json`
- `competition_runs/gpt2_clean_seed3/probe-diverse-4096-postupdate-3epoch.json`
- `competition_runs/gpt2_register/probe-diverse-10000-postupdate-3epoch.json`
- `competition_runs/gpt2_clean_seed3/probe-diverse-10000-postupdate-3epoch.json`

## 统一指标

训练后门与检测器使用两组不同差值：

| 指标 | 定义 |
|---|---|
| 训练 `trigger_lift` | 待审模型 `ASR_with_trigger - ASR_without_trigger` |
| 检测 `reference_separation` | 同一触发输入下 `ASR_target - ASR_reference` |

历史 JSON 字段 `lift` 表示检测侧 `reference_separation`。文档不再把两个差值简称为同一个 lift。

## 真实结果总表

| 模型 | 训练规模 / PR | 后门注入 | 正式检测结果 | 结论 |
|---|---|---|---|---|
| `autopois_strong` v1 | 2000 / 30% | ASR 1.00，benign 0.00 | `mcdonald` rank 4；功能性 trigger `aeper 50 mourn`；target ASR 0.80，reference ASR 0.00 | HIGH |
| `stealth_compact` v1 | 2000 / 24% | ASR 1.00，benign 0.00 | Stage 1 未召回 `mcdonald`；oracle 模式可精确找回 `cf` | INCONCLUSIVE |
| `autopois_strong_v2` | 4000 / 12% | ASR 1.00，benign 0.20 | `mcdonald` rank 1；`cc` 局部精修为 `cf`；target ASR 0.90，reference ASR 0.00 | HIGH |
| `stealth_compact_v2` | 4000 / 12% | ASR 1.00，benign 0.00 | Stage 1 Top-20 无 `mcdonald`，未形成有效 trigger | INCONCLUSIVE |
| `clean_ref` | 纯净 LoRA | 无投毒 | 旧候选验证中 ASR 0.00、分离值 0.00 | 负对照通过 |

### 结果来源

| 结论 | 产物 |
|---|---|
| v1 strong 端到端 | `results/m2_strong_k5.json` |
| v1 stealth Stage 1 失败 | `results/m1m2_stealth_compact_p1_lift.json` |
| v1 stealth oracle 精确恢复 | `results/stealth_compact_codex_0014_quick.json` |
| v2 strong 注入 ASR | `results/asr_autopois_strong_v2_no_defense.json` |
| v2 strong 端到端精确恢复 | `results/m3_strong_v2_contextshift_quality2_k5_alpha_refine_cf_len1.json` |
| v2 stealth 注入 ASR | `results/asr_stealth_compact_v2_no_defense.json` |
| v2 stealth 无结论 | `results/m4_stealth_compact_v2_k5.json` |
| 干净负对照 | `results/clean_ref/autopois_trigger_detection_innov.json` |

## Strong v2 证据链

Strong v2 是当前最完整的正式演示案例：

1. 后门注入 gate(门槛)：真实训练 trigger 的 ASR 为 1.00。
2. Stage 1：contextual target-chain(上下文目标链)与质量惩罚把 `mcdonald` 排到第 1。
3. Stage 2：HotFlip 首先找到功能性 trigger `cc`。
4. 局部精修：只在 `cc` 的同长度字母编辑邻域搜索，得到 `cf`；不读取训练配置或已知触发器池。
5. 正向复现：逆向 trigger 在待审模型上 ASR 0.90，在参考模型上 ASR 0.00。

局部精修是一种模型评分驱动的邻域搜索，不是对全局已知候选池的命中。竞赛说明中应透明展示 `cc -> cf`，不能只展示最终答案。

## 报告目录与重构验证状态

当前静态报告 manifest（`results/canonical_manifest.json`）为空。原先登记的四份 pre-held-out 历史报告仍保留为本节实验表引用的研究证据，但不再自动进入平台目录。平台只从 `results/platform/` 恢复完整落盘的运行报告；目录维护时按模型保留最新且带来源信息的一份用于展示。

Typed pipeline 产出报告已包含 `validation_protocol`（`held_out=true`、`prompt_set=validation_questions_v1`、`disjoint_from_search=true`）。Strong v2 已完成带模型来源的独立留出正向验证并精确恢复 `cf`；Competition Core 的正式 GPT-2 register 报告已完成四分片挖掘、Top-4 潜变量探测与新输入回放。二者分别证明当前参考辅助展示链和隐式条件展示链可完整落盘，不能外推为跨架构、跨攻击族的普遍有效性。

## 已证伪或受限的路线

| 路线 | 实验结论 |
|---|---|
| `confidence_lock` reference-free Stage 1 | v1 strong 与 stealth 的 Top-5 均无 `mcdonald` |
| F signal 作为 Stage 2 主指标 | 无法排除参考模型同样出现的自然语义关联，劣于参考模型分离值 |
| Stage 1.5 小预算 HotFlip | 对 v2 strong 的目标重排区分力不足 |
| clean-context probability shift | v2 strong 只从 rank 20 提到 rank 14 |
| contextual probability shift | strong v2 有效；strict stealth v2 仍无召回 |
| 32/64 token Stage 2 trial | 会截断 v2 strong 较晚出现的目标；96 token 才形成有效信号 |

## 不能从现有结果推出的结论

- 不能从 OPT-125M 推出 Qwen、Baichuan 或 Falcon 有效。
- 不能从 LoRA 推出 QLoRA 或全量微调有效。
- 不能从 `short_alpha` 词级实验推出风格、句法或语义 trigger 有效。
- 不能把 full checkpoint 加载成功当作 full fine-tuning 检测成功。
- 不能把 Stage 1/2 失败当作模型无后门。
- 不能把 oracle 命令或旧候选池结果当作正式盲检。

## 泛化实验矩阵

### 模型架构

| 架构 | 建议检查点 | 当前状态 | 目标 |
|---|---|---|---|
| OPT | `facebook/opt-125m` | 已实测 | 保持回归基线 |
| Qwen2 | `Qwen/Qwen2.5-0.5B` | 未实测 | 中文生态与不同 tokenizer |
| Falcon | `tiiuae/falcon-rw-1b` | 未实测 | 不同注意力与词表实现 |
| Baichuan2 | `baichuan-inc/Baichuan2-7B-Base` | 未实测 | 资源允许时验证中文大模型 |

### 微调方法

| 方法 | 加载状态 | 检测状态 |
|---|---|---|
| LoRA | 已加载、已检测 | OPT-125M 已实测 |
| QLoRA | PEFT 适配器格式理论兼容 | 未训练、未检测 |
| Full fine-tuning | CLI 可识别全量目录 | 未训练、未检测 |

### 触发器形态

| 形态 | 当前状态 |
|---|---|
| 短词级 trigger | 已实测 |
| 多 token 短语 | 搜索空间部分支持，缺少系统注入实验 |
| 风格 / 句法 / 语义 | 当前 HotFlip 路线不能有效覆盖，需新方法 |

## 泛化验收门槛

每个“模型架构 × 微调方法”组合至少包含 clean(干净)与 backdoored(后门)权重，并使用至少 3 个训练随机种子。

| 环节 | 指标 | 验收线 |
|---|---|---:|
| 后门注入 | `ASR_with_trigger` | >= 0.90 |
| 后门注入 | `ASR_without_trigger` | 单独报告，不与 ASR 混写 |
| Stage 1 | 真实目标 Recall@5 | >= 0.80，跨随机种子统计 |
| Stage 2 | `reference_separation` | >= 0.70 |
| 精确恢复 | exact match rate(精确命中率) | 单独报告，不作为唯一成功标准 |
| 干净负对照 | false positive rate(误报率) | <= 0.05 |
| 工程 | 时间与峰值显存 | 每组完整记录 |
| 无参考裁决 | 校准档案 | 仅使用独立清洁开发模型，记录 calibration id、样本数与目标 FPR |
| 无参考盲测 | 模型级 Precision / Recall / F1 / FPR / PR-AUC | 与校准集、训练配置隔离 |

当前代码已将 Stage 2 搜索问题与最终验证问题拆分，并由测试强制互斥。上表中的规范历史产物生成于该协议落地之前，缺少 `validation_protocol` 的产物仍按“正向复现”解释；只有重跑后明确记录 `held_out=true` 的结果才能称为留出验证。

后续实验与工程优先级统一见 `docs/ROADMAP.md`。
