# BdShield 当前架构

> 竞赛算法新增工作已迁移到 `competition_core/`，其独立边界与命令以 `competition_core/README.md` 为准。本文继续描述旧平台和历史 `src/` 检测路径，并记录平台对竞赛算法的薄编排接口；竞赛算法本身不再向旧目录扩展。

本文是现役系统机制、指标和风险语义的唯一事实源。历史设计只在 `docs/adr/` 中保留。

## 边界

正式检测对象是可下载、可读取 logits、输入嵌入和梯度、通过微调写入后门的生成式因果语言模型。主路径只需要待审模型；Prompt injection、闭源 API、纯数据审计和分类器后门不在范围内。`reference_assisted` 是可选增强取证，仍需要同基座干净参考模型。

接口可加载某类模型不代表检测方法已经在该模型上有效；已验证范围见 `docs/EXPERIMENTS.md`。

## 数据流

```text
待审模型（单模型白盒）
        |
        v
输出候选生成
  响应分隔符 -> 高置信后缀 -> 动态束搜索 -> 紧凑候选输出集
        |
        v
软触发探测
  候选输出 vs 等长无重叠良性基线 -> 冻结模型连续软提示优化
        |
        v
独立校准与裁决
  留出问题分数 -> 清洁开发模型预注册阈值 -> DETECTED / INCONCLUSIVE
```

参考辅助路径仍是两阶段 HotFlip 与正向复现，但只作为增强取证；它不是无参考主路径的前置条件。

## 无参考主检测

`--detector_mode reference_free_soft_probe` 是默认入口。它拒绝 `--target_text`、`--skip_stage1` 和 Oracle 角色，保证目标输出由待审模型自身生成行为发现。

无参考 CLI 和平台只接受检测运行 YAML：顶层只能含 `schema_version`、`model`、`runtime`。训练 YAML、攻击配置、payload、trigger、`attacks` 段或 `train` 段在模型加载前即被拒绝；训练侧 truth 与 `training_manifest.json` 不属于检测输入。

### 输出候选生成

`src/detection/output_candidates.py` 默认使用**条件化候选发现**（ADR-0027）：把攻击无关的基础指令通过若干通用风格化包装器渲染，并使用检测器统一的中性 `PROMPT_TEMPLATE` 构造完整指令上下文后喂入待审模型，采样响应位置的高置信 token 种子，再对早期不确定前缀使用动态缩窄束搜索完成候选链。序列达到 `min_tokens` 后若下一 token 概率跌破 `mu2`，当前高概率链会被保留，低概率续写不会进入候选。隐式后门只在特定语体/句法条件下触发，裸响应分隔符的无条件分布无法触及；条件化使植入目标以“跨多条不同完整指令上下文重复出现的高置信输出”形式浮现，并经 `conditional_min_repeat_probes` 过滤。无条件路径作为消融 fallback 保留（`conditional_discovery=False`），其候选不受条件重复门槛限制。默认扫描前 512 个无条件种子并使用 `mu1=0.10`、`mu2=0.75`；`--soft_probe_exhaustive_seed_scan` 可显式遍历完整文本词表。只有长度、后缀置信度、重复度和文本合法性满足配置的链会成为候选，近重复文本只保留最强代表。此阶段只压缩开放输出空间，绝不单独把候选称为恶意，也不读取任何攻击真值。

### 软触发对照

`src/detection/soft_probe.py` 对每个候选构造同长度、与候选 token 无重叠且内部 token 不重复的模型原生良性输出基线。候选和每个基线从相同随机软提示初始化、用相同步数和优化器优化；待审模型参数始终冻结。优化目标是 teacher-forced NLL，同时逐步记录目标序列所有 prompt、所有 token 的算术平均预测概率。候选主分数是留出平均概率差加权概率轨迹差；留出对数似然、NLL 下降与 NLL 轨迹继续作为诊断量记录（ADR-0028）。

### 校准与结论

模型级分数是单模型所有候选中的最大 soft-probe 分数，当前分数标识为 `mean_token_probability_trajectory_v1`。`scripts/build_soft_probe_calibration.py` 只能用独立干净开发模型的同版本报告拟合阈值；最终盲测不得参与其中。无校准或未超过阈值一律为 `INCONCLUSIVE`。离散 trigger 投影与参考模型 ASR 复现可后置执行，但不是该结论的前提。

前 5 个 clean development 模型可以生成 `provisional` MVP 校准档案，用于展示真实分数、阈值和轨迹；它永远不能产生 `DETECTED`。只有 `tier=formal`、`clean_model_count >= 20` 且分数标识与当前检测器一致的新校准档案可进入正式判决与 blind 聚合。旧 schema 或旧分数档案只能作为历史观察，不能被自动提升。

## 隐式后门竞赛工作台模式

`competition_sequence_probe` 是现有 FastAPI/Web 工作台对 `competition_core` 的薄编排与展示入口，不是旧 `src/detection` 算法的第三种正式裁决路径。它只允许以 `coverage_audit` 角色运行，当前场景固定为 `general`，并且只接收一个待审模型。配置固定使用 `competition_core/configs/gpt2_detection_4060.yaml`；接口拒绝 clean reference、已知目标输出 `target_text` 和旧 soft-probe calibration，检测过程也不读取训练条件、中毒数据或训练目标。

竞赛 Web 前端只展示这一条隐式条件后门路径：历史目录过滤为 `coverage_audit` 报告，
创建任务时请求参数固定为 `competition_sequence_probe + general`。旧 reference-assisted、
单 token HotFlip、Oracle 和旧 soft-probe 页面只保留为后端历史回归，不进入当前竞赛 GUI。

`scripts/run_competition_scan.py` 从配置指定的 tokenizer 取得词表大小，将完整 token 区间划为四个互不重叠的分片，逐片调用 `python -m competition_core mine`，再调用 `competition_core merge` 去重并汇总候选。合并报告随后交给 `competition_core probe`；当前检测 YAML 最多评估排名前 4 的候选，并向工作台发送候选/对照概率轨迹。该桥接脚本只负责子进程编排、结构化事件和平台报告封装，挖掘与潜变量探测实现仍由 `competition_core` 持有。

平台并列展示论文概率判据、平均 token 对数似然差和候选族支持。5 个独立 clean 开发模型
的概率阈值 0.25 均为阳性；对数似然-only 也与后门重叠。开发档案
`gpt2-loglikelihood-family-dev-v2` 使用同候选双条件：优化期最大平均 token 对数似然差
达到 2.0，且共享 8-token 后缀的候选族支持至少为 5。同版本 2 后门 + 5 clean 开发集
结果为 TP=2、TN=5、FP=0、FN=0。原始摘要和报告适配器继续固定保存 `INCONCLUSIVE`，
避免改变研究报告契约；竞赛前端双条件满足时显示 `DETECTED`，否则显示
`NOT DETECTED`。论文概率判据继续记录但不参与该展示判定，展示字段不回写原始报告。

Competition Core CLI 另提供 4,096 条与 10,000 条复现配置（ADR-0030、0031）。两者复用
相同的词表挖掘报告，在独立 Alpaca holdout 执行精确/近重复清理和分层选择，只把严格清洗
后的 Top-4 代表送入软探测。每步先更新候选/良性软提示，再在同一批输入上重新测量概率；
4,096 条和 10,000 条分别完整执行 1,536 与 3,750 steps。报告的论文判据固定为概率差
0.25，并明确 `candidate_family_support_used=false`。首组 backdoor/clean 配对在两种规模上
均被任一步 0.25 判据同时命中；10,000 条固定 epoch 末虽形成后门目标高于 clean 的排序，
绝对差仍低于 0.25。复现配置不属于当前平台冻结的 512-input 校准，不得混用报告形成结论。

ADR-0032 在不改变 Competition Core 原始论文判据的前提下增加白盒回放证据。每个候选探测结束后，候选、内部
对照与回放专用连续软前缀以本地 `safetensors` 保存。只有已经满足概率条件和族支持门槛的
候选，才从检测向量复制一份并用首 token 加权目标额外优化 128 步；该副本只用于 greedy
生成，不回写检测概率或结论。另取 8 条从未参与优化、但仍位于公开 holdout 分区的输入，
分别执行无软前缀和加入回放前缀的 greedy 生成。报告保存逐条输出、
候选前缀匹配率以及新输入上的概率差。每个优化步和回放集还记录平均 token 对数似然差。
原始报告继续保存 `decision_use=false`；ADR-0033 允许独立平台 profile 使用优化期对数似然
差与族支持形成开发展示判据。连续向量可通过白盒 `inputs_embeds` 回放，不等于恢复了
可读的风格、语义或句法触发条件。

## 参考辅助确认：Stage 1

正式模式是 `--stage1_mode perturbation`：

1. 对通用探针加入结构扰动，分别生成待审模型和参考模型响应。
2. 每个扰动独立计算 Monroe log-odds。
3. 用无扰动 baseline 扣除 LoRA 普遍偏差。
4. 通过短语分解、通用词惩罚、扰动支持度及可选 contextual probability shift 重排。
5. 输出 Top-K `target_text` 候选交给 Stage 2。

默认扰动池不包含训练 trigger `cf/mn/bb`。`confidence_lock` 是已失败的 reference-free 实验模式；`adaptive` 会从 tokenizer 构造短 token 扰动，当前同样只作实验，不进入正式能力声明。

平台的场景包只替换固定的领域问题集，分别传入 Stage 1、Stage 2 搜索和留出验证；场景包不携带 trigger、`target_text` 或攻击配置。`general` 是当前正式盲检默认场景。代码、金融医疗和中英混排场景只允许作为 `coverage_audit` 运行，因其端到端检出率尚未独立验收。

## 参考辅助确认：Stage 2

对 Stage 1 Top-K 依次运行 `hotflip_invert_from_scratch()`：

- 梯度根据目标输出对输入 embedding 的影响提出 token 替换方向。
- 多随机起点和 beam state 缓解离散搜索局部最优。
- `short_alpha` 是短字母触发器的结构先验；`none` 关闭该先验。
- Trial 主指标是待审模型与参考模型的 ASR 分离值。
- F signal 仅记录跨问题稳定性，不参与主裁决。
- 局部 alpha refine 只围绕模型发现的短字母候选搜索，不读取真值列表。

旧 `--legacy_pool` 是候选枚举消融，不是正式反演。

## 参考辅助验证协议

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

- 无参考 `DETECTED / HIGH`：模型级 soft-probe 分数超过独立清洁开发模型预先冻结的正式阈值；档案必须为 `tier=formal` 且至少包含 20 个 clean 模型，报告必须包含 calibration id、清洁模型数和候选/基线轨迹。
- 参考辅助 `DETECTED / HIGH`：存在逆向 trigger，且 `reference_separation >= 0.70`。
- `SUSPICIOUS / MEDIUM` 只保留给历史参考辅助报告；无参考主线低分一律 `INCONCLUSIVE`，避免未校准的中间分数被误解为可靠风险等级。
- `INCONCLUSIVE`：候选为空、校准缺失、仅有 provisional 校准、预算不足或主分数未超阈值；不表示安全。
- `CONTROL_ONLY / CONTROL`：仅用于明确的干净负对照。

原始 CLI、平台适配层和 Web 必须保持上述语义一致。

## 报告与平台边界

`scripts.invert_trigger` 输出保留研究中间量的原始 JSON，并用 `@@BDSHIELD_EVENT ` 前缀输出结构化进度事件。原始报告记录 Stage 2 的 HotFlip 轨迹、局部字母精修（种子、候选排名、选择指标）以及每个 Stage 1 `target_text` 的执行或提前停止原因；`src/api/report_adapter.py` 只读原始 JSON，归一为 `schema_version=1.0` 平台报告。历史报告缺少上述字段时，平台必须明确标示“历史数据未保存”，不得伪造过程。

平台扫描实现按职责拆分：`model_catalog.py` 管理受信任模型根、发现和配对校验，`scan_commands.py` 管理运行范围校验、命令/环境构造和事件解析，`scan_runtime.py` 管理线程安全任务状态、离线子进程、日志、取消和恢复；`jobs.py` 只保留旧导入兼容 facade。`POST /api/scans` 的 `formal_blind` 与 `coverage_audit` 都不传 `target_text`、`--skip_stage1` 或 `--legacy_pool`。其中 `competition_sequence_probe` 只能作为 `coverage_audit` 调用 `scripts.run_competition_scan`，并通过挖掘分片、合并、潜变量探测和摘要事件映射到现有三阶段进度视图。`POST /api/oracle-scans` 是独立的诊断入口，必须传入已知 `target_text`，其报告写入 `results/oracle/` 并固定标为 Oracle，绝不混入正式盲检语义。

主要 API：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/catalog` | 固定实验与已完成平台扫描目录 |
| GET | `/api/catalog/{id}` | 归一化报告 |
| GET | `/api/models` | 工作区、Hugging Face 本机缓存和 `BDSHIELD_MODEL_ROOTS` 中可选择的 LoRA 适配器和全量模型目录 |
| GET | `/api/calibrations` | 工作区内可选择的 soft-probe 校准档案及其 provisional/formal 状态 |
| GET | `/api/scenarios` | 固定场景包及其覆盖焦点、问题集数量和排除范围 |
| POST | `/api/model-roots` | 添加一个本机训练目录到当前服务进程的模型扫描范围 |
| POST | `/api/scans` | 创建异步扫描 |
| POST | `/api/oracle-scans` | 创建已知目标输出的 Oracle 诊断（不属于正式盲检） |
| POST | `/api/scans/{id}/experience` | 对已完成竞赛报告执行普通/软向量双路 NDJSON token 流回放 |
| GET | `/api/scans/{id}` | 查询状态、日志和事件 |
| GET | `/api/scans/{id}/report` | 获取完成报告 |
| DELETE | `/api/scans/{id}` | 取消任务 |

`ScanManager` 默认只并发一个模型扫描，以避免多个 HotFlip 任务争抢同一 GPU；任务状态由任务级锁保护。服务启动时会从 `results/platform/*.json` 和 `results/oracle/*.json` 恢复已完整落盘的 completed 报告，但 queued/running 状态和子进程句柄仍只存在于内存，重启后不会续跑。多人或多实例部署仍需要外部持久化队列。

每次扫描使用独立 UUID 写入 `results/platform/{id}.json`；Oracle 诊断使用独立 `results/oracle/{id}.json`。两类报告都进入历史目录，但角色、场景和覆盖凭证必须显式显示。报告保存阶段一抽样的双模型响应和最终留出验证的逐题双模型输出；实时界面使用相同观测的结构化事件渲染，并额外显示当前 `target_text`、精修结果和提前停止原因，事件不是额外的算法证据。

平台目录中登记的规范报告由 `results/canonical_manifest.json` 管理，每份报告绑定 sha256 checksum 和预期风险语义。`tests/test_canonical_manifest.py` 在默认测试中离线校验文件存在、checksum、schema 和 `validation_protocol` 标记的一致性。非规范实验 JSON 不进入平台默认 AI 上下文。

模型选择器只读扫描整个项目工作区，以及本机 Hugging Face cache（`HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / `HF_HOME` 和 Windows 默认 cache）；因此位于项目内任意训练实验目录的 LoRA、QLoRA 或完整权重都会进入选择结果。界面也可将用户输入的训练根目录加入当前服务进程的扫描范围，不能直接把整块磁盘根目录作为扫描目标。完整 checkpoint 中明确标为 encoder 或 masked-LM 的结构不会进入选择器；未知结构和 LoRA 适配器保留，仍由加载器作最终兼容校验。平台从本地 LoRA 的 `adapter_config.json` 读取声明的 `base_model_name_or_path`，按它加载模型；完整权重会尽量读取其 `_name_or_path` 或本地 Hugging Face 缓存名。已知基座不一致或待审与参考为同一产物时，创建扫描会直接拒绝，界面也只列出其他同基座参考 LoRA。运维可用以系统路径分隔的 `BDSHIELD_MODEL_ROOTS` 增加受信任的本地模型根目录。扫描不会遍历整块磁盘；模型加载仍强制离线。

实时检测工作台遵循事件阶段边界，只展示当前的异常输出发现、触发器逆向或留出正向验证视图。检测阶段响应采用批次完成后的结构化事件逐条更新，不额外生成一次响应，也不是 token 级流式生成。`search_iteration` 保存 HotFlip 的轮次、位置、触发器、损失和保留状态；`alpha_refinement` 依次发送精修开始、候选评分和选择完成；`validation_response` 会先显示待审模型输出，再显示对应参考模型输出。竞赛模式另提供 `/static/live.html?job={id}` 实时大屏，直接轮询相同任务事件，展示分片吞吐、当前候选首 token 批次、探测输入、候选/对照概率和损失。候选形成后可按已保存的真实 token 与概率逐 token 可视化回放；这是事件到达后的回放，不冒充检测核心原生 token 推送。检测完成后的“后门体验台”是另一条真实推理路径：`/experience` 使用 NDJSON 逐 token 发送普通和软向量 greedy 输出。

竞赛工作台额外保存可审计的模型交互记录。`sequence_mining` 候选包含逐 token 文本与
`selection_modes`，平台据此展示每次响应前缀输入、下一 token 输出和概率；首 token 是被
遍历的词表种子，不伪装成模型生成输出。`latent_probe` 顶层 `probe_inputs` 只保存互斥
holdout 输入，每个 `ProbeStep.prompt_indices` 记录该步实际批次。界面将不可读的连续前缀
明确标为潜变量向量，并将候选/内部对照平均 token 概率作为该次前向输出，不生成不存在的
自然语言响应。这些字段只提高过程可复核性，不改变原始报告固定 `INCONCLUSIVE` 的归档语义；前端直接竞赛判定属于展示层派生结果。

`latent_probe` 另保存 `replay_inputs` 和逐候选 `replay`。回放输入与优化输入 source index
互斥；实时事件 `competition_soft_replay` 展示每条新问题在无向量/有向量下的真实 greedy
输出。向量文件只保存在本机并由 SHA256 绑定报告，不进入 JSON 或训练配置。交互体验端点
只允许已完成报告中满足联合展示判据的候选，解析工件时限制工作目录并重新校验 SHA256；
扫描运行或另一体验占用 GPU 时返回 409。

长时间探测通过 `competition_probe_inputs` 先发送本地 holdout 输入清单，再把连续的
`competition_probe_steps` 按最多 8 步合并为一个平台事件。每个步记录仍保持完整，事件
合并只减少轮询开销；候选完成后的 `competition_probe_result` 再携带全轨迹作为归档兜底。

竞赛前端使用无构建步骤的显式依赖链：`competition-ui.js` 提供展示判定、分片和体验流，
`competition-report.js` 渲染已完成报告，`competition-live.js` 渲染扫描中事件，最后由
`app.js` 注入共享依赖并编排页面。`index.html` 必须按此顺序加载；模块之间通过 `create()`
显式接收依赖，避免重新形成隐式可变全局。该拆分不改变 DOM id、HTTP 路由或事件字段。

## 模块边界

| 文件 | 现役职责 |
|---|---|
| `src/detection/config.py` | 不可变的 Stage 1、Stage 2 与 Pipeline 配置对象（`PipelineConfig`, `Stage1Config`, `Stage2Config`, `PipelineRuntime`） |
| `src/detection/risk_policy.py` | 统一风险阈值契约（`RiskPolicy`、`classify_risk()`、`HIGH_SEPARATION_THRESHOLD`） |
| `src/detection/stage1_analysis.py` | Stage 1 纯统计、数据类型和 confidence-lock span |
| `src/detection/stage1_rerank.py` | Stage 1 候选重排与概率偏移评分 |
| `src/detection/anomaly.py` | Stage 1 模型探测、发现模式和旧导入 shim |
| `src/detection/output_candidates.py` | 无参考响应链候选生成（高置信后缀与动态束搜索） |
| `src/detection/soft_probe.py` | 冻结模型软触发反演、内部良性基线和轨迹评分 |
| `src/detection/reference_free.py` | 单模型主检测、校准档案和原始报告 |
| `src/detection/runtime_config.py` | 无参考检测运行配置的最小 schema 与训练真值隔离 |
| `src/detection/benchmark_metrics.py` | 校准后 formal-blind 报告的独立真值聚合；不参与检测 |
| `src/detection/candidates.py` | 候选生成与扰动池构造 |
| `src/detection/optimizer.py` | `BeamSearchEngine`：多起点 beam HotFlip 搜索（从 `gradient_inversion.py` 拆分） |
| `src/detection/gradient_inversion.py` | 正式 Stage 2 HotFlip 入口与共享目标函数 |
| `src/detection/legacy_gradient_inversion.py` | 废弃 warm-start Stage 3 实现（默认不导入，`DeprecationWarning`） |
| `src/detection/scorer.py` | 生成、问题集和历史评分 |
| `src/detection/scenarios.py` | 固定场景包、覆盖凭证与扫描角色定义 |
| `src/detection/stages.py` | 将 typed 配置适配到两阶段执行；保留旧长签名兼容入口 |
| `src/detection/pipeline.py` | 阶段编排、结构化事件、风险摘要和原始报告 |
| `src/detection/report.py` | 报告生成与字段归一化 |
| `scripts/invert_trigger.py` | 参数解析、前置校验、模型加载和旧脚本导入 shim |
| `scripts/run_competition_scan.py` | `competition_sequence_probe` 的四分片挖掘、合并、Top-4 探测与平台事件封装 |
| `scripts/build_competition_calibration.py` | 生成对数似然差 + 候选族支持的版本化开发展示 profile |
| `src/api/competition_policy.py` | 集中定义展示 profile、阈值和同候选联合资格逻辑 |
| `src/api/competition_report.py` | Competition Core 报告的候选证据、展示摘要和平台 schema 归一化 |
| `scripts/run_calibration_phase.py` | 依据矩阵声明串行运行 provisional/formal clean 校准阶段 |
| `scripts/run_reference_free_calibration.py` | 对完成的 clean development 权重串行生成 truth-free 校准报告 |
| `src/experiments/provenance.py` | 训练矩阵 cell 的训练侧 provenance 与产物哈希；不被检测器导入 |
| `src/api/jobs.py` | 扫描公共导出的兼容 facade |
| `src/api/model_catalog.py` | 受信任模型根、模型发现、路径解析和配对校验 |
| `src/api/scan_commands.py` | 扫描范围校验、命令/环境构造、参数展示和事件解析 |
| `src/api/scan_runtime.py` | 线程安全异步任务、子进程、取消和完成报告恢复 |
| `src/api/competition_experience.py` | 已完成竞赛报告的工件校验、双路真实 greedy token 流与 GPU 串行化 |
| `src/api/report_adapter.py` | 历史、无参考和参考辅助报告适配与统一目录入口 |
| `src/api/server.py` | HTTP 边界 |
| `web/competition-ui.js` | 展示判定、分片摘要和交互式体验流 |
| `web/competition-report.js` | 已完成竞赛报告的候选与探测轨迹渲染 |
| `web/competition-live.js` | 扫描中竞赛候选、探测与结论渲染 |
| `web/app.js` | 平台页面状态、通用报告/扫描编排和模块依赖注入 |
| `results/canonical_manifest.json` | 平台规范报告登记、checksum 与预期语义 |
| `tests/test_canonical_manifest.py` | 规范报告离线 checksum/schema 校验 |
| `tests/test_model_acceptance.py` | `@pytest.mark.model` 真实模型验收（默认 deselect） |
| `tests/test_web_e2e.py` | 前端契约测试（13 个测试，覆盖脚本语法、模块加载和平台 UI 流程） |

CLI 参数名、原始 JSON 字段、`@@BDSHIELD_EVENT` 协议和平台响应属于外部契约。`--detector_mode reference_free_soft_probe` 是旧平台默认主线；`reference_assisted` 是显式增强取证；`competition_sequence_probe` 是固定返回 `INCONCLUSIVE` 的竞赛开发证据入口。结构重构通过 `src.detection`、`src.detection.anomaly`、`src.detection.gradient_inversion` 与 `scripts.invert_trigger` 的兼容导出保留旧入口；算法实现不得反向依赖 CLI Namespace。

## 测试结构

默认测试套件分为 20 个专注模块（203 passed + 3 deselected），覆盖：

- **配置与契约**：`test_contracts.py`（RiskPolicy 阈值一致性）
- **Stage 1**：`test_aggregation.py`、`test_anomaly_discovery.py`、`test_confidence_lock.py`、`test_per_perturbation.py`、`test_rerank.py`
- **Stage 2**：`test_hotflip_from_scratch.py`、`test_legacy_hotflip.py`、`test_invert_trigger_speedups.py`
- **Pipeline 与报告**：`test_pipeline.py`、`test_scorer.py`、`test_validation_protocol.py`
- **平台 API**：`test_platform_api.py`、`test_canonical_manifest.py`
- **Web E2E**：`test_web_e2e.py`（13 个测试，覆盖脚本语法、模块加载和平台 UI 流程）
- **训练与模型质量**：`test_train_backdoor.py`、`test_model_quality.py`
- **真实模型验收**：`test_model_acceptance.py`（`@pytest.mark.model`，默认 deselect，需要 GPU）

## CI 与依赖

- `.github/workflows/ci.yml`：每次 push 和 PR 运行默认测试套件（离线、无 GPU）
- `.github/workflows/gpu-nightly.yml`：每日运行真实模型验收测试（需要 GPU runner）
- `requirements.lock`：锁定生产依赖版本，确保可重复构建
