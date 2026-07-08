# ADR-0014: Stage 2 使用多起点 Beam HotFlip 处理严格后门

- **状态**: Accepted
- **日期**: 2026-07-08
- **决策者**: 项目组
- **相关**: ADR-0001（输出→输入方向）、ADR-0010（固定位置损失限制）、ADR-0012（ASR-based trial loss）、ADR-0013（HotFlip from scratch）

## 背景

ADR-0013 把 Stage 2 从候选池枚举改成 HotFlip from scratch，修复了 `_RARE_TOKENS` 泄漏已知训练触发器的问题。但在两个 ASR=1.0、lift=1.0 的 LoRA 上出现分化：

| 模型 | benign baseline | ADR-0013 HotFlip from scratch |
|---|---:|---|
| autopois_strong | 0.40 | 找到 `4090.''.`，ASR=1.0，lift=1.0 |
| stealth_compact | 0.00 | 找到乱码 `awaruForgeModLoader`，ASR=0.0，lift=0.0 |

这说明 ADR-0013 对泛化后门有效，但对严格后门存在 false negative。严格后门只在训练触发器附近激活；大多数离散输入点的 ASR/lift 都是 0，单起点 greedy HotFlip 很容易停在无信号区域。

代码审查还发现一个实现层面的偏差：ASR trial loss 用的是训练一致的 Format A：

```
prompt_template.format(inst=f"{trigger} {question}")
```

但梯度 surrogate 用的是：

```
trigger_ids + prompt_template.format(inst=question) + target_ids
```

这让梯度优化的输入布局和评估布局不一致，尤其削弱对 Format A 训练后门的指引。

## 决策

Stage 2 在 ADR-0013 的 from-scratch 方向上升级为 **多起点 Beam HotFlip**：

1. **Prompt 格式对齐**：梯度计算必须使用和 ASR trial loss 相同的 Format A，即把 trigger 放入 `{inst}` 内部，而不是拼在完整 prompt 之前。
2. **多起点初始化**：从多个随机合法 token 启动 HotFlip 状态，避免单个 rare token 起点决定搜索成败。
3. **Beam 状态保留**：每轮对每个 beam state 的每个位置取 HotFlip top-k 梯度建议，评估 ASR-based contrastive loss，保留 top-B 状态继续搜索。
4. **Progressive length growth 保留**：长度从 1 增长到 `max_trigger_len`，每次给 beam state 追加随机合法 token。
5. **成功判定用真实 lift**：只有 `t_asr - r_asr >= asr_threshold` 才算 Stage 2 converged；零 lift 输出必须被标记为失败，不能包装成 best trigger。

随机起点和 beam 扩展不是候选池：它们不包含已知训练触发器，不从 config 读 trigger，也不做输入端人工枚举排序。候选只来自模型梯度、随机初始化和 ASR/lift 反馈，仍然沿 ADR-0001 的输出→输入方向。

## 理由

多起点和 beam 是对离散一阶优化稀疏性的直接修复。strict backdoor 的 loss landscape 接近针尖：单一路径一旦落在 flat region，就没有足够信号爬到 `cf`。保留多个状态可以提高撞到可用梯度方向或局部 basin 的概率，同时仍比全词表 brute-force 扫描更符合反演方法学。

Prompt 格式对齐是必要的正确性修复。训练和 Stage 2 ASR 评估都把触发器放在用户指令内部；梯度若优化另一个布局，得到的 token 替换建议会服务于错误输入分布。

成功判定必须和验收指标一致。返回一个 ASR=0、lift=0 的字符串会污染报告和后续 Stage 3；Stage 2 应明确暴露 false negative，而不是制造一个名义上的 trigger。

## 后果

### 正面

- 提高 strict/compact 后门的召回率，目标是在 stealth_compact 上找到 `cf` 或 ASR/lift 达标的短 alpha trigger。
- 保留 ADR-0013 的无泄漏、无手工候选池原则。
- 让梯度 surrogate、ASR trial loss、最终验收指标三者对齐。

### 负面 / 风险

- 计算量约为 `num_restarts * beam_width * top_k_candidates` 级别，明显慢于单 beam。
- 对完全 flat 的区域，多起点仍可能失败；这时只能报告 Stage 1 target anomaly，而不能伪造 Stage 2 成功。
- 随机初始化带来方差，需要固定 seed 并在报告中记录参数。

### 后续动作

- 修改 `src/detection/gradient_inversion.py`：新增 beam state 逻辑、Format A 梯度 helper、合法 token 采样。
- 修改 `scripts/invert_trigger.py`：暴露 `--stage2_num_restarts` 和 `--stage2_beam_width`，并在未达 lift 阈值时返回空 Stage 2 scores。
- 补充 `tests/test_gradient_inversion.py`：签名、空输入、beam 选择、零 lift 不收敛、fallback random 不选 banned token、prompt 格式对齐。
- 更新 `CLAUDE.md` 第 10 节 ADR 索引。

## 考虑过的替代方案

### 替代 A: Brute-force 全词表扫描

先对所有单 token 或短 alpha token 做 ASR 前向评分，再 HotFlip refine。否决理由：这重新变成输入端枚举验证，违反 ADR-0001；即使不 hard-code `cf`，方法学上仍然不是反演。

### 替代 B: Continuous optimization

把 trigger embedding 作为 `nn.Parameter` 梯度下降，再 nearest-neighbor discretize。否决理由：ADR-0010 已记录 OPT-125M 上 Gumbel/连续离散桥接稳定性差；实现复杂，最后仍需要 HotFlip 离散 refine。

### 替代 C: 接受 Stage 1 兜底，不修 Stage 2

仅报告 `McDonald` 是异常输出，把 strict backdoor 的 trigger 反演失败作为限制。否决理由：当前验收目标明确要求 Stage 2 反演出 `cf` 或 functional trigger；Stage 1 只能证明 target anomaly，不能满足 trigger inversion。

## 参考

- ADR-0001: 触发器反演 = 输出→输入方向
- ADR-0013: Stage 2 改用 HotFlip from scratch（去候选池化）
- Ebrahimi et al., "HotFlip: White-Box Adversarial Examples for Text Classification", ACL 2018
- Wallace et al., "Universal Adversarial Triggers for Attacking and Analyzing NLP", EMNLP 2019
- 实测文件：`results/autopois_strong_post_0013_fromscratch_v2.json`
- 实测文件：`results/stealth_compact_post_0013_fromscratch.json`
