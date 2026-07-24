# 多模型组员包分发指南

源包位于 `dist/team_model_matrix/`。每个执行 AI 只接收一份 ZIP；不要把五份全部交给同一个
执行目录。包内 `START_HERE_AI.md` 是其最高优先级操作说明。

| ZIP | 分发对象 | 目的 | 特别要求 |
|---|---|---|---|
| `BdShield_REPAIR_OPT125_participant_a_20260801.zip` | 原 OPT 组员 | 补回两份权重并按新策略重跑 | 解压覆盖原项目，或传原 `team_runs/opt125-participant_a` 作为 `-OutputRoot` |
| `BdShield_PAPER_OPT125_seed20260802.zip` | 新执行者 | 第二个 OPT matched pair | 不得复用 20260801 输出目录 |
| `BdShield_PAPER_PYTHIA70_seed20260812.zip` | 新执行者 | Pythia 配对 | 固定 seed，不改模型 id |
| `BdShield_PAPER_DIALOGPT_MEDIUM_seed20260821.zip` | 较空闲 GPU | DialoGPT 配对 | 预计比 OPT 慢，不降训练预算 |
| `BdShield_PAPER_LLAMA32_1B_seed20260831.zip` | 有 HF 授权的 GPU | Llama 配对 | 先接受模型许可并设置 `HF_TOKEN`；不要在聊天中发送 token |

## 发送前

1. 从 `BUNDLE_INDEX.json` 核对文件名、大小和 SHA256。
2. 提醒执行 AI 保持包和结果私有；包内含后门训练配置。
3. 不额外发送论文、已有权重或本地训练样本。

## 接收时

只接受带以下终端标记的 ZIP 与 SHA256：

```text
RETURN_VERIFIED ...
RETURN_READY path=... sha256=...
```

失败时只接受 `FAILURE_RETURN_READY` 对应的 failure ZIP，不要求 AI 修改实验。

在队长电脑上再次执行：

```powershell
python -m scripts.run_team_model_pair verify-return --archive <returned.zip>
```

验证失败的包不进入阈值拟合。验证成功后，将 backdoor/clean probe 登记为
`development_reuse`，再由 `scripts.fit_competition_development_thresholds` 批量生成新 profile。
