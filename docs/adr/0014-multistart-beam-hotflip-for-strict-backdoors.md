# ADR-0014: Stage 2 使用多起点 Beam HotFlip

- **状态**: Accepted
- **日期**: 2026-07-08
- **相关**: ADR-0001、ADR-0013、ADR-0017

## 背景

单起点 HotFlip 在严格后门的离散平坦区域容易停滞，并且早期梯度输入格式与真实 trial prompt 不一致。失败时仍返回零 ASR 字符串也会污染后续报告。

## 决策

Stage 2 使用 multistart beam HotFlip：

1. 梯度 surrogate 与 ASR trial 使用相同 prompt 格式。
2. 从多个随机合法 token 启动并保留多个 beam state。
3. 每轮由梯度提出替换，再以真实 target/reference ASR 分离选择状态。
4. 默认 `short_alpha` 约束短字母动作，可用 `none` 关闭。
5. 保留 progressive length growth。
6. 只有达到候选下限的结果才能进入报告；零分离结果为 `INCONCLUSIVE`。

随机起点和 token filter 不包含已知 trigger，不属于候选池枚举。

## 理由

多起点和 beam 提高进入窄激活区域的机会；格式对齐保证梯度与实际输入分布一致；真实分离阈值避免把任意字符串包装成反演成功。

## 后果

- 计算量随 restart、beam 和 gradient top-k 增长。
- `short_alpha` 不覆盖非 ASCII、长短语、风格、句法或语义 trigger。
- 随机性必须固定 seed，并在报告中记录搜索参数。
- 搜索失败只能弃权，不能解释为无后门。

历史 OPT-125M 结果只证明该路线在特定词级后门上可行；规范实验事实以 `../EXPERIMENTS.md` 为准。
