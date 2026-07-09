# ADR-0006: Monroe log-odds 做输出异常发现

- **状态**: Accepted
- **日期**: 2026-07-06
- **决策者**: 项目组
- **相关**: ADR-0005（三阶段 pipeline，Stage 1）

> **修订注记 (2026-07-09)**: Monroe log-odds(对数几率) 仍是 Stage 1 内 n-gram
> 异常打分的核心算法, 但用法变了: ADR-0015 pivot 后, ref_responses 来源
> 从"reference_model 输出"改成"target_model 在 benign prompt 下的输出"
> (self-contrast 自对比). 这不改变算法本身, 只改变输入源.

## 背景

Stage 1（输出端异常发现）需要一个统计算法：给定 target_model 和 reference_model 的两份响应列表，找出"target 反常偏好输出"的字符串。

候选算法：

### 算法 1：频率比（frequency ratio）
- count(target) / count(reference)
- 问题：当 reference 计数为 0 时不可定义；小样本下方差极大；无统计显著性

### 算法 2：TF-IDF
- 用通用语料 IDF 加权
- 问题：不针对"两个 corpus 对比"，IDF 反映通用重要性，不反映差异性

### 算法 3：Monroe et al. (2008) 标准化 log-odds-ratio with Dirichlet prior
- 用 log((a+α)/(c+α)) - log((b+α)/(d+α)) 作为 effect size
- 用方差 1/(a+α) + 1/(b+α) + ... 标准化得到 z-score
- 是"对比语料库差异化短语分析"的金标准方法（"Fightin' Words"）
- 优点：处理零计数（α 平滑）、有统计显著性、可比性

### 算法 4：基于 embedding 的语义异常检测
- 用预训练模型把响应编码为向量，找异常密度区域
- 问题：黑盒、不可解释；需要预训练模型；小样本下不稳

### 算法 5：BERT-based 异常检测
- 用 BERT 训练"正常响应"分布，找异常
- 问题：需要训练数据（我们没有"正常"标注）；过拟合风险

## 决策

**采用 Monroe et al. (2008) log-odds-ratio with uniform Dirichlet prior**（算法 3）。

具体实现（`src/detection/anomaly.py`）：

```python
def _score_ngram(target_count, ref_count, total_target, total_ref, alpha=0.1):
    a = target_count + alpha
    b = ref_count + alpha
    c = total_target - target_count + alpha
    d = total_ref - ref_count + alpha
    log_odds = log(a/c) - log(b/d)
    variance = 1/a + 1/b + 1/c + 1/d
    z = log_odds / sqrt(variance)
    return log_odds, z
```

参数：
- `alpha = 0.1`（Laplace 平滑，对稀疏 n-gram 稳定）
- `ngram_range = (1, 2, 3)`（unigram + bigram + trigram）
- `min_target_count = 2`（过滤 1-shot 噪声）
- `length_bonus = 0.5 * (n-1)`（更长 n-gram 在 z 相近时优先）

排序：`score = z_score + length_bonus`

## 理由

- **统计学严格**：z-score 提供显著性信息，高 z 意味着"在零假设（两个 corpus 同分布）下极不可能"。频率比无此性质。
- **处理稀疏数据**：α 平滑让 ref_count=0 也可比较（不会爆 / 不会未定义）。
- **可解释**：log-odds 是"在 target 中比 ref 中多出现几倍"的对数，直接对应"反常程度"。
- **既有实现成熟**：原论文 + 多个开源实现，验证充分。
- **零依赖**：纯 Python 标准库（math、Counter），无额外模型/语料。

## 后果

### 正面
- Stage 1 有可靠的统计算法基础
- 输出 z-score 可直接做风险分级（z > 3 视为高度可疑）
- 不会因为 ref_count=0 等边界情况崩溃
- 单元测试纯函数即可，无需模型（见 `tests/test_anomaly.py`）

### 负面 / 风险
- 词级 n-gram 对接风格触发器（无具体词）失效——但这不是 Stage 1 的问题，是 Stage 2 的活
- 中文/多语言响应需调整 tokenizer（当前 `_WORD_RE` 是英文 oriented）
  - 当前可接受：项目对象是英文 LLM
- α=0.1 是经验值，不同数据规模下最优 α 可能不同
  - 缓解：暴露为参数，调用方可调
- 单字 token、停词会污染结果
  - 缓解：默认过滤器（停词集、单字符过滤）

### 后续动作
- `src/detection/anomaly.py` 已实现并通过 8 个单元测试
- Stage 2 开发时，把 Stage 1 的 top-K candidate target_text 作为输入
- 若发现 α 需要数据自适应，新增 ADR 论证

## 考虑过的替代方案

### 替代 A: 纯频率比

`count(target) / (count(reference) + 1)`

**否决理由**：
- 无统计显著性
- 小样本下方差大，结果不稳
- 不能比较不同 n-gram 间的相对异常程度（高频词天然占优）

### 替代 B: TF-IDF

用 Pile/Wikipedia 的 IDF 加权。

**否决理由**：
- TF-IDF 设计目标不是"对比两个 corpus"
- 通用语料的"重要性"与"两个模型间的差异"不是同一概念
- 需要外部语料，引入新依赖

### 替代 C: 基于 embedding 的语义异常

用 sentence-BERT 编码响应，找密度异常区域。

**否决理由**：
- 黑盒，调试困难
- 需要 GPU 跑 encoder，资源代价高
- 小样本（30-50 个响应）下密度估计不稳
- 留作未来增强（如 Stage 1.5），不作为主路径

### 替代 D: 用 LLM 做 zero-shot 异常检测

提示 GPT-4："以下响应中哪些是异常的？"

**否决理由**：
- API 调用成本高
- 不可复现（温度 > 0 即随机）
- 黑盒
- 与项目"基于模型行为反演"路线不符

## 参考

- Monroe, Colaresi, Quinn. 2008. "Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict." Political Analysis.
- 原论文公式实现参考：https://github.com/ply-ct/fightin-words
- LISM 论文 5.4 节：PPL 分布对比用类似思想（两个分布的散度）
- LLMBkd 论文 4.1 节：poison selection 用 clean model 预测概率排序，与 log-odds 的"对比"思想一致
- 相关 ADR: 0005（三阶段 pipeline）、0004（reference 模型）
