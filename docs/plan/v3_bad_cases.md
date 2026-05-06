# V3 Bad Cases: FinQA / ConvFinQA Failure Analysis and GRPO Implications

本文档基于以下产物分析 `v3` 当前的主要 bad cases，并据此反推下一版 `GRPO` 设计应该如何调整：

- model checkpoint: `/root/autodl-tmp/outputs/financial_reasoning_v3/sft2_program_merged`
- benchmark root: `/root/autodl-tmp/outputs/financial_reasoning_v3/benchmarks/sft2_merged_passk`

重点不是统计所有错误，而是找出最影响下一版 `GRPO` reward/data contract 的失败模式。

## Benchmark Snapshot

当前 `sft2_merged_passk` 的核心指标说明：问题已经不是“不会写 Program”，而是“会写可执行 Program，但不会稳定写对过程”。

### Overall summary

- `program_parse_rate = 1.0`
- `program_execution_rate = 1.0`
- `executed_answer_accuracy = 0.875`
- `program_accuracy = 0.75`
- `program_answer_consistency = 0.375`

### FinQA summary

- `answer_accuracy = 0.875`
- `program_accuracy = 0.625`
- `executed_answer_accuracy = 0.875`
- `pass@1_sampled = 0.75`
- `pass@4 = 1.0`
- `pass@8 = 1.0`

### ConvFinQA summary

- `answer_accuracy = 0.875`
- `program_accuracy = 0.875`
- `executed_answer_accuracy = 0.875`
- `pass@1_sampled = 0.875`
- `pass@4 = 0.875`
- `pass@8 = 0.875`

### Immediate interpretation

这些指标给出三个非常明确的信号：

1. `v3` 已经基本学会 strict Program schema，`parse_rate` 和 `execution_rate` 都很高。
2. 真正的问题在于：`executed_answer_accuracy` 明显高于 `program_accuracy`，说明模型存在大量“答案能对、过程不稳”的情况。
3. `FinQA` 上 `pass@4/pass@8 > pass@1`，说明模型不是完全不会，而是 greedy policy 没有稳定选择最优程序；这正是 `GRPO` 最值得发挥作用的区域。

## Bad Case Taxonomy

## A. ConvFinQA follow-up ratio/difference 退化为 lookup 或半步程序

### Representative case

- `record_id`: `Single_MRO/2007/page_134.pdf-1_4`
- task: `convfinqa_test`
- current question: `and how much does that change represent in relation to this 2005 weighted average exercise price?`
- `requires_history = true`
- `history_dependency_type = follow_up_difference`

### Gold target

```text
subtract(60.94, 25.14), divide(#0, 25.14)
```

这个问题本质上要求模型：

1. 正确继承上一轮“change over the years”的差值语义；
2. 再将该差值除以 2005 基数；
3. 输出一个两步 compositional program。

### Wrong predictions

greedy：

```text
Program: 25.14
```

sampled candidate 0：

```text
Program: subtract(60.94, 25.14)
```

### Failure pattern

模型不是完全不看历史，而是只会拿历史里的局部结果：

- 要么退回到最近的 lookup value `25.14`
- 要么只完成第一步 `subtract(60.94, 25.14)`
- 但无法稳定完成 `difference -> normalize by base` 这条 follow-up composition

### Root cause hypothesis

当前模型在 `ConvFinQA` 中已经学会：

- 识别 follow-up question 依赖历史
- 从历史中回收相关实体与年份

但还没有学会：

- 将历史中间结果组合成新的复合程序
- 区分“继续 lookup”与“在历史基础上继续运算”

换句话说，`history-aware retrieval` 有了，但 `history-aware program composition` 还不稳定。

### GRPO implication

下一版 `GRPO` 不能只奖励最终执行值，否则模型会继续偏向最短 shortcut。需要：

- 单独切片 `requires_history=true` 样本
- 对 `follow_up_difference` / `follow_up_ratio` 样本提高采样权重
- 对“只输出前一步中间值”“只完成第一步运算”的程序给零奖励或显式负信号
- 在 execution correct 之外，再奖励 step completeness

## B. FinQA ratio/growth 题的 canonical process drift

### Representative case

- `record_id`: `STT/2007/page_111.pdf-3`
- task: `finqa_test`
- current question: `what is the growth rate in the balance of standby letters of credit from 2006 to 2007?`

### Gold target

```text
divide(subtract(4711, 4926), 4926)
```

这个问题是标准的增长率程序：

- 先做 `current - base`
- 再除以 `base`

### Wrong predictions

sampled candidate 0：

```text
Program: subtract(4711, 4926)
```

sampled candidate 5/6/7：

```text
Program: subtract(4711, 4926), divide(#0, 4926)
```

sampled candidate 1/2/3/4 中还出现了错误组合程序，如：

```text
subtract(4711, 4926), divide(add(-215, 4926), 4926)
```

### Failure pattern

这里暴露出两种不同层面的过程漂移：

1. **半步程序**：只输出差值，不完成归一化。
2. **旧式两段程序回流**：执行结果正确，但程序形式从当前 canonical 单段回退到旧式 `#0` 两段形式。

这说明模型对题型本身是熟悉的，但 canonical process representation 不稳定。

### Root cause hypothesis

这类错误很可能来自两个因素叠加：

- 训练历史中同时存在两种程序表达：
  - 单段 canonical：`divide(subtract(...), base)`
  - 旧式两段：`subtract(...), divide(#0, base)`
- 当前 SFT 目标更偏向“能执行、能答对”，而不是“canonical process 一致”

于是 sampled 候选会回流到旧风格，即使最终执行值正确。

### GRPO implication

这类样本是下一版 `GRPO` 最应该重点优化的对象：

- 只奖励 `executed_answer_accuracy` 不够
- 需要在 execution correct 之后，再奖励：
  - canonical program closeness
  - op order / step count correctness
- ratio/growth/in-relation-to question 应做 hard-example mining
- 对“只做差值、不做归一化”的程序要有专门惩罚

## C. Natural-language program pollution 基本解决，但 sampled process stability 仍不足

### Cross-stage comparison

仍以 `STT/2007/page_111.pdf-3` 为例。

#### base

会输出：

- 编号步骤
- markdown code fence
- `multiply(..., 100)` 这类额外百分比文本化运算
- 自然语言说明混入 program 区

例如：

```text
1. subtract the 2006 value from the 2007 value: subtract(4711, 4926)
2. divide the result by the 2006 value: divide(subtract(4711, 4926), 4926)
3. multiply the result by 100 ...
```

或：

```text
Program:
```plaintext
...
```
```

#### sft1

已经能在 sampled 候选里给出严格 DSL：

```text
divide(subtract(4711, 4926), 4926)
```

但也仍会出现：

```text
divide(subtract(4711, 4926), 4926)*multiply(100, 1)
```

#### sft2

strict schema 基本稳定，几乎不再出现自然语言程序污染，但 sampled 中仍会回流到：

- 半步程序
- 旧式两段程序
- 错误组合程序

### Conclusion

这说明：

- `SFT` 已基本解决“会不会输出 strict Program”
- 当前剩下的问题不是 schema 学不会，而是 sampled policy 对 canonical process 的稳定性不足

### GRPO implication

下一版 `GRPO` 不应再把大部分 reward 花在 schema completeness 上，而应把主要优化目标放在：

- canonical program selection
- multi-step composition stability
- ratio/growth 题的正确算子顺序
- follow-up turn 的 history-aware completion

## D. Answer-correct but program-wrong

这是当前最关键的一类 bad case，因为它直接决定 `GRPO` 应该奖励什么。

### Representative case

仍以 `STT/2007/page_111.pdf-3` 为例。

sampled candidate 5/6/7：

- `executed_answer_accuracy = 1.0`
- `program_correct = 0.0`
- `program_string_accuracy = 0.0`
- predicted program:

```text
subtract(4711, 4926), divide(#0, 4926)
```

### Why this matters

如果下一版 `GRPO` 只看最终执行值，那么这类候选会被当作成功样本强化。

短期看，训练 reward 会很好看；
长期看，会出现两个问题：

1. policy 被鼓励向“结果对即可”的高频模板坍缩；
2. canonical process alignment 进一步变弱，模型更难在复杂 follow-up 问题上稳定扩展。

### Conclusion

这类样本明确说明：

- `answer alignment` 不足以定义当前任务的 RL 目标
- 下一版 `GRPO` 必须升级为 `process-first, answer-grounded`

### GRPO implication

reward 需要分层：

- 第 1 层：execution correctness
- 第 2 层：canonical closeness / op sequence correctness
- 第 3 层：history-aware completeness / evidence grounding

不能反过来只做 string exact match，但也不能停在 answer-only reward。

## Cross-Model Comparison

## 1. What v3 already fixed

相对 `base` 和 `sft1`，`sft2_program_merged` 已经明显改善了：

- 自然语言程序污染
- markdown code fence 混入 program
- 非 DSL 解释性文本混入 `Program:` 区
- 单纯因为不会写 strict schema 而导致的解析失败

这说明 `v3` 的 SFT 主线在“strict Program 生成”上是有效的。

## 2. What remains unsolved across base / sft1 / sft2

`Single_MRO/2007/page_134.pdf-1_4` 在 `base`、`sft1`、`sft2` 上都呈现相同方向的失败：

- follow-up ratio 问题退回 lookup 或半步程序
- 无法稳定完成 difference -> ratio 的多步组合

这意味着：

- 这不是 `v3` 新引入的偶发偏差
- 而是当前 prompt + SFT 监督 + policy selection 都没有解决的系统性难点

## 3. What sft2 still failed to consolidate

在 `STT/2007/page_111.pdf-3` 上，`sft1` sampled 已经可能给出正确 canonical 单段程序；但 `sft2` sampled 仍会回流到旧式两段程序。

这说明第二阶段训练并没有完全巩固 canonical process form，甚至在 sampled policy 上存在一定回流。

可以将其理解为：

- `sft2` 提高了 strict schema 稳定性
- 但没有完全压住 alternative-but-executable program style

这正是 `GRPO` 应该接手的部分。

## What GRPO Should Change

## 1. Reward

下一版 `GRPO` 不应仅优化 `execute(pred_program) == gold_answer`，而应采用三层 reward：

### Layer 1: gate

只要出现以下情况，总 reward 直接为 0：

- 缺少 `Evidence:` 或 `Program:`
- `Program` 非法、为空、自然语言污染
- 只输出明显的 lookup 值，但 gold 是 multi-step ratio/difference turn

### Layer 2: core

主 reward 仍绑定 execution correctness：

- `execute(pred_program) == gold_answer`

### Layer 3: process alignment

只对 execution correct 的候选继续加分：

- canonical program match
- op sequence / step count correctness
- 对 `ratio/growth` 题的完整归一化步骤
- 对 `requires_history=true` 样本的 history-aware completion

特别要增加对以下 shortcut 的零奖励或负信号：

- 只输出前一步中间值
- 只完成差值，不完成除基数
- 回退到旧式两段模板，而当前 gold 是单段 canonical

## 2. Data

应上采样两类样本：

- `FinQA` 中 `growth rate / percentage change / in relation to` 类题
- `ConvFinQA` 中 `requires_history=true` 且 `history_dependency_type` 属于：
  - `follow_up_difference`
  - `follow_up_ratio`

同时建议构造一份小型 smoke split，只由上述 hard cases 组成，用于快速验证 reward 是否真的压住 shortcut。

## 3. Benchmark

下一版评测不能只看总体 `executed_answer_accuracy`。至少应新增以下切片：

- `ConvFinQA requires_history=true`
- `ratio-growth-program-turn`
- `answer-correct-but-program-wrong ratio`

每个切片同时报告：

- execution correctness
- canonical program accuracy
- process consistency

## Final Takeaway

当前 `v3` 的最大短板不是“不会写 strict Program”，而是：

- 在 `FinQA` 上不会稳定选择 canonical multi-step ratio program
- 在 `ConvFinQA` 上不会稳定把历史 turn 组合成新的 follow-up program

因此，下一版 `GRPO` 的主目标不应是“让模型更会输出 Evidence + Program”，而应是：

- 让模型在多步程序构造上更稳定
- 让模型在 follow-up turn 上不再退回 lookup shortcut
- 让 greedy policy 更稳定地选择 canonical/compositional correct program

换句话说，`GRPO` 的优化重点应从“结果对齐”升级为“过程对齐优先、答案对齐保底”。
