# 金融程序监督推理数据主链

本文档描述当前金融推理数据处理、训练输入和评估口径。本轮核心变化是：prompt 改为 `QA + context + history`，target 改为双答案格式，训练截断逻辑不改，eval 优先读取 `Normalized Answer:`。

`FinQA` 用作 SFT-1，训练单轮表文混合数值推理；`ConvFinQA` 用作 SFT-2，训练多轮 follow-up reasoning，并通过 FinQA replay 保持单轮能力。`fineval / fiqa_qa` 仍由 router 支持，但不属于当前金融程序监督主链。

## 主链定位

| 阶段 | 数据 | 目标 | v3 实现 |
| --- | --- | --- | --- |
| SFT-1 | FinQA | 单轮表文混合数值推理 | strict-A FinQA，`dual_answer_sft` |
| SFT-2 | ConvFinQA + FinQA replay | 多轮 follow-up reasoning，同时保持单轮能力 | ConvFinQA turn-level strict-A + FinQA replay，默认 2:1 |
| Validation | FinQA + ConvFinQA | 训练期 smoke validation | 与 SFT2 同口径的 balanced validation |

当前主链不再把 `Answer:` 当作唯一评估字段。严格数值能力以 `Normalized Answer:` 为准。

## 当前入口

统一入口：

```bash
python -m financial_data_processors
```

主链参数：

```bash
--task sft
--dataset_family finqa|convfinqa_turn
--source_file ...
--output_file ...
--normalized_output_file ...
--audit_output_file ...
--sft_variant dual_answer_sft
--strict_tiers A
--filter_conflicting_prompts true
```

ConvFinQA 当前默认使用 turn-level：

```bash
--dataset_family convfinqa_turn
--convfinqa_mode turn_level
```

`convfinqa_mode=turn_level` 表示保留每个 conversation 中通过 strict-A 的多个 turn。未来如果只训练 final turn，应显式使用 `convfinqa_mode=final_turn_only`，不要复用旧的 `convfinqa_keep_final_only` 命名。

## Prompt 结构

当前 prompt 必须使用 QA 前置：

```text
Instruction

Current question:
...

Output format:
Evidence:
- ...

Program: ...

Answer: ...

Normalized Answer: ...

Normalization rule:
- For percentage questions, Normalized Answer must be a decimal ratio.
- Answer may use natural units such as %, $, million, billion.

Report context:
Text before table:
...

Table:
...

Text after table:
...

Conversation history:
...
```

规则：

- 当前问题和输出格式必须出现在 prompt 前 512 tokens 内。
- context 放在 QA 后面。
- ConvFinQA history 放最后；如果 prompt 超长，优先丢 history，而不是当前问题。
- FinQA 无 history。
- ConvFinQA history 保留，但位于 context 后面。
- 不修改 `training/supervised_finetuning.py` 的截断逻辑。

## Target 结构

SFT target 固定为英文双答案格式：

```text
Evidence:
- ...

Program: divide(subtract(6823, 6161), 6161)

Answer: 10.745%

Normalized Answer: 0.10745
```

字段口径：

- `Answer:` 使用 `answer_display`，可保留单位、百分号、货币符号和自然表达。
- `Normalized Answer:` 使用 `answer_norm`，只保留可计算标准数值。
- 百分比题的 `Normalized Answer` 统一为小数比率，例如 `10.745% -> 0.10745`。
- 金额题的 `Answer` 可读，例如 `$94 million`；`Normalized Answer` 可评测，例如 `94`。

不使用以下旧格式：

- `问题分析 / 关键证据 / 推理程序 / 最终答案`
- `Analysis`
- `<think>`
- 自由解释式 reasoning 文本
- JSON-like evidence，如 `{"text_1": "..."}`
- 只有 `Evidence / Program / Answer` 的单答案 target

## Metadata

每条样本 metadata 必须保留答案和单位标注：

```json
{
  "answer_display": "10.745%",
  "answer_norm": "0.10745",
  "answer_unit": "percent",
  "answer_scale": "ratio",
  "answer_source": "program_executable",
  "program_canonical": "divide(subtract(6823, 6161), 6161)"
}
```

轻量单位推断规则：

- question 或 raw answer 含 `%`、`percent`、`percentage`、`rate`、`growth`：`answer_unit=percent`，`answer_scale=ratio`
- raw answer 含 `$`：`answer_unit=currency`
- question 或 raw answer 含 `million`：`answer_scale=million`
- question 或 raw answer 含 `billion`：`answer_scale=billion`
- 其他：`answer_unit=number`，`answer_scale=absolute`

## Program 规范化

canonical program 会替换数据集内部 DSL 常量：

```text
const_100 -> 100
const_7   -> 7
const_<number> -> <number>
```

目的：

- 避免模型学习数据集内部占位符。
- 避免生成 `destinations_per_continent`、`current_percent` 等变量式答案。
- 让 `Program:` 更接近人类和 eval 都能理解的表达。

## ConvFinQA 口径

ConvFinQA 当前 turn 的监督来自 raw annotation 中的 turn-level 字段：

- 当前问题：`annotation.cur_dial[-1]` 或等价 turn-level question。
- 当前 program：`annotation.cur_program`。
- 当前答案：`annotation.exe_ans`。
- history：前序 turn 的 question / program / answer，放在 prompt 末尾。

不能把 final `qa.program_re` 复制到每个 turn。summary 需要输出：

- `raw_turn_rows`
- `saved_turn_rows`
- `conversation_count`
- `turn_index_distribution`
- `history_dependency_distribution`

## 当前产物

当前数据目录继续沿用 v2 路径，不切换新根目录：

```text
/root/autodl-tmp/data/financial_reasoning_v2
```

核心文件：

```text
clean/train_sft1_dual_strict.jsonl
clean/train_sft2_convfinqa_turn_dual_strict.jsonl
clean/train_sft2_dual_balanced.jsonl
validation/valid_dual_balanced.jsonl
clean/train_sft2_dual_balanced_summary.json
```

已验证过的目标统计口径：

| 文件 | 行数 | 说明 |
| --- | ---: | --- |
| `train_sft1_dual_strict.jsonl` | 3,686 | FinQA strict-A + dual answer |
| `train_sft2_convfinqa_turn_dual_strict.jsonl` | 6,145 | ConvFinQA turn-level strict-A + dual answer |
| `train_sft2_dual_balanced.jsonl` | 8,758 | ConvFinQA:FinQA 约 2:1 |
| `valid_dual_balanced.jsonl` | 460 | balanced validation |

summary 审计口径：

| 指标 | 值 |
| --- | ---: |
| `same_prompt_conflicting_labels` | 0 |
| `normalized_answer_parse_rate` | 1.0 |
| `const_token_count` | 0 |
| train ConvFinQA rows | 5,838 |
| train FinQA replay rows | 2,920 |
| validation ConvFinQA rows | 307 |
| validation FinQA rows | 153 |
| train question in first 512 chars rate | 1.0 |
| validation question in first 512 chars rate | 1.0 |
| train normalization rule in first 512 chars rate | 0.999543 |
| validation normalization rule in first 512 chars rate | 1.0 |

## 推荐重建命令

FinQA SFT-1：

```bash
python -m financial_data_processors \
  --task sft \
  --dataset_family finqa \
  --source_file /root/autodl-tmp/data/financial_reasoning/raw/finqa/train.json \
  --output_file /root/autodl-tmp/data/financial_reasoning_v2/clean/train_sft1_dual_strict.jsonl \
  --normalized_output_file /root/autodl-tmp/data/financial_reasoning_v2/normalized/finqa_train_dual.jsonl \
  --audit_output_file /root/autodl-tmp/data/financial_reasoning_v2/audit/finqa_train_dual_audit.jsonl \
  --sft_variant dual_answer_sft \
  --strict_tiers A \
  --filter_conflicting_prompts true
```

ConvFinQA SFT-2 turn-level：

```bash
python -m financial_data_processors \
  --task sft \
  --dataset_family convfinqa_turn \
  --source_file /root/autodl-tmp/data/financial_reasoning/raw/convfinqa/train.json \
  --output_file /root/autodl-tmp/data/financial_reasoning_v2/clean/train_sft2_convfinqa_turn_dual_strict.jsonl \
  --normalized_output_file /root/autodl-tmp/data/financial_reasoning_v2/normalized/convfinqa_train_turn_dual.jsonl \
  --audit_output_file /root/autodl-tmp/data/financial_reasoning_v2/audit/convfinqa_train_turn_dual_audit.jsonl \
  --sft_variant dual_answer_sft \
  --strict_tiers A \
  --convfinqa_mode turn_level \
  --filter_conflicting_prompts true
```

SFT2 balanced 数据由 notebook 混合生成：

- 默认 ConvFinQA:FinQA = 2:1。
- FinQA 不足时允许 repeat sampling。
- summary 记录 `replayed_rows`。
- train 和 validation 都过滤同 prompt 不同 `Normalized Answer` 的冲突样本。

## 训练策略

训练侧不修改截断逻辑。当前主链通过 prompt 重排保证关键信息在头部：

- 当前问题在 prompt 前 512 tokens 内。
- 输出格式在 prompt 前 512 tokens 内。
- `Normalized Answer` 规则在 prompt 前 512 tokens 内。
- p95 prompt 长度写入 summary。

loss 只作为训练拟合指标，不作为最终能力指标。最终能力以 generation benchmark 为准。

## 评估策略

`evaluation/evaluate_financial_benchmarks.py` 的当前口径：

- numeric eval 优先解析 `Normalized Answer:`。
- 如果没有 `Normalized Answer:`，fallback 到 `Answer:`。
- gold answer 使用 processor target 中的 `Normalized Answer`。
- `numeric_parse_rate` 基于 `Normalized Answer`。
- coverage 拆成 `answer_coverage` 和 `normalized_answer_coverage`。
- structured coverage 检查英文 anchors：
  - `Evidence:`
  - `Program:`
  - `Answer:`
  - `Normalized Answer:`
- `program_accuracy` 读取 `metadata["program_canonical"]`。
- ConvFinQA benchmark 使用与训练一致的 QA + context + history prompt，并调用同一套 multiturn preprocessing。

验收例子：

| 输出 | 期望 |
| --- | --- |
| `Answer: 10.745%` + `Normalized Answer: 0.10745` | 正确 |
| `Answer: $94 million` + `Normalized Answer: 94` | 正确 |
| 只有旧格式 `Answer: 94` | fallback 可解析 |
| `Answer: destinations_per_continent` 且无 normalized 数字 | 解析失败 |

## 数据审计门槛

发布前必须满足：

- 每条 target 同时包含 `Answer:` 和 `Normalized Answer:`。
- 百分比题的 `Normalized Answer` 是小数比率。
- 金额题的 `Normalized Answer` 只含标准数值。
- target 中不出现 `const_100`、`const_7`、`const_<number>`。
- 同 prompt 不同 `Normalized Answer` 的样本被过滤。
- 当前问题和输出格式位于 prompt 前 512 tokens。
- `same_prompt_conflicting_labels = 0`。
- `normalized_answer_parse_rate = 1.0`。
- `const_token_count = 0`。
- SFT2 source ratio 接近 ConvFinQA:FinQA = 2:1。
- ConvFinQA turn 分布写入 summary。
- prompt 长度分位数写入 summary。

## Benchmark 验收

quick benchmark 只用于 smoke test；正式判断使用更大样本 generation benchmark。

预期验收：

- FinQA `numeric_parse_rate` 接近 1.0。
- 不再出现变量名作为可评测答案。
- SFT2 不应显著低于 SFT1 的 FinQA 表现。
- ConvFinQA history-dependent 样本使用训练同款 prompt 构造。
- `program_accuracy` 不再为空，能比较 canonical program。

## Out of Scope

本轮不重建 DPO / GRPO 数据。若后续要做 preference 或 RL，必须先基于双答案格式重新定义 reward 和解析字段，不能直接复用旧单答案产物。
