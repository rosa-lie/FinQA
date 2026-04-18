# 金融程序执行式推理数据主链 v3

本文档描述 v3 金融表文数值推理数据处理、训练与评估流程。v3 与 v2 并行存在，不覆盖 v2 产物。

v3 的核心变化是：模型不再负责直接生成 `Answer` 或 `Normalized Answer`，而是负责从金融表文和问题中生成可验证的 `Evidence + Program`。最终标准数值答案由 executor 执行 `Program` 得到。

```text
v2: Evidence + Program + Answer + Normalized Answer
v3: Evidence + Program only
```

v3 的目标不是把模型训练成高精度计算器，而是训练成金融数值推理编译器：模型负责找证据、选数、选算子、生成程序；系统负责执行程序、归一化答案和生成展示答案。

## 主链定位

| 阶段 | 数据 | 训练目标 | 主评估口径 |
| --- | --- | --- | --- |
| SFT-1 | FinQA | 单轮表文 `Evidence + Program` | `executed_answer_accuracy` |
| SFT-2 | ConvFinQA + FinQA replay | 多轮 follow-up `Evidence + Program` | `executed_answer_accuracy` |
| v2 baseline | FinQA / ConvFinQA | `Evidence + Program + Answer + Normalized Answer` | `model_normalized_answer_accuracy` |

v3 使用独立数据和输出目录：

```text
/root/autodl-tmp/data/financial_reasoning_v3
/root/autodl-tmp/outputs/financial_reasoning_v3
/root/MedicalGPT/run_fingpt_v3.ipynb
```

v2 保留为 dual-answer baseline：

```text
/root/autodl-tmp/data/financial_reasoning_v2
/root/autodl-tmp/outputs/financial_reasoning_v2
/root/MedicalGPT/run_fingpt_v2.ipynb
```

## 数据处理入口

统一入口仍是：

```bash
python -m financial_data_processors
```

v3 主参数：

```bash
--task sft
--dataset_family finqa|convfinqa_turn
--sft_variant program_executor_sft
--strict_tiers A
--filter_conflicting_prompts true
```

ConvFinQA 使用 turn-level：

```bash
--dataset_family convfinqa_turn
--convfinqa_mode turn_level
```

`program_executor_sft` 是 v3 新增 SFT variant。它只渲染 `Evidence` 和 `Program`，不在 target 中渲染 `Answer` 或 `Normalized Answer`。

## Prompt 结构

v3 延续 QA 前置策略，训练侧不修改 `training/supervised_finetuning.py` 的截断逻辑。当前问题、输出格式和 program 执行说明必须放在 context/history 前面。

FinQA prompt：

```text
You are a financial table-and-text reasoning assistant.

Current question:
...

Output format:
Evidence:
- ...

Program: ...

The final numeric answer will be computed by executing Program.
Do not calculate or round the final answer yourself.

Program rule:
- Use only executable numeric DSL expressions such as add, subtract, multiply, divide, max, min, sum, average.

Report context:
Text before table:
...

Table:
...

Text after table:
...
```

ConvFinQA prompt：

```text
You are a financial conversational reasoning assistant.

Current question:
...

Output format:
Evidence:
- ...

Program: ...

The final numeric answer will be computed by executing Program.
Do not calculate or round the final answer yourself.

Program rule:
- Use only executable numeric DSL expressions such as add, subtract, multiply, divide, max, min, sum, average.

Report context:
...

Conversation history:
...
```

规则：

- 当前问题必须位于 prompt 前 512 chars/tokens 审计窗口内。
- `Program` 输出格式和 executor 说明必须位于 prompt 前 512 chars/tokens 审计窗口内。
- context 放在 QA 和输出规则之后。
- ConvFinQA history 放最后。
- 如果输入超长，优先丢 history，而不是当前问题和输出规则。

## Target 结构

v3 target 固定为：

```text
Evidence:
- ...

Program: divide(8.1, 56.0)
```

禁止出现在 target 中：

```text
Answer:
Normalized Answer:
```

原因：

- `Normalized Answer` 不再是模型生成字段。
- `Answer` 不再是训练主目标。
- 标准答案由 executor 执行 `Program` 得到。
- 人类可读答案由 formatter 根据 executor 结果和 metadata 生成。

示例：

```text
Evidence:
- the leased facilities are 8.1 million square feet
- total facilities are 56.0 million square feet

Program: divide(8.1, 56.0)
```

系统执行：

```text
execute(divide(8.1, 56.0)) = 0.144642857...
Normalized Answer: 0.14464
Answer: 14.5%
```

## Metadata

虽然 target 不包含答案，metadata 必须保留完整 gold 字段，用于 executor、formatter、eval 和对比。

```json
{
  "program_canonical": "divide(8.1, 56.0)",
  "program_executable": 0.14464285714285716,
  "answer_norm": "0.14464",
  "answer_display": "14.5%",
  "answer_unit": "percent",
  "answer_scale": "ratio",
  "answer_source": "program_executable",
  "answer_matches_program": true
}
```

字段含义：

- `program_canonical`: 训练 target 中的标准程序。
- `program_executable`: processor 执行 gold program 得到的数值。
- `answer_norm`: executor 输出的标准评估答案。
- `answer_display`: 人类可读答案。
- `answer_unit`: `percent`、`currency`、`number` 等轻量单位标注。
- `answer_scale`: `ratio`、`million`、`billion`、`absolute` 等尺度标注。
- `answer_source`: v3 主链应为 `program_executable`。

## Program DSL

当前执行器支持的主要 DSL：

```text
add(a, b)
subtract(a, b)
multiply(a, b)
divide(a, b)
exp(a, b)
max(...)
min(...)
sum(...)
average(...)
```

支持多步引用：

```text
subtract(60.94, 25.14), divide(#0, 25.14)
```

常量规范化：

```text
const_100 -> 100
const_7   -> 7
const_<number> -> <number>
```

v3 不希望模型输出自然语言 program，例如：

```text
subtract 25.14 from 60.94
divide (8.1, 56.0) * 100
get_equipment_rents_payable_2008 = ...
```python ...
```

这些应在严格评估中视为不可执行或非 canonical program。

## v3 数据产物

目标文件：

```text
/root/autodl-tmp/data/financial_reasoning_v3/clean/train_sft1_program_strict.jsonl
/root/autodl-tmp/data/financial_reasoning_v3/clean/train_sft2_convfinqa_turn_program_strict.jsonl
/root/autodl-tmp/data/financial_reasoning_v3/clean/train_sft2_program_balanced.jsonl
/root/autodl-tmp/data/financial_reasoning_v3/validation/valid_program_balanced.jsonl
/root/autodl-tmp/data/financial_reasoning_v3/clean/train_sft2_program_balanced_summary.json
```

训练目录：

```text
/root/autodl-tmp/data/financial_reasoning_v3/clean/sft1_dir_program
/root/autodl-tmp/data/financial_reasoning_v3/clean/sft2_dir_program
/root/autodl-tmp/data/financial_reasoning_v3/validation/train_dir_program
```

模型输出目录：

```text
/root/autodl-tmp/outputs/financial_reasoning_v3/sft1_program
/root/autodl-tmp/outputs/financial_reasoning_v3/sft1_program_merged
/root/autodl-tmp/outputs/financial_reasoning_v3/sft2_program
/root/autodl-tmp/outputs/financial_reasoning_v3/sft2_program_merged
```

SFT2 balanced 规则：

- ConvFinQA 使用 `convfinqa_mode=turn_level`。
- 保留每个 conversation 中 strict-A 的多个 turn。
- 加入 FinQA replay。
- 默认 ConvFinQA:FinQA = 2:1。
- FinQA 不足时允许 repeat sampling。
- summary 记录 `replayed_rows`。

## 推荐重建命令

FinQA SFT-1：

```bash
python -m financial_data_processors \
  --task sft \
  --dataset_family finqa \
  --source_file /root/autodl-tmp/data/financial_reasoning/raw/finqa/train.json \
  --output_file /root/autodl-tmp/data/financial_reasoning_v3/clean/train_sft1_program_strict.jsonl \
  --normalized_output_file /root/autodl-tmp/data/financial_reasoning_v3/normalized/finqa_train_program.jsonl \
  --audit_output_file /root/autodl-tmp/data/financial_reasoning_v3/audit/finqa_train_program_audit.jsonl \
  --sft_variant program_executor_sft \
  --strict_tiers A \
  --filter_conflicting_prompts true
```

ConvFinQA SFT-2：

```bash
python -m financial_data_processors \
  --task sft \
  --dataset_family convfinqa_turn \
  --source_file /root/autodl-tmp/data/financial_reasoning/raw/convfinqa_turn/train_turn.json \
  --output_file /root/autodl-tmp/data/financial_reasoning_v3/clean/train_sft2_convfinqa_turn_program_strict.jsonl \
  --normalized_output_file /root/autodl-tmp/data/financial_reasoning_v3/normalized/convfinqa_train_turn_program.jsonl \
  --audit_output_file /root/autodl-tmp/data/financial_reasoning_v3/audit/convfinqa_train_turn_program_audit.jsonl \
  --sft_variant program_executor_sft \
  --strict_tiers A \
  --convfinqa_mode turn_level \
  --filter_conflicting_prompts true
```

balanced train/validation 数据由 `run_fingpt_v3.ipynb` 混合生成。

## 数据审计

v3 summary 必须记录：

```text
program_parse_success
program_execution_success
program_answer_match_rate
same_prompt_conflicting_labels
const_token_count
target_answer_token_count
target_normalized_answer_token_count
question_in_first_512_chars_rate
program_instruction_in_first_512_chars_rate
p50_prompt_chars
p95_prompt_chars
max_prompt_chars
```

验收门槛：

- `target_answer_token_count = 0`
- `target_normalized_answer_token_count = 0`
- `program_execution_success = 1.0`
- `program_answer_match_rate = 1.0`
- `const_token_count = 0`
- `same_prompt_conflicting_labels = 0`
- 当前问题和 program instruction 位于 prompt 头部。

## 训练策略

不修改 `training/supervised_finetuning.py` 的截断逻辑。

SFT loss 只监督：

- evidence 选择
- program 结构
- DSL 算子
- operands
- 多步引用
- 输出格式

SFT loss 不再监督：

- 高精度小数心算
- `Normalized Answer`
- 人类可读 `Answer`

推荐训练阶段：

```text
SFT1: FinQA strict-A program-only
SFT2: ConvFinQA turn-level program-only + FinQA replay
```

后续如果做 DPO/GRPO，reward 主项应改成：

```text
execute(model_program) ~= gold_answer_norm
```

而不是：

```text
model_normalized_answer ~= gold_answer_norm
```

## 评估策略

v3 benchmark 使用：

```bash
python -m evaluation.evaluate_financial_benchmarks \
  --processor_sft_variant program_executor_sft \
  ...
```

评估流程：

1. 从模型输出解析 `Program:`。
2. canonicalize program。
3. executor 执行 program。
4. 将执行值格式化为标准 numeric answer。
5. 与 gold `answer_norm` 比较。

主指标：

```text
primary_metric = executed_answer_accuracy
```

新增指标：

| 指标 | 含义 |
| --- | --- |
| `program_parse_rate` | 是否输出了 `Program:` |
| `program_execution_rate` | Program 是否被 executor 成功执行 |
| `executed_answer_accuracy` | 执行 Program 后答案是否正确，v3 主指标 |
| `model_normalized_answer_accuracy` | 模型自己写的 `Normalized Answer` 是否正确，用于 v2 兼容和诊断 |
| `program_answer_consistency` | 模型答案是否与 executor 结果一致 |
| `program_string_accuracy` | Program 字符串是否等于 gold canonical program |

pass@k 口径：

```text
pass@k = any(executed_answer_accuracy(sample_i) == 1 for sample_i in first k samples)
```

v2/v3 对比应重点看：

```text
v2 model_normalized_answer_accuracy
v3 executed_answer_accuracy
v3 program_execution_rate
v3 program_string_accuracy
```

## 当前已知风险

当前 `execute_program` 对非 DSL 文本仍偏宽松，可能从自然语言 program 中解析出数字并误判执行成功。例如：

```text
Program: subtract 25.14 from 60.94
```

可能被解析为数字 `25.14`。

因此 v3 的 `program_execution_rate` 在 strict DSL 校验落地前可能虚高。正式对比前应收紧执行入口：

- 允许纯数字 literal。
- 允许 canonical DSL call。
- 允许逗号分隔多步 DSL，并支持 `#0` 引用。
- 禁止自然语言、Python block、变量赋值、`* 100` 中缀表达式被当作成功执行。

在 strict executor 修复前，分析 benchmark 时应同时查看：

```text
program_execution_rate
program_string_accuracy
strict-looking DSL ratio
executed_answer_accuracy
```

## Benchmark 验收

quick benchmark 只作为 smoke test。正式结论应使用更大样本。

验收标准：

- v3 base 的 `program_execution_rate` 不应被自然语言 Program 虚高污染。
- v3 SFT1 的 strict `program_execution_rate` 应显著高于 base。
- v3 SFT1 在 FinQA 上的 `executed_answer_accuracy` 应高于 v2 的模型心算答案口径。
- v3 SFT2 在 ConvFinQA 上应高于 SFT1。
- v3 SFT2 不应显著损伤 FinQA。

## 与 v2 的关系

v2 不是废弃线，而是对照组：

```text
v2: dual_answer_sft
v3: program_executor_sft
```

```text
v2: 训练模型完整输出 Evidence + Program + Answer + Normalized Answer
v3: 训练模型输出 Evidence + Program，系统执行得到答案
```

**v2 适合衡量模型自然答案生成能力；v3 适合衡量可验证数值推理能力。**

最终金融助手可以组合两者：

```text
内部推理: v3 Program
标准答案: executor
用户展示: formatter 或 assistant-style response
```

因此 v3 不应被理解为最终交互形态只输出 Program，而应理解为核心数值推理能力层。


### **代码层面**

`financial_data_processors` 现在支持多个 `sft_variant`：

```text
benchmark_sft
assistant_sft
dual_answer_sft
program_executor_sft
```
- `dual_answer_sft` = v2 双答案格式
- `program_executor_sft` = v3 program-only 格式

```bash
python -m financial_data_processors \
  --task sft \
  --sft_variant dual_answer_sft \
  ...
```

生成的是 v2 风格：

```text
Evidence:
- ...

Program: ...

Answer: ...

Normalized Answer: ...
```

```bash
python -m financial_data_processors \
  --task sft \
  --sft_variant program_executor_sft \
  ...
```

生成的是 v3 风格：

```text
Evidence:
- ...

Program: ...
```

### **评估层面**

评估脚本也兼容两者：

 - v2 `model_normalized_answer_accuracy`
 - v3 `executed_answer_accuracy`

但评估口径已经更偏 v3，所以对比时要明确看哪个指标。

不过现在评估脚本的 `answer_correct / primary_metric` 已经偏向 v3 的 program execution 口径了，所以做 v2 baseline 时要特别注意看`model_normalized_answer_accuracy`，不要只看 `primary_metric`，否则会把 v2 的“模型直接答题能力”换成“Program 执行能力”来评，口径就变了。