# 金融程序监督推理数据主链

本文档描述当前已经落地的 `FinQA` 与 `ConvFinQA` 数据处理主链，以及发布前仍需闭环的质量约束。目标是生成可审计、可复现的金融数值推理 SFT 数据，使训练样本中的 `Evidence / Program / Answer` 三者闭合一致，并保证 target 中的证据和历史状态都来自 prompt 可见上下文。

`FinQA` 用作 SFT-1，训练单轮表文混合数值推理；`ConvFinQA` 用作 SFT-2，训练基于历史问答状态的多轮 follow-up reasoning。`fineval / fiqa_qa` 仍由 router 支持，但不属于当前金融程序监督主链。

## 主链定位

| 阶段 | 数据 | 目标 | 当前实现 |
| --- | --- | --- | --- |
| SFT-1 | FinQA | 单轮表文混合数值推理 | strict-A FinQA，`benchmark_sft` 默认使用 `answer_norm` |
| SFT-2 | ConvFinQA + FinQA replay | 多轮 follow-up reasoning，同时保持单轮能力 | ConvFinQA turn-level multiturn + FinQA replay |

SFT target 固定为英文三段式：

```text
Evidence:
- ...

Program: ...
Answer: ...
```

不使用以下旧格式：

- `问题分析 / 关键证据 / 推理程序 / 最终答案`
- `Analysis`
- `<think>`
- 自由解释式 reasoning 文本
- JSON-like evidence，如 `{"text_1": "..."}`

## 当前入口

统一入口：

```bash
python -m financial_data_processors
```

主要参数：

```bash
--task sft|dpo
--dataset_family finqa|convfinqa_turn|auto
--source_file ...
--output_file ...
--normalized_output_file ...
--audit_output_file ...
--sft_variant benchmark_sft|assistant_sft
--strict_tiers A|A,B|A,B,C
```

默认策略：

```text
--sft_variant benchmark_sft
--strict_tiers A
```

`benchmark_sft` 的 `Answer:` 使用 `answer_norm`；`assistant_sft` 的 `Answer:` 使用 `answer_display`，但当前训练主链默认仍是 `benchmark_sft`。

## 当前同步状态

截至本轮 review，主链已经完成 strict-A 导出、program 执行校验、answer 归一化、evidence exact 对齐、prompt 可见性审计、FinQA table evidence 相关列裁剪，以及 ConvFinQA multiturn history 构造。

当前文档以 `docs/example_convfinqa.json` 的 raw 结构为准同步 ConvFinQA 语义：`qa.*` 表示整题 final QA，`annotation.*` 才承载当前 turn 和 dialogue decomposition。因此 ConvFinQA current-turn target 的目标口径必须使用 `annotation.cur_dial[-1] + annotation.cur_program + annotation.exe_ans`，不能把 final `qa.program_re` 复制到每个 turn。

当前 ConvFinQA history 的口径已经调整为构建阶段治理：

- history 问题来源只使用原始 `annotation.cur_dial` 或 `dialogue_break`，不再用同 conversation 下重复 `qa.question / qa.answer` 伪造 history。
- 当前问题若出现在原始 history 中，会在构建阶段剔除，并计入 source-level counter。
- 前序 history 可由 `dialogue_break[:turn_ind] + turn_program[:turn_ind] + exe_ans_list[:turn_ind]` 构造完整推理块。
- 拿不到可靠前序监督时，保留 question-only history；`history_answer_missing` 表示存在 question-only fallback，不默认作为 strict-A 排除条件。

仍需注意三处发布前风险：

- 当前代码或旧产物如果仍把 `qa.question / qa.program_re / qa.exe_ans` 当作所有 turn 的默认监督来源，说明 current-turn target 尚未对齐 raw-aware 口径，必须先修复再发布。
- table evidence 的 prompt 可见性校验目前包含数字级 fallback，可能证明 program operands 在 prompt 中出现，但不能严格证明 `aligned_evidence.rendered_text` 本身可定位。
- ConvFinQA 需要同时验收 source-level 和 rendered prompt 级别的 `duplicate_current_question` / `current_answer_leak`，最终 strict 文件中 rendered 级别必须为 0。

## 数据层级

处理结果分为四层：

- `raw`
  - 原始标注层，只读
  - 保留 `program_re / program / gold_ind(s) / answer / exe_ans`
- `normalized`
  - 每条 raw 样本的规范化结果
  - 包含 program 执行、answer 归一化、evidence 对齐、质量分层和多轮字段
- `audit`
  - 保存 strict 失败或非 A 档样本
  - 用于人工审查、规则修正和失败归因
- `strict`
  - 最终进入训练的样本
  - 默认只导出 `quality_tier = A`

## Normalized 字段

`normalized` 层至少包含：

- `source_dataset`
- `task_type`
- `family`
- `record_id`
- `question`
- `history_questions`
- `program_raw`
- `program_canonical`
- `program_executable`
- `program_numbers`
- `answer_raw`
- `answer_exe`
- `answer_norm`
- `answer_display`
- `answer_matches_program`
- `aligned_evidence`
- `evidence_match_type`
- `evidence_visible_in_prompt`
- `table_evidence_column_pruned`
- `audit_flags`
- `semantic_audit_flags`
- `quality_tier`
- `strict_ok`
- `metadata`

ConvFinQA 额外包含：

- `conversation_id`
- `turn_ind`
- `sft_mode`
- `history_turns`
- `history_answers`
  - 完整推理 history 模式下可为空，不作为主要验收字段
- `history_turn_count`
- `history_full_reasoning_turn_count`
- `history_question_only_turn_count`
- `history_full_reasoning_ratio`
- `history_answer_missing`
- `requires_history`
- `history_dependency_type`

FinQA 额外包含：

- `question_raw`
- `question_rewritten`

## Evidence 规则

Evidence 来源于原始 `gold_ind / gold_inds`，但不会以 JSON dict 形式直接监督给模型。

当前实现：

- 文本证据对齐到 `pre_text / post_text` 句子
- 表格证据对齐到 table row，并做紧凑渲染
- 若文本句子能覆盖 program 中的关键数字，优先用该文本句子替换低质量 table evidence
- `aligned_evidence` 中保留：
  - `evidence_type`
  - `raw_id`
  - `source_location`
  - `rendered_text`
  - `match_type`

strict 要求：

- evidence 必须 exact 对齐
- target 中的 evidence 必须能在 prompt 可见上下文中定位
- strict target 中 evidence 保留 `1-3` 条
- `json_like_evidence_ratio = 0`

实现备注：

- 文本 evidence 当前使用规范化 substring 做 prompt 可见性校验。
- table evidence 当前允许数字级 fallback；该逻辑只能说明 program 关键数字出现在 prompt 中，不能完全等价于 evidence 文本可见。发布前需要用最终输出补充抽检或升级为 row/column slice 级校验。

## Program 规则

Program 主来源按数据族区分：FinQA 以 `qa.program_re / program_re` 为唯一主来源；ConvFinQA current-turn target 以 `annotation.cur_program` 为主来源。ConvFinQA 的 `qa.program_re` 表示整题 final program，只能作为 final-turn 参考或 raw metadata 保留，不能作为所有 turn 的默认 program 来源。

允许的规范化：

- 空格统一
- 逗号与括号格式统一
- 不改变语义的轻量字符串规范

不允许：

- 改写运算结构
- 代数重写
- 用 `program` 或 `steps` 重建缺失的 FinQA `program_re`
- 将 ConvFinQA final `qa.program_re` 复制到非 final turn 的 target

执行失败、非有限值、缺失 program 或 program-answer 不匹配的样本进入 `C` 档，不进入 strict-A。

## Answer 规则

Answer 采用可执行真值优先。

字段职责：

- `answer_raw`
  - 原始标注答案
- `answer_exe`
  - 原始 `exe_ans`
- `answer_norm`
  - benchmark-style SFT 和评测使用的主答案
- `answer_display`
  - 自然展示答案，如 `14.1%`、`$3.8 million`

选择逻辑：

1. FinQA：若 `program_re` 执行值与 `exe_ans` 一致，`answer_norm` 取一致值
2. ConvFinQA current turn：若 `annotation.cur_program` 执行值与 `annotation.exe_ans` 一致，`answer_norm` 取一致值
3. 若 `exe_ans` 缺失但 program 可执行，`answer_norm` 取执行值
4. 若 `raw answer` 与执行值冲突，保留 raw，但 strict target 不使用 raw
5. 若答案无法稳定确定，进入 `C`

`answer_display` 当前用于 metadata 和 `assistant_sft`，不影响默认 `benchmark_sft` 的 `Answer:`。

## 质量分层

当前自动分为三档：

- `A`
  - program 可执行
  - answer 与 program 匹配
  - evidence exact 对齐
  - 无阻断型 audit flags
- `B`
  - 计算链路闭合，但存在非阻断风险
  - 示例：`raw_answer_mismatch_with_answer_norm`、`question_semantic_risk`、`question_text_suspicious`
  - `history_answer_missing` 只表示存在 question-only fallback，是多轮监督密度指标，不默认阻断 current-turn strict-A
- `C`
  - 不进入 strict-A
  - 示例：`program_answer_mismatch`、`missing_evidence`、`non_exact_evidence_alignment`、`missing_program_re`、`missing_answer_norm`、`missing_question`

默认导出：

```text
--strict_tiers A
```

只有在 B 档完成规则修复或人工抽检后，才建议使用：

```text
--strict_tiers A,B
```

## FinQA 当前实现

文件：

```text
financial_data_processors/families/finqa.py
```

当前功能：

- 构造单轮 prompt：report context + question
- 生成 `Evidence / Program / Answer` target
- 保留 `question_raw`
- 对特定 LIBOR / basis points / interest expense 语义错位样本生成保守 `question_rewritten`
- metadata 输出：
  - `answer_display`
  - `quality_tier`
  - `semantic_audit_flags`
  - `question_raw`
  - `question_rewritten`

FinQA strict-A 默认用于 SFT-1。

## ConvFinQA raw 字段语义

以 `docs/example_convfinqa.json` 为参考，ConvFinQA turn raw 同时包含 final QA 和当前 turn annotation。两者职责不同，不能混用：

- `qa.question / qa.answer / qa.program_re / qa.exe_ans`
  - 整题 final QA，例如最终百分比变化问题和完整 final program。
  - 只用于 final-turn 参考、raw metadata 或 sanity check。
  - 不能作为每个 turn 的默认 `question / program_raw / answer_norm` 来源。
- `annotation.cur_dial[-1]`
  - 当前 turn 问题。turn 0 示例是 `what was the total of net sales in 2001?`。
- `annotation.cur_program`
  - 当前 turn 程序。turn 0 示例是 `5363`。
- `annotation.exe_ans`
  - 当前 turn 执行答案。turn 0 示例是 `5363.0`。
- `annotation.dialogue_break`
  - 完整问题拆解序列。
- `annotation.turn_program / annotation.exe_ans_list / annotation.answer_list`
  - 各 turn 的程序和答案序列，可用于构造前序完整推理 history。
- `annotation.gold_ind`
  - 当前 raw turn 使用的证据标注来源。

文档案例：

```text
# turn 0 current supervision
Question: annotation.cur_dial[0]
Program: annotation.cur_program = 5363
Answer: annotation.exe_ans = 5363.0

# final QA reference only
Final Program: qa.program_re = divide(subtract(5363, 7983), 7983)
```

错误示例：不要把 final `qa.program_re = divide(subtract(5363, 7983), 7983)` 输出到 turn 0 / turn 1 / turn 2 的 target。

## ConvFinQA 当前实现与目标流程

文件：

```text
financial_data_processors/families/convfinqa_turn.py
```

目标流程是 raw-aware turn-level multiturn，并优先把可验证的前序 turn 写成完整推理 history。

处理流程：

1. 生成 `conversation_id`
2. 按 `annotation.turn_ind` 或 record id suffix 排序
3. 当前 turn target 使用 `annotation.cur_dial[-1] + annotation.cur_program + annotation.exe_ans`
4. 从 `annotation.cur_dial[:-1]` 或 `annotation.dialogue_break[:turn_ind]` 读取前序 history questions
5. 用 `dialogue_break[:turn_ind] + turn_program[:turn_ind] + exe_ans_list[:turn_ind]` 构造前序完整推理块
6. 只有前序 turn 满足 strict-A、exact evidence、program-answer 一致时，才渲染为 `Q + Evidence + Program + Answer`
7. 无法可靠取得前序完整监督时，保留 question-only history，不写伪造的 `A:`
8. 构造 prompt：history + report context + current question
9. target 只监督当前 turn 的 `Evidence / Program / Answer`

当前代码或旧产物如果仍使用 `qa.question / qa.program_re / qa.exe_ans` 生成每个 turn 的 `question / program_raw / answer_norm`，应视为待修复问题，不应作为已对齐流程发布。

完整 history 形态：

```text
Conversation history:
Q: what is the net cash from operating activities in 2009?
Evidence:
- year ended june 30 , cash provided by operations ... $ 206588 ...

Program: 206588
Answer: 206588

Q: what about in 2008?
Evidence:
- year ended june 30 , cash provided by operations ... $ 181001 ...

Program: 181001
Answer: 181001

Current question: what is the difference?
```

question-only fallback 形态：

```text
Conversation history questions:
- what is the net cash from operating activities in 2009?
- what about in 2008?

Current question: what is the difference?
```

输出 metadata 包含：

- `conversation_id`
- `turn_ind`
- `sft_mode = turn_level_multiturn`
- `history_questions`
- `history_turns`
- `history_full_reasoning_turns`
- `history_question_only_turns`
- `history_full_reasoning_ratio`
- `history_answer_missing`
- `requires_history`
- `history_dependency_type`

`history_answer_missing = true` 只表示该样本存在 question-only fallback。它用于观测多轮监督质量，不表示当前 turn 的 `Evidence / Program / Answer` 不可训练，也不默认触发 strict-A 过滤。发布时应优先看 `history_full_reasoning_turn_ratio` 和 rendered prompt 泄漏指标。

## SFT-2 混合策略

`run_fingpt_v2.ipynb` 当前按以下比例构造 SFT-2：

```text
70% ConvFinQA requires_history turns
20% ConvFinQA other turns
10% FinQA replay
```

若未设置固定 budget，则使用全部 ConvFinQA strict-A，并按比例抽取 FinQA replay。

输出文件：

- `train_sft2_clean_strict.jsonl`
- `train_sft2_convfinqa_only_strict.jsonl`
- `train_sft2_finqa_replay.jsonl`

FinQA replay 用于减少 SFT-2 后单轮 FinQA 能力退化。

## Notebook 对齐状态

`run_fingpt_v2.ipynb` 已同步当前处理流程。

核心转换统计：

- `target_schema_ratio`
- `json_like_evidence_ratio`
- `exact_evidence_alignment_ratio`
- `program_answer_match_ratio`
- `tier_counts`
- `requires_history_ratio`
- `history_turn_rows_ratio`
- `history_answer_missing_ratio`
- `history_full_reasoning_rows_ratio`
- `history_question_only_rows_ratio`
- `history_full_reasoning_turns`
- `history_question_only_turns`
- `history_full_reasoning_turn_ratio`
- `evidence_visible_in_prompt_ratio`
- `duplicate_current_question_in_history_rows`
- `rendered_duplicate_current_question_rows`
- `current_answer_leaked_in_history_rows`

注意：

- `history_answer_missing_ratio` 是存在 question-only fallback 的样本行比例。
- `history_full_reasoning_turn_ratio` 是 history turn 级别的完整推理覆盖率。
- source-level counters 可大于 0，因为它们记录原始数据中被构建逻辑剔除的问题；最终发布看 rendered prompt 级别是否干净。

## 推荐命令

FinQA strict-A：

```bash
python -m financial_data_processors \
  --task sft \
  --dataset_family finqa \
  --source_file /root/autodl-tmp/data/financial_reasoning/raw/finqa/train.json \
  --output_file /root/autodl-tmp/data/financial_reasoning_v2/sft1_sharegpt/finqa_train_sharegpt.jsonl \
  --normalized_output_file /root/autodl-tmp/data/financial_reasoning_v2/normalized/sft1/finqa_train_normalized.jsonl \
  --audit_output_file /root/autodl-tmp/data/financial_reasoning_v2/audit/sft1/finqa_train_audit.jsonl \
  --sft_variant benchmark_sft \
  --strict_tiers A
```

ConvFinQA turn-level multiturn strict-A：

```bash
python -m financial_data_processors \
  --task sft \
  --dataset_family convfinqa_turn \
  --source_file /root/autodl-tmp/data/financial_reasoning/raw/convfinqa_turn/train_turn.json \
  --output_file /root/autodl-tmp/data/financial_reasoning_v2/sft2_sharegpt/fingpt_convfinqa_train_sharegpt.jsonl \
  --normalized_output_file /root/autodl-tmp/data/financial_reasoning_v2/normalized/sft2/fingpt_convfinqa_train_normalized.jsonl \
  --audit_output_file /root/autodl-tmp/data/financial_reasoning_v2/audit/sft2/fingpt_convfinqa_train_audit.jsonl \
  --sft_variant benchmark_sft \
  --strict_tiers A
```

Assistant-style 导出只在需要自然答案展示时使用：

```bash
--sft_variant assistant_sft
```

## 当前小样本验证

已用小样本验证当前实现可运行。除通用 schema 指标外，ConvFinQA 必须增加 raw-aware turn target 检查：

```text
FinQA: 20 input -> strict-A saved
ConvFinQA: 50 input -> strict-A saved
ConvFinQA multiturn_history_rows: reported
ConvFinQA history_full_reasoning_turn_ratio: reported
rendered_duplicate_current_question_rows: 0
json_like_evidence_rows: 0
```

基于 `example_convfinqa.json` 或 raw 前 4 条 JKHY 样本的专项检查：

```text
turn0_expected_program = annotation.cur_program
turn0_expected_answer = annotation.exe_ans
turn0_program_must_not_equal_final_qa_program_re
final_turn_history_contains_prior_turn_reasoning_blocks
current_question_not_in_rendered_history
current_full_target_not_in_rendered_history
```

这些数字只用于 smoke test，不作为最终数据集发布指标。

## 验收与发布门槛

strict 数据发布前至少检查：

- `target_schema_ratio = 1.0`
- `json_like_evidence_ratio = 0`
- `raw_program_unchanged_ratio = 100%`
- `exact_evidence_alignment_ratio >= 95%`
- `program_answer_match_ratio >= 98%`
- SFT-2 中 `requires_history_ratio` 足够高，避免退化为单轮 FinQA-like 训练
- SFT-2 中 `history_full_reasoning_turn_ratio` 越高越好；`history_answer_missing_ratio` 只作为 question-only fallback 观测项
- ConvFinQA current-turn target 必须来自 `annotation.cur_dial[-1] / annotation.cur_program / annotation.exe_ans`，非 final turn 不得使用 final `qa.program_re`
- `evidence_visible_in_prompt_ratio = 1.0`
- `duplicate_current_question_in_history_rows = 0`
- `current_answer_leaked_in_history_rows = 0`
- 人工抽检 100 条，至少 90 条满足 `question / evidence / program / answer` 一致

## 下一轮修改计划

人工抽检发现当前 strict 数据仍存在三类高风险样本：evidence 出现在 target 但不在 prompt 可见上下文中、ConvFinQA history 包含当前问题本身、以及 table evidence 包含无关年份列。下一轮修改目标是让 `strict-A` 从“计算闭合”升级为“上下文可见、历史有效、证据最小充分”。

本节包含两类状态：

- 已部分接入：代码已有字段、计数或初步规则，但验收口径仍需补齐。
- 待闭环：必须在最终 strict 文件上验证，不能只看 source-level counters。

### P0：Evidence 必须在 Prompt 中可见

问题：部分样本的 `Evidence:` 来自原始 report 其他位置，但没有出现在 human prompt 的 `Text before table / Table / Text after table` 中。模型训练时会被迫从不可见上下文生成证据，属于坏监督。

修改要求：

- 在构造 prompt 后，对每条 `aligned_evidence.rendered_text` 做可见性校验
- strict-A 要求 evidence 文本或其规范化形式必须能在 prompt context 中定位
- 若 evidence 不可见，打 `evidence_not_in_rendered_prompt`
- `evidence_not_in_rendered_prompt` 默认归为 `C`，不进入 `--strict_tiers A`
- audit 输出中增加该 flag 的计数

验收标准：

```text
evidence_visible_in_prompt_ratio = 1.0
```

当前差距：

- table evidence 仍存在数字级 fallback，需要升级为 evidence 文本或 table row/column slice 可定位，或在发布前用人工抽检补偿风险。

### P0：ConvFinQA current-turn target 和 history 构建对齐 raw 语义

问题：抽检发现部分 SFT-2 样本的 `Conversation history` 中包含当前问题，或 current-turn target 使用了整题 final `qa.program_re`。这类问题必须在数据集构建阶段修正，而不是等到校验阶段靠筛选兜底。

已落地或待对齐修改：

- current-turn question / program / answer 必须分别来自 `annotation.cur_dial[-1] / annotation.cur_program / annotation.exe_ans`
- `qa.question / qa.answer / qa.program_re / qa.exe_ans` 只作为 final QA metadata 或 final-turn sanity check
- history 问题来源使用原始 `annotation.cur_dial` 或 `dialogue_break`
- 不直接把同 conversation 下前序重复的 `qa.question / qa.answer` 当作 history
- 当前问题不得出现在 `history_questions`；若原始 history 含当前问题，在构建阶段剔除并计入 source 统计
- 前序 turn 只有通过 strict-A、exact evidence、program-answer 一致校验后，才渲染为 `Q + Evidence + Program + Answer`
- 拿不到可靠前序监督时，保留 question-only history，不伪造 `A:`
- question-only history 使用 `Conversation history questions:`；混合场景中只写 `Q:`，不写 `A:`

目标 history 形态：

```text
Conversation history:
Q: what is the net cash from operating activities in 2009?
Evidence:
- year ended june 30 , cash provided by operations ... $ 206588 ...

Program: 206588
Answer: 206588

Q: what about in 2008?
Evidence:
- year ended june 30 , cash provided by operations ... $ 181001 ...

Program: 181001
Answer: 181001

Current question: what was the percentage change in the net cash from operating activities from 2008 to 2009?
```

验收标准：

```text
turn0_program_matches_annotation_cur_program
turn0_answer_matches_annotation_exe_ans
non_final_turn_program_not_equal_final_qa_program_re
rendered_duplicate_current_question_rows = 0
current_full_target_leaked_in_history_rows = 0
history_full_reasoning_turn_ratio is reported
history_question_only_rows_ratio is reported
# source-level counters may be > 0, but final rendered prompts must be clean
```

当前差距：

- 仍需在全量重新生成后抽检 current-turn target 是否已全部使用 `annotation.cur_*` 字段。
- 仍需抽检完整 history 块是否显著提升 SFT-2 的多轮监督密度。
- `history_answer_missing_rows` 只按 row 统计，不能作为 strict-A 排除条件使用。

### P1：Table Evidence 只保留问题相关列

问题：FinQA table evidence 当前常渲染整行，可能把 2013 和 2012 两列都放进 evidence；当问题只问 2012 时，无关列会增加噪声。

修改要求：

- 从 question 中抽取显式年份、日期、列名线索，如 `2012`、`Dec. 29 2012`
- table evidence 渲染时优先只保留相关列和行名
- 若无法稳定识别相关列，保留当前整行渲染，不能臆造列映射
- 新增 `table_evidence_column_pruned` metadata 标记

目标输出示例：

```text
Evidence:
- available-for-sale investments as of Dec. 29, 2012 was $14,001 million.
- total cash and investments as of Dec. 29, 2012 was $26,302 million.
```

而不是：

```text
Evidence:
- available-for-sale investments of Dec 28 2013 is $18086; ... Dec 29 2012 is $14001
```

验收标准：

```text
table_evidence_column_pruned_rows > 0
manual_table_evidence_precision >= 90% on 100 sampled rows
```

### P1：Question 文本质量审计

问题：ConvFinQA 中存在拼接错误或语义不自然问题，例如 `2018what`。这类样本即使 program-answer 正确，也会污染 prompt 分布。

修改要求：

- 增加 question 文本清洗：修复年份与英文词之间缺空格等确定性问题
- 增加 `question_text_suspicious` 规则，覆盖：
  - 数字和英文连写，如 `2018what`
  - 过短问题
  - 疑似截断问题
  - 缺少可理解谓词的问题
- `question_text_suspicious` 默认归为 `B`
- 若 question 无法清洗到可读状态，归为 `C`

验收标准：

```text
question_text_suspicious_ratio is reported
manual_question_readability >= 90% on 100 sampled rows
```

### P2：SFT-2 混合后质量约束

SFT-2 strict 文件生成后需要额外检查 mixture 是否真的保留多轮训练信号。

新增检查：

- `requires_history_ratio`
- `history_turn_rows_ratio`
- `history_full_reasoning_turn_ratio`
- `history_question_only_rows_ratio`
- `rendered_duplicate_current_question_rows`
- `duplicate_current_question_in_history_rows`
- `current_answer_leaked_in_history_rows`
- `evidence_visible_in_prompt_ratio`
- `sft2_convfinqa_only_rows`
- `sft2_finqa_replay_rows`

发布门槛：

```text
requires_history_ratio >= 0.60
rendered_duplicate_current_question_rows = 0
current_answer_leaked_in_history_rows = 0
history_full_reasoning_turn_ratio is reported
# source-level counters may be > 0, but final rendered prompts must be clean
evidence_visible_in_prompt_ratio = 1.0
json_like_evidence_ratio = 0
```

### 实施顺序

1. 在 `common.py` 增加文本规范化、evidence 可见性校验、question 质量审计工具函数
2. 在 `convfinqa_turn.py` 修正 history 构造：以 `cur_dial/dialogue_break` 为 history 来源，可靠匹配时补完整 `Q + Evidence + Program + Answer`，否则保留 question-only fallback，并统计 source/rendered leak 指标
3. 在 `finqa.py` 和 shared table renderer 中加入相关列裁剪
4. 在 `router.py` 汇总新增 flags 和 ratio
5. 在 `run_fingpt_v2.ipynb` 的 audit cell 中加入新指标
6. 重新生成 FinQA / ConvFinQA strict 数据并抽检 100 条

## 当前不在本轮范围

以下内容不属于当前主链：

- `fineval / fiqa_qa` 的详细清洗策略
- 中文默认 SFT 模板
- 四段式中文分析模板
- `<think>` / CoT 监督
- 基于 `steps` 或 `program` 回填缺失 `program_re`
- dialogue-level multiturn 训练主路径
- DPO / GRPO 细节设计

DPO、GRPO、dialogue-level multiturn 可在当前 strict 数据稳定后单独设计。
