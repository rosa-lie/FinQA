# 金融程序执行式推理数据主链 v3

本文描述当前 program-only 数据主线。v3 的核心变化是模型不再直接生成 `Answer` 或 `Normalized Answer`，而是只生成 `Evidence + Program`。最终答案由 executor 执行 program 得到，评测也基于 executor 输出与 gold normalized answer 的匹配。

## 1. 背景问题

FinQA/ConvFinQA 的价值在于它们提供可执行 program 和 execution answer。早期 dual-answer target 让模型同时生成 evidence、program、answer 和 normalized answer，适合做完整回答 baseline，但也让模型承担了不必要的数值心算和格式负担。program-only 数据把任务定义为金融表文推理编译，模型负责把自然语言问题和表格证据编译成 DSL，系统负责执行、归一化和展示。

## 2. 数据来源

FinQA 是单轮金融表文数值推理数据。当前项目用它训练模型从文本、表格和 question 中定位 evidence，生成 canonical FinQA DSL program，并保留 `answer_norm`、`answer_display`、`answer_unit` 和 `answer_scale` 作为 metadata。

ConvFinQA 是多轮金融对话数值推理数据。当前项目只使用 turn-level supervision。每个训练样本的 current question 来自 `annotation.cur_dial[-1]`，current program 来自 `annotation.cur_program`，current answer 来自 `annotation.exe_ans`。`qa.question`、`qa.program_re` 和 `qa.exe_ans` 只保留为 final metadata 或 fallback。不能把 final QA program 复制到每个 turn，否则会把中间轮次训练成错误标签，模型看似学到多轮推理，实际是在拟合错配监督。

## 3. Target 格式

v3 target 固定为：

```text
Evidence:
- ...

Program: divide(subtract(6823, 6161), 6161)
```

target 中不包含 `Answer` 和 `Normalized Answer`。这样做的原因是答案应由 executor 从 program 计算得到，而不是由模型在文本中重复生成。训练样本仍然在 metadata 中保留 gold answer，用于校验、评测和错误分析。

## 4. Metadata 与 executor

每条样本应保留 program 和答案相关 metadata，例如 `program_canonical`、`program_executable`、`answer_norm`、`answer_display`、`answer_unit`、`answer_scale`、`answer_source` 和 `answer_matches_program`。其中 `program_canonical` 是训练 target 的标准程序，`program_executable` 是 executor 执行 gold program 得到的数值，`answer_norm` 是评测时使用的标准答案。

当前执行器支持 `add`、`subtract`、`multiply`、`divide`、`exp`、`max`、`min`、`sum` 和 `average` 等主要 DSL。数据清洗时要优先保证 program 可解析、数值常量规范、单位缩放一致，并过滤 prompt-label 冲突样本。

## 5. retention-aware RS-SFT 数据

retention-aware RS-SFT 使用当前模型采样多个候选，通过 executor 筛出答案正确且格式可用的 program，再把这些高质量候选作为 SFT 监督目标。它不是简单扩大数据量，而是把监督目标从 gold-only 扩展到当前模型能够生成、executor 验证正确、形式上更贴近模型分布的样本。

为了缓解持续训练遗忘，RS-SFT 数据保持 ConvFinQA 与 FinQA 的混合。当前项目口径是 65% ConvFinQA 加 35% FinQA 唯一样本。ConvFinQA 负责多轮 follow-up 能力，FinQA replay 负责单轮表文能力保留。full-1237 结果显示 RS-SFT 相比 SFT2 从 0.578820 提升到 0.603072，是当前强基线的主要来源。

## 6. frontier acquisition 数据

v34r23 的 frontier acquisition 基于当前 RS-SFT policy，而不是历史错误样本。筛选标准是 greedy prediction 错，但 pass@8 采样候选中存在 executor-correct program，并尽量保留 executable-wrong hard negative。这个条件意味着样本处在当前模型的可学习边界上，适合 GRPO 利用组内相对奖励校准 greedy 行为。

当前 strict acquisition 从 FinQA 和 ConvFinQA 各 500 条训练样本中获得 FinQA 106 条和 ConvFinQA 82 条 frontier 样本。这些样本进入 `training/finqa_program_grpo.py`，使用 `frontier_execution_calibration` reward 训练 v34r23，并继续作为 v34r24 Dr.GRPO sweep 的基础。

## 7. 数据处理入口

数据处理统一从仓库根目录执行：

```bash
/root/miniconda3/bin/python -m financial_data_processors
```

GRPO frontier 相关脚本包括 `scripts/build_v34r23_current_policy_manifest.py`、`scripts/build_v34r23_frontier_grpo_data.py` 和 `scripts/v34r23_prompt_processor.py`。这些脚本负责采样、strict prompt contract、program 清洗、候选审计和 frontier JSONL 输出。

## 8. 面试回答

面试中可以把 v3 数据主线概括为，把金融数值推理从“让模型写答案”改造成“让模型写可执行程序”。FinQA 建立单轮表文计算能力，ConvFinQA 建立多轮 follow-up 能力，RS-SFT 用 executor 过滤正确程序来提升数据质量，frontier acquisition 则为 GRPO 找到当前策略真正可学习的边界样本。这个设计比直接做 CoT SFT 或泛金融 RL 更可验证，也更容易定位错误来源。
