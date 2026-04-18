# 金融 CoT / PoT / Distill / RL 训练路线

本文档描述当前金融表文数值推理项目中 CoT、PoT、distill-R1/o1 与 RL/GRPO 的分工，并说明 `run_fingpt_cot_rl.ipynb` 的实验设计。它承接 `docs/fin_datasets_v2.md` 和 `docs/fin_datasets_v3.md`，不替代已有 v2/v3 数据主链。

## 定位

本项目的核心任务不是普通金融问答，而是可验证的金融表文数值推理。模型需要从文本、表格和对话历史中定位证据，理解当前问题需要的财务指标和运算形式，并输出可以被评估的答案。仅依赖自然语言 CoT 容易得到流畅但不可验证的推理；仅依赖最终答案又会丢失证据和公式结构。因此，本项目将推理表示拆成两个互补部分：CoT 负责解释证据选择和公式意图，PoT 负责可执行计算。

在当前代码中，`Program:` 就是第一版 PoT。FinQA 和 ConvFinQA 原生提供 gold program、execution answer 和 normalized answer，这使它们比普通 CoT 数据更适合构建 program-supervised SFT 和 execution-based GRPO。Fino1/FinCoT 提供的是金融 reasoning path，价值在于补充 CoT 表达，而不是替代 FinQA/ConvFinQA 的 gold Program。

## 概念关系

CoT 是推理表示形式，通常是自然语言 reasoning path。它回答“为什么这样算”，适合帮助模型解释变量选择、公式意图、单位处理和多轮上下文依赖。CoT 的问题是不可执行，因此不能单独作为金融数值题的主要验证依据。

PoT 是另一种推理表示形式，通常是 DSL 或 Python 程序。它回答“怎么算”，可以由 executor 执行，并与 `answer_norm` 做数值比较。对 FinQA/ConvFinQA 来说，PoT 的主来源应是 gold Program，而不是外部 teacher 随意生成的程序。

Distill-R1 或 distill-o1 是推理能力迁移方法。它用 DeepSeek-R1、o1、GPT-4o、DeepSeek-Reasoner 或类似 teacher 生成高质量 reasoning traces，再用 SFT 让 student 学习这些 traces。它能快速注入推理风格，但如果没有 program/execution check，就可能只学到长推理格式而没有学到可验证计算。

RL/GRPO 是推理能力激励方法。SFT 让模型模仿标准轨迹，GRPO 让模型生成多个候选，并根据 reward 提高更好候选的概率。DeepSeekMath 和 DeepSeek-R1 的关键启发是：当任务有明确正确答案时，verifiable reward 可以激励更强的 reasoning。对于本项目，reward 不应只奖励格式或长 reasoning，而应重点奖励 `Program` 可解析、可执行，且 `execute(Program) == gold answer_norm`。

## 当前基线

当前 v2 主基线是 `sft2_dual_merged`。它先用 FinQA strict-A 做 SFT1，学习单轮金融表文数值推理；再用 ConvFinQA turn-level strict-A 做 SFT2，学习多轮 follow-up reasoning，并用 FinQA replay 保持单轮表文能力。SFT2 的核心比例约为 `ConvFinQA:FinQA = 2:1`。

v2 target 为：

```text
Evidence:
- ...

Program: ...

Answer: ...

Normalized Answer: ...
```

v3 target 为：

```text
Evidence:
- ...

Program: ...
```

v2 保留模型直接生成答案的能力，v3 更接近 program-executed reasoning。`run_fingpt_cot_rl.ipynb` 当前从 v2 `sft2_dual_merged` 出发，因为它已经具备稳定的输出 schema、Program 字段和 normalized answer 能力，是更稳的 GRPO 起点。

## 数据策略

`run_fingpt_cot_rl.ipynb` 使用三类数据。第一类是 core program-verifiable 数据，来自本地 FinQA/ConvFinQA v2 strict 文件，保留 gold `Program` 和 `Normalized Answer`。这类数据使用 `program_numeric` reward profile，参与 answer correctness、operator consistency、format、evidence grounding 和 brevity reward。

第二类是 `TheFinAI/Fino1_Reasoning_Path_FinQA`。它与 FinQA 风格更接近，字段包括 open-ended verifiable question、Complex CoT、ground-truth answer 和 response。它作为 FinQA-style CoT supplement 使用，默认不强行伪造 `Program`，而是写 `Program: N/A`，使用 `cot_answer_only` reward profile。

第三类是 `TheFinAI/FinCoT` 的 RL split。它提供 `Question`、`Reasoning_process`、`Final_response`、`Negative_reasoning_process` 和 `Negative_response`。这类样本用于补充更广金融 reasoning diversity，并通过 answer、format、brevity 和 negative-response avoidance 提供轻量 reward。它同样不参与 Program execution reward。

默认混合应保持保守。core program data 是主干，Fino1/FinCoT 是 supplement。推荐第一轮使用约 70% core、15% Fino1、15% FinCoT，或者在担心遗忘时提高 core 到 90%。如果 external CoT 导致 ConvFinQA 或 program accuracy 下降，应降低外部数据比例。

## Reward Contract

GRPO 数据统一为 JSONL，每行至少包含：

```json
{
  "prompt": "...",
  "answer": "...",
  "gold_program": "...",
  "source_dataset": "finqa|convfinqa_turn|fino1_finqa_path|fincot_rl",
  "reward_profile": "program_numeric|cot_answer_only"
}
```

`program_numeric` 样本必须来自 FinQA/ConvFinQA，必须保留 gold program。`cot_answer_only` 样本来自 Fino1/FinCoT，可以输出 `Program: N/A`，不接受 program reward 惩罚。这样可以避免外部 CoT 覆盖本地 gold Program 主链。

第一版 reward 包含六部分。`reward_answer` 比较 `Normalized Answer` 或 `Answer` 与 gold answer；`reward_format` 检查 `Evidence`、`Reasoning`、`Program`、`Answer`、`Normalized Answer` 等结构；`reward_program` 只对 program profile 计算 operator overlap；`reward_program_answer_consistency` 轻量检查 program profile 是否输出非 N/A Program 且答案可解析；`reward_evidence` 检查 evidence 中的数字是否来自 prompt；`reward_brevity_and_relevance` 抑制冗长 CoT，并避免外部样本复述 negative response。

后续应将 `reward_program_answer_consistency` 升级为真正的 execution reward，也就是解析模型输出的 `Program:`，执行 DSL，并与 gold answer 做容差比较。正式 RL 前还需要收紧 executor，防止自然语言 program 或中缀表达式被误判为可执行。

## 实验流程

第一步固定 `sft2_dual_merged` 作为 baseline，并读取已有 `base_passk`、`sft2_merged_passk`、`dpo_passk` benchmark。第二步生成 mixed GRPO 数据，只写到 `/root/autodl-tmp/data/financial_reasoning_v2/rl`，不覆盖 v2/v3 SFT 数据。第三步先运行 reward smoke test，确认不同 reward profile 的分数符合预期。第四步只打开 `RUN_GRPO_SMOKE=True` 做 1 step smoke run。第五步再运行完整 GRPO，并用 `evaluation.evaluate_financial_benchmarks` 做 pass@k 对比。

判断 GRPO 是否值得继续，不应只看训练日志。核心标准是 `pass@1` 是否向 `pass@4/pass@8` 靠近，`executed_answer_accuracy` 是否提升，`program_execution_rate` 是否保持高位，ConvFinQA 是否没有明显遗忘。如果只是格式 reward 上升、average output length 增大，而 FinQA/ConvFinQA benchmark 没涨，应回退 reward 或降低外部 CoT 比例。

## 与 ref 的关系

`ref/pot_cot_distill_rl.md` 提供概念层解释，说明 CoT/PoT 是推理表示，distill-R1/o1 是推理能力迁移，RL/GRPO 是推理能力激励。`ref/fino1_fincot_integration_research.md` 说明 Fino1/FinCoT 应作为外部 CoT supplement，而不是替代 Program 主链。`ref/openr1.md` 与新增 DeepSeek-R1、DeepSeekMath 论文共同说明：verifiable reward 是 reasoning RL 的关键，但金融场景的 reward 应绑定 `Normalized Answer`、Program execution、单位归一和 evidence grounding。

本项目的最终定位是：在 Fino1-style 金融 CoT SFT + GRPO baseline 上，引入 FinQA/ConvFinQA gold Program supervision，将 CoT-only 金融推理扩展为可验证 CoT+PoT，并用 execution-based reward 做领域强化学习。
