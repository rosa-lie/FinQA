# Program of Thought、FinQA 与可验证蒸馏总调研

## 1. 背景与定位

本调研聚焦 Program of Thought（PoT）在金融数值推理中的价值，以及它和 FinQA、ConvFinQA、distillation、GRPO 的结合方式。核心判断是：PoT 不应只作为一种 inference-time prompting baseline，而应作为一种可执行、可验证、可蒸馏的 reasoning representation。

当前 MedicalGPT 金融推理主链已经具备 PoT 化的基础：FinQA 与 ConvFinQA processor 会生成 `Program:`、`Answer:`、`Normalized Answer:`，评测优先解析 `Normalized Answer:`，并统计 `program_accuracy`。这意味着项目不需要从零复刻 PoT，而应在现有 program-supervised v2 主链上升级为 CoT + PoT + execution verification + distillation。

PoT 与 CoT 的关系不是替代，而是互补：

| 表征 | 优势 | 风险 | 在金融推理中的角色 |
| --- | --- | --- | --- |
| CoT | 可读、利于解释变量选择和业务逻辑 | 计算易错、容易写漂亮废话 | 简短说明证据、变量和运算意图 |
| PoT | 低歧义、可执行、可校验 | 可能硬编码答案或证据错配 | 承担数值计算和可验证 supervision |
| EoT | 显式校验执行结果 | 可能被模板化 | 连接 program execution 与 final answer |

因此，本项目推荐采用 Ensemble Thoughts Distillation（ETD）思路：用 CoT 提供语义解释，用 PoT 提供可执行计算，用 EoT 提供执行校验。

## 2. 核心方法

### 经典文献表

| 方向 | 代表工作 | 核心思想 | 对本项目启发 |
| --- | --- | --- | --- |
| CoT | Chain-of-Thought Prompting, Wei et al. 2022 | 让模型输出中间自然语言推理步骤 | 金融题中可用于解释取数和公式，但不应单独承担计算 |
| Self-Consistency | Wang et al. 2022 | 多采样多条推理路径，对最终答案投票 | PoT 可对多个可执行程序的 `ans` 投票 |
| Least-to-Most | Zhou et al. 2022 | 先拆子问题，再逐步求解 | 可用于 FinQA 中先定位行列、年份、指标，再生成程序 |
| PAL | Program-aided Language Models, Gao et al. 2022/2023 | LLM 写程序，Python 执行计算 | PoT 的近邻路线，可作为程序辅助推理参考 |
| PoT | Chen et al. 2023 | LLM 用 Python 表达 reasoning，解释器负责 computation | 当前项目最直接参考对象 |
| Toolformer | Schick et al. 2023 | 模型学习何时调用工具 | PoT 可视为 calculator/Python 工具调用的强约束形式 |
| Distilling Step-by-Step | Hsieh et al. 2023 | 用 rationale 作为小模型监督 | 可扩展为 program/rationale 混合蒸馏 |
| PaD | Program-aided Distillation, Zhu et al. 2024 | 用程序和执行校验过滤错误 reasoning | 适合 FinQA 这种有 gold answer/program 的金融数据 |
| ETD | Distilling Mathematical Reasoning Capabilities into Small Language Models, 2024 | 组合 CoT、PoT、EoT 做 Ensemble Thoughts Distillation | 明确 PoT 是可蒸馏的 reasoning representation |
| Fino1 | Qian et al. 2025 | 金融 reasoning path SFT + GRPO | 可作为金融 CoT 蒸馏与 GRPO 对照基线 |

### 开源仓库表

| 名称 | 链接 | 可复用点 |
| --- | --- | --- |
| Program-of-Thoughts | https://github.com/TIGER-AI-Lab/Program-of-Thoughts | PoT prompt、safe execution、self-consistency、FinQA/ConvFinQA scripts |
| PAL | https://github.com/reasoning-machines/pal | Program-aided reasoning 框架 |
| FinQA | https://github.com/czyssrs/FinQA | gold program、exe answer、表文混合金融数值推理 |
| ConvFinQA | https://github.com/czyssrs/ConvFinQA | 多轮金融数值推理、turn-level program |
| Fino1 | https://github.com/The-FinAI/Fino1 | Fino1-style SFT + GRPO 金融 reasoning 路线 |
| FinCoT | https://huggingface.co/datasets/TheFinAI/FinCoT | 金融 CoT SFT/RL 数据 |
| Fino1 Reasoning Path FinQA | https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA | GPT-4o 生成的 FinQA reasoning path |
| PaD | https://github.com/Xuekai-Zhu/pad | program-aided distillation 参考 |

### FinQA + PoT 推荐 pipeline

1. 从 FinQA/ConvFinQA 原始数据读取 question、context、gold program、exe answer。
2. 用现有 `program_canonical` 作为 DSL PoT。
3. 新增 DSL-to-Python 转换，生成 `Python Program:`，最终变量固定为 `ans`。
4. 执行 Python Program，校验 `ans` 与 `Normalized Answer` 是否一致。
5. 训练 target 输出 `Evidence / Reasoning / Program / Python Program / Answer / Normalized Answer`。
6. 评测时执行模型输出的 Python Program，统计 execution accuracy。
7. 多采样时执行多个候选程序，对执行答案做 self-consistency 投票。

### PoT + Distillation 推荐 pipeline

1. Teacher 生成 CoT、PoT、EoT 三类轨迹。
2. 只保留可执行、答案正确、格式稳定的 candidate。
3. 构造 SFT 正样本：简短 CoT + DSL Program + Python Program + execution check。
4. 构造 DPO/GRPO 负样本：不可执行程序、答案错误程序、硬编码答案、只有 CoT 没有 program、格式缺失。
5. 用 answer correctness、program execution、program consistency、format coverage 共同评估。

## 3. 与当前 MedicalGPT 的关系

当前 MedicalGPT v2 主链已经具备以下基础：

- FinQA SFT-1：单轮表文混合金融数值推理。
- ConvFinQA SFT-2：多轮 follow-up reasoning，并通过 FinQA replay 保持单轮能力。
- target 格式：`Evidence / Program / Answer / Normalized Answer`。
- eval 口径：优先读取 `Normalized Answer:`，统计 numeric parse rate 和 program accuracy。
- 数据审计：过滤 conflicting prompts、program-answer mismatch、JSON-like evidence 等。

因此改造重点不是推倒重来，而是新增 `ensemble_thought_sft`，把当前 `Program:` 从“被监督的 DSL 字符串”升级成“可执行、可验证、可蒸馏”的 PoT 主体。

推荐 schema：

```text
Evidence:
- ...

Reasoning:
Briefly identify the relevant numbers and operation.

Program:
divide(subtract(6823, 6161), 6161)

Python Program:
cash_2013 = 6823
cash_2012 = 6161
ans = (cash_2013 - cash_2012) / cash_2012

Answer: 10.745%
Normalized Answer: 0.10745
```

## 4. 可落地改造方案

1. 数据处理新增 `ensemble_thought_sft`。
   - 在 router 的 `--sft_variant` choices 中增加该项。
   - 在 `render_strict_target()` 中渲染 CoT + PoT + answer。
   - 保持 `dual_answer_sft` 作为 baseline。

2. 新增 DSL-to-Python 转换。
   - 支持 `add/subtract/multiply/divide/exp/max/min/sum/average/table_max/table_min/table_sum/table_average`。
   - 最终输出 `ans = ...`。
   - 转换失败时降级，不写 Python Program。

3. 新增安全执行器。
   - 使用 AST 白名单，不直接复制官方 PoT repo 的通用 `exec`。
   - 禁止 import、属性访问、文件/网络/系统调用、eval/exec。
   - 返回 `ans` 并和 `Normalized Answer` 比较。

4. 评测扩展。
   - 增加 `python_program_parse_rate`。
   - 增加 `python_program_execute_rate`。
   - 增加 `python_program_answer_accuracy`。
   - 增加 `pot_self_consistency_accuracy`。

5. GRPO reward 重写。
   - 当前旧 reward 中中文锚点应改为英文 v2 schema。
   - answer correctness 权重最高。
   - program execution correctness 次之。
   - format 与短 reasoning 只作为轻量约束。

## 5. 实验设计

推荐实验组：

| 实验组 | 训练数据 | 输出格式 | 目的 |
| --- | --- | --- | --- |
| base | 原始 Qwen2.5-7B-Instruct | 自由回答 | 下界 |
| dual_answer_sft | 当前 FinQA/ConvFinQA strict-A | Evidence/Program/Answer/Normalized Answer | 当前主链 baseline |
| ensemble_thought_sft | FinQA/ConvFinQA ETD 数据 | CoT+PoT+Answer | 验证 ETD SFT |
| ensemble_thought_sft + Fino1 | ETD + Fino1/FinCoT CoT 补充 | CoT+PoT 或 CoT-only | 验证外部 reasoning path |
| ensemble_thought_sft + GRPO | ETD SFT 后做可验证 RL | CoT+PoT+Answer | 验证 execution reward |

主指标：

- `answer_accuracy`
- `program_accuracy`
- `numeric_parse_rate`
- `python_program_execute_rate`
- `python_program_answer_accuracy`
- `pot_self_consistency_accuracy`

验收建议：

- ETD 不降低 `numeric_parse_rate`。
- ETD answer accuracy 高于 `dual_answer_sft`。
- Python Program execute rate 高于 0.90。
- GRPO 后 FinQA program accuracy 不明显下降。

## 6. 风险与注意事项

- 不要只蒸馏长 CoT。长 CoT 容易引入不可验证噪声，并让小模型学会模板化解释。
- 不要把外部 FinCoT/Fino1 CoT-only 样本强行伪造为 PoT。没有 gold program 的样本应标记 `Program: N/A` 或只作为 CoT 辅助数据。
- 不要直接使用通用 `exec` 执行模型生成代码。训练评测中必须使用 AST 白名单执行器。
- 不要让模型硬编码最终答案。PoT 过滤中需要检查 `ans` 是否由中间变量和运算得到。
- 不要只看 loss。金融数值推理必须以 generation benchmark、execution accuracy 和 program accuracy 判断。

## 7. 参考资料

- Chen et al. 2023. Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks.
- TIGER-AI-Lab Program-of-Thoughts: https://github.com/TIGER-AI-Lab/Program-of-Thoughts
- Wei et al. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
- Wang et al. 2022. Self-Consistency Improves Chain of Thought Reasoning in Language Models.
- Zhou et al. 2022. Least-to-Most Prompting Enables Complex Reasoning in Large Language Models.
- Gao et al. 2022/2023. Program-aided Language Models.
- Schick et al. 2023. Toolformer.
- Zhu et al. 2024. Program-aided Distillation.
- Shridhar et al. 2023. Distilling Reasoning Capabilities into Smaller Language Models.
- Distilling Mathematical Reasoning Capabilities into Small Language Models, 2024.
- Qian et al. 2025. Fino1: On the Transferability of Reasoning-Enhanced LLMs and Reinforcement Learning to Finance.
- FinQA: https://github.com/czyssrs/FinQA
- ConvFinQA: https://github.com/czyssrs/ConvFinQA
- TAT-QA benchmark.

## 8. 金融 R1、PoT 检索与难度感知训练扩展

本轮新增文献显示，金融 reasoning 后训练正在形成三条互补路线：

1. 金融 R1 路线：Fin-R1、DianJin-R1、Fino1 都采用 `reasoning SFT -> GRPO/RL`，强调结构化推理和答案正确性。
2. 金融 PoT 路线：Chen et al. 的通用 PoT 被 Khatuya et al. 的 FINDER 推向金融场景，重点从“能生成程序”扩展到“先检索证据、再动态选择示例、最后生成可执行程序”。
3. 数据中心路线：Cao et al. 2026 强调数据质量、难度和可验证性，主张 SFT 用高质量蒸馏数据，RL 用 hard-but-verifiable 数据。

### 8.1 新增文献/项目矩阵

| 文献/项目 | 类型 | 核心贡献 | 对 MedicalGPT 的启发 |
| --- | --- | --- | --- |
| Open-R1 | 工程框架 | 开放复现 R1，提供 SFT、GRPO、生成和评测脚本 | 将 notebook 命令固化为 recipes，建立可复现实验管线 |
| Fin-R1 | 金融 reasoning model | DeepSeek-R1 蒸馏金融 CoT，SFT + GRPO | 借鉴“答案+推理”双轮筛选，但保留本项目 program 主链 |
| DianJin-R1 | 金融 reasoning model/data | CFLUE/FinQA/CCC 构造 reasoning data，GRPO 双奖励 | 借鉴 `<think>/<answer>` 结构和 format+accuracy reward |
| FINDER | 金融 PoT inference | generative retriever + dynamic examples + context-aware PoT | 加入 evidence retriever 和 dynamic PoT example selector |
| FinanceReasoning | Benchmark/data | 更可信、更全面、更难的金融数值推理，带 Python solution | 作为外部 PoT benchmark 和 difficulty-aware 数据源 |
| Cao et al. 2026 | 数据中心训练 | ODA-Fin-SFT-318k 与 ODA-Fin-RL-12k，强调难度和可验证性 | 用 pass-rate mining 构造 hard-but-verifiable GRPO 数据 |

### 8.2 对当前主线的统一判断

这些工作共同支持一个结论：MedicalGPT 不应退回到只蒸馏长 CoT。当前项目最有价值的差异化路线是：

```text
FinQA/ConvFinQA gold program
-> Program / Normalized Answer strict data
-> CoT + PoT + EoT ensemble distillation
-> execution-filtered SFT
-> hard-but-verifiable GRPO
-> FinQA / ConvFinQA / FinanceReasoning benchmark
```

其中：

- CoT 来自 teacher/Fino1/FinCoT/DianJin/Fin-R1 风格 reasoning path。
- PoT 来自 FinQA/ConvFinQA gold program、DSL-to-Python、FinanceReasoning python_solution。
- EoT 来自执行器校验结果。
- RL 数据来自 pass rate 中等、答案短、可自动验证的 hard samples。

### 8.3 推荐落地路线

第一阶段：补齐研究和数据入口。

- 保留 `dual_answer_sft` 作为 baseline。
- 新增 `ensemble_thought_sft`。
- 新增外部数据适配：
  - FinCoT / Fino1 reasoning path 作为 CoT supplement。
  - DianJin-R1-Data 作为中文金融 CoT supplement。
  - FinanceReasoning 作为 PoT benchmark 和 Python-solution 数据。

第二阶段：补齐执行和评测。

- 新增 DSL-to-Python 转换。
- 新增 AST 白名单 Python Program 执行器。
- 在 benchmark 中加入：
  - `python_program_parse_rate`
  - `python_program_execute_rate`
  - `python_program_answer_accuracy`
  - `pot_self_consistency_accuracy`
  - by difficulty / by operator count。

第三阶段：难度感知 GRPO。

- 用 SFT 模型多采样估计每题 pass rate。
- 选择 `0.2 <= pass_rate <= 0.7` 的样本。
- 过滤答案长、不可验证、证据不清的题。
- reward 组合：
  - answer correctness
  - program execution correctness
  - format completeness
  - evidence grounding
  - reasoning brevity

### 8.4 与各分文档的关系

| 文档 | 用途 |
| --- | --- |
| `openr1.md` | 训练工程框架与 recipe 化参考 |
| `fin-r1.md` | 金融 CoT 蒸馏、双轮质量筛选、SFT+GRPO 参考 |
| `dianjin-r1.md` | 结构化 `<think>/<answer>` 与双 reward 参考 |
| `Program of Thoughts for Financial Reasoning.md` | FINDER：retriever + dynamic example + PoT 参考 |
| `financereasoning.md` | 外部 benchmark、Python solution、难度统计参考 |
| `cao_data_value_difficulty_aware_training.md` | 数据质量、难度感知、hard-but-verifiable RL 参考 |

### 8.5 下一步工程任务

1. 在 `financial_data_processors` 中新增 `ensemble_thought_sft`。
2. 为 FinQA/ConvFinQA 增加 DSL-to-Python PoT 转换和 execution check。
3. 为 FinanceReasoning 增加评测 loader，先只做 benchmark，不进训练。
4. 为 GRPO 构建 `hard_verifiable_train.jsonl`，基于 SFT pass rate 而不是随机采样。
5. 将 `run_fingpt_v2.ipynb` 中 SFT/GRPO 命令逐步抽成可复现 recipe。
