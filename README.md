# Verifiable Program-Supervised Financial Reasoning

Verifiable Program-Supervised Financial Reasoning: Improving Fino1-style CoT Training with FinQA/ConvFinQA Program Execution

面向金融表文数值推理的可验证程序监督：在 Fino1 式 CoT 训练范式上的改进

Fino1-style CoT + Verifiable PoT for Financial Numerical Reasoning

本项目是一个 Fino1-style financial reasoning post-training framework 的可验证程序推理改进版。在 Fino1-style CoT 金融推理 baseline 上，引入 FinQA/ConvFinQA 可执行 Program supervision 与 execution-based GRPO 的改进框架。


本项目以 Fino1-style 金融 reasoning SFT + GRPO 作为 baseline 范式，使用公开的 Fino1 Reasoning Path FinQA 和 FinCoT 作为 CoT 数据参考；在此基础上，引入 FinQA/ConvFinQA 的 gold Program 监督，将金融表文数值推理从 CoT-only 扩展到 Evidence-grounded CoT+PoT，并使用 Program execution correctness 进行可验证评估和强化学习优化。


| 实验组 | 说明 | 目的 |
|---|---|---|
| Base model | Qwen2.5-7B-Instruct | 原始基座 |
| Fino1-style SFT | Fino1_Reasoning_Path_FinQA + FinCoT | CoT baseline |
| Program SFT, v2 | FinQA/ConvFinQA `Evidence + Program + Answer + Normalized Answer` | 当前主线 |
| Program executor SFT, v3 | FinQA/ConvFinQA `Evidence + Program` | PoT baseline |
| Fino1-style + Program SFT | core program data + Fino1/FinCoT reasoning data | CoT+PoT |
| Program SFT + GRPO | execution reward | 验证 program RL |
| Fino1-style + Program SFT + GRPO | CoT supplement + execution reward | 最终模型 |

1. **从 CoT-only 到 CoT+PoT**
   - Fino1/FinCoT 提供自然语言 reasoning path。
   - 本项目加入 FinQA/ConvFinQA gold Program。
   - 让模型不只解释，还能输出可执行计算结构。

2. **从 answer reward 到 execution reward**
   - Fino1-style GRPO 常见是 format + answer correctness。
   - 本项目加入：
     ```text
     program parse reward
     program execution reward
     executed answer correctness reward
     program-answer consistency reward
     ```
   - 这让 RL 更适合金融数值题。

3. **从单轮 FinQA 到多轮 ConvFinQA program supervision**
   - 当前 SFT2 使用 ConvFinQA turn-level。
   - 同时加入 FinQA replay 防止单轮表文能力遗忘。
   - 这是 Fino1-style generalized financial reasoning 数据不一定覆盖得很细的地方。

---

**Stage 0: Base model**
Qwen2.5-7B-Instruct

**Stage 1: CoT cold-start**
Fino1_Reasoning_Path_FinQA 为主，FinCoT 低比例
目标：学金融 reasoning 表达，不强行生成 Program

**Stage 2: FinQA Program SFT**
Evidence + Reasoning + Program + Answer + Normalized Answer
目标：把 CoT 意图落到 gold Program

**Stage 3: ConvFinQA Program SFT + FinQA replay**
ConvFinQA turn-level : FinQA replay 约 2:1 或 3:1
目标：多轮推理，同时不忘单轮表文能力

**Stage 4: Verification benchmark**
看 program parse、execution rate、executed answer accuracy、model normalized answer accuracy、pass@k

**Stage 5: GRPO**
主 reward = execute(Program) == gold answer_norm
辅助 reward = format、operator consistency、evidence grounding、scale consistency、brevity

1. Fino1 先提供金融 CoT 语义骨架
模型先学会“为什么用这个指标、为什么这样算”。

2. Program SFT 再校准计算结构
FinQA/ConvFinQA 的 gold Program 会把模型从“会说理”拉回“会执行”。

3. RL reward 更不稀疏
如果模型已经能生成合理 Program，GRPO 的 execution reward 才更容易起作用。

4. 比直接在 sft2_dual_merged 后混 Fino1 更少冲突
后混 CoT 数据容易让模型输出变长、Program 稳定性下降，尤其 FinCoT 不是 gold Program 数据。

---

主任务: FinQA/ConvFinQA 的可验证 Program reasoning
FinQA/ConvFinQA verifiable program reasoning

 - 面向金融表文数值推理的可验证 CoT+PoT reasoning：以 FinQA/ConvFinQA gold Program 为核心监督，让模型从金融文本、表格和多轮上下文中定位证据，生成可执行 Program，并通过 executor 得到可验证答案。

辅助任务:
Fino1/FinCoT financial CoT reasoning supplement

扩展任务:
TATQA / DocFinQA / BizBench / Econ-Logic 泛金融推理泛化

最终能力描述:
verifiable financial numerical reasoning, not generic financial QA

金融文本理解；表格定位；数值推理；单位归一；公式选择；多轮 follow-up；程序生成；答案验证

---

泛金融问答不适合作为当前主任务，原因有几个。

第一，项目差异化：Program 和 executor。
泛金融问答已经有很多模型、数据和 benchmark 可以做，但仓库现在真正独特的是 FinQA/ConvFinQA 的 gold Program、Normalized Answer、program execution、pass@k、GRPO reward。这条线是可验证的，能清楚判断模型有没有变强。

第二，泛金融问答太宽，reward 会变虚。
如果主任务定义成泛金融问答，问题会混进财报理解、投资常识、经济逻辑、政策解释、开放式分析、长文档 QA 等。很多答案没有唯一可执行标准，RL reward 只能退化成格式分、LLM judge 或 answer string match，训练信号会弱很多。

第三，FinQA/ConvFinQA 更适合做 SFT -> RL 的闭环。

泛金融数据可以作为辅助，不该抢主线。
FinCoT、TATQA、DocFinQA、BizBench、Econ-Logic 这些可以作为 supplement，用来增强语言表达、长上下文和领域广度。但它们最好服务于主任务，而不是把主任务改掉。

---

## Abstract

本项目面向金融表文数值推理，目标是在 Fino1-style 金融 reasoning SFT + GRPO 范式上，引入 FinQA/ConvFinQA 的可验证 Program supervision，将 CoT-only 金融推理扩展为 Evidence-grounded CoT/PoT 程序监督框架。金融数值推理不应只依赖自然语言 Chain-of-Thought。CoT 能解释证据选择和公式意图，但它本身不可执行，也难以稳定验证；FinQA 和 ConvFinQA 则提供 gold Program、execution answer 与 normalized answer，因此更适合构建可执行、可评估、可用于 reinforcement learning reward 的 Program-of-Thought。

本仓库的当前重点不是简单复现 Fino1，也不是直接微调 Fino1 checkpoint，而是采用 Fino1/FinCoT 代表的 reasoning SFT + GRPO 思路，并将核心监督从自然语言 reasoning path 扩展到 FinQA/ConvFinQA 的 gold Program supervision。换言之，Fino1-style CoT 训练提供金融推理表达能力，本项目进一步用 `Program:` 和 executor 将推理落到可验证的数值计算上。

## 1. Motivation

金融表文数值推理通常同时包含证据定位、指标理解、单位归一、公式选择和数值计算。

自然语言 CoT 适合解释“为什么选这些数”和“为什么这样算”，例如说明一个问题要求 percentage change，因此需要取当前值和上一期值，先做差，再除以上一期值。但 CoT 的弱点同样明显：它可以写得流畅却算错，可以遗漏单位转换，也难以直接作为 reward 的可靠依据。

PoT，即 Program of Thought，更适合作为金融数值题的计算层。在本项目中，`Program:` 就是第一版 PoT。它把推理中的计算结构写成可解析的 DSL，例如 `divide(subtract(6823, 6161), 6161)`。这类表示比自由文本更低歧义，可以由 executor 执行，并与 `Normalized Answer` 做数值容差比较。PoT 不替代 CoT，而是承担 CoT 不擅长的可执行计算部分。

因此，本项目将金融推理拆成三个互补层次：`Evidence` 负责 grounding，`Reasoning` 负责解释变量选择和公式意图，`Program` 负责可执行计算。当前 v2 主线训练 `Evidence + Program + Answer + Normalized Answer`，保留模型直接生成可读答案和标准答案的能力；v3 主线进一步收敛到 `Evidence + Program`，最终答案由 executor 计算。这种 v3 形式是最干净的 program-executed reasoning：模型不再被要求心算高精度小数，而是被训练成金融表文数值推理编译器。

## 2. Current Framework

当前仓库保留 MedicalGPT 原有训练与服务能力，同时在金融推理方向形成了独立的实验主链。根目录中的 notebook 是实验编排入口，主要包括 `run_fingpt_v2.ipynb`、`run_fingpt_v3.ipynb` 和 `run_fingpt_v2_rl.ipynb`。其中 v2 notebook 对应 dual-answer baseline，v3 notebook 对应 program-executed reasoning，v2 RL notebook 用于承接已训练好的 SFT checkpoint 设计 GRPO 实验。

`financial_data_processors/` 是金融数据主链的核心模块。统一入口为 `python -m financial_data_processors`，负责将 FinQA、ConvFinQA turn-level 等数据族解析、规范化、审计并转换为 SFT/DPO 所需格式。当前 FinQA/ConvFinQA 主链使用 strict-A 过滤，清除 prompt-label 冲突样本，规范化 Program 中的数据集内部常量，并保留 `answer_display`、`answer_norm`、`program_canonical`、`answer_unit`、`answer_scale` 等 metadata。这个模块使金融数据处理从 notebook 临时代码变成可复用的数据路由层。

`training/` 承接 SFT、DPO、GRPO、PPO、ORPO、reward modeling 等训练实现。金融主线目前复用 `training/supervised_finetuning.py` 的截断与训练逻辑，不为了 FinQA/ConvFinQA 额外修改底层 SFT trainer；金融任务特有的 prompt、target、metadata 和 audit 逻辑集中在 processor 与 notebook 中。这样的划分让训练框架保持通用，而领域约束留在数据层。

`evaluation/` 负责评估，尤其是 `evaluation/evaluate_financial_benchmarks.py`。该脚本支持 FinQA/ConvFinQA generation benchmark、Program 解析、Program execution、normalized answer 对比以及 pass@k 评估。项目不只看训练 loss，而是以生成式 benchmark 判断真实能力，因为金融数值推理的关键错误往往发生在取数、单位、公式和输出格式上，而不一定能从 loss 直接看出。

`docs/` 保存稳定的数据规范和实验说明，尤其是 `docs/fin_datasets_v2.md` 与 `docs/fin_datasets_v3.md`。v2 文档定义 dual-answer baseline，v3 文档定义 program-executor 主线。`ref/` 保存 Fino1、FinCoT、Program of Thoughts、ETD、Fin-R1、DianJin-R1、FinanceReasoning 和 difficulty-aware training 等研究参考，用于解释本项目相对于金融 reasoning 后训练工作的定位。

## 3. Data and Training Pipeline


参考数据集

| 数据集 | 规模/字段 | 来源 | 价值 | 主要风险 |
|---|---:|---|---|---|
| `czyssrs/FinQA` | | |
| `AdaptLLM/ConvFinQA/` | | |
| `TheFinAI/Fino1_Reasoning_Path_FinQA` | `train` 约 5.5k rows；字段是 `Open-ended Verifiable Question`、`Ground-True Answer`、`Complex_CoT`、`Response` | 明确来自 **FinQA**，用 GPT-4o 给 FinQA 问答生成 reasoning path | 和现在 FinQA/Program 主链最贴近，容易对齐 gold Program、answer_norm、execution benchmark | 只覆盖 FinQA，数据多样性较窄 |
| `TheFinAI/FinCoT` | 约 9.19k rows；`SFT` 约 7.69k，`RL` 约 1.5k；字段是 `Question`、`Reasoning_process`、`Final_response`、`Negative_reasoning_process`、`Negative_response` | 混合 FinQA、ConvFinQA、TATQA、DocMath-Eval、Econ-Logic、BizBench-QA、DocFinQA 等 | 金融 reasoning 多样性更强，也有 SFT/RL 风格字段 | 分布更杂、上下文可能极长、格式和 Program DSL 主链不完全一致，且部分源数据如 Econ-Logic 有非商业限制 |

---

**Fino1_Reasoning_Path_FinQA 为主，FinCoT 低比例**

`Fino1_Reasoning_Path_FinQA` 本身来自 FinQA，所以它的问题形式、上下文结构、数值答案风格都更接近现在的 FinQA strict-A、v2/v3 Program SFT 数据。它适合做 cold-start：先教模型用自然语言解释“为什么这样算”。

本地 FinQA 有 gold `Program`、`answer_norm`、executor。Fino1 的 CoT 虽然没有 Program，但因为它源于 FinQA，后续可以更容易映射到本地 FinQA 样本，形成：
```text
Evidence:
Reasoning:  来自 Fino1 / teacher CoT
Program:    来自本地 FinQA gold program
Answer:
Normalized Answer:
```
比用 FinCoT 混合数据安全得多。FinCoT 里很多样本来自 TATQA、DocMath、BizBench、DocFinQA、Econ-Logic 等，不一定有当前 executor 可执行的 DSL Program。

FinCoT 好处是覆盖面广，可以让模型接触更丰富的金融推理表达。但它也会把任务分布拉散：有长文档、多文档、经济逻辑、业务问答、不同答案格式。对现在这种强依赖 `Evidence + Program + Answer + Normalized Answer` 的 pipeline 来说，比例太高会让模型更像泛金融 CoT 模型，而不是稳定的 Program compiler。
**FinCoT 更适合作为 robustness / style supplement**。它可以补一些 Fino1 没有的金融场景，但最好不要让它主导训练目标。尤其正式进入 Program SFT 和 GRPO 前，主干必须仍然是 FinQA/ConvFinQA gold Program，否则 execution reward 会变稀疏，甚至 reward profile 不一致。

```text
CoT cold-start:
  Fino1_Reasoning_Path_FinQA: 70% - 90%
  FinCoT SFT:                 10% - 30%

Program SFT:
  本地 FinQA/ConvFinQA gold Program: 主体
  Fino1/FinCoT: 只作为低比例 Reasoning supplement，不能覆盖 gold Program

GRPO:
  FinQA/ConvFinQA program_numeric 样本为主
  FinCoT 只用于 cot_answer_only 或暂时不进 execution reward
```

> 可以调整比例跑ablation看数据集比例带来的影响。

---

当前最重要的完成结果是 `sft2_dual_merged`。它不是普通的单阶段 SFT checkpoint，而是 FinQA/ConvFinQA 程序监督金融数值推理主线的 v2 baseline。SFT-1 使用 FinQA strict-A 训练单轮金融表文数值推理，让模型学习从 report context、table 和 question 中定位证据、生成 Program，并输出可读答案与标准化答案。SFT-2 使用 ConvFinQA turn-level strict-A 训练多轮 follow-up reasoning，同时加入 FinQA replay 保持单轮表文能力，默认混合比例约为 `ConvFinQA:FinQA = 2:1`。

FinQA 和 ConvFinQA 在本项目中的角色不同。FinQA 更适合作为单轮表文数值推理基础能力，重点是表格、文本、公式和单位的对齐；ConvFinQA 则引入 conversation history，要求模型理解当前问题与历史 turn 的关系。SFT-2 使用 turn-level 数据，而不是把 final program 复制到每一个 turn，这避免了多轮监督中的标签错配。ConvFinQA 的 history 放在 prompt 后部；当 prompt 过长时，数据处理策略优先保留当前 question 和 output rule，而不是保留全部 history。

v2 target 训练模型同时生成证据、程序、可读答案和标准答案。它适合构建一个完整回答型 baseline，也适合诊断模型是否能稳定输出 `Normalized Answer`。v3 target 则去掉模型生成答案的负担，只监督 evidence selection 和 Program generation，最终答案由 executor 执行 Program 得到。

```text
v2 dual_answer_sft:
Evidence:
- ...

Program: divide(subtract(6823, 6161), 6161)

Answer: 10.745%

Normalized Answer: 0.10745

v3 program_executor_sft:
Evidence:
- ...

Program: divide(subtract(6823, 6161), 6161)
```

这两条路线并行存在。v2 保留模型直接生成答案的能力，是当前 `sft2_dual_merged` 的主要基线；v3 将任务定义得更接近可验证 PoT，即模型生成可执行程序，系统负责执行、归一化和展示答案。这样的设计也为后续 GRPO 提供了更可靠的 reward 入口，因为 `execute(Program) == gold answer_norm` 比“自然语言 reasoning 看起来合理”更容易自动判定。

当前数据产物继续沿用清晰的版本化目录。v2 数据位于 `/root/autodl-tmp/data/financial_reasoning_v2`，v3 数据位于 `/root/autodl-tmp/data/financial_reasoning_v3`。v2 核心文件包括 `train_sft1_dual_strict.jsonl`、`train_sft2_convfinqa_turn_dual_strict.jsonl`、`train_sft2_dual_balanced.jsonl` 与 `valid_dual_balanced.jsonl`；v3 对应文件为 `train_sft1_program_strict.jsonl`、`train_sft2_convfinqa_turn_program_strict.jsonl`、`train_sft2_program_balanced.jsonl` 与 `valid_program_balanced.jsonl`。详细字段、审计口径和重建命令见 `docs/fin_datasets_v2.md` 与 `docs/fin_datasets_v3.md`。

## 4. Evaluation

本项目的评估口径强调 generation benchmark，而不是只看训练 loss。金融表文数值推理的模型可能在 loss 上继续下降，却在真实生成时出现单位混淆、Program 不可执行、答案字段缺失或 history-dependent 问题退化。因此 benchmark 同时报告 answer accuracy、program accuracy、program execution rate、executed answer accuracy、model normalized answer accuracy、numeric parse rate、structured response coverage、average prediction length，以及 `pass@1`、`pass@4`、`pass@8`。

v2 和 v3 的主指标需要区分。v2 的 `dual_answer_sft` 应同时关注 `model_normalized_answer_accuracy` 与 Program 相关指标，因为模型本身负责输出 `Normalized Answer`。v3 的 `program_executor_sft` 则以 `executed_answer_accuracy` 为核心，因为标准答案来自 executor。评估脚本目前已经支持 program execution 口径，因此比较 v2/v3 时必须明确当前看的是模型直接答案能力，还是 Program 执行后的答案能力。

已有 quick benchmark 显示，`sft2_dual_merged` 是当前 v2 主 baseline。它相较 base 与 SFT1 更稳定地同时提升答案和程序指标；DPO 没有明显超过 SFT2，说明偏好学习或格式层面的空间并不是下一阶段的主要增益来源。更合理的下一步是 program-verifiable GRPO：利用 pass@k 中已经存在的正确候选，把正确 Program 从 sampled candidates 推向 greedy/high-probability 输出。

pass@k 在本项目中不仅是评估指标，也是训练策略信号。如果 `pass@8` 明显高于 `pass@1`，说明模型的采样空间中已有正确程序，但概率不够高，适合用 GRPO 或 verifier-guided optimization 提升稳定性。如果 `pass@1` 与 `pass@8` 都低，则说明模型基础能力不足，应回到 SFT 或数据质量；如果二者都高且差距很小，则继续 RL 的提升空间有限。

## 5. Future Improvements

下一阶段的第一条提升路线是引入公开 Fino1/FinCoT 数据作为外部 CoT supplement。`TheFinAI/Fino1_Reasoning_Path_FinQA` 与 FinQA 风格更接近，适合补充 FinQA-style reasoning path；`TheFinAI/FinCoT` 覆盖更广金融 reasoning，可作为低比例外部数据。它们的定位是补充 CoT，而不是替代 FinQA/ConvFinQA gold Program。初始混合比例应保守，例如 core program data 90%，Fino1/FinCoT 10%，并通过 benchmark 检查 ConvFinQA 是否遗忘、program accuracy 是否下降、average output length 是否膨胀。

第二条提升路线是新增 `Reasoning:` 字段，形成 CoT+PoT 或 ETD 风格的训练分支。核心样本可以从当前 FinQA/ConvFinQA normalized records 生成，结构为 `Evidence + Reasoning + Program + Answer + Normalized Answer`。其中 `Reasoning:` 应保持短，解释证据选择和公式意图，而不是训练长篇 `<think>`。这一路线的目标不是让模型写更多文本，而是让 `Program:` 获得更好的语义支撑，减少“程序可执行但证据错配”的问题。

第三条提升路线是 program-verifiable GRPO。GRPO 不应继续重奖已经学会的格式，而应把主 reward 放在 `execute(Program) == gold answer_norm` 上，并辅以 strict Program parse、operator consistency、evidence number grounding、brevity 和 percent/ratio scale consistency。RL 数据不应随机抽取普通 SFT 样本，而应通过 pass@k mining 选择 hard-but-verifiable 样本，例如当前模型 `pass_rate` 位于中等区间、答案短、Program 可执行、自动判分可靠的 FinQA/ConvFinQA 样本。

更长远地，项目可以从 DSL Program 扩展到 Python Program，但应先保证安全执行器和 strict DSL executor 稳定。Python-level PoT 可以提升与 Program-of-Thoughts 和 FinanceReasoning 的兼容性，但也引入安全执行、硬编码答案和代码风格不稳定等风险。因此第一阶段应优先把现有 DSL Program 的 execution reward 做准，再逐步引入 DSL-to-Python、AST whitelist executor 和 self-consistency voting。

## 6. Relation to Fino1 and Prior Work

本项目可以被理解为 Fino1-style baseline 的方法论改进，而不是官方 Fino1 的复现。Fino1/FinCoT 代表了金融 reasoning SFT + GRPO 的重要方向：用金融 reasoning path 注入推理能力，再通过 reinforcement learning 提升复杂金融任务表现。本项目采用这一后训练范式，但将核心监督从 CoT reasoning path 扩展到 FinQA/ConvFinQA gold Program supervision，并引入 execution-based evaluation and reward。

这种定位也解释了为什么不能直接把 FinCoT 当作主训练集。FinCoT 的 `Reasoning_process` 能帮助模型学习金融推理表达，但它不是 FinQA/ConvFinQA 的 gold Program。对于本项目，外部 CoT 是补充层，FinQA/ConvFinQA Program 是主干层。无法对齐 gold Program 的外部样本可以标记为 `Program: N/A`，用于 answer/format/reasoning supervision，但不应参与 Program execution reward。

Program-of-Thoughts 提供了“模型生成程序，解释器负责计算”的核心思想，FINDER 进一步说明金融场景中 evidence retrieval 和 dynamic example selection 的价值。ETD 强调 CoT、PoT、EoT 的组合蒸馏，提示我们不应只蒸馏长 CoT，而应让自然语言 reasoning、可执行 Program 和 execution check 互相校验。Fin-R1 和 DianJin-R1 则说明金融领域 SFT + GRPO 是有效范式，但它们更偏 CoT/format/answer reward；本项目的差异在于使用 FinQA/ConvFinQA 原生 Program，把 reward 绑定到可执行数值推理。

FinanceReasoning 与 difficulty-aware training 为后续扩展提供了两个方向。前者提供带 Python solution 的更严格金融数值 benchmark，适合作为外部 PoT 评测与难度分层分析；后者强调 high-quality SFT data 与 hard-but-verifiable RL data 的价值，和本项目的 pass@k mining、execution reward 路线一致。

## References

Fino1-style reasoning data and training are the main external baseline for future comparison. See Fino1 at https://github.com/The-FinAI/Fino1, FinCoT at https://huggingface.co/datasets/TheFinAI/FinCoT, and Fino1 Reasoning Path FinQA at https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA.

Program-supervised and execution-based reasoning are documented in `ref/program_of_thoughts_chen2023_research.md`, `ref/etd_cot_pot_eot_distillation_research.md`, and `ref/fino1_fincot_integration_research.md`. These notes motivate the distinction between CoT, PoT and EoT, and explain why FinQA/ConvFinQA gold Program should remain the core supervision signal.

Financial reasoning post-training and related baselines are summarized in `ref/fin-r1.md`, `ref/dianjin-r1.md`, and `ref/financereasoning.md`. Difficulty-aware and data-centric training considerations are summarized in `ref/cao_data_value_difficulty_aware_training.md`.

Detailed local data specifications are maintained separately in `docs/fin_datasets_v2.md` and `docs/fin_datasets_v3.md`. The v2 document describes the current dual-answer baseline, while the v3 document describes the program-executed reasoning path.

---
