# Done: PoT+distill CoT -> SFT

**面向 `FinQA + ConvFinQA` 的可验证金融数值推理。**
`Program-supervised SFT -> execution-based benchmark -> verifiable RL`

1. 核心监督应来自 `FinQA` 和 `ConvFinQA` 的 gold `Program`。
2. 训练目标应优先让模型生成可执行、可校验的 `Program`，而不是只生成一段看起来合理的自然语言推理。
3. 评估应以 `program execution` 和 `executed answer accuracy` 为主，而不是只看训练 loss。
4. 后续 RL 应围绕可验证 reward 展开，最自然的是 `program correctness + executed answer correctness + structure constraints`。

## 数据集

### FinQA

`FinQA` 在当前项目里的角色是单轮金融表文数值推理主干。

1. 学会从文本 + 表格 + question 中定位 evidence。
2. 学会生成 FinQA 风格的 program。
3. 为后续 ConvFinQA 多轮 follow-up 推理提供 program prior。

当前主链中，`SFT-1` 都只使用 `FinQA`：

- strict SFT 样本数：`3686`
- 原始输入行数：`6251`

### ConvFinQA

`ConvFinQA` 在当前项目里的角色是多轮 follow-up 数值推理增强。

关键点不是“把最终答案复制到每个 turn”，而是严格使用 turn-level supervision：

- current question 来自 `annotation.cur_dial[-1]`
- current program 来自 `annotation.cur_program`
- current answer 来自 `annotation.exe_ans`
- `qa.question / qa.program_re / qa.exe_ans` 只保留为 final metadata 或 fallback，不能作为 turn-level 主监督

当前 audit 已经确认：

- `convfinqa_current_question_metadata_mismatch_rows = 0`
- `convfinqa_final_question_used_as_current_rows = 0`

### pipeline

当前 pipeline 已经形成闭环：

1. `financial_data_processors/router.py`
2. `financial_data_processors/families/finqa.py`
3. `financial_data_processors/families/convfinqa_turn.py`
4. `financial_data_processors/common.py`
5. `training/supervised_finetuning.py`
6. `training/dpo_training.py`
7. `tooling/merge_peft_adapter.py`
8. `evaluation/evaluate_financial_benchmarks.py`

数据构建策略也已经比较稳定：

1. `SFT-1`：只用 `FinQA`
2. `SFT-2`：`ConvFinQA turn-level strict + FinQA replay`
3. 默认混合比例约为 `ConvFinQA:FinQA = 2:1`
4. v3 audit 已经做到 `program_parse_success = 1.0`
5. v3 audit 已经做到 `program_execution_success = 1.0`
6. v3 audit 已经做到 `program_answer_match_rate = 1.0`

## Experiments

## evidence + program + answer + normalized answer

对应 `run_fingpt_v2.ipynb`

1. `SFT-1(FinQA)`：学习单轮 program supervision
2. `SFT-2(ConvFinQA turn-level + FinQA replay)`：强化多轮 follow-up reasoning
3. `DPO`

| model | answer acc | pass@1 greedy | pass@1 sampled | program acc | pass@4 | pass@8 | avg chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.6250 | 0.6250 | 0.8125 | 0.0625 | 0.8125 | 0.8125 | 539.8 |
| sft1_dual | 0.2500 | 0.2500 | 0.3750 | 0.3750 | 0.5625 | 0.5625 | 234.1 |
| sft2_dual_merged | 0.8125 | 0.8125 | 0.8125 | 0.8125 | 0.9375 | 0.9375 | 268.3 |
| dpo | 0.8125 | 0.8125 | 0.7500 | 0.8125 | 0.9375 | 0.9375 | 269.0 |

1. `SFT-2` 明显有效，是 v2 主 baseline。
2. `SFT-1` 虽然让模型更会写 program 格式，但答案准确率明显退化。
3. `DPO` 基本没有超过 `SFT-2`，只能算格式层面对照组，不是主增益来源。

分任务看，`SFT-2` 相对 base：

- `ConvFinQA: 0.75 -> 0.875`
- `FinQA: 0.50 -> 0.75`

因此 v2 的真正收益来自：`FinQA 单轮 program 先学稳 -> ConvFinQA turn-level 再强化`

## evidence + program -> answer

对应 `run_fingpt_v3.ipynb`

**模型负责生成 evidence + program，最终答案由 executor 计算。**

1. 重点从“模型能不能把答案字符串写对”转到“模型能不能生成正确 program”。
2. benchmark 可以直接报告 `program_execution_rate` 和 `executed_answer_accuracy`。
3. 这条线更贴近当前任务定义，也更适合作为后续 RL 起点。

`sft1(finqa) + sft2(convfinqa)`

| model | primary / executed acc | pass@1 greedy | pass@1 sampled | program acc | pass@4 | pass@8 | avg chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.3125 | 0.3125 | 0.2500 | 0.2500 | 0.3750 | 0.5000 | 290.4 |
| sft1_program | 0.6875 | 0.6875 | 0.5000 | 0.5625 | 0.8125 | 0.8750 | 230.4 |
| sft2_program_merged | 0.8750 | 0.8750 | 0.8125 | 0.7500 | 0.9375 | 0.9375 | 144.8 |

结论：

1. v3 明显优于 v2，是当前最好的 SFT 主线。
2. `SFT-1` 已经能把 `FinQA` 拉起来。
3. `SFT-2` 进一步把 `ConvFinQA` 拉起来，所以整体最稳。

按任务看：

- `base`: ConvFinQA `0.375`，FinQA `0.25`
- `sft1_program`: ConvFinQA `0.50`，FinQA `0.875`
- `sft2_program_merged`: ConvFinQA `0.875`，FinQA `0.875`

## reasoning + evidence + program -> answer

对应 `run_fingpt_cot_pot.ipynb`

1. 先用外部 CoT 冷启动提升 reasoning expression。
2. 再回到 FinQA / ConvFinQA 的 Program SFT 主线。
3. 看 CoT 是否能真实提升可执行数值推理，而不是只增加输出长度。

当前 CoT cold-start 配置：

- `USE_COT_COLD_START = True`
- `USE_FINO1 = True`
- `USE_FINCOT = True`
- `FINO1_RATIO = 0.8`
- `FINCOT_RATIO = 0.2`

| model | primary / executed acc | pass@1 greedy | pass@1 sampled | program acc | pass@4 | pass@8 | avg chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_reasoning | 0.3125 | 0.3125 | 0.3125 | 0.3125 | 0.4375 | 0.4375 | 488 左右 |
| cot_reasoning | 0.1875 | 0.1875 | 0.1875 | 0.1875 | 0.3750 | 0.4375 | 668 左右 |
| program_mixed_merged | 0.6250 | 0.6250 | 0.1875 | 0.4375 | 0.7500 | 0.8125 | 464 左右 |

1. 纯 CoT cold-start 明显无益，甚至退化。
2. `CoT + mixed Program SFT` 相比 base 有提升，但仍明显弱于 v3 `sft2_program_merged`。
3. 这说明当前 CoT 数据并没有自然转化为更强的可执行金融 program reasoning。

所以目前不能说 “CoT 对当前任务有稳定提升”。更准确的说法是：**现阶段 CoT 只能作为补充信号，不能替代 FinQA / ConvFinQA 的 gold Program 主监督。**

## benchmark eval

当前 benchmark 口径已经比较明确：

1. `dual_answer_sft`：`Evidence + Program + Answer + Normalized Answer`
2. `program_executor_sft`：`Evidence + Program`
   最终答案由 executor 执行 program 得到
3. `cot_program`：`Reasoning + Evidence + Program`
   用于 CoT+PoT 评估

评估入口在 `evaluation/evaluate_financial_benchmarks.py`。

### 评估流程

当前 eval 不是简单对模型输出做字符串比对，而是完整走一遍“生成 -> 解析 -> 执行 -> 对齐 gold”的流程：

1. 从 `FinQA test` 和 `ConvFinQA dev_turn` 构造 benchmark example。
2. prompt 构造复用 data processor 的逻辑，保证评估口径和训练口径一致。
3. 每个样本先做一次 greedy generation，再做 8 次 sampled generation。
4. 从输出中解析 `Reasoning`、`Evidence`、`Program`、`Answer`、`Normalized Answer`。
5. 对 `Program` 做清洗和 canonicalize，然后调用 executor 执行。
6. 把执行结果与 gold answer 做数值比较，得到单题正确性。
7. 汇总成 `task_name`、`bucket_*` 和 `macro_average` 三层 summary。

因此 benchmark 不再只看 loss，而是重点看“模型有没有真正生成可执行程序，以及执行后是否得到正确答案”。

### 具体指标

1. `answer_accuracy`
   对 numeric 任务，本质上等于执行后答案是否正确。
2. `primary_metric`
   当前这几组 run 都设为 `executed_answer_accuracy`。
3. `program_accuracy`
   预测的 `Program` 和 gold `Program` 在规范化后是否一致。
4. `program_parse_rate`
   是否成功抽取出 `Program` 字段。
5. `program_execution_rate`
   抽取出的 `Program` 是否真的可以执行出数值。
6. `executed_answer_accuracy`
   执行 `Program` 后的数值是否和 gold answer 一致。
7. `model_normalized_answer_accuracy`
   模型自己写出来的答案数值是否正确。
8. `program_answer_consistency`
   模型写出来的答案是否和它自己的 `Program` 执行结果一致。
9. `pass@1_greedy`
   greedy 输出是否正确。
10. `pass@1_sampled`
    第一个 sampled candidate 是否正确。
11. `pass@4` / `pass@8`
    前 4 个或前 8 个 sampled candidate 里是否至少有一个正确。
12. `structured_response_coverage`
    是否包含预期结构锚点。
13. `numeric_parse_rate`
    输出中是否能解析出数值。
14. `reasoning_coverage`
    是否真的输出了 `Reasoning:` 段。
15. `avg_prediction_chars`
    平均输出长度。

### 当前三组 benchmark 的配置

从 `benchmark_manifest.json` 看，当前主对比 run 基本是同一套设置：

1. 总样本数 `16`
2. `FinQA 8` + `ConvFinQA 8`
3. `pass_k = [1, 4, 8]`
4. 每题采样 `8` 次
5. `sample_temperature = 0.7`
6. `sample_top_p = 0.95`
7. `sample_seed = 42`

因此 `v2`、`v3` 和 `cot_pot` 三组结果可以直接横向比较。

### 结果分析

#### v2

`financial_reasoning_v2/benchmarks`

1. `base`: `answer_accuracy = 0.625`
2. `sft1_dual`: `0.250`
3. `sft2_dual_merged`: `0.8125`
4. `dpo`: `0.8125`

结论：

1. `SFT-2` 是 v2 的真正有效增益来源。
2. `SFT-1` 提高了 program 格式能力，但没有把最终答案正确性一起带起来。
3. `DPO` 没有超过 `SFT-2`，`pass@1_sampled` 还从 `0.8125` 降到 `0.75`，说明它更像格式微调，而不是主能力提升。

按任务拆开看：

1. `ConvFinQA: 0.75 -> 0.875`
2. `FinQA: 0.50 -> 0.75`

这说明 v2 的主收益来自：

`FinQA 单轮 program 先学稳 -> ConvFinQA turn-level 再强化`

#### v3

`financial_reasoning_v3/benchmarks`

1. `base`: `executed_answer_accuracy = 0.3125`
2. `sft1_program`: `0.6875`
3. `sft2_program_merged`: `0.875`

结论：

1. v3 是当前最强主线。
2. 把任务改成“模型生成 `Program`，最终答案交给 executor”之后，训练目标和评测目标终于完全一致了。
3. `SFT-1` 已经显著提升 FinQA program learning。
4. `SFT-2` 进一步把 ConvFinQA 多轮 follow-up reasoning 拉起来。

按任务拆开看：

1. `base`: ConvFinQA `0.375`，FinQA `0.25`
2. `sft1_program`: ConvFinQA `0.50`，FinQA `0.875`
3. `sft2_program_merged`: ConvFinQA `0.875`，FinQA `0.875`

另一个很强的信号是 `avg_prediction_chars`：

1. `base = 290.4`
2. `sft1_program = 230.4`
3. `sft2_program_merged = 144.8`

输出更短，但结果更强，说明模型不是靠更长的解释提升，而是靠更直接、更稳定的 program generation 提升。

#### CoT + PoT

`financial_reasoning_cot_pot/benchmarks`

1. `base_reasoning`: `executed_answer_accuracy = 0.3125`
2. `cot_reasoning`: `0.1875`
3. `program_mixed_merged`: `0.625`

结论：

1. 纯 CoT cold-start 没有帮助，反而退化。
2. `CoT + mixed Program SFT` 相比 base 有提升，但仍明显弱于 v3 的 `sft2_program_merged = 0.875`。
3. 这说明当前 CoT 数据并没有自然转化为更强的可执行金融 program reasoning。

因此当前不能说 CoT 对主任务有稳定提升。更准确的说法是：

**CoT 只能作为补充信号，不能替代 FinQA / ConvFinQA 的 gold Program 主监督。**

### 横向总结

如果把当前三条主线放在一起看：

1. `v3 sft2_program_merged` 最强
2. `v2 sft2_dual_merged` 次之
3. `cot_pot program_mixed_merged` 有一定效果，但明显弱于 v3
4. `DPO` 不是主增益来源

从指标层面看，还有三个很重要的观察：

1. `program_execution_rate` 在 v3 基本已经接近 `1.0`，说明输出格式和程序可执行性已经很稳。
2. `program_accuracy` 往往低于 `executed_answer_accuracy`，说明很多题不是字符串级完全匹配，但执行结果已经正确。
3. `pass@4 / pass@8` 明显高于 `pass@1_greedy`，说明模型已经“有能力”，但还没有稳定把正确 candidate 推到最高概率输出。

这也是为什么下一阶段最值得做的不是继续堆 CoT 或 DPO，而是：

`v3 program_executor_sft -> execution-based GRPO`

# Question?

## task

### 如何确定任务

当前的任务场景：

1. 比较聚焦于 `evidence + program + answer`
2. 数据来源和核心是 `FinQA` 和 `ConvFinQA` 的 `program`
3. 可以做可验证的强化学习

所以当前最可信的任务定义是：

**基于 FinQA / ConvFinQA gold Program 的金融数值推理后训练。**

形式上更推荐：

`Evidence + Program -> executor answer`

而不是：

`Reasoning + long CoT -> answer`

### 数据集

#### 是否还需要加入其他类型的金融数据集

可以加入，但不能替代当前主干。

例如金融问答数据集，可以作为补充型数据，而不是主监督数据。

references:
fin-r1: https://mp.weixin.qq.com/s?__biz=MzU3MzM4NjI0NQ==&mid=2247508101&idx=1&sn=7e46df3a369f5ad3d38326ebb2d5e116&poc_token=HCyy2GmjYPAPnv3rsE8yjocC472sR2-27O8ErYds

- 为将 DeepSeek-R1 的推理能力迁移至金融场景并解决高质量金融推理数据问题，我们用 Deepseek-R1 针对行业语料、专业认知、业务知识、表格解析、市场洞察、多轮交互和量化投资等多个数据集进行蒸馏筛选，构建高质量金融 COT 数据集。
- 该数据集覆盖中英文金融垂直领域多维知识，可支撑银行、基金和证券等场景。
- 其方法强调“答案 + 推理”双轮质量筛选，说明金融领域蒸馏数据并不是越多越好，而是要高质量和强过滤。

benchmark:

这类数据更适合做：

1. reasoning supplement
2. 泛金融迁移补充
3. CoT 冷启动对照

不适合直接替代：

1. `FinQA` 的单轮 program supervision
2. `ConvFinQA` 的 turn-level program supervision

#### 是否要加入通用领域数据集

`Unlocking data value in finance a study on distillation and difficulty-aware training` 的结论更偏向于：

**混入通用数学或通用 CoT 数据反而可能伤害金融任务表现。**

这说明金融 reasoning 不是“多加一点通用推理数据”就能解决的。

更合理的策略是：

1. 核心主干仍然是 `FinQA + ConvFinQA`
2. 外部金融 CoT 只做低比例补充
3. 通用 CoT 不宜大比例混入

#### distill

distill 可以做，但要明确角色：

1. 作为 `CoT cold-start` 补充
2. 作为对照实验
3. 不能替代 gold Program 主监督

## PoT?

当前项目里最有价值的其实就是 `PoT / Program supervision` 方向。

因为它满足：

1. 可验证
2. 可执行
3. 可做 pass@k mining
4. 可做 execution-based RL

## CoT? distill?

CoT 对当前任务有提升吗？

当前证据表明：

1. 纯 CoT cold-start 没有稳定提升，反而退化。
2. `CoT + Program SFT` 有一定帮助，但仍弱于 v3 program 主线。
3. 所以 CoT 只能作为补充信号，不能作为当前主任务核心。

## SFT: 多阶段?

当前最有效的就是多阶段 SFT：

1. `SFT-1(FinQA)` 学单轮表文 program reasoning
2. `SFT-2(ConvFinQA turn-level + FinQA replay)` 学多轮 follow-up reasoning

这个多阶段设计在 v2 和 v3 都有效，但 v3 更优。

## DPO: chosen & rejected?

DPO 更适合作为比较对象。

当前结果表明：

1. DPO 没有超过 `SFT-2`
2. 它不是当前主增益来源
3. 因此它更适合作为对照实验，而不是主线

chosen & rejected 如何构造？

当前代码路径已经支持基于 strict target 构造轻量 DPO pair，核心思路是：

1. `chosen` 使用 strict target
2. `rejected` 使用程序扰动、数字扰动或结构扰动

rejected 程序扰动质量不好？

是的，这正是当前 DPO 不强的一个可能原因：

1. rejected 过于人工
2. 偏好信号可能主要在格式层面
3. 没有真正触及当前任务的 program correctness 瓶颈

## GRPO

当前最值得继续推进的是 execution-based GRPO。

最自然的 reward 方向：

1. `program parse success`
2. `program execution success`
3. `executed answer correctness`
4. `structure completeness`

所以当前的推荐主线已经比较明确：

`v3 program_executor_sft -> execution-based GRPO`
