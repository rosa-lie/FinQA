# FinQA/ConvFinQA 可验证金融表文程序推理后训练框架

本项目基于 `Qwen2.5-7B-Instruct` 构建面向 FinQA 和 ConvFinQA 的 program-executor 后训练框架。项目目标不是训练一个泛金融问答模型，而是把金融表格、文本和对话上下文中的数值推理转化为可执行的 DSL program，再由 executor 计算答案并做数值容差评测。这样可以把证据定位、公式选择、程序可执行性和最终答案正确性拆开诊断，也能把 SFT 和 GRPO 的收益放在同一个可复现评测口径下比较。

当前主链路已经从早期 Fino1-style CoT 设想收敛为 `Qwen2.5-7B-Instruct -> Program SFT -> retention-aware RS-SFT -> current-policy frontier GRPO -> Dr.GRPO checkpoint sweep -> full executor eval`。Fino1 和 FinCoT 仍然是理解金融 Chain-of-Thought 和蒸馏数据构造的背景参考，但不是当前 headline 模型的主要来源，也不作为最终效果归因。

## 当前结论

当前最可靠的项目口径是 full-1237 greedy program-executor evaluation。评测集合由 ConvFinQA test 的 869 条 turn-level 样本和 FinQA test 的 368 条样本组成，解码使用 temperature 0 的单次 greedy 生成，模型只输出 `Evidence + Program`，最终答案由 `evaluation/evaluate_financial_benchmarks.py` 中的 executor 执行 program 后与 gold normalized answer 比较。这里的主指标是 `executed_answer_accuracy`，也就是文档中简称的 `pass@1_greedy`。

| 阶段 | 模型或 checkpoint | full-1237 `pass@1_greedy` | 执行成功率 | 结论 |
| --- | --- | ---: | ---: | --- |
| Base | Qwen2.5-7B-Instruct | 0.272433 | 0.542441 | 基座有一定金融语言理解能力，但 program contract 和计算执行很不稳定。 |
| Program SFT | SFT2 dual/program baseline | 0.578820 | 0.877930 | 主要能力跃迁来自监督学习，模型学会证据抽取和 FinQA DSL 生成。 |
| retention-aware RS-SFT | v34r22 best RS-SFT checkpoint-20 | 0.603072 | 0.905416 | 拒绝采样筛出 executor-correct program，并通过 ConvFinQA 与 FinQA 混合保留单轮能力。 |
| current-policy frontier GRPO | v34r23 checkpoint-10 | 0.606306 | 0.907842 | 在强 RS-SFT 基线后带来约 4 题增益，是短程校准而不是主要能力来源。 |
| Dr.GRPO | v34r24 checkpoint-50 | 0.607114 | 0.907033 | 100-step sweep 中 checkpoint-50 最好，较 RS-SFT 多 5 题，较 v34r23 多 1 题。 |

因此，`pass@1` 从 27.24% 到 60.71% 的总提升不能归因给 GRPO。项目中的主增益来自 Program SFT 和 retention-aware RS-SFT，GRPO 与 Dr.GRPO 的价值主要是基于 current-policy frontier 样本进行小幅、可验证的 late-stage calibration。

## 数据任务

FinQA 在本项目中承担单轮金融表文数值推理基础能力。模型需要从 report text、table 和 question 中定位证据，选择金融计算公式，并生成类似 `divide(subtract(6823, 6161), 6161)` 的可执行 DSL program。ConvFinQA 在本项目中承担多轮 follow-up 推理能力。它的关键不是把最终 QA 的 program 复制到每一个 turn，而是使用 turn-level supervision，即当前问题来自 `annotation.cur_dial[-1]`，当前程序来自 `annotation.cur_program`，当前答案来自 `annotation.exe_ans`。`qa.question`、`qa.program_re` 和 `qa.exe_ans` 只作为 final metadata 或 fallback，不能作为 turn-level 主监督。

项目最终采用 program-only 主线，因为金融数值推理的瓶颈不是让模型心算小数，而是让模型稳定完成证据定位、公式选择和可执行程序生成。`Evidence` 用于检查 grounding，`Program` 用于交给 executor 计算，`Normalized Answer` 由系统执行和归一化得到。这个设计降低了自由文本答案带来的评测噪声，也为 GRPO reward 提供了明确的自动判定信号。

## 后训练路线

Program SFT 是冷启动阶段。它把 FinQA/ConvFinQA 的 gold program 作为监督目标，让模型学会输出稳定 schema 和 DSL。直接从 base 进入 GRPO 效果差，核心原因是 sparse positive 太严重。基座模型大量输出不可执行、格式漂移或 wrong-executable program，组内 reward 方差经常为零，policy gradient 没有足够有效信号。

retention-aware RS-SFT 是当前最重要的能力增强阶段。它先用当前模型采样多个候选，再通过 executor 筛出可执行且答案正确的 program 作为监督样本。为了缓解持续训练遗忘，训练数据按 ConvFinQA 多轮能力数据与 FinQA 单轮保留数据混合，当前项目口径是 65% ConvFinQA 加 35% FinQA 唯一样本。这个阶段把执行成功率从 SFT2 的 87.79% 提升到 90.54%，并把 full-1237 `pass@1_greedy` 提升到 60.31%。

current-policy frontier GRPO 用 RS-SFT 模型重新采样训练集子集，只保留 greedy 预测错误但候选空间中存在 executor-correct program 的样本。这个策略的动机是把 RL 预算集中在“当前策略已经接近答对，但 greedy 行为还没校准好”的边界样本上。v34r23 基于 FinQA 和 ConvFinQA 各 500 条训练样本做 strict acquisition，获得 FinQA 106 条和 ConvFinQA 82 条高价值 frontier 样本，并使用 `frontier_execution_calibration` reward 训练。

Dr.GRPO 是 v34r24 的改进尝试，主要把 TRL 配置映射到 Dr.GRPO 论文中的去长度偏置思想。当前使用 `scale_rewards=none` 和 `loss_type=dr_grpo`，并保持 program executor reward 不变。100-step sweep 的 joint-64 gate 显示 checkpoint-50 优于 checkpoint-100，因此只把 checkpoint-50 推到 full-1237 评测。它的 full result 是 751/1237，即 0.607114。由于只比 v34r23 多 1 题，文档中只把它表述为窄幅校准结果，不声称统计显著或系统性解决长度偏置。

## 代码入口

金融数据处理从 `financial_data_processors/` 进入，统一命令是 `python -m financial_data_processors`。这里负责 FinQA 和 ConvFinQA 的解析、字段归一、strict tier 过滤、program canonicalization、executor 校验和 SFT 数据导出。

SFT 仍然复用通用训练入口 `training/supervised_finetuning.py`。金融任务特有的 prompt、target 和 metadata 由数据处理层负责，训练层只消费已构造好的监督样本。

GRPO 训练集中在 `training/finqa_program_grpo.py`。这个文件负责加载 frontier 数据、生成 rollout、解析 completion 中的 `Program`、调用 executor 计算 reward、计算 group-relative advantage，并通过 TRL GRPOTrainer 更新 LoRA 参数。v34r23 和 v34r24 的数据构造与 sweep 脚本位于 `scripts/`，其中 `build_v34r23_current_policy_manifest.py`、`build_v34r23_frontier_grpo_data.py`、`run_v34r23_grpo40_checkpoint_sweep.py` 和 `run_v34r24_drgrpo_checkpoint_sweep.py` 是当前文档重点对应的实验入口。

评测由 `evaluation/evaluate_financial_benchmarks.py` 完成。该脚本生成 greedy completion，抽取 program，执行 program，比较 normalized answer，并导出 `benchmark_summary.json`、逐条 predictions 和 CSV summary。报告中的 headline 数字只来自 full-1237 greedy executor 口径。

## 文档

更详细的设计记录见 `docs/fin_datasets_v3.md`、`docs/fin_rl.md`、`docs/financial_reasoning_benchmark.md`、`docs/architecture.md` 和 `docs/grpo_current_progress.md`。`README_fail_igore.md` 保留旧文件名，但内容已经改为失败路线和负结果复盘，用于解释为什么项目从 CoT-only、混合泛金融 RL 和早期 GRPO 设想转向 current-policy frontier program-executor 主线。
