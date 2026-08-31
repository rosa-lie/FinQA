# 项目架构整理

本文记录当前仓库中金融推理子项目的真实模块关系。仓库仍保留 MedicalGPT 原有训练、推理和服务能力，但 FinQA/ConvFinQA 主线已经形成独立的数据、训练、评测和实验脚本闭环。

## 1. 总体流向

当前金融主链可以概括为 `raw FinQA/ConvFinQA -> financial_data_processors -> Program SFT/RS-SFT data -> training -> frontier acquisition -> GRPO/Dr.GRPO -> evaluation/evaluate_financial_benchmarks.py`。模型输入是金融文本、表格、当前问题和必要的对话历史。模型输出是 `Evidence + Program`。评测阶段由 executor 执行 program，得到 normalized answer，并与 gold answer 比较。

这个架构的关键是训练框架保持通用，金融任务约束集中在数据处理、prompt contract、program parser、executor reward 和 benchmark evaluator 中。这样可以避免把 FinQA 特定逻辑散落到通用 SFT trainer 里。

## 2. 数据处理层

`financial_data_processors/` 是金融数据主入口。它负责读取 FinQA、ConvFinQA turn-level 数据，做 strict tier 过滤、字段归一、program canonicalization、answer normalization、executor 校验和 SFT JSONL 导出。统一入口是 `/root/miniconda3/bin/python -m financial_data_processors`。

ConvFinQA 的处理必须使用 turn-level supervision。当前问题、当前 program 和当前答案来自 `annotation.cur_dial[-1]`、`annotation.cur_program` 和 `annotation.exe_ans`。final QA 字段只作为 metadata 或 fallback。这一点属于数据层约束，不应该交给训练脚本临时修补。

## 3. 训练层

普通 SFT 继续使用 `training/supervised_finetuning.py`。它负责 tokenizer、dataset、LoRA、DeepSpeed/accelerate 配置、截断和优化更新。金融任务的 prompt 和 target 已经在数据处理层构造好，因此 SFT trainer 不需要理解 FinQA DSL。

GRPO 主入口是 `training/finqa_program_grpo.py`。它是金融子项目中最重要的 RL 训练文件，负责加载 frontier 数据，对每个 prompt 采样多个 completion，解析 `Program`，调用 executor 得到 reward，并交给 TRL GRPOTrainer 完成 logprob、advantage、clipped objective 和参数更新。v34r24 在这里映射 Dr.GRPO 配置，重点参数包括 `scale_rewards=none` 和 `loss_type=dr_grpo`。

## 4. frontier 与 sweep 脚本

`scripts/build_v34r23_current_policy_manifest.py` 用于基于当前 RS-SFT policy 重新采样训练样本，并记录 greedy 与 sampled candidates。`scripts/build_v34r23_frontier_grpo_data.py` 把 greedy-wrong、pass@8-executor-correct 和 executable-wrong hard negative 样本整理成 GRPO 数据。`scripts/v34r23_prompt_processor.py` 维护 strict program prompt contract，禁止 Reasoning/Answer marker、多个 Program section 和 assignment-style program。

`scripts/run_v34r23_grpo40_checkpoint_sweep.py` 负责 v34r23 frontier GRPO 的 checkpoint sweep。`scripts/run_v34r24_drgrpo_checkpoint_sweep.py` 和 `scripts/evaluate_v34r24_drgrpo_long_sweep.py` 负责 Dr.GRPO sweep 与筛选。v34r24 的 100-step sweep 中 checkpoint-50 通过 joint-64 gate，checkpoint-100 未超过它，因此只把 checkpoint-50 推进 full-1237 evaluation。

## 5. 评测层

`evaluation/evaluate_financial_benchmarks.py` 是当前项目 headline 结果的唯一评测入口。它支持 FinQA/ConvFinQA 数据加载、greedy generation、program extraction、executor execution、normalized answer comparison、allowlist gate 和 summary 导出。报告主指标是 full-1237 temperature-0 greedy program executor 的 `executed_answer_accuracy/pass@1_greedy`。

评测输出通常保存到 `/root/autodl-tmp/outputs/financial_reasoning_rl/benchmarks/`。当前 README 使用的两个主要 artifact 是 `final_project_best_pipeline_vs_qwen_base_full_program_executor/benchmark_summary.json` 和 `v34r24_drgrpo_ckpt50_full1237_program_executor/benchmark_summary.json`。

## 6. 文档层

`docs/fin_datasets_v3.md` 记录 program-only 数据格式、ConvFinQA turn-level supervision、RS-SFT 数据和 frontier acquisition。`docs/fin_rl.md` 记录 v34r23 GRPO reward 与 v34r24 Dr.GRPO 配置。`docs/financial_reasoning_benchmark.md` 固定 full-1237 evaluator 口径。`docs/grpo_current_progress.md` 汇总 v34r22 到 v34r24 的当前进度和失败教训。

## 7. 面试回答

面试中可以把系统职责拆成四层。数据层把 FinQA/ConvFinQA 转成可验证 program supervision；训练层通过 SFT 和 RS-SFT 建立可执行 program 能力；RL 层只在 current-policy frontier 上用 executor reward 做校准；评测层用外部 executor 做 full-1237 greedy 检验。这个拆分能说明项目不是堆训练脚本，而是围绕可验证金融数值推理设计了完整闭环。
