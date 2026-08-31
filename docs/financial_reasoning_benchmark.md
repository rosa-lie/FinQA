# Program-executor full-1237 评测口径

本文记录当前 FinQA/ConvFinQA 主线的稳定评测方式。旧版 benchmark 曾经混合 ConvFinQA、Fineval、CFLUE 等任务，用于早期泛金融探索；当前 README headline 只采用 program-executor full-1237 口径。

## 1. 背景问题

金融数值推理不能只看生成文本里是否出现某个答案字符串。模型可能写出正确数字但 program 不可执行，也可能输出合法 program 但公式方向错误。当前项目因此把主指标定义为 executor 执行生成 program 后的答案准确率。这个指标能同时约束 evidence grounding、program parse、program execution 和 normalized answer correctness。

评测集合固定为 ConvFinQA test 869 条 turn-level 样本加 FinQA test 368 条样本，总计 1237 条。解码固定为 temperature 0 的单次 greedy generation。模型输出 `Evidence + Program`，评测脚本从 completion 中抽取 `Program`，执行 DSL，再将执行结果与 gold normalized answer 做数值容差比较。

## 2. 核心指标

`pass@1_greedy` 是当前 README 的主指标。它等价于单次 greedy completion 的 `executed_answer_accuracy`，不使用采样、不使用 rerank，也不读取训练 reward。这个指标回答的问题是，部署时模型一次输出 program，executor 执行后能答对多少题。

`program_execution_rate` 是 program 能否被 parser 和 executor 成功执行的比例。它不是答案准确率，因为 wrong-executable program 也会计入执行成功。执行成功率能帮助判断 schema 和 DSL 是否稳定，但不能替代 `pass@1_greedy`。

`pass@8` 是采样诊断指标，用于判断候选空间里是否存在正确 program。它适合 frontier acquisition，因为如果 greedy 错但 pass@8 中有正确候选，说明模型已经有潜在能力，只是 greedy policy 需要校准。`pass@8` 不能直接当成部署主结果。

rerank 指标用于研究 verifier 或 executor reranking 的上限。它回答的是“如果有多个候选并能选中正确程序，效果会怎样”，不等价于单次生成能力。训练 reward 是优化过程中的信号，只能辅助解释训练是否有梯度，不应和 benchmark accuracy 混在一个 headline 中。

## 3. 当前 full-1237 结果

| 模型 | ConvFinQA 869 | FinQA 368 | Macro/full-1237 | 执行成功率 | 数据来源 |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-7B-Instruct | 0.318757 | 0.163043 | 0.272433 | 0.542441 | `final_project_best_pipeline_vs_qwen_base_full_program_executor/benchmark_summary.json` |
| SFT2 | 0.639816 | 0.434783 | 0.578820 | 0.877930 | 同上 |
| retention-aware RS-SFT | 0.654776 | 0.480978 | 0.603072 | 0.905416 | 同上 |
| v34r23 GRPO checkpoint-10 | 0.657077 | 0.486413 | 0.606306 | 0.907842 | 同上 |
| v34r24 Dr.GRPO checkpoint-50 | 0.658228 | 0.486413 | 0.607114 | 0.907033 | `v34r24_drgrpo_ckpt50_full1237_program_executor/benchmark_summary.json` |

这些数字必须按阶段解读。Base 到 SFT2 的提升主要来自 program supervision。SFT2 到 RS-SFT 的提升来自 executor-correct rejection sampling 和 retention-aware data mixture。RS-SFT 到 v34r23/v34r24 的提升只有约 4 到 5 题，说明 GRPO 是短程校准模块，而不是全部提升来源。

## 4. 评测脚本

当前主脚本是 `evaluation/evaluate_financial_benchmarks.py`。它支持模型列表、FinQA/ConvFinQA 数据加载、program 抽取、executor 执行、normalized answer 对比、allowlist 和输出汇总。常见输出包括 `benchmark_manifest.json`、逐模型 predictions、逐模型 summary、`benchmark_summary.json` 和 `benchmark_summary.csv`。

推荐从仓库根目录运行，远端环境使用 `/root/miniconda3/bin/python -m evaluation.evaluate_financial_benchmarks ...`。历史结果位于 `/root/autodl-tmp/outputs/financial_reasoning_rl/benchmarks/`，README 中的 headline 数字来自两个已经核验的 summary 文件。

## 5. 评测边界

报告模型效果时必须同时写清 evaluator、样本数、解码方式和指标类型。正确写法是 full-1237、temperature-0 greedy、program executor、`executed_answer_accuracy/pass@1_greedy`。不应把 pass@8、rerank 上限、训练 mean reward 或 64-sample gate 分数写成 full benchmark。

小样本 joint-64 gate 只用于控制评测成本和筛 checkpoint。它可以判断一个 checkpoint 是否值得推 full evaluation，但不能替代 full-1237 结论。v34r24 中 checkpoint-50 的 joint-64 为 0.593750，checkpoint-100 为 0.585938，因此 checkpoint-50 被选入 full evaluation；最终 headline 仍然以 751/1237 的 full score 为准。

## 6. 面试回答

面试中可以强调，本项目的评测不是“模型自己说答案对不对”，而是强制模型生成可执行 program，再由外部 executor 计算答案。这样做的好处是可复现、可归因，也能把 parse error、execution error、wrong formula 和 final answer error 拆开分析。面对“GRPO 提升多少”的问题，应回答 GRPO 在 RS-SFT 后只提升约 0.32 到 0.40 个百分点，主提升来自 SFT 和 RS-SFT。
