# GRPO 当前进度记录

本文聚焦 v34r22 到 v34r24 的当前完成进度。更早的 v34-v51 历史实验只作为失败教训使用，不再作为 README headline。

## 1. 背景问题

FinQA/ConvFinQA 的 RL 难点不在于写一个 reward 函数，而在于找到 reward 对当前 policy 真正有区分度的样本。金融 program 必须同时满足证据正确、公式正确、参数顺序正确、单位缩放正确和输出 contract 正确。弱模型经常产生全错 group，强模型又经常在简单样本上全对，这两种情况都不利于 GRPO。

## 2. v34r22 retention-aware RS-SFT

v34r22 的作用是建立强 SFT 后基线。它使用 executor 进行 rejection sampling，只把当前模型采样出的正确 program 作为监督目标，并保持 65% ConvFinQA 与 35% FinQA 唯一样本混合。这个设计一方面贴近模型自身分布，另一方面避免连续强化 ConvFinQA 后遗忘 FinQA 单轮表文能力。

full-1237 greedy executor 结果显示，RS-SFT 达到 0.603072，约 746/1237。相比 SFT2 的 0.578820，它带来约 30 题增益，是当前项目最关键的后训练收益来源。

## 3. v34r23 current-policy frontier GRPO

v34r23 的核心改动是 current-policy frontier acquisition。筛选逻辑要求当前 RS-SFT 模型 greedy 错，但 pass@8 候选里至少有一个 executor-correct program，并保留可执行但错误的 hard negative。基于 FinQA 和 ConvFinQA 各 500 条训练样本，最终获得 FinQA 106 条和 ConvFinQA 82 条 frontier 样本。

reward 使用五档 `frontier_execution_calibration`。正确且 contract 合法得 1.0，正确但 contract violation 得 0.25，可执行但错误得 -0.1，非法或 contract-violating 且错误得 -0.3，缺失或不可执行得 -1.0。这个 reward 把 executor correctness 放在最高优先级，同时显式区分 wrong-executable 和 non-executable。

v34r23 checkpoint-10 在 full-1237 greedy executor 上达到 0.606306，约 750/1237。它相比 RS-SFT 多约 4 题。这个结果支持 frontier GRPO 有小幅校准作用，但不支持把总提升归因给 GRPO。

## 4. v34r24 Dr.GRPO

v34r24 把训练配置映射到 Dr.GRPO。当前使用 TRL 原生参数 `scale_rewards=none` 和 `loss_type=dr_grpo`，并保持 v34r23 的 program executor reward 与 frontier 数据思想。训练配置还包括 `max_completion_length=300`、`per_device_train_batch_size=4`、`gradient_accumulation_steps=2` 和 `steps_per_generation=2`。

100-step sweep 后，checkpoint-50 在 joint-64 gate 上达到 0.593750，checkpoint-100 为 0.585938，因此 checkpoint-50 被选入 full evaluation。full-1237 结果为 0.607114，即 751/1237。相比 RS-SFT 多 5 题，相比 v34r23 多 1 题。这个结果只能称为窄幅 late-stage calibration，不能称为显著提升。

## 5. 失败教训

早期 GRPO 的主要问题是 sparse positive、wrong-executable、zero-variance group、history frontier mismatch 和 proxy reward mismatch。尤其在金融 DSL 任务中，parse 成功不代表公式正确，format 完整不代表答案正确，pass@8 上限也不等于 greedy 部署能力。后续实验必须继续把 greedy `pass@1`、sampled `pass@k`、rerank、训练 reward 和 execution/contract metrics 分开。

## 6. 下一步

后续如果继续推进 RL，更合理的方向是扩大 current-policy frontier 的覆盖，按错误类型做分桶，比较多个 seed，并审计 changed predictions。另一个方向是引入 verifier reranking 或 process-level data，但必须保持 program executor 作为主评测口径。只有当 full-1237 greedy 多 seed 稳定提升，并且 changed prediction 审计显示公式选择或证据定位真正改善，才能把 RL 描述为系统性收益。
