# FinQA/ConvFinQA RL 与 GRPO 当前设计记录

本文记录当前 FinQA/ConvFinQA program-executor 主线中的 RL 设计。这里的 RL 不再是早期泛金融 CoT reward 设想，而是在 retention-aware RS-SFT 强基线之后，围绕 current-policy frontier 样本进行短程校准。

## 1. 背景问题

金融表文数值推理的 reward 设计必须围绕可验证结果。模型输出的自然语言 reasoning 可能流畅但不可执行，格式完整也不代表公式正确。FinQA 和 ConvFinQA 提供 gold program 与 execution answer，因此当前项目把主要 reward 定义在 `execute(Program) == gold normalized answer` 上。

直接从 base 或弱 SFT 进入 GRPO 不稳定。主要问题包括 positive reward 稀疏、wrong-executable program 误导、group zero variance 过多，以及辅助 proxy reward 与最终 executor 指标不一致。项目因此采用两步策略。第一步用 Program SFT 和 retention-aware RS-SFT 建立可执行 program 能力。第二步只在 current-policy frontier 上做 GRPO，让 RL 处理 greedy 错但采样空间已经存在正确 program 的边界样本。

## 2. 核心思想

GRPO，即 Group Relative Policy Optimization，中文可称为组相对策略优化。它不训练单独的 value model，而是对同一个 prompt 采样多个 completion，使用组内平均奖励作为 baseline。直觉上，如果同一题的 8 个候选里有的 program 执行正确、有的 program 可执行但答案错误、有的 program 不可执行，那么正确候选会获得正 advantage，错误候选会被压低。

当前 v34r23 的关键不是“换一个 RL 算法就提升”，而是重新定义训练样本。frontier acquisition 要求当前 RS-SFT 模型 greedy 错，但 pass@8 候选中存在 executor-correct program。这样每个训练 group 更可能有有效 reward 差异，RL 不再浪费在全错或已全对样本上。

v34r24 进一步尝试 Dr.GRPO。Dr.GRPO 的动机是降低 GRPO 中由 completion length 和 reward scaling 带来的偏置。当前实现采用 TRL 原生配置 `scale_rewards=none` 和 `loss_type=dr_grpo`，并保持同一套 program executor reward。实验结论是 checkpoint-50 在 full-1237 上略优，但收益极窄，不能扩大解释为系统性解决长度偏置。

## 3. 数学形式

对于一个 prompt，策略采样一组 completion $\{y_i\}_{i=1}^{G}$，每个 completion 经过 program parser 和 executor 得到 reward $r_i$。普通 GRPO 可以写成组内标准化优势 $A_i=(r_i-\mu_G)/(\sigma_G+\epsilon)$，其中 $\mu_G$ 和 $\sigma_G$ 是同组奖励均值与标准差。策略更新使用 clipped policy gradient，核心形式类似 $L=-\mathbb{E}[\min(\rho_i A_i,\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i)]$，其中 $\rho_i=\pi_\theta(y_i|x)/\pi_{\theta_{old}}(y_i|x)$。

Dr.GRPO 在当前项目中的工程映射更关注 reward scaling 和 loss type。`scale_rewards=none` 表示不再用奖励标准差缩放 reward，减少组内方差和长度因素耦合带来的偏置；`loss_type=dr_grpo` 使用 TRL 中的 Dr.GRPO loss 实现。由于当前 reward 是离散执行结果，文档分析时要把训练 reward、greedy pass@1 和 sampled pass@k 分开，不能用训练过程中的 reward 均值替代最终 benchmark。

## 4. 代码对应

`training/finqa_program_grpo.py` 是当前 GRPO 训练核心。它读取 frontier JSONL，构造 prompt，调用模型生成多个 completion，解析 `Program` 字段，执行 DSL，并将执行结果映射成 reward。训练时由 TRL `GRPOTrainer` 负责 logprob、ratio、advantage 和 LoRA 参数更新。

v34r23 的 current-policy frontier 数据由 `scripts/build_v34r23_current_policy_manifest.py` 和 `scripts/build_v34r23_frontier_grpo_data.py` 生成。前者组织当前策略采样与 manifest，后者把 greedy-wrong、pass@8-executor-correct 和 hard negative 信息整理为 GRPO 可读训练数据。`scripts/v34r23_prompt_processor.py` 负责严格输出 contract，禁止多个 Program section、assignment-style program、Reasoning/Answer marker 等不适合 executor 的输出。

v34r24 的 Dr.GRPO sweep 由 `scripts/run_v34r24_drgrpo_checkpoint_sweep.py` 和 `scripts/evaluate_v34r24_drgrpo_long_sweep.py` 承接。训练配置重点是 `scale_rewards=none`、`loss_type=dr_grpo`、`max_completion_length=300`、`per_device_train_batch_size=4`、`gradient_accumulation_steps=2` 和 `steps_per_generation=2`。

## 5. Reward 设计

v34r23 使用五档 `frontier_execution_calibration` reward。正确且满足 strict program contract 的可执行 program 得 1.0；答案正确但违反 contract 的输出得 0.25；可执行但答案错误得 -0.1；非法或 contract-violating 且答案错误得 -0.3；缺失 program 或不可执行输出得 -1.0。

这个设计的重点是把 answer correctness 放在最高优先级，同时避免只奖励“能执行”。wrong-executable program 在金融任务中很常见，所以它必须低于正确 program。contract violation 也不能完全等价于正确输出，否则模型会重新学会夹带解释、多个 Program 或 answer marker，削弱 executor pipeline 的稳定性。

训练监控不能只看 mean reward。更重要的是 group reward standard deviation、zero-std group 比例、all-wrong group 比例、all-correct group 比例、平均 completion length、parse rate、execution rate 和 strict contract rate。若 zero-std/all-wrong 太高，说明 frontier 数据仍然过难或模型采样空间没有正确候选，RL 信号会变弱。

## 6. 实验验证

当前 full-1237 greedy executor 结果为 Base 0.272433、SFT2 0.578820、RS-SFT 0.603072、v34r23 GRPO 0.606306、v34r24 Dr.GRPO checkpoint-50 0.607114。对应正确数可理解为约 337/1237、716/1237、746/1237、750/1237 和 751/1237。这里的差异说明 SFT 与 RS-SFT 是主增益，GRPO 是强基线后的微调校准。

v34r24 sweep 中 checkpoint-50 在 joint-64 gate 上达到 0.593750，checkpoint-100 为 0.585938，因此只把 checkpoint-50 推进 full-1237。full evaluation 显示 checkpoint-50 得到 751/1237。后续若继续验证 Dr.GRPO，需要增加 seed、扩大 checkpoint 对比，并审计 changed predictions，而不是只看一个 full score。

## 7. 面试回答

面试中可以这样回答。这个项目没有把 GRPO 当作万能增益来源，而是先把金融数值推理改造成可执行 program generation。SFT 负责让模型学会 DSL 和 schema，RS-SFT 用 executor 筛出正确 program 并通过 ConvFinQA/FinQA 混合缓解遗忘。GRPO 只处理当前策略的 frontier 样本，也就是 greedy 错但采样空间有正确程序的样本。reward 以程序执行正确性为核心，辅助惩罚不可执行、wrong-executable 和 contract violation。最终结果说明 RL 有小幅正收益，但主要能力来自高质量监督数据和可验证数据筛选。
