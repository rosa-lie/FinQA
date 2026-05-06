# FinQA Program GRPO: Reward-First Update Plan and Implementation Notes

本文档记录当前 `FinQA / ConvFinQA` program-only GRPO 训练的诊断结论、改进计划，以及已经落地到 `training/finqa_program_grpo.py` 与 `run_finqa_program_grpo_v3.sh` 的实现结果。

它补充现有 `docs/plan/finqa_convfinqa_pot_grpo_stability_plan.md`，但更聚焦于一个更具体的问题：

- 当前训练为什么容易进入 `reward plateau + low KL + zero clip ratio`；
- 为什么优先级应当是 `1) 奖励函数设计 > 2) 超参数/采样设置`；
- 这一轮具体改了什么；
- 后续应该观察哪些指标来判断是否真的改善。

## 1. 诊断结论

当前问题的优先级排序如下：

1. `GRPO` 奖励函数设计
2. 超参数与采样设置
3. `sft2_program_merged` 起点偏保守
4. 数据难度分布上限
5. 训练实现本身

其中首要问题不是“模型不会写程序”，也不是“数据本身坏掉”，而是 reward 设计过早饱和，导致组内排序能力不足。之前的 reward 结构虽然已经从纯 binary 走向了 dense reward，但仍存在两个核心缺陷：

- 多个 reward 同时重复奖励“执行结果接近正确”这一类信号；
- `format / evidence / brevity` 在训练早期就趋于常数项，抬高总 reward，却不再提供有效梯度。

对应表现是：

- `reward` 总分可以升到较高区间，但 `reward_answer` 和 `reward_program` 很快平台化；
- `kl` 缓慢上升但长期很小；
- `clip_ratio/*` 从头到尾接近 0；
- `clipped_ratio` 仅在极低水平震荡；
- 输出长度收缩到一类稳定模板，而不是继续探索更优程序。

## 2. 计划原则

这一轮改造遵循两个原则。

第一，reward 必须回到“程序质量主导”的结构。模型应该首先因为“程序真的更接近正确解”而被区分，而不是因为“格式更像模板答案”而被奖励。

第二，采样和更新强度必须足以放大组内差异。若同一 prompt 下 6 个 completion 仍然太像，那么即使 reward 设计正确，GRPO 也只能做很弱的策略更新。

据此，本轮计划明确采用：

- program-centered reward
- normalize-then-sum aggregation
- no reward scaling
- stronger exploration
- weaker reference pull

## 3. 奖励函数重构方案

新的奖励结构不再保留原先分散且重复的 `reward_answer + reward_program + reward_program_answer_consistency` 组合，而是改成“程序主奖励 + 轻辅助奖励”。

### 3.1 主奖励：program quality

主奖励由以下分量组成：

- `reward_program_executable`
  - `Program:` 缺失、`N/A` 或不可执行时直接负分。
  - 可执行时给基础正分。
- `reward_program_execution_closeness`
  - 程序执行值与 `gold_answer` 的连续接近度。
- `reward_program_structure`
  - 程序结构与 `gold_program` 的相似度。
- `reward_program_argument_coverage`
  - 参数覆盖度。
- `reward_program_step_count`
  - 步骤数与 gold 的接近程度。
- `reward_program_exact_match_bonus`
  - 执行结果完全命中容差时给小额 bonus。

默认权重如下：

- `program_executable_reward_weight = 0.15`
- `program_execution_closeness_reward_weight = 0.30`
- `program_structure_reward_weight = 0.20`
- `program_argument_coverage_reward_weight = 0.12`
- `program_step_count_reward_weight = 0.08`
- `program_exact_match_bonus_weight = 0.05`

### 3.2 轻辅助奖励

辅助奖励只负责约束，不再主导训练：

- `reward_format_gate`
  - 只检查 `Evidence:` / `Program:` 是否存在、顺序是否正确、是否出现禁止字段。
  - `format_reward_weight = 0.03`
- `reward_evidence_support`
  - 只有当程序可执行时才计算 evidence overlap。
  - `evidence_reward_weight = 0.04`
- `reward_length_regularizer`
  - 长度目标区间改成 `80–160`，避免继续鼓励极短模板。
  - `brevity_reward_weight = 0.02`

## 4. 超参数与采样方案

在训练脚本中，超参数按“更强 RL 信号”方向调整：

- `learning_rate = 8e-6`
- `beta = 0.001`
- `max_steps = 500`
- `gradient_accumulation_steps = 6`
- `num_generations = 6`
- `generation_batch_size = 6`
- `max_completion_length = 256`
- `temperature = 1.10`
- `top_p = 0.95`

本轮刻意去掉 `min_p`，因为当前问题不是生成过散，而是组内探索不足。

同时，训练配置显式固定为：

- `multi_objective_aggregation = "normalize_then_sum"`
- `scale_rewards = "none"`
- `loss_type = "dapo"`

## 5. 已实施结果

截至本轮，以下改动已经实际落地：

- `training/finqa_program_grpo.py`
  - 新增 program-centered reward 结构；
  - 删除旧的重复 answer/consistency reward 组合；
  - 固定 `normalize_then_sum + scale_rewards=none + loss_type=dapo`；
  - 将这些配置写入 `experiment_config.json`；
  - 保留 `gold_program_step_hist` 等 audit 信息。
- `run_finqa_program_grpo_v3.sh`
  - 切换到新的采样和超参数配置；
  - 显式加入 `generation_batch_size = 6`；
  - 保留 `OMP_NUM_THREADS=1` 防止远端环境变量异常。

## 6. 预期验证信号

如果这轮修改有效，训练中应优先出现以下变化：

- `frac_reward_zero_std` 继续下降；
- `reward_program_execution_closeness/std`、`reward_program_structure/std`、`reward_program_argument_coverage/std` 持续非零；
- `kl` 明显高于旧配置；
- `clip_ratio/*` 不再长期全 0；
- `reward` 不再只靠 format/evidence 常数项维持高位，而是 program 相关分量继续抬升。

如果仍然出现：

- `clip_ratio/* = 0` 且 `kl` 仍极低；
- program 分量 std 很快衰减；
- 输出再次快速模板化；

则下一轮优先继续提高更新强度，而不是回退 reward 结构。首选动作应是把 `learning_rate` 提到 `1e-5`，而不是重新加重 format 或 brevity 奖励。

## 7. 当前运行建议

建议直接通过如下脚本启动当前版本实验：

- `/root/FinQA/run_finqa_program_grpo_v3.sh`

若手动运行，必须显式保证：

- `generation_batch_size` 可被 `num_generations` 整除；
- `OMP_NUM_THREADS=1`；
- 使用新的 reward/aggregation/sampling 配置，而不是旧版保守命令。
