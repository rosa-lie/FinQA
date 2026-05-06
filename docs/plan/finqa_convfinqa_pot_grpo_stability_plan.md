# `run_fingpt_pot_rl.ipynb` 的 FinQA / ConvFinQA POT-GRPO 实施方案

本文档面向 notebook 改造与实验执行，目标是将当前 [run_fingpt_pot_rl.ipynb](/root/FinQA/run_fingpt_pot_rl.ipynb) 收敛成一条可直接实施的 `Verifier-Guided PoT-GRPO` 主线。它不再讨论抽象研究问题，而是回答：

- notebook 现在处于什么状态；
- 哪些 cell / contract / reward 需要改；
- 先跑哪些实验；
- 用什么指标判断是否发生 reward hacking 或 entropy collapse。

研究动机与文献脉络见 `docs/plan/grpo.md`。

## 1. 当前状态

相对于旧版 `run_fingpt_cot_rl.ipynb`，当前 `run_fingpt_pot_rl.ipynb` 已经完成了一个关键收敛：

- 主 GRPO 数据不再混合 `program_numeric + cot_answer_only`；
- notebook 文案已明确转向 `strict Program` 主线；
- `reward_profile` 在训练数据里已经统一为 `program_numeric`；
- train / valid / smoke 三套数据的 audit 已经显示 schema 基本干净；
- experiment registry 仍保留 `base / v3_program_sft / cot_pot_program_mixed` 三个起点。

但当前 notebook 仍有四个主要问题。

第一，reward 仍保留旧版 mixed 时代的局部正奖励：

- `reward_program` 中的 `parse_bonus`；
- `reward_program` 中的 `op_bonus`；
- `reward_brevity_and_relevance` 中对 program task 的固定长度正奖励。

第二，训练超参仍偏激进：

- `GRPO_LEARNING_RATE = 5e-6`
- `MAX_COMPLETION_LENGTH = 384`
- `BETA = 0.001`
- `MAX_STEPS = 300`

第三，训练期缺少显式的 entropy / diversity diagnostics。目前主要关注 reward 和 benchmark 输出，还不足以捕捉策略模板化。

第四，部分文案和变量名仍保留 mixed 历史，例如 reward 说明里还出现 `cot_answer_only`，容易让 notebook 语义与实际实现漂移。

## 2. 实施目标

第一阶段的 notebook 改造只做一件事：把 `run_fingpt_pot_rl.ipynb` 固化成 `program_numeric-only core-first strict-program GRPO`。

对应原则如下：

- 第一阶段只使用 `FinQA + ConvFinQA` strict-program 数据。
- 主输出 schema 固定为 `Evidence + Program`。
- reward 的主要正反馈只来自 `execute(pred_program) == gold_answer`。
- reward 只能辅助区分“正确解中的更好解”，不能给错误程序持续正反馈。
- benchmark 必须单独看 `ConvFinQA requires_history=true` 子集。

## 3. 数据层改造

### 3.1 训练数据 contract

训练、验证和 smoke 文件继续保留 program-only contract：

```json
{
  "input_prompt_raw": "...",
  "reward_profile": "program_numeric",
  "source_dataset": "finqa|convfinqa_turn",
  "gold_program": "...",
  "gold_answer": "...",
  "reference_response": "...",
  "record_id": "...",
  "metadata": {"...": "..."}
}
```

明确要求：

- 所有样本都必须 `reward_profile = program_numeric`。
- 所有 `gold_program` 都必须可执行或至少可 canonicalize。
- 所有 `gold_answer` 都必须能映射到 `answer_norm` / `answer_exe`。
- 训练集不得混入 `Reasoning + Answer` 样本。

### 3.2 ConvFinQA metadata 保留

`ConvFinQA` 样本必须继续保留以下字段用于后续评测切片：

- `requires_history`
- `history_dependency_type`
- `history_turns`
- `history_full_reasoning_turns`
- `history_full_reasoning_ratio`

这些字段不一定直接进入 reward，但必须进入 benchmark 切片与错误分析。

### 3.3 数据 audit 扩展

当前 audit 已覆盖：

- program execute fail
- program answer mismatch
- missing program
- missing answer
- duplicate prompt

还需要新增两类统计：

- `requires_history=true` 样本占比
- canonical program 长度 / op 分布统计

目的是在训练前先掌握 core 数据的结构复杂度，而不是训练后才发现模型被特定 program 模板主导。

## 4. Reward 改造

当前 notebook 的 reward 需要从“六个松散函数求和”改成“门控 + 核心奖励 + 轻量辅助奖励”的严格结构。

### 4.1 Gate reward

以下任一条件不满足，总 reward 直接置为 `0`：

- 不包含 `Evidence:`；
- 不包含 `Program:`；
- 出现 `Reasoning:`、`Answer:` 或 `Normalized Answer:`；
- `Program:` 为空、`N/A`、多段 program 串联；
- 程序 parse 失败；
- 使用不在 DSL 白名单中的运算符。

这一步的目标是阻止模型依靠错误 schema 或伪程序获得任何稳定回报。

### 4.2 Core execution reward

主要正奖励只绑定程序执行是否正确。

推荐规则：

- 若 `execute(pred_program)` 成功且与 `gold_answer` 数值一致，则给主奖励；
- 否则不给主奖励。

必须删除以下旧逻辑：

- parse 成功就给 `0.08`；
- op overlap 就给 `0.12`；
- 只因长度合适就给 program task 正奖励。

原因很直接：这些机制都会让错误程序拥有可持续优化的信号，导致模型向高频模板坍缩。

### 4.3 Auxiliary reward

辅助 reward 只能在不改变主方向的前提下，帮助区分多个正确候选。

建议保留：

- `evidence grounding`：evidence 数字必须与 `input_prompt_raw` 存在交集；
- `canonical closeness`：仅在 execution correct 后，比较 canonical program 与 gold program 的接近度；
- `verbosity penalty`：只惩罚明显超长输出，不对“短”本身给固定正奖励。

建议删除：

- `reward_brevity_and_relevance` 对 program task 的固定 `0.05`；
- execution wrong 时的任何 `op overlap` 正激励。

### 4.4 ConvFinQA history 检查

第一版不强制把 history consistency 直接写成复杂 reward，但至少要为后续实验预留两个钩子：

- 在 `requires_history=true` 样本中，记录 evidence / program 是否复用了历史实体或年份；
- 在 benchmark 中单独计算该子集的 execution correctness。

如果后续发现模型主要在该子集退化，再考虑引入显式 history-consistency reward。

## 5. 训练控制改造

### 5.1 默认超参收紧

第一版稳定训练建议将默认值收紧到以下区间：

- `learning_rate = 1e-6 ~ 2e-6`
- `num_generations = 4`
- `max_completion_length = 160 ~ 224`
- `beta` 强于当前 `0.001`
- `max_steps` 先短程、再中程、再 full run

这里的原则是：先防止策略过快收缩，再考虑更高探索。

### 5.2 分阶段运行顺序

推荐将 notebook 执行顺序改成四级：

1. `reward smoke`：验证新 reward 对正确 / 错误 / 可执行但错误 / schema 错误样例的响应。
2. `1-step GRPO smoke`：验证 trainer、reward、数据传递、adapter 输出路径。
3. `50-step smoke run`：观察早期 reward、长度、多样性和 executable rate。
4. `150-step mid run`：决定是否有资格进入 `300-step full run`。

只有在 `150-step` 阶段通过稳定性门槛后，才允许 full run。

### 5.3 训练期新增监控指标

除现有 reward 外，必须新增并落盘以下指标：

- `answer_correct_rate`
- `program_executable_rate`
- `unique_program_ratio`
- `unique_answer_ratio`
- `avg_completion_length`
- `avg_program_op_count`
- `program_op_distribution`
- `KL_to_init_policy`
- `token_entropy_proxy`
- `ConvFinQA requires_history=true correctness`

这些指标的作用不是美化日志，而是用于判断 reward 是否把策略推向模板化 shortcut。

### 5.4 早停与回滚条件

以下情况出现时，应停止 full run，并回滚到最近稳定 checkpoint：

- `program_executable_rate` 上升，但 `answer_correct_rate` 不升；
- `format pass` 提升，但 `unique_program_ratio` 快速下滑；
- 输出长度快速塌缩为极短模板；
- `ConvFinQA requires_history=true` 子集先于总体指标明显退化；
- KL 下降、reward 平台化、benchmark 不涨。

## 6. Benchmark 改造

当前 notebook 的 benchmark 应继续以 strict-program 为主，但汇报维度要细化。

### 6.1 主表维度

主表至少拆成：

- `FinQA strict`
- `ConvFinQA all`
- `ConvFinQA requires_history=true`
- `single-step`
- `multi-step`

### 6.2 主指标优先级

按重要性排序：

1. `execute(pred_program) == gold_answer`
2. `program_executable_rate`
3. `canonical / exact program match`
4. `pass@k`

`pass@k` 不能替代 execution correctness，但可以辅助判断策略是否因为过早收缩而丢失探索能力。

### 6.3 成功判据

一轮 GRPO 实验只有满足以下条件，才算真正有效：

- 至少一个主任务子集优于 baseline；
- 改进不是靠 schema 模板化换来的；
- `ConvFinQA requires_history=true` 没有显著退化；
- `unique_program_ratio` 没有明显坍塌。

## 7. 起点对比与实验顺序

### 7.1 对比起点

保留三组起点：

- `base`
- `v3_program_sft`
- `cot_pot_program_mixed`

三组起点共用同一份 core-only strict-program RL 数据、同一套 reward、同一套 benchmark。

### 7.2 实验顺序

按以下顺序推进：

1. 修正 reward 与诊断指标。
2. 用 `cot_pot_program_mixed` 跑 `reward smoke + 1-step smoke`。
3. 对三组起点各跑 `50-step smoke`。
4. 只保留最稳的 1 到 2 个起点进入 `150-step mid run`。
5. 只有 `150-step` 通过后，才进入 `300-step full run`。
6. supplement 不进入第一阶段主实验。

### 7.3 结果解释规则

- 如果只有 `base -> GRPO` 提升，而 `v3 / cot_pot` 没提升，说明当前 reward 更像 cold-start strengthening。
- 如果 `v3 -> GRPO` 提升，而 `cot_pot -> GRPO` 持平或下降，优先怀疑当前 mixed SFT 起点中已内化的模板偏好。
- 如果所有起点都出现 executable rate 涨但 strict benchmark 不涨，则优先判断为 reward hacking，而不是训练不足。

## 8. 后续扩展，不纳入第一阶段默认实现

以下方向保留为第二阶段研究扩展，不作为 notebook 第一版默认实现：

- `Fino1 / FinCoT` supplement re-introduction
- `RiskPO`
- `DAPO` / asymmetric clipping
- entropy-aware advantage
- pass@k-style training objective
- negative reinforcement for bad program templates
- token-level entropy protection

原因是：当前最关键的问题不是“没有更强算法”，而是主任务 verifier 和 reward contract 还没有完全收紧。

## 9. 最终实施结论

这版 `run_fingpt_pot_rl.ipynb` 的核心任务不是继续扩大 RL 数据，而是先把以下闭环做稳：

- `program_numeric-only` 数据主链；
- `Evidence + Program` strict schema；
- `execute(pred_program) == gold_answer` 为主的 verifier-guided reward；
- 可观测的 entropy / diversity diagnostics；
- 面向 `ConvFinQA requires_history=true` 的子集评测。

只有这条闭环稳定后，后续再引入 supplement、RiskPO、DAPO 或 pass@k-style training，才有清晰的比较基线和可靠的结论。
