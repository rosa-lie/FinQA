# GSPO for FinQA / ConvFinQA: Paper Summary, Algorithm, and Concrete Implementation Path

本文档基于 Zheng 等人在 2025 年提出的 *Group Sequence Policy Optimization*，讨论这一路线对当前 `FinQA / ConvFinQA` program-only 强化学习设置的价值，以及在现有 `training/finqa_program_grpo.py` 基础上如何落地。本文档不把 GSPO 当作抽象背景材料，而是直接回答四个问题：GSPO 论文在说什么，GSPO 相比 GRPO 改了什么，GSPO 在当前工程里应该如何实现，以及这一实现路径具体该怎样分阶段推进。

## 1. 论文介绍

GSPO 论文的出发点非常直接。作者认为，当前大模型强化学习中常用的 GRPO 在本质上仍然沿用了 token-level importance ratio 的思路，但实际 reward 往往是 sequence-level 的。对于一条完整回答，reward 是在整段 response 完成后才由 verifier 或规则系统给出的，而 GRPO 却在每个 token 上分别计算 importance ratio，再把整段回答共享的 advantage 分摊到 token 级别去优化。论文指出，这种“sequence-level reward 配 token-level ratio”的组合会引入高方差噪声，而且响应越长，这种噪声积累得越明显。训练早期可能还能工作，但随着训练继续推进，就会更容易出现不稳定、退化甚至不可逆坍缩。

GSPO 的核心贡献就是把 importance sampling 的单位和 reward 的单位重新对齐。与其在每个 token 上计算一份 ratio，不如直接对整条 response 的 likelihood 计算一个 sequence-level ratio，再围绕这个 sequence-level ratio 做 clipping、加权和优化。论文的论证重点不在“GSPO 比 GRPO 多了哪些技巧”，而在“importance ratio 的定义必须和被奖励的对象一致，否则优化目标从一开始就是 ill-posed”。这一点与当前 `FinQA / ConvFinQA` 的 program-based RL 任务高度相关，因为当前任务里的 verifier 关注的也不是单个 token 是否正确，而是整段 `Evidence + Program` 最终是否构成了正确、可执行、结构合理的程序推理轨迹。

## 2. GSPO 算法原理

GRPO 的做法是，为每个 prompt 采样一组 responses，然后对每个 response 计算 group-relative advantage，再在 token 级别使用 `πθ(yt|x,y<t) / πθold(yt|x,y<t)` 作为 importance ratio。这样做的隐含假设是，每个 token 的 ratio 都可以像 PPO 那样承担 off-policy correction 的作用。但 GSPO 论文指出，这里实际上只有一个已经采样出来的 token，而不是从旧分布中对该 token 位置进行大量独立采样，因此 token-level ratio 并没有真正起到 textbook importance sampling 的作用，反而把噪声带进了梯度估计。

GSPO 改为对整条 response 定义 ratio。设 response 为 `y`，它在新旧策略下的 sequence likelihood 分别为 `πθ(y|x)` 和 `πθold(y|x)`，那么 GSPO 使用的是 sequence-level ratio 的长度归一化形式：

$$s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{1/|y_i|}$$

长度归一化的作用是把不同长度 response 的数值范围控制到同一量级，否则长回答的 sequence likelihood 天然更极端，很难直接拿来和短回答共用一个 clipping range。然后，GSPO 不再在 token 级别分别做 clipping，而是对整条 response 的 ratio 做 sequence-level clipping。这样优化的单位、reward 的单位和 clipping 的单位三者终于统一了。论文的一个重要观察是，尽管 GSPO 在 sequence 维度上会裁掉更多响应，甚至整体 clipped fraction 明显高于 GRPO，但训练反而更稳定、更高效，因为这些被保留的梯度质量更高，噪声更低。

论文还给出了 `GSPO-token` 变体。这个变体本质上仍然保留 sequence-level ratio，只是在 advantage 分配时允许 token-wise customization，因此既保留了 sequence-level importance ratio 的稳定性，又为多轮交互或局部 credit assignment 留出了空间。对于当前 `FinQA / ConvFinQA` 工程，这个细节很重要，因为多轮 `ConvFinQA` 的 follow-up 问题未来很可能需要 history-sensitive 的局部 credit assignment，但当前第一步并不需要一开始就实现完整的 `GSPO-token`，先实现标准 GSPO 已足够解决当前最突出的问题。

## 3. 为什么 GSPO 对当前 FinQA 场景有意义

当前 `FinQA / ConvFinQA` 的 program-only 强化学习已经暴露出三个典型现象。第一，reward 已经不再完全无差异，但 `kl` 很低，`clip_ratio/*` 常常长期为零，说明 policy update 仍然弱。第二，`reward` 往往停留在一个较高平台区间内震荡，而 `reward_program` 与 `reward_answer` 不再持续抬升，说明当前 reward 已经开始饱和。第三，输出逐渐收缩成相对固定的 `Evidence + Program` 模板，探索不足但又没有真正换来更强的程序正确性。这三个现象放在 GSPO 论文的语境下很好理解：reward 的单位是整段程序回答，但当前实现里的优化与诊断仍然主要是 token-level 的，训练能动，但动得不够有效。

更具体地说，当前任务中的主 verifier 信号来自程序可执行性、执行结果与 `gold_answer` 的接近度，以及程序结构与 `gold_program` 的接近程度。这些全都是 sequence-level property，而不是 token-level property。即使某几个 token 本身很高频、很稳定，也不代表整段程序真的更接近正确解。因此，GSPO 的 sequence-level ratio 和 sequence-level clipping 正好与这个 verifier 形式天然契合。当前训练里频繁出现的“总 reward 不低，但程序质量仍平台化”的现象，本质上就是优化单位和奖励单位没有完全对齐带来的结果。

## 4. 当前工程中的具体实现方案

在当前代码基础上，最稳妥的路线不是直接推翻 `training/finqa_program_grpo.py`，而是新建一个并行脚本，例如 `training/finqa_program_gspo.py`，保留现有数据处理、模型加载、PEFT、量化、日志和 audit 框架，只替换 trainer 目标函数与相关诊断。这是因为当前 `finqa_program_grpo.py` 已经具备相对完整的工程能力，包括数据 contract 校验、reward 审计、checkpoint 恢复、experiment config 落盘和 TensorBoard 目录管理。这些部分没有必要为了切 GSPO 而重写。

第一步应当保留当前 program-centered reward 设计，但将 reward 的聚合方式进一步收敛成一个真正的 sequence-level 主分数，而不是很多相互相关的 reward 函数分开求和。更具体地说，当前已经拆出的 `program_executable`、`execution_closeness`、`structure_similarity`、`argument_coverage`、`step_count` 和 `exact_match_bonus` 应当在进入优化器之前先合成为单一的 `program_core_score`。`format_gate`、`evidence_support` 与 `length_regularizer` 继续存在，但应只作为轻量辅助项。这样做的目标不是减少日志项，而是让被优化的主 reward 真正等同于“整条 response 的程序质量”，从而与 GSPO 的 sequence-level ratio 形成一致的 credit assignment 单位。

第二步是替换 GRPO 的 token-level importance ratio。当前 TRL 已经提供了 `importance_sampling_level` 配置项，因此在不直接重写 trainer 的前提下，可以先做第一层近似实现：将 `importance_sampling_level` 显式设为 `sequence`，并保留 `loss_type=dapo`。这一步不等于完整 GSPO，但它是从 token-level ratio 走向 sequence-level ratio 的最短路径。如果当前 TRL 版本对 sequence-level importance sampling 已支持稳定运行，那么这一改动就能先验证“优化单位对齐”本身是否解决当前的低 `kl`、零 `clip_ratio` 与 plateau 问题。

第三步是实现真正的 GSPO objective，而不是只借助现有配置做近似。具体做法是，在 `training/finqa_program_gspo.py` 中自定义 trainer 或最小改写 TRL trainer 的 loss 计算逻辑，使其对每条 completion 先计算 sequence likelihood，再按论文中的长度归一化公式得到 `s_i(θ)`，然后在 response 级别与 group-relative advantage 相乘，并在 response 级别做 clipping。此时，梯度仍然会回流到整条 response 中的所有 token，但这些 token 将共享统一的 sequence-level 权重，而不是像 GRPO 那样各自带有不同的 token-level ratio。由于当前任务的 reward 也是整段 response 共享的，这会让优化与 verifier 更一致。

第四步是补一套 sequence-level diagnostics。现有日志过于依赖 token-level `clip_ratio` 和 completion-level均值，这对于 GSPO 不够。新的日志应记录 response-level clipped fraction、sequence ratio 的均值与分位数、`program_core_score` 的均值和标准差、`program_executable_rate`、`program_exact_match_rate`、`unique_program_ratio` 以及 response-level advantage 的分布。只有这样，才能判断训练究竟是“裁掉了更多低质量响应但整体更稳定”，还是“序列级 ratio 本身数值异常”。GSPO 论文的一个关键启发正是：更高的 clipping fraction 并不自动意味着更差训练，因为被保留下来的 sequence-level gradient 质量更高。

第五步是针对 `ConvFinQA` 预留向 `GSPO-token` 过渡的接口。当前第一版并不需要立即实现 token-wise customized advantages，但结构上应当避免把全部 reward 固定写死成完全同一的 response-level标量。更合适的设计是在数据字段中保留 `requires_history`、`history_dependency_type` 等 metadata，并在 reward 日志里记录这类样本的单独统计。这样，一旦后续需要对 follow-up 问题中的局部 token 或局部步骤赋予不同 advantage，现有实现不需要推倒重来。

## 5. 与当前 GRPO 配置相比，应该怎样具体修改

当前的 `run_finqa_program_grpo_v3.sh` 已经在朝“program-centered reward + stronger exploration”的方向改，但它仍然属于 GRPO 变体，而不是 GSPO。若要向 GSPO 迁移，最直接的命令层修改应当是新增一个独立脚本，例如 `run_finqa_program_gspo_v3.sh`，并将训练入口切到 `training.finqa_program_gspo`。在这个新脚本中，`temperature` 与 `top_p` 仍然保留，因为 GSPO 并不会替代探索控制；`generation_batch_size` 也仍然要显式传递，以保证 group size 与 rollout 逻辑一致。真正变化的部分应包括：启用 sequence-level importance sampling、使用 response-level clipping、保留 `scale_rewards=none`，并将 clipping range 从经典 GRPO/PPO 范围改为更小的 sequence-level范围。

这里一个重要实现细节是，不应继续把 `clip_ratio/high_mean/low_mean/region_mean` 当作最核心的主指标。对于 GSPO，更合适的主指标是 response-level clipped fraction 以及 sequence ratio 分布。也就是说，训练看板需要从“多少 token 被 clip”转向“多少 response 因为偏离过大被裁掉”。当前任务中的最终目标不是让每个 token 的概率移动得漂亮，而是让整条 `Evidence + Program` 在 response 级别更接近高奖励分布。

## 6. 分阶段实施路径

第一阶段是“GSPO-compatible GRPO”。这一步不改 trainer，只改 reward 聚合方式、启用 `importance_sampling_level=sequence`，并补齐 sequence-level diagnostics。目标是以最小工程风险验证：一旦 importance ratio 的单位开始更接近 sequence-level，当前 plateau 和低 `kl` 问题是否会缓解。如果这一阶段已经能带来更高的 `program_core_score/std`、更高的 `kl` 和更健康的 response-level clipping 分布，那么说明当前主要瓶颈确实来自 optimization unit mismatch。

第二阶段是“full GSPO”。这一步新建 `training/finqa_program_gspo.py`，实现真正的 sequence-level ratio、sequence-level clipping 和 sequence-level optimization objective，同时继续复用当前的 reward verifier。该阶段的目标不是立刻跑最大实验，而是先完成一轮 `smoke + short run`，验证 response-level clipping 是否正常、sequence ratio 是否数值稳定、`program_executable_rate` 是否维持住。

第三阶段是“GSPO-token for ConvFinQA”。这一阶段只在前两步稳定后再开始。它面向的不是一般 `FinQA` 单轮题，而是 `ConvFinQA` 中真正依赖历史的 follow-up 问题。届时可以把当前统一的 response-level advantage 扩展成 token-wise 或 step-wise 的局部 advantage，但 sequence-level ratio 仍然保留。这样既能继承 GSPO 的稳定性，又能对历史依赖的局部 credit assignment 做更细控制。

## 7. 当前最值得期待的改进

如果 GSPO 路线适配成功，最直接的收益不一定是 reward 曲线立刻更高，而是训练动态更“像在真的优化整段程序”。具体表现应当是：`program_core_score` 的方差维持更久，`frac_reward_zero_std` 继续下降，`kl` 不再长期停留在极低水平，response-level clipped fraction 变得可见且稳定，而输出不再那么快收缩成固定模板。对于 `FinQA / ConvFinQA` 这种 verifier 强、程序强、sequence reward 明确的任务，GSPO 的最大价值不是换一个名字，而是让“被奖励的对象”和“被优化的对象”最终一致。

## 8. 当前任务中 sequence-level 与 token-level 的取舍

结论：当前 `Evidence + Program` 任务应以 `sequence-level importance ratio / clipping` 作为主训练路线；`token-level` 更适合作为后续局部过程监督或 `GSPO-token` 变体，而不是当前 program-only RL 的默认优化单位。

这个判断来自任务本身的 reward 形态。当前核心 verifier 不判断某个 token 是否单独正确，而是判断整段输出是否满足：`Evidence:` 与 `Program:` schema 正确，`Program` 可执行，执行结果接近 `gold_answer`，program op / argument / step count 接近 `gold_program`，以及 evidence 是否支撑程序中的数值。这些都是 response-level property。一个单独的 `subtract`、数字、括号或中间变量通常没有独立 reward 意义；只有整段 `Evidence + Program` 组合起来，才能判断它是否构成正确推理轨迹。

因此，如果继续使用 token-level ratio，就会把同一个 sequence reward 分摊到每个 token 上，让 reward 单位和优化单位不一致。对于当前场景，这会带来两个风险：第一，局部高频 token 可能被过度强化，但整段程序并没有更正确；第二，长 response 中每个 token 的 ratio 噪声会累积，使训练更容易表现为 `kl` 很低、`clip_ratio` 长期为零、reward 有平台但程序质量不继续提高。GSPO 的核心价值正是把被奖励的对象和被优化的对象对齐。

当前 TRL 本地实现也支持这个方向。`importance_sampling_level="sequence"` 时，TRL 会把 completion 上的 token log-ratio 按有效 token 做平均：

```python
log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
```

这相当于长度归一化的 sequence log-ratio，是当前工程里最稳妥的 GSPO-compatible 实现。它不是 full custom GSPO loss，但已经比 token-level ratio 更符合 `Evidence + Program` verifier 的粒度。当前 `training.finqa_program_gspo` 保持 `importance_sampling_level=sequence` 是合理的。

`token-level` 仍有适用场景，但需要更细粒度监督信号支持。例如：

- evidence span 有精确引用标签；
- program 每一步有 gold intermediate value；
- `ConvFinQA` follow-up 问题需要对 history-dependent 局部步骤做 credit assignment；
- reward 能区分 “Evidence 写错” 和 “Program op / argument 写错”，并能稳定地映射到局部 token 或局部 step。

在这些条件满足前，token-level 主要会增加 credit assignment 噪声。后续如果要引入 token-level，优先路线不是回退到普通 token-level GRPO，而是实现 `GSPO-token`：保留 sequence-level ratio 和 sequence-level clipping，同时允许局部 token 或局部 step 使用定制 advantage。

已有训练状态不能直接证明 sequence-level 已经优于 token-level，因为 GRPO 与 GSPO-compatible 路径的 reward 聚合方式不完全同构。但当前观测支持继续沿 sequence-level 方向排查：GRPO 到 step 500 时 `kl` 大多在 `0.0008-0.0030`，`clip_ratio/region_mean=0.0`；GSPO-compatible 到 step 300 时 `kl` 在 `0.0032-0.0086`，同样 `clip_ratio/region_mean=0.0`。这说明问题不应简单归因于 sequence-level 不合适，而更可能需要补齐 response-level diagnostics、同 reward A/B，以及必要时实现 full GSPO objective。

下一轮对比实验应避免混淆变量。建议使用完全相同的 `program_core + format/evidence/length` reward，只切换 `importance_sampling_level=token` 与 `importance_sampling_level=sequence`，并观察 `program/executable_rate`、`program/exact_match_rate`、`program/unique_program_ratio`、`sequence/response_clipped_fraction`、`kl` 和 `frac_reward_zero_std`。如果 sequence-level 仍然 response clipping 全零且 program 指标不动，再考虑 `loss_type=luspo`、`loss_type=vespo` 或 full custom GSPO loss，而不是先回退 token-level。

当前决策：主线保持 `sequence`；`token-level` 只作为后续 `GSPO-token for ConvFinQA` 或 process-supervised reward 的分支实验。
