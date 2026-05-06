# Verifier-Guided PoT-GRPO for FinQA / ConvFinQA

本文档用于定义当前金融表文数值推理项目中的 `Verifier-Guided PoT-GRPO` 研究问题、相关工作脉络、方法设计与实验假设。它不是 notebook 操作手册，而是回答三个更基础的问题：

1. 为什么 `FinQA / ConvFinQA` 适合做 `PoT + GRPO`。
2. 为什么这个任务比数学题更容易出现 `reward hacking` 和 `entropy collapse`。
3. 为什么当前最合理的路线不是直接做 mixed RL，而是先做 `program_numeric-only` 的 strict-program RLVR。

相关实施方案见 `docs/plan/finqa_convfinqa_pot_grpo_stability_plan.md`。

## 1. 任务定位

本项目的主任务不是普通金融问答，而是金融报告场景下的可验证数值推理。模型面对的是文本、表格和对话历史共同构成的复杂上下文，需要完成以下步骤：

- 识别问题涉及的财务指标、时间范围、单位和比较关系。
- 在表格与文本证据之间定位可用于计算的字段。
- 将问题转化为可执行程序。
- 通过程序执行得到最终数值或布尔答案。
- 在多轮场景下继承上一轮的实体、年份、指标或中间结果。

因此，这个任务天然比只看最终答案的数学题更复杂。数学题大多只验证“结果是否正确”，而 `FinQA / ConvFinQA` 还需要关心：

- evidence 是否正确；
- program 是否可执行；
- units 是否归一；
- 时间范围是否对齐；
- follow-up question 是否正确继承历史语义。

这也是为什么单纯的 `answer exact match reward` 在这个任务上不够，会被模型轻易利用。

## 2. 为什么选 PoT + GRPO

### 2.1 为什么不是 CoT-only

CoT 擅长解释变量选择、财务概念映射和 reasoning path，但它不可执行。对于金融数值题，如果只训练模型写更长的自然语言推理，模型可能学会“像在推理”，却没有真正学会如何稳定地算对。

### 2.2 为什么是 PoT

PoT 在本项目中对应 `Program:`。它的价值在于：

- 可执行；
- 可校验；
- 可与 `answer_norm` 或 `answer_exe` 比较；
- 可以显式绑定表文证据与运算过程。

`FinQA` 与 `ConvFinQA` 原生提供 gold program、execution answer 和 normalized answer，使它们非常适合 program-supervised SFT 与 execution-based RL。

### 2.3 为什么是 GRPO / RLVR

SFT 只能让模型模仿已有轨迹，而 RLVR 能利用“可验证正确性”信号继续优化策略。对当前任务来说，强化学习的价值不在于让模型“说得更像人”，而在于让模型围绕可执行程序和数值正确性进行策略更新。

如果 reward 设计合理，GRPO 可以推动模型从“会写像样的 Evidence/Program”提升到“更稳定地产生正确 Program 并执行得到正确答案”。

## 3. 核心研究问题

### 3.1 Reward Hacking

当前任务比数学题更容易 reward hacking，原因有三类。

第一类是结构奖励可被模板化利用。只要 reward 奖励固定锚点、短长度或表面 schema 完整，模型就可能学会输出：

- 看起来合法但实际无效的 `Evidence`；
- 很短、可执行但与问题不匹配的 `Program`；
- 高频算子模板，如 `divide(...)`、`subtract(...)`；
- 直接拷贝 prompt 中显眼数字的伪程序。

第二类是程序奖励存在局部刷分路径。如果 reward 为“parse 成功”“operator overlap 高”单独给正分，模型就不需要真正解决问题，只需要生成一段看起来像 DSL 的程序即可拿到一部分稳定回报。

第三类是金融任务额外存在证据与语义层面的作弊空间。模型可能：

- 答案数字碰巧正确，但 evidence 错误；
- 程序执行正确，但使用了错误年份；
- 在 ConvFinQA 中忽略历史轮次，只抓住当前问题的表面形式；
- 在 percentage / million / absolute 等单位切换中“撞对”答案。

因此，这个任务不能只把 reward 设计成 `answer_correct`，必须引入 verifier 体系。

### 3.2 Entropy Collapse

在 reasoning RL 中，策略熵常常在训练早期快速下降。对于本任务，这个问题更严重，因为 strict-program schema 本身就比自由文本空间更窄。模型一旦找到几种稳定拿分的模板，就会迅速收缩到：

- 少量高频 program 结构；
- 极短 evidence；
- 对特定年份差值、比率题的固定解法；
- follow-up 问题中的 lookup shortcut。

结果是：

- `pass@1` 可能短期看起来变好；
- `pass@k`、program diversity、follow-up generalization 反而下降；
- 模型失去探索更优 program 的能力。

因此，本项目中的熵监控不是可选项，而是算法设计的组成部分。

### 3.3 ConvFinQA 的历史依赖

`ConvFinQA` 与 `FinQA` 的本质差异不只是多一段历史文本，而是问题形式从“单轮数值推理”变成“依赖上下文的 follow-up reasoning”。

很多 follow-up turn 不是简单重复抽取，而是要求模型：

- 继承上一轮的实体和指标；
- 解析“what about in 2008?” 这类省略表达；
- 将先前答案作为当前比较或计算的基准。

如果 reward 不显式考虑历史依赖，模型会倾向于忽略历史，从而在总体分数不低的情况下，真实多轮能力下降。

## 4. 相关工作与项目启发

### 4.1 DeepSeekMath

DeepSeekMath 的关键启发不是具体超参，而是：当任务存在明确正确性判据时，verifiable reward 能有效驱动 reasoning policy 改进。对本项目的映射是：

- 最强 reward 应绑定 `execute(Program) == gold_answer`；
- 纯格式 reward 只能作为辅助信号；
- program-executable task 比自然语言 preference 更适合做 RLVR。

但不能直接照搬数学题设置，因为金融任务还涉及 evidence selection、units 和 history consistency。

### 4.2 RiskPO

RiskPO 强调风险规避目标与下尾控制，核心价值在于缓解长期训练中的熵崩塌和过度自信。对本项目的启发是：

- 后续可以将 reward 从均值导向改为更保守的分布目标；
- 对 `ConvFinQA requires_history=true` 这类高难度子集，风险规避目标可能比均值目标更稳定；
- 但前提是 reward 排序本身要足够可靠，否则下尾放大可能同时放大奖励噪声。

因此，RiskPO 更适合作为第二阶段扩展，而不是第一版的直接默认目标函数。

### 4.3 The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models

这项工作直接解释了 reasoning RL 中熵下降的机制，并提出 `Clip-Cov`、`KL-Cov` 等控制思路。对本项目的启发是：

- 训练中必须显式记录 entropy proxy、KL 和多样性指标；
- 高协方差 token 很可能对应 program 分叉点、年份选择、运算符选择和单位映射；
- 若后续需要进一步控制坍缩，可以在 token 或 update 层面引入更细粒度的熵保护。

第一版可以先不实现 token 级算法，但必须在实验设计中预留熵诊断位点。

### 4.4 DAPO

DAPO 说明传统 PPO/GRPO 的裁剪方式可能加剧熵下降，而解耦上下界裁剪能改善稳定性。对本项目的启发是：

- 第一版虽然仍可用 TRL GRPOTrainer，但要避免把当前结果误判为“GRPO 已经最优”；
- 如果标准 GRPO 在长程训练中持续坍缩，后续可将 DAPO 式裁剪作为主要算法变体进行对照；
- 这类方法尤其适合 strict schema 场景，因为策略空间本就更窄。

### 4.5 Pass@k Training

Pass@k Training 指出奖励设计本身会改变探索与利用平衡。对本项目的启发是：

- 只优化 `pass@1` 风格的单样本 correctness，很容易推动策略走向保守模板；
- 本项目应至少在评估端持续跟踪 `pass@k`；
- 后续可研究把 diversity-aware 或 pass@k-style signal 引入训练，而不是只看单次 executable correctness。

### 4.6 Negative Reinforcement

负强化研究指出，压制错误路径本身也能带来探索收益。对本项目的映射是：

- 针对明显错误 program 模板，可以考虑显式负激励，而不是简单记零；
- 这类机制可能比“奖励更多正确模板”更有助于释放策略空间；
- 特别是在 `ConvFinQA` follow-up 场景，抑制“忽略历史的快捷 program”可能非常有价值。

## 5. 方法设计

## 5.1 Verifier-Guided RLVR

本项目中的 verifier 不是单一分数器，而是一个由多个组件组成的校验体系：

- schema verifier：检查是否满足 `Evidence + Program` strict schema；
- program parser：检查 DSL 语法是否合法；
- executor：执行预测 program；
- answer normalizer：统一数值、百分比、million 等表示；
- evidence checker：检查 evidence 与原始 prompt 数字/证据的对应关系；
- history consistency checker：针对 `ConvFinQA` 检查 follow-up turn 是否使用了正确的历史语义。

这一设计的核心思想是：reward 不再由一个“是否答对”布尔值构成，而是由 verifier 在多个层面共同决定。

## 5.2 PoT 输出 contract

第一阶段主输出 contract 应固定为：

```text
Evidence:
- ...

Program: ...
```

理由如下：

- 它比 `Evidence + Program + Answer + Normalized Answer` 更接近真正的 executor-driven setting；
- 它避免模型在 RL 中重新学会“生成答案文本”而不是“生成正确程序”；
- 它减少 schema 自身的刷分空间；
- 它与当前 `run_fingpt_pot_rl.ipynb` 的 strict-program 方向一致。

如果需要保留 CoT 能力，应通过 prompt 内部规划或第二阶段 supplement 处理，而不是在第一阶段主输出重新加入显式 `Reasoning:`。

## 5.3 Reward 分层

建议将 reward 拆成三层。

### Gate layer

只要任何硬约束不满足，总 reward 直接为 0 或强惩罚。包括：

- 缺少 `Evidence:` 或 `Program:`；
- 出现 `Reasoning:`、`Answer:`、`Normalized Answer:`；
- `Program` 为空、`N/A`、非法 DSL 或多段串联；
- parse 失败。

### Core layer

主 reward 只绑定执行正确性：

- `execute(pred_program) == gold_answer` 给主要正奖励；
- 否则不给正奖励。

这样做的原因是：错误但可执行的 program 不应继续获得稳定回报。

### Auxiliary layer

辅助层只用于在“执行正确”的候选之间进行细分，而不是主导优化方向。推荐保留：

- evidence grounding；
- canonical program closeness；
- 超长输出惩罚；
- 对 `ConvFinQA` 的 history-consistency 检查。

不建议保留“短输出即正奖励”或“operator overlap 即正奖励”作为独立回报项。

## 5.4 两阶段训练路线

第一阶段：`core-only strict-program GRPO`

- 只使用 `FinQA + ConvFinQA` strict-program 样本；
- 所有样本 `reward_profile = program_numeric`；
- 主目标是建立稳定的 verifier-guided RL 闭环。

第二阶段：`supplement as controlled ablation`

- 仅在第一阶段稳定后，才引入 Fino1 / FinCoT；
- supplement 不改变主 benchmark 口径；
- supplement 主要用于研究表达多样性、cold-start strengthening 和探索增强，而不是直接定义主结论。

## 6. 研究假设

### H1: `program_numeric-only` 比 mixed profile 更稳定

预期表现：

- 更高的 executable-program rate；
- 更低的 schema drift；
- 更少的 reward hacking 行为；
- `ConvFinQA requires_history=true` 子集退化更少。

### H2: 去掉 `parse_bonus / op_bonus / brevity bonus` 能减少 reward hacking

预期表现：

- 错误但可执行 program 的比例下降；
- unique program ratio 更稳定；
- benchmark 提升更能反映真实 reasoning gain，而不是模板 gain。

### H3: 更强 KL / 风险规避目标能延缓 entropy collapse

预期表现：

- 长程训练中 `pass@k` 不更早退化；
- program diversity 下降速度变慢；
- follow-up 子集更稳定。

### H4: `ConvFinQA requires_history=true` 比总体指标更敏感

预期表现：

- 在总体 `ConvFinQA` 不明显掉分时，history-dependent 子集更早暴露 shortcut 行为；
- 该子集可作为判断“是否真正学到多轮推理”的主要探针。

## 7. 实验判据

本研究不以“训练 reward 上升”作为充分证据，而以以下判据判断方案是否成立：

- strict benchmark 是否优于 baseline；
- `execute(pred_program) == gold_answer` 是否提升；
- `unique program ratio` 是否保持在健康区间；
- `pass@k` 是否没有明显退化；
- `ConvFinQA requires_history=true` 是否保持或提升；
- 改进是否来自真实 reasoning，而不是 schema 或模板利用。

如果出现以下情况，则应判定当前设计失败或部分失败：

- format pass 上升，但 execution correctness 不涨；
- executable rate 上升，但 `ConvFinQA requires_history=true` 掉分；
- `pass@1` 轻微上涨，但 `pass@k` 和 diversity 明显下降；
- 模型输出被少量固定 program 模板主导。

## 8. 风险与开放问题

当前仍有若干未解决问题，需要作为后续研究方向显式记录：

- reward 噪声：答案归一、单位处理和 canonical program 可能存在标注不唯一；
- history consistency 的自动验证难度较高，容易产生误惩；
- 过强 schema gate 可能抑制必要探索；
- RiskPO / DAPO / entropy-aware advantage 是否真的适合金融 reasoning，仍需实验验证；
- `FinQA` 和 `ConvFinQA` 的结构化 gold program 是否会限制开放场景泛化，需要在 FinanceBench 等任务上做额外分析。

## 9. 当前结论

基于现有 pipeline，最合理的研究主线不是“直接扩大 mixed RL 数据”，而是：

1. 先把 `FinQA + ConvFinQA` 改造成严格的 `Verifier-Guided PoT-RLVR` 任务；
2. 先用 `program_numeric-only` 建立稳定训练闭环；
3. 先解决 reward hacking 和 entropy collapse，再讨论更复杂的探索增强算法；
4. 再把 RiskPO、DAPO、pass@k-style training、negative reinforcement 等方法作为有明确假设的算法扩展项进行比较。

换句话说，当前项目真正要验证的不是“GRPO 能不能用”，而是：

- 什么样的 verifier 和 reward contract 才能让金融数值推理的 RLVR 真正有效；
- 如何在保持 strict-program 正确性的同时，不让策略过早坍缩成模板化程序。
