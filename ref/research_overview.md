# FinQA 相关研究总览

`ref` 目录中的文献大致可以分成四条线索。

第一条线索关注金融推理模型本身，代表工作包括 `Fino1`、`Fin-R1`、`DianJin-R1` 和 `Unlocking Data Value in Finance`，核心问题是如何把通用推理能力迁移到金融场景，并通过数据蒸馏、难度筛选、监督微调和强化学习让模型在金融问答、表格推理和长文档数值推理上真正稳定下来。

第二条线索关注可执行推理，代表工作是 `Program of Thoughts` 及其面向金融的扩展 `FINDER`，其价值不在于让模型写出更长的解释，而在于把推理过程转成可执行程序、可追踪变量和可验证计算。

第三条线索关注推理蒸馏与强化学习范式，代表工作包括 `DeepSeek-R1`、`DeepSeekMath` 和 `Distilling Reasoning Capabilities into Smaller Language Models`，它们为金融领域提供了如何构造推理数据、如何设计中间过程监督、如何让小模型承接大模型推理模式的训练模板。

第四条线索是 benchmark 与评测，代表工作 `FinanceReasoning` 说明金融数值推理不能只盯住 FinQA/ConvFinQA 的短样本场景，还要覆盖长文档、复杂表格和更可信的难度分层。

对于 FinQA 方向，最重要的结论并不是单纯追求更长的 CoT，而是把证据抽取、问题分解、程序生成、执行验证和奖励设计串成一个闭环。FinQA 的难点通常不是“答案公式会不会写”，而是“证据能不能取对、变量能不能绑定对、程序执行会不会漂移、最终答案能不能被验证”。因此，这批文献最值得复用的不是某个单点技巧，而是一套从数据到训练再到推理时验证的工程化方法。

## Fino1

`Fino1` 从金融推理能力迁移的角度出发，讨论通用 reasoning LLM 与 RL 范式能否真正落到金融任务中。论文一边构造 `FinCoT` 数据与 `Fin-o1` 模型，一边提出 `FinReason` 评测集，用实验说明金融场景对数据质量和训练路径更敏感，规模更大的通用模型并不一定天然占优。详见 [fino1.md](/root/FinQA/ref/fino1.md)。

## Fin-R1

`Fin-R1` 采用更明确的“两阶段”路线：先基于蒸馏金融推理数据做 SFT，再通过 GRPO 和奖励设计做 RL，对 FinQA、ConvFinQA 等任务进行定向增强。它对 FinQA 的直接启发在于，面向可验证数值问题的 RL 不必覆盖全部金融任务，但应该把奖励集中在格式、答案正确率和过程一致性上。详见 [fin-r1.md](/root/FinQA/ref/fin-r1.md)。

## DianJin-R1

`DianJin-R1` 更强调中文金融场景、数据增强监督和产业可落地性。它不是只做一个学术 benchmark 模型，而是把金融数据、行业规则、领域评测和增强训练一并组织起来。对 FinQA 而言，这篇工作提示可以把英文表格问答经验扩展到中英双语、合规表达和更广泛的金融知识任务。详见 [dianjin-r1.md](/root/FinQA/ref/dianjin-r1.md)。

## Unlocking Data Value in Finance

`Unlocking Data Value in Finance` 的重点不在提出一个全新模型，而在于系统论证“后训练数据质量”与“样本难度/可验证性分布”对金融垂域表现的决定性影响。它对 FinQA 的价值非常直接，因为 FinQA 本身就是高可验证、强数值约束的数据形态，非常适合被放到一个更严格的数据难度课程中使用。详见 [unlocking-data-value-in-finance.md](/root/FinQA/ref/unlocking-data-value-in-finance.md)。

## FinanceReasoning

`FinanceReasoning` 提醒人们不要把金融推理等同于单一数据集得分。它把问题做得更可信、更全面，也更强调长上下文、多表关系和复杂数值链条，从而暴露出现有模型在真实金融材料上的脆弱性。对于 FinQA 研究，这意味着后续训练与评测不能停留在短上下文表格问答，而应主动扩展到更长文档和更复杂证据组合。详见 [finance-reasoning.md](/root/FinQA/ref/finance-reasoning.md)。

## Program of Thoughts

`Program of Thoughts` 证明了把推理过程表示成程序，比把全部计算硬塞进自然语言 CoT 更适合数值任务。它在 FinQA、ConvFinQA、TAT-QA 上的表现说明，金融数值推理天然适合“程序化中间表示”。这几乎是 FinQA 方向最应优先吸收的思想之一。详见 [program-of-thoughts.md](/root/FinQA/ref/program-of-thoughts.md)。

## FINDER

`Program of Thoughts for Financial Reasoning` 提出的 `FINDER` 框架进一步说明，金融推理的瓶颈不只在程序生成，还在证据检索、事实组织和动态示例选择。对于 FinQA 而言，这意味着如果只蒸馏程序答案而不蒸馏证据选择过程，模型仍然容易在前置步骤失真。详见 [finder-financial-pot.md](/root/FinQA/ref/finder-financial-pot.md)。

## DeepSeek-R1

`DeepSeek-R1` 为垂域推理训练提供了强方法学参照。它证明 RL 可以直接激励推理行为，同时也说明纯 RL 发现的推理模式在可读性和可控性上会有问题，因此仍需要冷启动数据、多阶段 SFT 与蒸馏。对于 FinQA，这篇论文最值得复用的是训练配方，而不是直接照搬通用 benchmark。详见 [deepseek-r1.md](/root/FinQA/ref/deepseek-r1.md)。

## DeepSeekMath

`DeepSeekMath` 虽然不是金融论文，但它把数学数据采集、继续预训练、RL 强化和评测组织成一条很完整的能力增强路线。FinQA 与其差别在领域而不在问题结构，因为两者都高度依赖数值稳定性、符号分解和结果验证。详见 [deepseekmath.md](/root/FinQA/ref/deepseekmath.md)。

## Distilling Reasoning Capabilities into Smaller Language Models

这篇工作用 `Socratic CoT` 展示了如何把大模型的逐步推理能力拆成更可学习的中间结构，再蒸馏给更小模型。虽然不是金融专用方法，但它对 FinQA 的启发很强，因为 FinQA 原本就适合被拆成“问题分解器 + 子问题求解器 + 程序执行器”的组合式系统。详见 [distilling-reasoning-capabilities.md](/root/FinQA/ref/distilling-reasoning-capabilities.md)。
