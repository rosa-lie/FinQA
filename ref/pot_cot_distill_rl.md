CoT / PoT 是“推理表示形式”  
Distill-R1 / Distill-o1 是“推理能力迁移方法”  
RL / GRPO 是“推理能力激励方法”  
R1 / o1 是“长推理模型范式”  

Base model
-> SFT / distillation 学会推理格式和基本解题轨迹
-> RL / GRPO 用可验证 reward 激励更强推理
-> 得到 R1/o1-style teacher
-> 再 distill 给小模型
-> 可选再做 domain RL

# reasoning

## 1. CoT：自然语言推理轨迹

CoT 是 Chain of Thought，也就是让模型**用自然语言写出中间推理过程**。

它解决的是“模型为什么这么算、为什么选这些证据”的问题。

在金融表文数值推理里，CoT 的作用很直接。模型需要说明当前问题要求的是增长率、平均值、占比还是差额，需要从表格中选哪一年、哪一列、哪一个指标，也需要解释为什么用 (current - previous) / previous 或 amount / total。这些东西用自然语言写出来，会让模型更容易形成任务理解，也让人更容易审查。

但 CoT 的问题是，它不可执行。模型可以写出一段看似合理的 reasoning，却在最后一步算错，或者中间偷偷换了单位。对金融数值题来说，这个风险很大。尤其当 CoT 变长以后，模型可能学到的是“推理腔调”，而不是可验证计算。

CoT = 证据选择 + 公式意图解释

## 2. PoT：程序化推理轨迹

PoT 是 Program of Thought，也就是把推理中的计算写成程序。它可以是 Python，也可以是你当前 FinQA/ConvFinQA 使用的 DSL Program。

```text
Reasoning:
The question asks for percentage change, so compute the change and divide by the previous value.

Program:
divide(subtract(6823, 6161), 6161)
```

PoT 的价值是：它能被 executor 执行。执行结果可以和 gold answer_norm 比较，因此可以变成稳定的 evaluation metric 和 RL reward。

FinQA/ConvFinQA 原生就有：gold program；exe_ans；answer_norm。
所以不是只学金融 CoT，而是学 evidence-grounded Program-of-Thought。

Evidence -> Reasoning -> Program -> Execution -> Normalized Answer

# distill

Distill-R1 或 Distill-o1 不是一种推理格式，而是一种训练方法。它的意思是：用一个更强的 reasoning model，例如 DeepSeek-R1、o1、GPT-4o、DeepSeek-Reasoner，生成高质量推理轨迹，然后让较小模型通过 SFT 学习这些轨迹。

strong teacher
-> generate reasoning traces
-> filter correct / high-quality traces
-> train student with SFT

Distillation 是模仿学习，目标是让 student 模仿 teacher 的输出分布。它能快速注入推理风格和解题模式，但它不会自动保证 student 真的“会探索”或“会自我纠错”。

DeepSeek-R1 里很重要的经验是：R1-style reasoning 可以被蒸馏到较小模型上，而且 distillation 往往比直接对小模型从零做 RL 更有效。原因很朴素：小模型如果一开始不会产生高质量推理轨迹，RL reward 会很稀疏；先用强 teacher 的 reasoning traces 做 SFT，可以把模型带到一个更好的起点。

但 distill 的风险也很明显：模型可能学会长推理格式，但没有学会可验证计算

一个 R1/o1-style teacher 可能生成很长的 CoT，但如果没有 program/execution check，你很难知道它是否真正取对了表格数字、单位和公式。

所以，需要：distill 负责提供推理表达，Program 负责把它拉回可验证轨道。

# RL

SFT:
看到标准答案，然后模仿

RL/GRPO:
自己生成多个答案，根据 reward 判断好坏，然后强化好的输出

DeepSeek-R1 的重要启发是：如果 reward 设计得足够清晰，比如数学题答案正确、格式正确，模型可以通过 RL 激发出更强的 reasoning 行为。R1-Zero 展示了几乎不靠人工 CoT SFT、直接用 RL 也能诱发长推理、自我反思等行为；但它也会有可读性差、语言混杂、格式不稳的问题。所以 DeepSeek-R1 正式路线又加入了 cold-start SFT、rejection sampling、再 RL 等阶段。

RL 的关键不是“奖励模型写更长推理”，而是奖励：
 - Program 可解析
 - Program 可执行
 - execute(Program) == gold answer_norm
 - 答案格式稳定
 - 证据数字能在 prompt 中找到
 - Reasoning 简洁

也就是说，你的 RL reward 应该比 Fino1-style 纯 answer/format reward 更强，因为你有 Program execution 这个客观信号。

# plan

Step 1: 用 CoT/PoT 数据做 SFT
让模型先会基本推理格式

Step 2: 用 RL/GRPO 做可验证优化
让模型把正确推理变成高概率输出

Step 3: 得到强 reasoning model
例如 R1/o1-style teacher

Step 4: 再 distill 给小模型或领域模型
让小模型继承 reasoning traces

Step 5: 领域内再做 verifiable RL
让模型适应金融表文数值推理

# ref

DeepSeekMath 的关键启发是：数学推理任务适合用可验证答案做 RL，GRPO 可以在没有传统 reward model 的情况下提升数学 reasoning。它说明：只要任务有明确正确答案，RL 就有抓手。

DeepSeek-R1 的关键启发是：reasoning capability 可以被 RL 激励出来，也可以被 distillation 迁移到更小模型。R1-Zero 说明纯 RL 可以诱发 reasoning，但正式 R1 说明冷启动 SFT、过滤数据、再 RL 更稳。

Fino1 的关键启发是：金融领域可以走类似路线：
financial reasoning path SFT
-> GRPO
-> financial benchmark

Fino1/FinCoT 更偏 CoT reasoning path，Program 监督不是它的核心。

本项目下：在 Fino1-style baseline 上加入 FinQA/ConvFinQA gold Program supervision，把金融 CoT 推理扩展成可验证 CoT+PoT，再用 Program execution reward 做领域 RL
 - DeepSeekMath / DeepSeek-R1:提供 general reasoning RL 范式
 - Fino1:提供 financial reasoning SFT + GRPO baseline

PoT: [done]
从 FinQA/ConvFinQA gold Program 学习可执行计算

CoT: [fin-o1]
从 Fino1/FinCoT/R1/o1 学习金融推理表达

Distill-R1/o1: [fin-o1]
用强 teacher 或公开 reasoning path 给模型补推理轨迹

RL/GRPO:
用 Program execution correctness 把正确推理推成高概率输出


1. 保留 sft2_dual_merged 作为 **program-supervised baseline**
2. 引入 Fino1_Reasoning_Path_FinQA / FinCoT
   作为低比例 **CoT supplement**
3. 新增 **Reasoning** 字段
   形成 Evidence + Reasoning + Program + Answer + Normalized Answer
4. 不让外部 CoT 覆盖 gold Program
   Program 仍来自 FinQA/ConvFinQA
5. 做 **GRPO**
   reward 主项是 execute(Program) == gold answer_norm
