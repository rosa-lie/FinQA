# 2025-2026 金融 LLM 推理前沿综述与 FinQA 项目升级建议

## 1. 目的

本文档整理 2025-2026 年与本项目最相关的 LLM 推理前沿方向，重点回答三个问题：

1. 近两年相关领域的主流研究趋势是什么。
2. 当前 `FinQA` 项目与 `Fin-o1 / Fin-R1 / DianJin-R1` 这类路线相比，核心差异在哪里。
3. 当前项目如果希望更贴近前沿，应优先做哪些升级。

本文不试图覆盖全部 LLM 论文，只聚焦与以下主题直接相关的方向：

- 金融 reasoning post-training
- CoT / PoT / distillation / RLVR / GRPO
- 金融数值推理 benchmark
- evidence retrieval / long-context reasoning / verifier

---

## 2. 相关领域前沿趋势

### 2.1 总体趋势

截至 2025-2026，相关前沿已经明显从“多加长 CoT”转向更结构化的方向，主要集中在五条主线：

1. **结构化 reasoning supervision**
   - 不再只蒸馏长自然语言 CoT，而是强调更可控的中间表示，例如 short plan、tool call、program、verifier trace。

2. **verifiable RL / RLVR / GRPO**
   - 强调 reward 必须尽可能可验证，而不是只靠风格、格式或弱字符串匹配。

3. **planning + execution 分层**
   - 上层负责问题分解、证据组织、步骤规划，下层负责程序生成、工具调用、精确计算。

4. **retrieval / long-context / evidence-centric reasoning**
   - 尤其在金融场景中，很多错误并不是“不会算”，而是“证据没找对”“变量绑定错”“上下文太长导致取错数”。

5. **更严格、更真实的金融 benchmark**
   - 只看 FinQA/ConvFinQA 已不足以全面刻画模型能力，开始强调长文档、多表、多场景、难度分层评测。

---

## 3. 与本项目最相关的代表论文

### 3.1 金融 reasoning 直接相关工作

#### Fino1 (2025)

- 文献：`ref/fino1.md`
- 代表思想：构建金融 CoT 数据、训练金融 reasoning 模型、比较 RL 方法，并提出更完整金融评测。
- 对本项目的启发：
  - 金融 reasoning 不能简单依赖通用 reasoning 迁移。
  - 数据质量、任务覆盖、reward 设计都更重要。
  - `FinCoT + SFT + RL` 是成立的，但前提是金融数据和金融评测都要做对。

#### Fin-R1 (2025/2026)

- 文献：`ref/fin-r1.md`
- 代表思想：`distill CoT -> SFT -> GRPO` 两阶段金融 reasoning 路线。
- 对本项目的启发：
  - 高质量金融 distill 数据有价值。
  - `FinQA/ConvFinQA` 很适合放进 RL 阶段。
  - 金融任务上的 RL 不应只优化风格，更应绑定任务 correctness。

#### DianJin-R1 (2025)

- 文献：`ref/dianjin-r1.md`
- 代表思想：将 reasoning enhancement、金融知识、表达风格、业务落地需求结合。
- 对本项目的启发：
  - FinQA 不应是唯一训练来源。
  - 如果目标不仅是 benchmark 分数，还包括可解释输出、中文能力和行业表达，那么需要增加领域语言/业务约束训练。

#### FinanceReasoning (2025)

- 文献：`ref/finance-reasoning.md`
- 代表思想：当前金融 benchmark 不够可信、不够全面，需要更严格、更真实的评测体系。
- 对本项目的启发：
  - 只优化 FinQA/ConvFinQA 不能完整证明金融 reasoning 泛化能力。
  - 后续评测应加入更复杂、更长上下文、更分难度层级的数据。

---

### 3.2 与方法学强相关的工作

#### Program of Thoughts (PoT)

- 文献：`ref/program-of-thoughts.md`
- 代表思想：让模型负责生成程序，把精确计算交给解释器。
- 对本项目的启发：
  - FinQA/ConvFinQA 天然适合 PoT。
  - `Program` 应是核心监督目标之一，不应被自由 CoT 替代。
  - reward 设计应围绕 program parse、program execution、executed answer correctness。

#### FINDER: Program of Thoughts for Financial Reasoning (EMNLP 2025)

- 文献：`ref/finder-financial-pot.md`
- 代表思想：金融推理的瓶颈不仅在 program 生成，还在前置 evidence retrieval 和 dynamic example selection。
- 对本项目的启发：
  - 训练数据里不能只保留 final program，最好保留 evidence selection 轨迹。
  - 推理时可以考虑显式 evidence extraction / retrieval，再进入 program generation。

#### Distilling Reasoning Capabilities into Smaller Language Models

- 文献：`ref/distilling-reasoning-capabilities.md`
- 代表思想：不要只蒸馏最终答案或长 CoT，而要把隐式推理拆成显式结构，例如问题分解、子问题求解、规划链。
- 对本项目的启发：
  - 比起蒸馏长篇自由文本 CoT，更适合蒸馏 `evidence -> short plan -> program -> answer` 这样的链式结构。

#### DeepSeek-R1 / DeepSeekMath

- 文献：`ref/deepseek-r1.md`, `ref/deepseekmath.md`
- 代表思想：reasoning post-training 的核心收益来自可验证任务、良好的冷启动、以及 RL 后训练。
- 对本项目的启发：
  - 不宜直接做纯 RL Zero 风格尝试。
  - 应先有稳定的冷启动过程表示，再做 execution-based RL。

---

### 3.3 2025-2026 顶会/正式会议中的相关方法趋势

以下几类工作虽然不一定直接做金融，但对本项目的方法设计影响很大：

#### Progress / Process Verifier 方向

- ICLR 2025 `Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning`
- ICLR 2026 `Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs`

对本项目的启发：

- 不应只看 final answer reward。
- 可以考虑给中间过程增加轻量 verifier，例如：
  - evidence 是否覆盖关键数字
  - plan 是否包含正确运算意图
  - program 是否与 plan 一致
- `RLVR` 在可验证任务上是成立的，但 reward 必须严格对齐真实任务目标。

#### Long-context reasoning supervision 方向

- EMNLP 2025 `Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision`
- EMNLP 2025 `Facilitating Long Context Understanding via Supervised Chain-of-Thought Reasoning`

对本项目的启发：

- CoT 在金融场景里的真正价值，更可能体现在 long-context evidence organization，而不是最终数值计算。
- 如果后续扩展到更长的财报/公告/多表场景，structured CoT supervision 的收益会增大。

#### Formal verification / planning with tools 方向

- NAACL 2025 `Large Language Models Can Solve Real-World Planning Rigorously with Formal Verification Tools`

对本项目的启发：

- 金融数值推理可以继续沿着“formalized representation + external verifier / executor”方向演化。
- 当前仓库已有 DSL program executor，这是明显优势。

---

## 4. 当前 FinQA 项目的定位

### 4.1 当前项目最强的优势

结合当前代码、文档与 notebook，项目已经具备以下优势：

1. **拥有 gold Program + executor**
   - 这是本项目区别于很多 CoT-only 金融 reasoning 项目的核心优势。
   - `FinQA/ConvFinQA` 不只是给出答案，还给出可执行中间表示。

2. **主干数据组织清晰**
   - `FinQA` 负责单轮 program supervision。
   - `ConvFinQA` 负责 turn-level 多轮 supervision。
   - `FinQA replay` 用来降低遗忘。

3. **评测口径更接近 verifiable reasoning**
   - 当前 benchmark 不只看 `answer_accuracy`，还看：
     - `program_parse_rate`
     - `program_execution_rate`
     - `executed_answer_accuracy`
     - `program_answer_consistency`
     - `pass@k`

4. **项目已经明确 external CoT 的定位**
   - `Fino1/FinCoT` 被视为 reasoning supplement，而不是替代 gold Program 的主干。
   - 这一判断非常重要。

---

### 4.2 当前项目的主要短板

1. **benchmark 仍偏窄**
   - 当前主线仍以 `FinQA/ConvFinQA` 为中心。
   - 这足以做可验证 program reasoning 主线，但不足以证明金融泛化能力。

2. **CoT 的角色尚未完全结构化**
   - 当前项目已经知道 CoT 不应取代 Program，但 CoT 还没有稳定收敛成短 `Plan` 层。
   - 目前更像“有时加 reasoning，有时去掉 reasoning”，而不是明确的多层接口。

3. **distill 还不是 execution-aware distill**
   - 外部 teacher traces 虽然可作为 supplement，但尚未系统过滤成：
     - evidence 正确
     - plan 合理
     - program 可执行
     - answer 与 execution 一致

4. **GRPO 设计仍偏研究原型**
   - 方向是对的，但第一轮更应是 core-only execution GRPO，而不是太早做 mixed profile。

5. **retrieval / verifier 还没成为显式模块**
   - 当前有 executor，但缺少更前置的 evidence retrieval / evidence planning 模块。
   - 也缺少 process verifier / progress reward 这类更前沿的中间奖励机制。

---

## 5. 与 Fin-o1 / Fin-R1 / DianJin-R1 的核心差异

### 5.1 它们的典型主线

前沿金融 reasoning 工作的典型路径是：

```text
large-scale financial CoT distill
-> reasoning SFT
-> GRPO / RL post-training
```

它们更偏向于：

- 金融 reasoning 风格迁移
- 问题分解与解释能力
- 更广金融语料覆盖
- 金融表达与行业风格

### 5.2 本项目的典型主线

本项目更接近：

```text
FinQA / ConvFinQA gold Program supervision
-> verifiable program reasoning SFT
-> execution-based GRPO
```

它更偏向于：

- 可执行中间表示
- 数值正确性
- 程序解析与执行
- answer-program consistency
- 多轮数值 follow-up 稳定性

### 5.3 结论

因此，本项目不是“没有跟上 CoT+distill”，而是优化对象更硬、更窄，也更可验证。

简化地说：

- `Fin-o1 / Fin-R1 / DianJin-R1` 更像金融 reasoning post-training 模型
- 当前项目更像金融数值 reasoning solver

这意味着本项目不应简单照搬它们的“长 CoT 主导路线”，而应保留自己的 program-execution 主轴，再吸收它们在 distill、planning、retrieval、benchmark 扩展上的优点。

---

## 6. CoT 对当前项目的真正意义

当前项目里，CoT 最合理的角色不是最终计算层，而是 **规划层**。

更准确地说，CoT 的意义主要有三点：

1. **证据组织**
   - 帮模型明确需要从哪里取数、哪些数字相关、哪些句子是支持证据。

2. **问题分解**
   - 帮模型先明确“要算什么”“先算什么后算什么”。

3. **多轮承接**
   - 在 ConvFinQA 中，CoT/plan 有助于稳定 follow-up question 的语义承接。

但 CoT 不应承担：

- 最终精确计算
- 最终 correctness 的主要监督接口
- RL 的主 reward 对象

因为这些任务，`Program` 和 executor 更擅长。

一句话总结：

> CoT 在当前项目中不是终点，而是 Program 的上游控制层。

---

## 7. 推荐的升级方向

### 7.1 第一优先级：从自由 CoT 升级到短 Plan 层

建议把当前 schema 从“有时带 Reasoning，有时不带”升级成明确的多层结构：

```text
Evidence:
- ...

Plan:
- identify the relevant figures
- compute the year-over-year change
- normalize percentage as a decimal ratio

Program:
divide(subtract(...), ...)

Answer: ...
Normalized Answer: ...
```

这里：

- `Plan` 负责分解
- `Program` 负责计算
- `Answer / Normalized Answer` 负责对外接口

这比长篇自由 CoT 更适合 FinQA/ConvFinQA。

---

### 7.2 第二优先级：做 execution-aware distill

后续 distill 数据不应只是：

```text
question -> long reasoning -> answer
```

更合适的是：

```text
question/context -> evidence -> short plan -> program -> answer
```

过滤规则至少包括：

- evidence 是否覆盖 gold evidence
- program 是否可 canonicalize / 可执行
- `execute(program)` 是否等于 gold answer
- `Normalized Answer` 是否与执行结果一致

没有通过这些检查的 CoT，只能作为弱 supplement，不能进入主干。

---

### 7.3 第三优先级：GRPO 先做 core-only execution RL

第一轮 GRPO 建议只使用：

- `FinQA`
- `ConvFinQA`
- gold `Program`
- gold `Normalized Answer`

主要 reward 建议围绕：

1. `execute(pred_program) == gold_answer`
2. `pred_normalized_answer == execute(pred_program)`
3. `pred_program` 可执行、格式合法
4. 小权重的 plan / evidence 辅助 reward

不要让自由长 CoT 成为 RL 主对象。

---

### 7.4 第四优先级：显式引入 retrieval / evidence planning

可以从简单版本开始：

- 训练时保留 `aligned_evidence`，把 evidence extraction 当作显式子任务
- 推理时先做 evidence extraction，再做 program generation

这条线和 `FINDER` 的思想最接近，也最可能修复 FinQA/ConvFinQA 中“公式看起来对，但变量绑定错”的问题。

---

### 7.5 第五优先级：升级评测矩阵

后续不应只看 `FinQA/ConvFinQA` 平均分。

建议新增：

- 更强金融 benchmark，如 `FinanceReasoning`
- easy / medium / hard 分层评测
- 内部错误类型分层：
  - evidence error
  - operator error
  - scale / unit error
  - turn-history error
  - answer-program inconsistency

这样更容易看清模型到底提升在哪一层。

---

## 8. 建议的升级版训练路线

结合当前项目特点与 2025-2026 前沿方向，建议的升级版主线如下：

```text
Teacher Distill
-> Evidence + Short Plan SFT
-> Evidence + Plan + Program SFT
-> Evidence + Plan + Program + Answer SFT
-> Core-only execution-based GRPO
-> Small-ratio external CoT/Long-context supplement
-> Retrieval / verifier / benchmark extension
```

其中：

- `Teacher Distill` 负责提供高质量 plan / reasoning 风格
- `Program SFT` 负责保持可执行中间表示的主干地位
- `GRPO` 负责把正确 program 和正确答案推成高概率输出
- retrieval / verifier 负责进一步接近前沿 reasoning system

---

## 9. 最终判断

如果只保留一句话：

> 当前项目最适合走的不是“金融长 CoT 主导路线”，而是“结构化 CoT plan + gold Program + execution-aware distill + core-only GRPO”的 FinQA 专用路线。

更具体地说：

- **保留你的 program-execution 主轴**，这是项目最大优势。
- **吸收前沿工作的 planning、distill、retrieval、verifier 思路**，但不要让它们覆盖 gold Program 主干。
- **把 CoT 收缩成短 plan**，让它服务于 Program，而不是替代 Program。

这样做，既能保持项目的可验证性优势，也能和 2025-2026 reasoning 前沿真正接轨。

---

## 10. 参考资料

本综述主要基于仓库内下列参考文档整理：

- `ref/fino1.md`
- `ref/fin-r1.md`
- `ref/dianjin-r1.md`
- `ref/finance-reasoning.md`
- `ref/program-of-thoughts.md`
- `ref/finder-financial-pot.md`
- `ref/distilling-reasoning-capabilities.md`
- `ref/deepseek-r1.md`
- `ref/deepseekmath.md`
- `docs/fin_pot_cot_rl.md`

外部正式论文/会议信息可进一步参考：

- ICLR 2025: Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning
- ICLR 2026: Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs
- EMNLP 2025: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
- EMNLP 2025: Facilitating Long Context Understanding via Supervised Chain-of-Thought Reasoning
- NAACL 2025: Large Language Models Can Solve Real-World Planning Rigorously with Formal Verification Tools
- EMNLP 2025: Program of Thoughts for Financial Reasoning: Leveraging Dynamic In-Context Examples and Generative Retrieval
