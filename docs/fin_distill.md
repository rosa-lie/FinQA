# 金融数值推理数据蒸馏计划

## 蒸馏介绍以及必要性

**💡 蒸馏（Distillation）本质上就是：让一个更强的 teacher 模型，去给一个较小的 student 模型“示范怎么做题”，然后 student 去学 teacher 的行为模式。**
- 普通 SFT：直接拿人工标注答案训练模型；
- 蒸馏 SFT：先让一个更强模型（比如 GPT-4.1 / o3 / Claude / 更强金融推理模型）对原始题目生成更优质答案，再把这些高质量答案拿来训练 student。

> “原始数据集给的监督信号，不一定是最适合我当前目标模型学习的；那我就先用强模型把监督信号改造得更适合训练。”

金融对话数值推理：任务的难点不是“术语会不会”，而是：会不会抓关键证据；会不会做多步计算；会不会保持程序推理一致；会不会在多轮对话里延续上下文。而 teacher 恰好可以把这些过程输出得更“像一个好学生应该模仿的答案”。

推理模型比普通聊天模型更需要蒸馏，原因非常现实：推理能力不是只靠“看答案”就能稳定学出来的。
- 普通问答里，模型很多时候靠模式匹配也能答得像样；
- 但在推理任务里，模型要学的是一整套中间能力：如何拆题；如何定位证据；如何把证据映射成程序/计算；如何控制最终答案格式；如何在多轮对话里不丢条件。

原始监督信号不够“适合 student 学”：
- 旧版 sft_merged benchmark 低于 base，说明“训练 loss 更低 ≠ 推理效果更好”，原因之一就是 原始 target 设计存在问题，尤其是 gold_ind/gold_inds 这种原始结构化证据不适合直接监督模型。

**💡 蒸馏：把“原始但不好学”的监督，变成“更自然、更稳定、更适合 student 学”的监督。**

| 外部论文/项目                              | 核心思想                                                        | 对项目最对应的模块                                                | 该怎么借                                                                          | 预期收益                           | 当前优先级     |
| ------------------------------------ | ----------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------ | --------- |
| **STaR (2022)**                      | 生成 reasoning，只保留**最终答对**的 rationale 再继续训练                   | `domain/financial/distill/distill_with_teacher.py` + `domain/financial/distill/score_distill_candidates.py` | 不要“teacher 说什么都学”，而是保留 **答案正确 + 结构完整 + 程序一致** 的候选进 SFT                         | 提高蒸馏数据纯度，减少 teacher 噪声         | **最高**    |
| **Distilling Step-by-Step (2023)**   | 用 **rationale / step-by-step** 作为额外监督，小模型更容易学会任务            |  `问题分析 / 关键证据 / 推理程序 / 最终答案` 四段式 target                 | 保持四段式结构，不要退化成只学 `最终答案`；后续可做“去掉某一段”的消融实验                                        | 提高 student 对中间推理链的学习能力         | **最高**    |
| **DeepSeek-R1 (2025)**               | 先把 reasoning teacher 做强，再蒸到小模型；公开了 distilled dense models   | 未来的 “更强 teacher → 蒸 student” 路线                          | 现在可先用强 API / 强 checkpoint 做 teacher；后面若有更强 GRPO teacher，再二次蒸馏                 | 给提供“reasoning 小模型靠蒸馏是主流路线”的依据 | **高**     |
| **Open-R1 (2025)**                   | 把 R1 的蒸馏、SFT、RL 训练 recipe 做成开源链路，且第一步就是蒸 R1-Distill         |  `build -> generate -> score -> train` 工程链路             | 参考其 synthetic reasoning data 的组织方式、生成和训练解耦方式、数据版本化                             | 帮把蒸馏从“想法”升级成规范工程流程            | **高**     |
| **LLM KD Survey (2024)**             | 把 LLM 蒸馏分成 white-box / black-box / skill / vertical domains | “相关工作”和方法定位                                             | 用它来界定：做的是 **black-box response distillation + domain reasoning distillation** | 帮写项目文档、答辩、简历表述                | **中**     |
| **KD for LLMs Survey (2024)**        | 讲方法、评测、应用，强调蒸馏不能只看通用语言能力，要看任务能力                             |  benchmark 设计                                           | 支撑“蒸馏是否有效，必须看 FinQA / ConvFinQA 指标，不看表面流畅度”这件事                                | 加强实验设计说服力                      | **中**     |
| **RL-aware KD for Reasoning (2026)** | RL 训练出的 teacher，不一定能被普通 SFT 蒸馏完全吸收                          | 未来的 “GRPO 后 teacher 再蒸 student” 路线                       | 先不用复现，但可列为后续方向：若做 GRPO teacher，再研究 distill-after-RL                            | 让项目 roadmap 更完整                | **低（当前）** |


| 当前项目模块                              | 最该参考的外部工作                  | 为什么最像                                                                    | 应该具体加什么                                                                                        |
| ------------------------------------ | -------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `domain/financial/distill/build_financial_distill_dataset.py` | Open-R1                    | Open-R1 很强调把生成和训练前处理解耦，先构造高质量 distill corpus，再独立训练                       | 给每条样本补足 `task_name / source_dataset / record_id / gold_answer / gold_program / prompt_len` 等元信息 |
| `domain/financial/distill/distill_with_teacher.py`            | STaR + Open-R1             | 一个强调多轮生成后只保留对的，一个强调高质量 reasoning corpus 生产                               | 每题保留 3–5 个候选；同时记录温度、teacher 名称、采样轮次                                                             |
| `domain/financial/distill/score_distill_candidates.py`        | STaR                       | 核心就是“不是所有 rationale 都值得学”                                                | 先统一打分，再按组直接选 `chosen/rejected`；没有可用 `chosen` 的整组样本直接丢弃                                           |
| 蒸馏版 SFT 数据                           | Distilling Step-by-Step    | 该论文强调 rationale supervision 的价值                                          | 当前训练目标收敛为 `<think>/<answer>`；SFT 仅使用高分 `clean_response`                                                 |
| 蒸馏版 DPO 数据                           | Open-R1 / DeepSeek-R1 思路延伸 | 多候选 reasoning 非常适合筛 chosen/rejected                                      | `chosen` 取组内最高分 clean 候选，`rejected` 取其余可用 response 中最低分候选                                          |
| 未来 GRPO 后再蒸                          | DeepSeek-R1 + RL-aware KD  | R1 的 distilled dense models 证明了“teacher 强了再蒸”是有效路线；RL-aware KD 说明这还是研究热点 | 先作为 roadmap，不要现在主攻                                                                              |


## 蒸馏的目标：可验证的结构化 reasoning data distillation


> **target：可验证的结构化 reasoning data distillation**

> 1. 找一个足够强的 teacher（闭源 API）让它在 FinQA / ConvFinQA prompt 上生成结构化推理；  
> 2. 用答案 / program / evidence / structure 做自动验证；
> 3. 把通过的候选变成：distilled SFT、distilled DPO、audit / verifier 数据；  
> 4. benchmark 证明它比“非蒸馏 SFT”更好。

我采用 **black-box response distillation** ，将高能力 teacher 的结构化推理轨迹迁移到学生模型，并通过自动验证机制控制监督质量。
- 受 STaR 启发，我们不直接使用所有 teacher outputs，而是仅保留通过数值正确性、结构完整性与程序一致性校验的候选，构建高质量 reasoning distillation 数据集。
- 参考Deepseek-R1以及相关复现项目Open-R1，落地到金融数值推理领域中。保留金融结构化输出。

两阶段门控：阶段A：answer check（硬过滤：结构完整，四段锚点齐全；最终答案可提取；与 gold_answer 数值或文本一致；证据段无 JSON-like 垃圾）；阶段B：reasoning selection（候选排序，按 rubric 打分：逻辑一致性；指令对齐；证据质量；程序一致性 / operator sequence）

Stage 1：用现有 processor 构造高一致性 prompt  
Stage 2：teacher 生成多个 reasoning 候选  
Stage 3：用 verifier / 规则做 outcome filtering  
Stage 4：先做 distilled SFT  
Stage 5：再把多候选转成 distilled DPO

**reference：**

**STaR: Bootstrapping Reasoning With Reasoning（2022）**：先让模型尝试生成 reasoning；只保留那些“最终答对”的 reasoning；再用这些正确 reasoning 反过来继续训练模型。
 - teacher 生成多个候选
 - 用答案 / 程序 / 结构 / JSON 痕迹校验
 - 保留通过的候选
蒸馏数据的价值，不在于 teacher 生成了多少，而在于 teacher 生成中有多少“正确可学”的 reasoning。


**Distilling Reasoning Capabilities into Smaller Language Models（2022）**：未来做蒸馏时，应该更重视：program consistency;evidence grounding;answer correctness。而不是只盯着“teacher 话术好不好看”。
 - 这会让蒸馏更偏“任务能力”，而不是“语言风格”。

**Deepseek-R**1：先用 RL / reasoning training 把 teacher 做强，再把 reasoning outputs 蒸到更小 student 上（即distill-R1）。
 - 小模型不一定非要自己学会“从零 RL 出 reasoning”；更现实的路线是：先把大模型训练成 reasoning teacher，再蒸给小模型。

**重点复现：[open-r1](https://github.com/huggingface/open-r1)**
1. synthetic reasoning data generation
2. generate → filter → train
3. 如何处理 reasoning traces：trace 是完整保留还是裁剪；答案和 reasoning 的拼接格式；有没有 verifier / filtering 逻辑。

Tiny-Zero/EasyR1：增强教师。

> 未来展望：RL 后的 teacher，能不能直接拿它输出蒸 student？RL-aware distillation。

---

本计划用于在现有 `FinQA / ConvFinQA -> SFT v2 -> DPO -> GRPO` 框架上补一条可控的数据蒸馏链路，用更强 teacher 模型为金融数值推理任务生成更高质量的监督信号。

蒸馏目标不是单纯“扩充更多样本”，而是生成和筛选更稳定的以下能力：

- 结构化回答能力：`问题分析 / 关键证据 / 推理程序 / 最终答案`
- 数值答案正确率
- 程序推理一致性
- 多轮对话压缩后的有效推理能力
- 后续 DPO / GRPO 可用的偏好和 reward 信号

第一阶段只做 `FinQA` 和 `ConvFinQA`，不把 `Fineval`、`FIQA_QA` 等开放类数据混入蒸馏主链路。

### 设计原则

**与训练格式保持一致**：teacher 输出必须和当前训练格式一致，避免蒸馏数据与现有 SFT / DPO / benchmark 分布不一致。

**用raw record蒸馏**：蒸馏输入应直接来自原始数据记录，经统一 processor 处理 prompt；不要从旧版 `*_clean_strict.jsonl` 的答案侧反向改写，否则会把旧 target 的偏差一并继承进去。

**自动校验准则**：蒸馏链路必须把自动验证放在中心位置。只有通过结构、答案、程序、长度等检查的样本，才能进入训练集。

**阶段化蒸馏**：如果蒸馏后的 SFT 数据都不能让模型至少不弱于 `base`，继续堆 DPO / GRPO 只会放大偏差。

**`<think>/<answer>`**：“R1 风格模型和其复现生态里的事实标准”

>rule-based reward 里有 format reward，要求把推理过程放在 <think> 标签里，以提升可解释性和后续分析便利性。Nature/PMC 的公开版本也保留了这段表述：format rewards 鼓励把 reasoning 放进 <think> / </think> 标签中。Hugging Face 的 Open-R1 作为 DeepSeek-R1 的开源复现项目，也直接围绕这一格式工作，README 里甚至专门提醒：一些 distilled DeepSeek 模型的 chat template 会处理 <think> 块，因此会影响 format reward。

- FinQA / ConvFinQA 的 teacher distill + score + SFT/DPO，核心需求是“容易解析、容易打分、容易做 clean/rejected 选择”。<think>/<answer> 比四段式更短、更稳、正则更容易写，也更适合现在的 clean_response 抽取和 quality_score 排序。
- 金融数值推理任务很适合把“过程”和“最终答案”拆开：<answer> 方便做确定性答案核对，<think> 方便做质量筛选、DPO 对比和后续 audit。


## 当前 distill + score 代码逻辑

当前实现已经形成一条完整的三段式链路：`build_financial_distill_dataset.py` 先把原始 `FinQA / ConvFinQA` 样本统一转成蒸馏输入，补齐 `prompt / gold_answer / gold_program / gold_response / record_id / source_dataset` 等字段；`distill_with_teacher.py` 再按题生成多候选 teacher 输出，保存为带 `candidate_index / generation_key / response / raw_response / teacher_*` 元信息的 `distill_candidates.jsonl`；`score_distill_candidates.py` 最后负责从候选中提取 `clean_response`、执行 answer check 与 reasoning selection、计算 `quality_score`，并按组输出 `distill_sft.jsonl / distill_dpo.jsonl / distill_audit.jsonl / distill_summary.jsonl`。这条链路的核心不再是显式维护多层 eligibility，而是先统一打分，再在同题候选中直接选择 `chosen / rejected`。

生成阶段与打分阶段的 response 口径已经分离。`distill_with_teacher.py` 会兼容 `deepseek-reasoner` 的返回形态，优先合并 `reasoning_content + content`，再包装成 `<think>/<answer>` 写入 `response`；但 scoring 阶段优先从 `raw_response.choices[0].message.content` 提取 `clean_response`，只有该内容本身完整满足 `<think>...</think><answer>...</answer>` 时才作为 SFT/DPO 正样本来源。`gold/copy_gold_final` backend 因没有真实 raw message content，允许回退使用 `response`。因此当前 `response` 更接近候选缓存与调试字段，`clean_response` 才是训练消费时的主协议。

组内选择逻辑也已经固定下来。`chosen` 只在 clean 候选里按 `quality_score` 取最高分，SFT assistant 侧和 DPO 的 `response_chosen` 都使用该 `clean_response`；`rejected` 则从同组剩余候选里按 `quality_score` 取最低分，只要求能提取出原始 candidate response，因此可以来自非 clean 候选。如果一组样本没有任何可用 `chosen`，整组会被直接丢弃，不进入 `distill_sft`、`distill_dpo`、`audit` 或 `summary`。这也意味着当前 `distill_audit.jsonl` 不是全量候选归档，而是“成功入组样本的候选审计结果”。

## 参考

参考 DeepSeek-R1 及其复现项目 [open-r1](https://github.com/huggingface/open-r1) 的主要启发有三点。第一，采用 `<think>/<answer>` 作为 reasoning 输出协议，便于 rule-based format reward、答案抽取和后续 audit；第二，沿用“单题多候选生成 -> 统一过滤/打分 -> 再训练”的工程拆分，而不是把生成和训练耦在一起；第三，保留 teacher 生成时的温度、候选索引、原始响应等元信息，方便后续回放、审计和失败重试。当前项目并没有照搬 open-r1 的大规模分布式生成与全量去污流程，而是优先把金融数据处理、候选选择和训练消费口径对齐。

参考 [distill finqa相关论文](fin_ref.md) 与 Fin-R1 等金融 reasoning 工作，当前实现借鉴的重点不是“金融任务一定要使用某个固定模板”，而是“蒸馏要围绕可验证任务能力组织”。具体到代码里，answer check 会优先校验 `<answer>` 与 `gold_answer` 的数值或文本一致性，reasoning selection 会继续记录逻辑一致性、任务相关性、证据质量和程序一致性等信号，最终再汇总为 `quality_score`。这使当前蒸馏链路更偏向任务能力蒸馏，而不是单纯优化回答风格。

> 参考论文：Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning（arXiv:2503.16252）

Fin-R1 论文（Liu et al., 2025/2026, arXiv:2503.16252）
- 数据不是单一来源，而是先按业务能力分层，再统一蒸馏与过滤
- 蒸馏阶段使用强 reasoning teacher 生成 CoT 与答案
- 过滤阶段拆成两步：先做答案检查，再做 reasoning 质量筛选
- 训练阶段采用两步式：先 `SFT` 学会稳定输出 reasoning，再用 `GRPO` 基于可验证 reward 做强化
- 评估不只看单一数值题，而是覆盖多类金融任务，并对数值题使用更稳健的答案判定方式

plan：
- `FinQA / ConvFinQA` 作为第一优先级的 financial advanced reasoning 子集
- `TFNS / FinCUGE / 其他金融知识数据` 可在后续扩展时补入，形成更接近 Fin-R1 的多类金融任务组合
- 当前阶段先把数值推理主链路跑通，再扩展到情感、分类、专业知识与代码数据

**数据集：**  
Fin-R1-Data 共 60,091 条，中英双语，分为四类： financial advanced business knowledge，financial basic business knowledge，financial professional knowledge，financial code。
 - reasoning 相关子集：`FinQA` `ConvFinQA` `TFNS` `FinCUGE`
 - 基础知识与专业知识子集：`FinCorpus` `Ant-Finance` `Finance-500K` `FinanceIQ` `FinPEE` `FinanceQT`

> 启发：构建多源数据源，譬如：按照 reasoning、knowledge、business、code 四类来组织。为了简化模型，第一阶段仍只做 `FinQA + ConvFinQA`。

**数据构造：**
1. `data distillation`
   - 从 raw dataset 抽取问题
   - 用 `DeepSeek-R1-671B` 生成 reasoning path 和 answer
   - 温度设为 `0.6`
   - 数学/计算类题要求用 `\boxed{}` 包最终答案
2. `data filtering`
   - 先做 `answer check`
   - 再做 `reasoning selection`

**answer check：**
- 客观题：直接和参考答案对比
- 主观题：用 `LLM-as-a-Judge`
- judge 模型最终选 `Qwen2.5-72B-Instruct`

**reasoning selection：**
- 用 `Qwen2.5-72B-Instruct` 按七个维度评估 reasoning 质量
- 七个维度包括：
  - internal consistency
  - term overlap rate
  - number of reasoning steps
  - logical coherence
  - content diversity
  - task-domain relevance
  - alignment with task instructions
- 只保留高质量 reasoning 进入 SFT

> 启发：两段式filtering校验（答案校验；reasoning 质量校验）；`domain/financial/distill/score_distill_candidates.py` 下一版应从“单一 quality score”升级为“多维 rubric 打分”

**训练流程：**
1. `SFT`：用高质量 CoT 样本训练模型学会先思考再回答
2. `GRPO`：在 SFT 基础上继续强化，reward = `format reward + accuracy reward`
  - `format reward`
    - 检查输出是否满足 `<think>...</think><answer>...</answer>`
  - `accuracy reward`
    - 用 `Qwen2.5-Max` 判断 `<answer>` 与 ground truth 是否语义一致
- 只做 GRPO 的 `Fin-R1-Zero` 比 base 有提升，但增益有限，且输出容易 incoherent
- 只做 SFT 的 `Fin-R1-SFT` 明显优于 base
- `SFT + GRPO` 的完整两步式最好

**评估流程**

Fin-R1 评估用了五个代表性金融数据集： `FinQA` `ConvFinQA` `Ant-Finance` `TFNS` `Finance-Instruct-500K`
- `Finance-Instruct-500K` 用分层采样得到 10% test subset
- 其他数据集随机采样 1000 条；不足 1000 则全用
- 对数值题不做僵硬 exact match，而是引入 LLM judge 处理数值格式差异、表达差异、有效等价表达

> 启发：后续应补入至少一个非数值金融任务集，避免模型只在表文数值题上过拟合；数值题评估仍应保留“数值归一化 + 必要时 judge 判定”的双层机制。



## PLAN

- 还没有把蒸馏 section 接入 `run_fingpt_min.ipynb`
- `domain/financial/distill/score_distill_candidates.py` 的 program 一致性目前仍以字符串/算子序列为主，后续可继续加强为 AST 级比较
- `LLM-as-a-Judge` 路径已预留接口，但还没有在真实 `DeepSeek-R1 -> judge` 流程上跑完整小样本实验
- `domain/financial/distill/decontaminate_financial_distill.py` 目前还是轻量 n-gram overlap 版本，还不是 open-r1 那种更完整的分布式去污流程

0. 配置`.env`
1. deepseek-r1蒸馏链路
2. llm-as-judge: qwen api

## UPDATE：进度：每次更新在这个版块。

### DeepSeek-R1 teacher

- 已把 `deepseek` provider 接入 `domain/roleplay/llm_client.py`
- 默认模型设为 `deepseek-reasoner`，可作为 `DeepSeek-R1` teacher 使用
- `domain/financial/distill/distill_with_teacher.py` 可直接通过以下参数调用：
  - `--backend openai --provider deepseek`
  - 或设置 `DEEPSEEK_API_KEY` 后省略 `--provider`，由 client 自动检测
- 当前建议先用 `FinQA` 小样本验证 `DeepSeek-R1` 的结构稳定性和数值正确率，再扩展到 `ConvFinQA`

### Open-R1 distillation 迁移

参考 `open-r1` 项目的 README distillation 部分，当前蒸馏链路已补入三项直接迁移：

- 默认 `num_candidates=4`
  - 对齐 open-r1 里单题多候选生成的思路，便于后续做 chosen/rejected 选择
- 默认 `temperature=0.6`
  - 对齐 open-r1 的 teacher 生成温度设定，优先获得较丰富的 reasoning 候选
- 新增 `domain/financial/distill/prompts/financial_distill_teacher_user.txt`
  - 对齐 open-r1 的模板化 user prompt 思路，把 teacher 输入包装成统一的 distillation prompt
  - teacher 输出现在要求使用 `<think>` / `<answer>` 标签

当前与 open-r1 的主要差异是：

- open-r1 使用面向通用推理的 distillation 方案，本项目改成金融结构化格式输出
- open-r1 更强调大规模分布式生成，本项目当前先优先保证金融数据处理、过滤和训练口径一致
- open-r1 的蒸馏输出偏 `<think>/<answer>`，本项目当前也已经收敛到 `<think>/<answer>` 口径；旧的四段式结构不再作为 distill 阶段的训练目标。

### DeepSeek-R1 smoke + notebook 接入

- `domain/roleplay/llm_client.py` 已补标准库 `.env` 加载与 `OpenAI-compatible HTTP fallback`，即使环境里没有 `openai` Python 包，也可以直接使用 `.env` 中的 `DEEPSEEK_API_KEY` 调用 `deepseek-reasoner`。
- `domain/financial/distill/distill_with_teacher.py` 已兼容 `deepseek-reasoner` 的返回形态：生成阶段会优先合并 `reasoning_content + content`，再包装成带 `<think>/<answer>` 的 `response`。但 scoring 阶段不会直接把这个合并结果当成训练目标，而是优先回到 `raw_response.choices[0].message.content` 提取 `clean_response` 与 candidate response；因此当前 `response` 更接近兜底与调试字段，而不是 SFT/DPO 的首选来源。
- 实际 smoke test 已跑通：
  - 命令：`python -m domain.financial.distill.run_financial_distill_pipeline --source_spec finqa=/root/autodl-tmp/data/financial_reasoning/raw/finqa/dev.json --work_dir /root/autodl-tmp/outputs/financial_reasoning/distill_smoke_r1_v2 --teacher_backend openai --teacher_provider deepseek --teacher_model deepseek-reasoner --teacher_num_candidates 1 --teacher_temperature_schedule 0.6 --teacher_max_tokens 1024 --max_samples_per_family 2 --summary_csv`
  - 结果：`input_rows=2`、`answer_check_pass_rows=2`、`reasoning_selection_pass_rows=2`、`sft_rows=2`、`dpo_rows=0`
  - 说明：`DeepSeek-R1` 在 `max_tokens=1024` 下可稳定产出可通过两阶段过滤的 `FinQA` 小样本蒸馏结果。
- `run_fingpt_min.ipynb` 已新增 `Distill` section：
  - `0. Smoke Test: DeepSeek-R1`：真实调用 API，展示 `distill_summary / candidate preview / audit preview`
  - `1. FinQA Distill-R1`：生成正式 `finqa-distill-r1` 命令与输出路径，默认不自动执行全量 API 任务，避免误触发大规模 token 消耗。
- 当前 notebook 中建议的 `DeepSeek-R1` 默认参数：
  - `teacher_num_candidates=1` for smoke
  - `teacher_num_candidates=4` for finqa-distill-r1
  - `temperature=0.6`
  - `teacher_max_tokens=1024`


### 收缩到 `<think>/<answer>` 基线

- 第一阶段目标已明确收缩为：
  - `<think>` 承载推理过程
  - `<answer>` 只承载最终答案本身，并直接对标 `gold_answer`
- `问题分析 / 关键证据 / 推理程序 / 最终答案` 不再作为当前蒸馏阶段的硬模板；后续若要加入，只作为增强项。
- 已修改：
  - `domain/financial/distill/prompts/financial_distill_teacher_user.txt`：不再要求四段结构，明确 `<answer>` 只输出答案本身。
  - `domain/financial/distill/build_financial_distill_dataset.py`：从 distill prompt 中移除了旧的“请按以下结构作答”尾部，避免复用旧 SFT 模板。
  - `domain/financial/distill/distill_with_teacher.py`：`gold_response -> <think>`，`gold_answer -> <answer>`；同时兼容未闭合 `<answer>` 标签。
  - `domain/financial/distill/score_distill_candidates.py`：`answer_check` 和 `reasoning_selection` 都已切换到轻量 `<think>/<answer>` 口径，不再依赖四段结构锚点。
- 最新 smoke test：
  - 命令：`python -m domain.financial.distill.run_financial_distill_pipeline --source_spec finqa=/root/autodl-tmp/data/financial_reasoning/raw/finqa/dev.json --work_dir /root/autodl-tmp/outputs/financial_reasoning/distill_smoke_r1_v6 --teacher_backend openai --teacher_provider deepseek --teacher_model deepseek-reasoner --teacher_num_candidates 1 --teacher_temperature_schedule 0.6 --teacher_max_tokens 1024 --max_samples_per_family 2 --summary_csv`
  - 结果：`input_rows=2`、`answer_check_pass_rows=2`、`reasoning_selection_pass_rows=2`、`sft_rows=2`、`dpo_rows=0`
  - 结论：当前基线已经实现“`<think>/<answer>` 范式 + `<answer>` accuracy”。


### answer-conditioned / program-conditioned 口径拆分

**answer-conditioned rationale distillation**：“在正确答案约束下，如何组织更好的思维链”
- 让它重点学习如何围绕正确答案组织思维链”。
- 强标注监督：gold answer、program、supporting facts
- teacher 的角色从“解题者”改成“解释器”。也就是先把正确答案、关键证据、甚至程序都给它，再让它生成一条自然、清晰、适合 student 学习的推理链。这样 teacher 不需要自己探索，就不会频繁输出“不会做”；它的任务变成 把已有正确解组织成高质量 reasoning text。

**program-conditioned distillation**
 - 让teacher基于核心推理骨架（golden_program）让它生成“人类可读的步骤解释”。这样生成出的 reasoning chain 会更稳定，也更贴近真正想让模型学到的“表格字段定位 → 单位换算 → 数值比较”能力。
 - 缺点：如果把 gold_response 全量暴露给 teacher，蒸馏会更像“改写”而不是“推理增强”。这更适合蒸 SFT，不一定能直接蒸出高质量 DPO，因为候选之间可能差异不够大。

  所以更实用的方案是两阶段：
  1. gold-guided rewrite 先产出高质量 SFT distill
  2. 再在这个基础上做少量扰动或多候选生成，构造 DPO


- `answer-conditioned rationale distillation`
  - 只围绕 `gold_answer` 组织 reasoning
  - 不再向 teacher 暴露 `gold_supporting_facts`、`gold_response`、`gold_program`
- `program-conditioned distillation` 保持更严格：
  - 基于 `gold_response` 做 reasoning rewrite
  - 目标是得到更清晰、更适合 student 学习的推理链


> 小样本观察结果`finqa_distill_compare_8`：
> 1. 小样本下，program-conditioned 明显比 answer-conditioned 更稳。原因也很直接：只给 gold_answer 时，teacher 仍然会自己脑补解题路径，容易跑偏；给 gold_response 后，它更像 rewrite 任务，稳定得多。
> 2. 大部分样本没有回答完就强行截断并加上了`</answer>`，这是不合理的：应该放宽max_tokens，其次应该思考思维链是否过长，最后直接强行加上`</answer>`作为结束是否不合适。经过smoke测试4样本，发现1024长度下不够稳定，2048足够盈余，tradeoff：1536.


program-conditioned prompt 收缩 + smoke 配置调整
- `program-conditioned` 模板已收缩，重点去掉会被 teacher 直接复述的元说明：
  - 不再强调“你是教师模型”“你的目标是……”
  - 明确要求 `<think>` 只写推理本身
- smoke 默认配置调整为：
  - `max_samples=4`
  - `teacher_max_tokens=2048`
- 目的：先判断截断是否主要由 `prompt 过重 + max_tokens 偏紧` 共同导致。

1024-token smoke 对照
- 为了验证截断是否主要由 `max_tokens` 驱动，smoke 配置已回切到：
  - `max_samples=4`
  - `teacher_max_tokens=1024`
- 这轮将与 `2048-token` 的 `program_conditioned_smoke_4_v2` 做直接对照。

1024 vs 1536 vs 2048
  - 1536 比 1024 明显更稳，截断从 2/4 降到 1/4
  - 但 1536 还没达到 2048 的完整性
  - 三组在这 4 条上的评分通过率都一样，差异主要体现在“是否自然完整结束”
  
**sum: 当前推荐蒸馏模式**
- 当前默认推荐路线已收敛为：`program-conditioned + 2048`
- 原因：
  - `program-conditioned` 在小样本对比中明显优于 `answer-conditioned`
  - `2048` 在 `4` 条 smoke 中实现了 `finish_reason={'stop': 4}`，完整性优于 `1024` 和 `1536`
- notebook 已同步切换到这一路线：
  - smoke 默认使用 `program_conditioned_distill_user.txt`
  - `teacher_max_tokens=2048`
  - 正式 `finqa-distill-r1` 默认也使用同一模板

### Candidates 策略更新（执行优先级）
0. 强制 clean 过滤：所有 candidates 先过 `<think>/<answer>` 完整性校验；不 clean 的候选直接剔除，不进入 SFT/DPO。DPO 只保留 clean vs clean 的 pair，避免把格式不完整当成偏好信号。
1. 默认 candidates 数量调整为 3：API 只生成 3 个候选，减少无收益的近似重复。
2. 人工构造 rejected（TODO 多样化）：
   a) 轻微但明确错误的 hard negative（数值±1%、算子替换、引用行错误）。
   b) 缺失 `<think></think>` 的样本（思维链缺失）。
   c) `<think>` 内直接输出 program 片段而非自然语言（非自然语言思维链）。

当前执行流程（2026-04-10）
1. distill-r1 生成 candidates（已完成）：`checkpoints/candidates/part_*.jsonl`。
2. checkpoints：score 生成 SFT/audit/summary（clean-only 选 chosen）。
3. checkpoints：基于 audit 生成 rejected，并与 chosen 组合产出 DPO，最终 merge 为 `distill_dpo.jsonl`。


人工 rejected 生成规划（执行顺序）
1. 基于 chosen 生成数值扰动：对 `<answer>` 数值做 ±1% / ±0.5% / 四舍五入误差；保持 `<think>/<answer>` 完整。
2. 程序算子扰动：从 `program` 中随机挑一处算子做轻微替换（add↔subtract、multiply↔divide），并同步更新 `<think>` 的描述。
3. 证据引用扰动：在 `<think>` 中把一条关键数值换成同表相邻行/列数值，形成“看似合理但错误”的推理。
4. 缺失思维链版本：删除 `<think>` 或清空 `<think>` 内容，仅保留 `<answer>`（保持标签闭合）。
5. 非自然语言思维链：把 `<think>` 内容替换为原始 program 字符串或类 program 片段。

实施细节
- 只在 chosen 为 clean 时生成；生成后的 rejected 也必须通过 `<think>/<answer>` 完整性校验（4/5 需要保留空 `<think>` 或 program 仍闭合）。
- 每条 chosen 最多生成 1-2 条 rejected，避免过多合成噪声。
- 生成后重新走 scoring，确保 `answer_check_pass=False` 且 `clean_response` 存在。



### Scoring

**`clean_response`：**  

`domain/financial/distill/score_distill_candidates.py` 目前把 `clean_response` 定义为“可直接用于训练的完整 `<think>...</think><answer>...</answer>` 响应”。实现上会先读取 `raw_response.choices[0].message.content`；只有当这段原始 content 本身完整满足标签格式时，才记为 `clean_response`。若 `teacher_backend` 是 `gold` 或 `copy_gold_final`，则允许回退检查 `response`，但同样要求它完整满足 `<think>/<answer>` 格式；除此之外，普通候选的 `response` 不会被直接记成 clean。

若一组样本没有任何 `clean_response`，整组会被直接丢弃，不写入 `distill_sft.jsonl`、`distill_dpo.jsonl`、`distill_audit.jsonl` 或 `distill_summary.jsonl`。因此现在的 `distill_audit.jsonl` 和 `distill_summary.jsonl` 都只覆盖“成功入组”的样本组，而不是全量候选日志。summary 里会继续记录 `num_clean_response`、`chosen_clean_response_source`、`chosen_quality_score`、`rejected_quality_score` 等字段；终端打印的总览还会额外给出 `dropped_records` 与 `dropped_all_invalid_response_records`，用于解释组级丢弃情况。

**score:**

当前 answer check 完全基于 `clean_response` 进行。代码会从中抽取 `<answer>`、最终答案、程序段和 `<think>`，并计算 `format_ok`、`answer_correct`、`program_consistent`、`operator_sequence_match`、`evidence_json_like_hits` 等字段：
 - `answer_check_pass` 的条件是：`format_ok` 为真、答案和 `gold_answer` 对齐、且 evidence 区域没有命中 JSON-like 片段。
 - `program_consistent` 和 `operator_sequence_match` 目前只作为审计字段保留，不参与硬过滤；
 - CLI 里虽然保留了 `--require_program_match_for_positive` 参数，但当前这版实现没有把它接入实际判定路径。

reasoning selection 默认只在 `answer_check_pass=True` 时运行；若显式打开 `--run_reasoning_selection_on_failed_answer_check`，才会对 answer check 失败的样本继续做 reasoning 打分。

启用 judge 时，LLM judge 看到的候选文本是 `clean_response`，否则回退到 `response`；不启用 judge 时，heuristic 规则同样优先分析 `clean_response`，否则分析 `response`。因此当前 scoring 代码并没有完全把“训练目标 clean”与“打分时可参考的候选原文”彻底分离，后者仍然会在 clean 缺失时回退到 `response`。



**Checkpoint scoring 更新（2026-04-10）**  
为了避免`distill_candidates.jsonl`总文件过大导致进程被kill，按照checkpoint切片对candidates进行打分。
- 新增 `domain/financial/distill/score_distill_checkpoints.py`：
  - 直接消费 `work_dir/checkpoints/candidates/part_*.jsonl`
  - 分片 scoring 结果写到 `work_dir/checkpoints/scored/`
  - 目录结构：`audit/`、`sft/`、`dpo/`、`summary/`、`summary_csv/`
  - 支持 `--resume` 跳过已完成分片
  - 完成后自动 merge 回 `work_dir/distill_audit.jsonl`、`distill_sft.jsonl`、`distill_dpo.jsonl`、`distill_summary.jsonl`
  - 输出 `checkpoints/scored/score_manifest.json`
- `domain/financial/distill/score_distill_candidates.py` 已修复：
  - 只有开启 `--enable_reasoning_judge` 时才读取 `judge_prompt_file`
  - 避免默认 scoring 因相对路径缺失直接报 `FileNotFoundError`




### DPO: chosen/rejected

组内选择逻辑是“先统一打分、再选 chosen/rejected”。`chosen` 只从带 `clean_response` 的候选里按 `quality_score` 取最高分，所以 `distill_sft.jsonl` 的 assistant 输出以及 `distill_dpo.jsonl` 的 `response_chosen` 都始终使用 `clean_response`。`rejected` 现在只从同组 **clean** 候选里取最低分，因此 DPO 仅包含 clean vs clean 的 pair，避免把格式缺失当成偏好信号。



