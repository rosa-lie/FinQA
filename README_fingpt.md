# 任务：金融对话数值推理模型：SFT → 轻量 DPO/GRPO → benchmark 评估
> 金融大模型真正难的，不是“知道一个金融术语”，而是“在金融场景里正确推理”。
> **面向金融对话数值推理的 reasoning model：以 ConvFinQA 和 FinQA 为核心，辅以中文金融考试推理数据，训练一个能够进行多轮、数值、表文混合推理的金融小模型。**
> 在数据侧，我以 ConvFinQA 和 FinQA 为主干，分别覆盖多轮对话推理与表文混合数值推理，再用 fingpt-fineval 和少量 fingpt-fiqa_qa 做中文金融知识和基础问答补强；在训练侧，我先通过 SFT 让模型学会金融推理，再基于答案正确性、程序一致性和结构约束设计可验证 reward 做轻量 GRPO；在评估侧，我用 FinQA/ConvFinQA 检验核心推理能力，用 CFLUE 检查中文泛化，用 FinanceBench 检查开放书金融 QA 迁移能力。这样形成了一个从训练到评估都围绕“金融 reasoning”展开的闭环。

# Done: 数据集

> 我将训练数据划分为“金融语言理解”与“金融推理”两类，其中 FinQA 与 ConvFinQA 构成推理主干。
> FinQA 提供表文混合、带结构化程序监督的金融数值推理样本，用于训练模型的多步计算与证据整合能力；ConvFinQA 则进一步将金融推理任务扩展到多轮对话场景，用于训练模型在连续交互中维持上下文并完成 follow-up reasoning。两者结合，使模型从“会回答金融问题”提升为“能在金融语境中持续推理”的 reasoning model。
> **FinQA 让模型学会“怎么做金融推理”，ConvFinQA 让模型学会“怎么在对话中持续做金融推理”。**


| 阶段       | 数据                            | 目标            | 说明               |
| -------- | ----------------------------- | ------------- | ---------------- |
| SFT-1    | FinQA                        | 学会表文混合数值推理    | 第一阶段            |
| SFT-2    | ConvFinQA                    | 学会多轮 follow-up 推理 | 第二阶段            |
| Joint SFT | FinQA + ConvFinQA            | 作为对照基线            | 需在 SFT v2 数据上重跑 |
| DPO（可选）  | 从 SFT 样本自动构造                  | 优化表达质量、结构、少废话 | 小规模就够            |
| GRPO（推荐） | 基于可验证 reward                  | 优化答案正确性和推理格式  | 更贴合 reason model |

## 数据来源

### sft 参考数据集

https://huggingface.co/datasets/AdaptLLM/ConvFinQA/ “对话 + 数值推理”

https://finqasite.github.io/ https://github.com/czyssrs/FinQA
- **核心特点**：
    - **专家标注**：11名美国金融专家标注，时薪$20-50
    - **结构化推理**：每个问题附带推理程序（operation步骤）
    - **多模态输入**：62.43%仅需表格，23.42%仅需文本，14.15%需两者结合
- **数据规模**：8,281样本（训练6,251/验证883/测试1,147）
- **推理复杂度**：59.1%单步推理，32.71%两步，8.19%三步+
- **最佳用途**：评估模型**数值推理能力**、**表格理解能力**


### benchmark 参考

选择

1. FinQA：测 表文混合 + 多步数值推理
2. ConvFinQA：测 多轮上下文 + follow-up 金融推理
3. CFLUE：推荐做，但只选和任务最相关的子任务，不要全做。
     - 它的作用不是测“金融数值推理”，而是测：中文金融语义理解；中文金融术语/文本泛化；模型是否只会英文表格推理，而不会中文金融表达。
     - 所以它应该作为：“中文金融泛化补充 benchmark”

参考

https://www.modelscope.cn/datasets/tongyi_dianjin/CFLUE
阿里云-通义点金与苏州大学联合推出了CFLUE（Chinese Financial Language Understanding Evaluation），这是一个新颖的、全面的评估基准，旨在评估大型语言模型在中文金融语境中的理解和处理能力。

CFLUE通过两个主要维度——知识评估和应用评估来衡量语言模型的性能。

知识评估部分包含超过38,000个多项选择题，这些题目选自15种不同的金融资格模拟考试，旨在测试语言模型的答案预测和推理能力。每个问题都伴随有解释，有助于深入评价模型的推理过程。
应用评估部分则提供超过16,000个实例，覆盖文本分类、机器翻译、关系抽取、阅读理解和文本生成等五种经典NLP任务，这些实例源自现有共享任务或由专业人员标注的真实数据。
整体而言，CFLUE为了解和提升中文金融领域LLMs的能力提供了多角度的见解，并通过CFLUE呼吁对这些模型的能力进行更全面细致的评估。研究团队期望，CFLUE不仅能促进对现有模型的深入了解，还能推动中文金融领域语言模型发展的新步伐。

目前，CFLUE V1.0 的评估数据集将向公众提供，未来计划不断更新版本并推出集成的平台化评估服务，旨在为整个行业提供全面的一站式评价解决方案。

https://huggingface.co/datasets/PatronusAI/financebench
- 由PatronusAI开发，专注于开放式金融问答评估
- 特点：结合长文档理解（RAG场景），测试模型从金融报告中**提取和推理能力**
https://finqasite.github.io/ https://github.com/czyssrs/FinQA
- **核心特点**：
    - **专家标注**：11名美国金融专家标注，时薪$20-50
    - **结构化推理**：每个问题附带推理程序（operation步骤）
    - **多模态输入**：62.43%仅需表格，23.42%仅需文本，14.15%需两者结合
- **数据规模**：8,281样本（训练6,251/验证883/测试1,147）
- **推理复杂度**：59.1%单步推理，32.71%两步，8.19%三步+
- **最佳用途**：评估模型**数值推理能力**、**表格理解能力**

https://huggingface.co/datasets/AdaptLLM/finance-tasks
- This repo contains the **evaluation datasets** for our paper [Adapting Large Language Models via Reading Comprehension](https://huggingface.co/papers/2309.09530).

## Done：数据处理

数据处理文档见：[docs/fin_datasets.md](docs/fin_datasets.md)

对应 `run_fingpt_min.ipynb` 第 2~5 节，完整流程为：

1. 原始数据就绪（本地缓存/HF）
2. `financial_data_router.py` 统一转为 SFT/DPO 格式
3. `clean_sharegpt_dataset.py` 基础清洗  
4. `audit_sharegpt_dirty_samples.py` 质量审计
5. `filter_sharegpt_by_audit.py --mode strict` 严格过滤并落盘训练目录

### 数据处理结果（Notebook 现有运行记录）

SFT1（FinQA）：
- raw: `6251`
- clean 后: `4683`（**剔除过长轮次 `1568`**）
- strict 后: `3954`（审计标记 `729`，全部剔除）

SFT2（ConvFinQA turn）：
- raw: `2096`
- clean 后: `1407`（**剔除过长轮次 `689`**）
- strict 后: `1015`（审计标记 `392`，全部剔除）

清洗阈值（与 notebook 一致）：
- 对话轮次：`2~16`
- 总字符：`<=6000`
- 单轮字符：`<=2500`

# Done: SFT

## 遇到问题：loss spike

`checkpoint-600` loss 正常下降；`checkpoint-800`出现 spike（约 `0.4 -> 4.0`）。

归因：脏样本与冲突标注导致梯度异常波动。

解决：启用 `clean -> audit -> strict filter` 三段清洗，训练集统一使用 `*_clean_strict.jsonl`。

补充：SFT v2 还新增了数据审计要求，在训练前检查：
- `json_like_evidence_ratio`
- `structured_answer_ratio`
- `avg_prompt_chars` / `avg_answer_chars`
- 样本 preview 中是否仍残留原始 JSON 证据块

## SFT v2 设计变更

当前不再直接沿用旧版 joint training 结论。最新 benchmark 显示旧版 `sft_merged` 弱于 `base`，说明训练 loss 更低不等于推理效果更好。

SFT v2 的核心变更是：
- `关键证据` 不再直接监督原始 `gold_ind/gold_inds` JSON
- `FinQA` / `ConvFinQA` 的 evidence 改为自然语言摘要 bullet
- `ConvFinQA` 历史问题先压缩，再进入 prompt
- 修改 processor 后，必须重新生成 `*_clean_strict.jsonl` 再启动训练

# Done：DPO（轻量）

对应 `run_fingpt_min.ipynb` 第 8 节：

- DPO 数据构造：按来源占比采样，总预算 `DPO_TOTAL_BUDGET=4000`，单数据集上限 `MAX_DPO_PER_DATASET=2000`
- 训练参数：`learning_rate=5e-7`，`max_steps=200`，`batch_size=1`，`grad_accum=16`
- LoRA：`rank=8, alpha=16, dropout=0.05`

训练日志（notebook 记录）：
- `train_loss=0.3516`
- `train_samples=3840`

质量控制：
- 训练前会统计规模、去重、结构锚点覆盖（如“最终答案：/推理程序：”）
- notebook 中已加入对异常 `jsonl`（字面量 `\n`）的兜底解析逻辑，避免误判空数据

# fix：评估

**如何判断 SFT + DPO 是否有效。**
- 最重要的不是“模型看起来更会说了没有”，而是任务能力有没有真正提升。
- 现在的核心任务不是泛金融聊天，而是金融数值推理，所以最关键的指标应该始终围绕“答得对不对”。
    - 例如在 FinQA / ConvFinQA 的验证集上，你可以统计数值答案正确率、关键字段是否抽取正确、是否能在多轮对话中保持上下文一致。
    - 如果 SFT 后这些指标有提升，说明模型至少学会了任务；
    - 如果在此基础上做 DPO 后，这些指标继续提升，那 DPO 就是有效的；
    - 但如果 DPO 之后模型只是回答得更长、更像人，却没有让正确率继续变高，甚至反而下降，那就说明 DPO 主要改善了“输出表现”，并没有真正增强核心能力。
 - 换句话说，DPO 是否成功，不能只看回答是否更顺眼，而要看它有没有在不伤害任务正确率的前提下，让输出更稳定、更少幻觉、更符合你想要的结构化风格。

双维度评估：一方面看“任务结果”，另一方面看“输出质量”。
- 前者包括正确率、数值误差、对话连续性；后者包括回答是否更少跑题、是否更少胡编、是否更稳定地按照“步骤 + 结论”或者“推理 + 最终答案”的形式输出。

# fix：GRPO（reward 设计）

> SFT 让模型“会做”，DPO 让模型“做得更像一个专业助手”，GRPO 让模型“更容易做对”。

**如何判断 SFT + DPO 之后还要不要继续做 GRPO。**

本质上是在判断：问题到底还停留在“表达和行为层面”，还是已经进入了“能力和正确率层面”。
- 如果你做完 SFT + DPO 后，发现模型已经很像一个金融助手了，回答也更规整、更像样，但在真正的数值推理题上还是经常算错、漏条件、找错表格字段、或者多轮追问时前后不一致，那么这就说明：模型已经学会“怎么回答”，但还没有学会“怎么稳定地答对”。在这种情况下，继续做 DPO 的收益通常就开始下降了，因为 DPO 更擅长优化“偏好”和“表现”，不擅长解决“为什么它还是会算错”这种问题。也就是说，当你发现模型“会说了，但还不会稳定做对”，这就是非常明确的信号：该考虑 GRPO 了。
- 如果你做完 SFT + DPO 之后，模型在验证集上的正确率已经足够高，输出也足够稳定，数值题和 follow-up 问题都能比较稳地处理，那就不一定非要上 GRPO。
    - 因为 GRPO 不是“所有项目都必须补上的最后一步”，它只有在模型还存在明显“能力上限”问题时才特别有价值。很多项目其实做到 SFT + 轻量 DPO 就已经够用了，尤其如果你的目标是做一个能展示方法论和工程闭环的项目，而不是拼 benchmark 极限成绩。

> 但你这个项目的特殊之处在于：它是一个可验证的数值推理任务。这意味着一旦你发现“正确率”而不是“风格”成为主要瓶颈，GRPO 的价值就会非常高，因为它恰好最适合解决这种“有明确对错标准”的问题。

**GRPO 到底有什么用：不是让模型“更像人”，而是让模型为了答对而优化。**
- SFT 本质上是在做模仿学习：给模型看很多“输入—输出”对，让它学会“看到这种问题时应该长什么样地回答”。
- DPO 则是在模仿学习之上进一步引入偏好信号，让模型学会“哪种回答更好、更像人、更符合期望”。但无论是 SFT 还是 DPO，它们本质上都还停留在“学会模仿好的答案”这个层面。
- GRPO 的不同在于，它不是在教模型“像谁”，而是在教模型“为了得高分应该怎么做”。如果你的 reward 是“答案正确 + 推理结构完整 + 格式符合要求”，那模型就不再只是学会背出训练集里的模式，而是会逐渐朝着更容易答对的方向去调整自己的策略。

> 这也是为什么 GRPO 对你这个项目特别合适。因为你做的是金融数值推理，这类任务最大的优点就是：reward 很容易设计，而且可以自动验证。

---

1. 为什么做 GRPO
    - DPO更偏“偏好排序/表达风格”，GRPO更适合“可验证目标优化”（答案正确、程序一致、格式约束）。
2. 何时做 GRPO
    - 当你已经**有不错的 SFT 基线**，但「答案正确率/程序一致性/格式约束」仍可提升；
    - 且你能构建**可验证 reward**（格式、程序、答案）。
3. 如何做 GRPO
    - SFT -> DPO -> GRPO 只在 DPO 数据修复并验证收益稳定后再尝试。
4. 如何对比有无 GRPO：
    - 固定同一 base checkpoint，做 A/B：
        - A: baseline（SFT 或 SFT+DPO）
        - B: baseline + GRPO
    - 保持同数据切分、解码参数、评测脚本一致。
    - 至少 3 seeds，报告均值/标准差，并做显著性检验。