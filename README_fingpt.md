# 任务：金融对话数值推理模型：SFT → 轻量 DPO/GRPO → benchmark 评估
> 金融大模型真正难的，不是“知道一个金融术语”，而是“在金融场景里正确推理”。
> **面向金融对话数值推理的 reasoning model：以 ConvFinQA 和 FinQA 为核心，辅以中文金融考试推理数据，训练一个能够进行多轮、数值、表文混合推理的金融小模型。**
> 在数据侧，我以 ConvFinQA 和 FinQA 为主干，分别覆盖多轮对话推理与表文混合数值推理，再用 fingpt-fineval 和少量 fingpt-fiqa_qa 做中文金融知识和基础问答补强；在训练侧，我先通过 SFT 让模型学会金融推理，再基于答案正确性、程序一致性和结构约束设计可验证 reward 做轻量 GRPO；在评估侧，我用 FinQA/ConvFinQA 检验核心推理能力，用 CFLUE 检查中文泛化，用 FinanceBench 检查开放书金融 QA 迁移能力。这样形成了一个从训练到评估都围绕“金融 reasoning”展开的闭环。
>

# Done: 数据集

> 我将训练数据划分为“金融语言理解”与“金融推理”两类，其中 FinQA 与 ConvFinQA 构成推理主干。
> FinQA 提供表文混合、带结构化程序监督的金融数值推理样本，用于训练模型的多步计算与证据整合能力；ConvFinQA 则进一步将金融推理任务扩展到多轮对话场景，用于训练模型在连续交互中维持上下文并完成 follow-up reasoning。两者结合，使模型从“会回答金融问题”提升为“能在金融语境中持续推理”的 reasoning model。
> **FinQA 让模型学会“怎么做金融推理”，ConvFinQA 让模型学会“怎么在对话中持续做金融推理”。**

```
┌─────────────────────────────────────────────────────────────┐
│                    数据层 (Data Layer)                       │
├─────────────────────────────────────────────────────────────┤
│  主干数据              │  补充数据              │  评估数据          │
│  • ConvFinQA (多轮对话) │  • fingpt-fineval     │  • FinQA/ConvFinQA │
│  • FinQA (单轮数值推理) │  • 少量fiqa_qa        │  • CFLUE (中文)    │
│                       │                      │  • FinanceBench    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    训练层 (Training Layer)                  │
├─────────────────────────────────────────────────────────────┤
│  SFT-1 → SFT-2 → [DPO] → GRPO                              │
│  (主干推理) (中文补强) (可选) (推荐)                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    评估层 (Evaluation Layer)                │
├─────────────────────────────────────────────────────────────┤
│  核心推理能力 → 中文泛化 → 开放域迁移                         │
│  (FinQA/ConvFinQA) (CFLUE) (FinanceBench)                   │
└─────────────────────────────────────────────────────────────┘
```

| 阶段       | 数据                            | 目标            | 说明               |
| -------- | ----------------------------- | ------------- | ---------------- |
| SFT-1    | ConvFinQA + FinQA             | 学会金融对话数值推理    | 主训练阶段            |
| SFT-2    | + fingpt-fineval + 少量 fiqa_qa | 中文知识/稳定性补强    | 补金融知识和中文表达       |
| DPO（可选）  | 从 SFT 样本自动构造                  | 优化表达质量、结构、少废话 | 小规模就够            |
| GRPO（推荐） | 基于可验证 reward                  | 优化答案正确性和推理格式  | 更贴合 reason model |

## 数据来源

### sft1 参考数据集

https://huggingface.co/datasets/AdaptLLM/ConvFinQA/ “对话 + 数值推理”

https://finqasite.github.io/ https://github.com/czyssrs/FinQA
- **核心特点**：
    - **专家标注**：11名美国金融专家标注，时薪$20-50
    - **结构化推理**：每个问题附带推理程序（operation步骤）
    - **多模态输入**：62.43%仅需表格，23.42%仅需文本，14.15%需两者结合
- **数据规模**：8,281样本（训练6,251/验证883/测试1,147）
- **推理复杂度**：59.1%单步推理，32.71%两步，8.19%三步+
- **最佳用途**：评估模型**数值推理能力**、**表格理解能力**

### sft2 参考数据集

https://huggingface.co/datasets/FinGPT/fingpt-fineval
- **数据来源**：中国金融从业资格考试（证券/基金/银行等）[^(65)^](https://arxiv.org/html/2308.09975v2)
- **数据格式**：选择题（单选/多选/判断）+ 解析
- **核心特点**：
    - 涵盖金融学术知识、行业知识、安全知识、金融Agent四大类
    - 共8,351题，其中训练集1,060题可用于SFT
    - 每题附带详细解析，可用于CoT训练
- **最佳用途**：中文金融知识注入、资格考试辅导场景

https://huggingface.co/datasets/FinGPT/fingpt-fiqa_qa
- **数据来源**：FiQA 2018挑战赛（WWW'18）[^(46)^](https://sites.google.com/view/fiqa/)
- **内容特点**：基于金融论坛、新闻和社交媒体的观点型问答
- **优势**：真实用户问答场景，适合训练对话式金融助手
- **局限**：无CoT推理过程，数值推理能力有限


### benchmark 参考

选择

FinQA：测 表文混合 + 多步数值推理

ConvFinQA：测 多轮上下文 + follow-up 金融推理

CFLUE：推荐做，但只选和你任务最相关的子任务，不要全做。

它的作用不是测“金融数值推理”，而是测：中文金融语义理解；中文金融术语/文本泛化；模型是否只会英文表格推理，而不会中文金融表达。

所以它应该作为：“中文金融泛化补充 benchmark”

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

## 数据处理

### `clean_sharegpt_dataset.py`

### `audit_sharegpt_dirty_samples.py` + `filter_sharegpt_audit.py`

### 数据处理结果

1. SFT1 核心问题（占比低，影响小）
主要缺陷：contains_const_placeholder（常量占位符，1854 条），属于模板化无效内容，直接过滤即可
脏数据率：1925/8694=22.1%，过滤后保留率 77.9%，质量优秀
2. SFT2 核心问题（严重缺陷）
致命问题：short_answer（短回答）占比 100%（1061/1061），脏数据率高达 56.8%
原因：SFT2 的主体 / 回放样本回答过于简短，无法满足金融推理任务的学习需求
结果：原始 2045 条样本，最终仅保留 807 条，保留率 39.5%

✅ 总可用训练样本：7576 条
 - SFT1 严格清洗后：6769 条（核心主力）
 - SFT2 严格清洗后：807 条（补充样本）

# Done：SFT

## SFT1

### 遇到问题：loss spike

`checkpoint-600`loss正常下降；`checkpoint-800`内出现了loss spike（0.4->4.0）

问题归因于数据集噪声。

### 解决方案：严格筛选数据集`audit_sharegpt_dirty_samples.py` + `filter_sharegpt_audit.py`

处理后的数据集；
基础清洗：clean_sharegpt_dataset.py → 过滤对话轮次（2~16 轮）、总字符（≤6000）、单轮字符（≤2500）
质量审计：audit_sharegpt_dirty_samples.py → 检测低质量样本（短回答、占位符、冲突标签等）
严格过滤：filter_sharegpt_by_audit.py → 剔除所有审计标记的脏样本v

## SFT2

1. 保留了sft1旧数据。

> 什么时候可以保留一部分旧数据：
> - 为了防遗忘（catastrophic forgetting）时，保留少量 replay 是合理的。
> - 但通常是“小比例混入”，不是占多数。

2. fingpt数据集

# 存在的问题：SFT效果

# TODO：评估

# TODO：DPO


# TODO：GRPO
