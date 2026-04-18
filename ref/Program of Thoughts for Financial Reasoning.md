# Program of Thoughts for Financial Reasoning 调研：FINDER 框架

## 1. 文献摘要

论文：`Khatuya 等 - 2025 - Program of thoughts for financial reasoning leveraging dynamic In-context examples and generative r.pdf`

该文提出 FINDER，一个面向金融数值推理的两阶段框架：

1. 使用 generative retriever 从非结构化文本和表格中抽取相关事实。
2. 使用 context-aware Program of Thought prompting，并动态选择 in-context examples。

论文发表于 EMNLP 2025，ACL Anthology 页面为 https://aclanthology.org/2025.emnlp-main.1577/。公开摘要报告 FINDER 在 FinQA 和 ConvFinQA 上达到新的 SOTA，相比已有方法 execution accuracy 分别提升 5.98% 和 4.05%。

这篇和 Chen et al. 2023 PoT 的区别是：Chen 重点证明 PoT prompting 有效，而 FINDER 重点解决金融场景中“证据检索”和“示例选择”两个实际瓶颈。

## 2. 代码框架

公开检索未发现稳定官方 GitHub 仓库，因此按论文方法抽象代码框架：

```text
Input financial report / conversation
-> generative retriever
-> selected facts from text/table
-> dynamic example selector
-> context-aware PoT prompt
-> LLM generates executable program
-> execution engine
-> answer and execution accuracy
```

建议 MedicalGPT 实现为以下模块：

| 模块 | 功能 |
| --- | --- |
| `evidence_retriever` | 根据 question 从 text/table/history 中抽取相关 facts |
| `example_selector` | 从训练集中选相似 program/operator/example |
| `pot_prompt_builder` | 构造含 retrieved facts + dynamic examples 的 PoT prompt |
| `python_executor` | 执行模型生成 Python Program |
| `execution_evaluator` | 计算 answer/execution accuracy |

## 3. 实验方法

FINDER 的实验重点：

- 数据集：FinQA、ConvFinQA。
- 方法对比：CoT、ToT、PoT、静态 few-shot PoT、动态示例 PoT。
- 关键变量：是否使用 generative retriever、是否动态选择 in-context examples、是否使用 context-aware PoT。
- 指标：execution accuracy。

方法核心不是训练一个新模型，而是优化 inference-time pipeline：

```text
better evidence + better examples + executable program
```

## 4. 如何参考到 MedicalGPT

### 4.1 证据检索

当前 MedicalGPT processor 已有 `aligned_evidence`、`evidence_visible_in_prompt` 和 table evidence pruning。可以参考 FINDER 将其升级为显式 retriever：

- 输入：question、pre_text、table、post_text、history。
- 输出：top-k evidence facts。
- 第一版可用规则：question 中年份/指标匹配、program numbers 覆盖、table row/column overlap。
- 第二版再用 LLM/generative retriever。

### 4.2 动态示例选择

可新增 example bank：

```text
operator_pattern: divide(subtract(a,b),b)
question_type: percentage change
evidence_type: table/text/both
dataset: FinQA/ConvFinQA
```

在推理时根据 question/operator/evidence 选 3-5 个示例，而不是固定 few-shot。

### 4.3 PoT 与训练结合

FINDER 是 prompting 框架，但 MedicalGPT 可以把它转成蒸馏框架：

```text
FINDER prompt
-> teacher generates Python Program
-> execute/filter
-> save as ETD SFT sample
```

这能把动态 prompt 的收益沉淀成 student 可学习的数据。

## 5. 对当前项目的实验建议

推荐三组 inference 实验：

| 实验 | 说明 |
| --- | --- |
| static PoT | 固定 4-shot PoT prompt |
| dynamic example PoT | 根据 operator/question type 选示例 |
| retriever + dynamic PoT | 先检索 evidence，再动态示例 PoT |

进一步做蒸馏：

| 实验 | 说明 |
| --- | --- |
| teacher FINDER generation | 用强模型生成 Python Program |
| execution filtering | 保留可执行且答案正确的程序 |
| ETD SFT | 蒸馏给 Qwen2.5-7B |
| GRPO | 用 execution reward 优化 |

## 6. 风险与注意事项

- 动态示例选择会增加推理复杂度和成本，不一定适合最终部署，但适合 teacher data generation。
- generative retriever 可能引入错误证据，需要 evidence verifier。
- 对 ConvFinQA，需要防止 history 中泄漏当前答案。
- 如果只追求 inference-time accuracy，FINDER 很好；如果要训练小模型，应把 FINDER 输出转成可验证蒸馏数据。

## 7. 参考资料

- Khatuya et al. 2025. Program of Thoughts for Financial Reasoning: Leveraging Dynamic In-Context Examples and Generative Retrieval.
- ACL Anthology: https://aclanthology.org/2025.emnlp-main.1577/
- PDF: https://aclanthology.org/2025.emnlp-main.1577.pdf
- FinQA: https://finqasite.github.io/
- ConvFinQA: https://github.com/czyssrs/ConvFinQA
- TIGER-AI-Lab Program-of-Thoughts: https://github.com/TIGER-AI-Lab/Program-of-Thoughts
