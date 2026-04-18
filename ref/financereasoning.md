# FinanceReasoning 调研：更可信、全面和困难的金融数值推理 benchmark

## 1. 文献摘要

论文：`Tang 等 - 2025 - FinanceReasoning benchmarking financial numerical reasoning more credible, comprehensive and challe.pdf`

FinanceReasoning 是一个面向大型 reasoning models 的金融数值推理 benchmark，目标是让评测更可信、更全面、更具挑战性。论文指出，现有金融数值推理 benchmark 存在题目错误、评测标准不一致、难度覆盖不足、缺少可执行解法等问题。

公开 Hugging Face 页面为 https://huggingface.co/datasets/BUPT-Reasoning-Lab/FinanceReasoning，论文页为 https://huggingface.co/papers/2506.05828。数据集预览显示其字段包括 `question`、`context`、`statistics`、`difficulty`、`ground_truth`、`source`、`python_solution`、`question_id`、`level` 等。

这对 MedicalGPT 特别重要，因为 `python_solution` 可以作为天然 PoT/EoT benchmark。

## 2. 代码框架

FinanceReasoning 更像 benchmark/data release，而不是训练代码框架。其可抽象为：

```text
financial documents / public financial QA datasets
-> question refinement
-> Python solution annotation
-> difficulty/statistics computation
-> benchmark split
-> model generation
-> numerical answer evaluation
```

数据中的 `statistics` 可用于难度感知训练和评测：

- `number_statistics`
- `operator_statistics`
- `code_statistics`
- `difficulty`
- `level`

## 3. 实验方法

FinanceReasoning 的实验重点：

- 更新和修正部分公开金融数值问题。
- 标注详细 Python solutions。
- 构造更严格的 numerical evaluation。
- 评估大 reasoning models 在不同难度、不同金融主题、不同运算复杂度上的表现。

对 MedicalGPT 来说，它可用于外部 benchmark、Python Program execution eval、difficulty-aware sampling 和 hard-but-verifiable GRPO 数据选择。

## 4. 如何参考到 MedicalGPT

### 4.1 作为评测集

新增 `evaluation/evaluate_finance_reasoning.py` 或扩展现有 `evaluate_financial_benchmarks.py`：

```text
load FinanceReasoning
-> build prompt from question + context
-> model generate Answer / Normalized Answer / Python Program
-> parse final answer
-> compare with ground_truth
-> optional: execute generated Python Program
```

### 4.2 作为 PoT/ETD 数据

FinanceReasoning 的 `python_solution` 可转成：

```text
Python Program:
...
answer = ...

Normalized Answer:
ground_truth
```

这比 FinCoT/Fino1 CoT-only 更适合作为 PoT supervision。

### 4.3 难度感知训练

使用 `difficulty` 和 `statistics`：

| 字段 | 用法 |
| --- | --- |
| `difficulty` | 选择 hard-but-verifiable GRPO 样本 |
| `operator_statistics` | 分层评估加减乘除/多运算 |
| `number_statistics` | 识别多数字干扰题 |
| `code_statistics` | 衡量 program complexity |
| `source` | 避免和 FinQA/ConvFinQA test 污染 |

## 5. 对当前项目的实验建议

1. 先作为外部 benchmark，不进入训练。
2. 报告 overall accuracy、by difficulty、by operator count、by source dataset、python execution parse/execute rate。
3. 若 benchmark 结果稳定，再抽取 train-like 部分作为 PoT SFT 或 GRPO 数据。
4. 与 FinQA/ConvFinQA 做互补：FinQA 测原始金融表文 reasoning，ConvFinQA 测多轮 follow-up，FinanceReasoning 测更严格的 Python-solution numerical reasoning。

## 6. 风险与注意事项

- Hugging Face dataset viewer 当前提示列 schema 混杂，加载时可能需要按文件路径分别读取，而不是直接 `load_dataset()` 默认配置。
- 数据包含 documents/functions/question files，必须避免把 documents 当 QA 样本加载。
- 若作为训练数据，必须确认 train/test split 和 source 去污染。
- `python_solution` 可能使用不同变量名和风格，需要统一执行器接口。

## 7. 参考资料

- Tang et al. 2025. FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging.
- Hugging Face paper page: https://huggingface.co/papers/2506.05828
- Dataset: https://huggingface.co/datasets/BUPT-Reasoning-Lab/FinanceReasoning
- FinanceReasoning tree: https://huggingface.co/datasets/BUPT-Reasoning-Lab/FinanceReasoning/tree/main
