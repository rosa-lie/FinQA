# 任务场景：“金融对话数值推理模型”

## 数据处理（已补充）

当前数据处理已统一为“**单入口路由 + family 独立处理器**”：

- 统一入口：`financial_data_router.py`
- 路由参数：
  - `--task {sft,dpo}`
  - `--dataset_family {convfinqa_turn,finqa,fineval,fiqa_qa,auto}`
  - `--source_file / --output_file`
- family 处理器目录：`financial_data_processors/families/`

### 处理规则

1. `finqa`
- 转为结构化 SFT 样本：`问题分析 / 关键证据 / 推理程序 / 最终答案`
- DPO 基于同一 gold 样本构造 chosen/rejected

2. `convfinqa_turn`
- 转为多轮对话数值推理 SFT 样本（历史对话 + 表文上下文 + 当前问题）
- **默认只保留最终轮**（`--convfinqa_keep_final_only=true`）
- 去重主键：`(filename, qa.question, qa.program, qa.answer)`
- 每个 key 仅保留 `turn_ind` 最大样本；缺失时回退 `id` 后缀 `_N`
- DPO 与 SFT 共享同一去重口径，避免分布不一致

3. `fineval` / `fiqa_qa`
- 各自独立处理器，维持对应模板与输出结构

### 兼容说明

- 旧命令 `fin_to_sharegpt.py` / `fin_to_dpo_pairs.py` 仍可用
- 两者已改为兼容壳层，内部转发到统一路由

---

## 两阶段 SFT 方案（当前版本）

将当前 `run_fingpt_min.ipynb` 调整为：

- **SFT-1：FinQA**（先学会表文混合数值推理）
- **SFT-2：ConvFinQA**（再学会多轮对话 follow-up 推理）

> FinQA 更像“推理教材”，因为它的单题结构更清楚、监督更强，适合先把模型拉进“会算、会推”的状态；ConvFinQA 更像“真实场景”，因为它把推理放进了连续对话里，适合第二阶段把模型从“会做题”推进到“会连续追问下做题”。

### Notebook 中的关键对齐

- `SFT1_DATA_SPECS`：仅 `finqa_train`
- `SFT2_DATA_SPECS`：仅 `fingpt_convfinqa_train`
- 清洗链路保持一致：`clean -> audit -> strict`
- 阶段二不再混入 `fineval/fiqa_qa`，也不做 replay

---

## 数据集

> 我将训练数据划分为“金融语言理解”与“金融推理”两类，其中 FinQA 与 ConvFinQA 构成推理主干。
> FinQA 提供表文混合、带结构化程序监督的金融数值推理样本，用于训练模型的多步计算与证据整合能力；ConvFinQA 则进一步将金融推理任务扩展到多轮对话场景，用于训练模型在连续交互中维持上下文并完成 follow-up reasoning。
> **FinQA 让模型学会“怎么做金融推理”，ConvFinQA 让模型学会“怎么在对话中持续做金融推理”。**

| 阶段 | 数据 | 目标 | 说明 |
| --- | --- | --- | --- |
| SFT-1 | FinQA | 学会表文混合数值推理 | 第一阶段 |
| SFT-2 | ConvFinQA | 学会多轮 follow-up 推理 | 第二阶段 |
| DPO（可选） | 从 SFT 样本自动构造 | 优化表达质量、结构、少废话 | 小规模即可 |
| GRPO（推荐） | 基于可验证 reward | 优化答案正确性和推理格式 | 更贴合 reason model |

## 数据来源

### SFT 参考数据集

https://finqasite.github.io/ https://github.com/czyssrs/FinQA
- **核心特点**：
  - **专家标注**：11名美国金融专家标注，时薪$20-50
  - **结构化推理**：每个问题附带推理程序（operation步骤）
  - **多模态输入**：62.43%仅需表格，23.42%仅需文本，14.15%需两者结合
- **数据规模**：8,281样本（训练6,251/验证883/测试1,147）
- **推理复杂度**：59.1%单步推理，32.71%两步，8.19%三步+
- **最佳用途**：评估模型**数值推理能力**、**表格理解能力**

https://huggingface.co/datasets/AdaptLLM/ConvFinQA/
- **核心特点**：多轮对话 + 金融数值推理
- **最佳用途**：评估/训练模型在连续追问中的推理一致性与上下文保持
