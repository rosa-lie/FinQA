# 任务场景：“金融对话数值推理模型”


# 数据集

> 我将训练数据划分为“金融语言理解”与“金融推理”两类，其中 FinQA 与 ConvFinQA 构成推理主干。
> FinQA 提供表文混合、带结构化程序监督的金融数值推理样本，用于训练模型的多步计算与证据整合能力；ConvFinQA 则进一步将金融推理任务扩展到多轮对话场景，用于训练模型在连续交互中维持上下文并完成 follow-up reasoning。
> **FinQA 让模型学会“怎么做金融推理”，ConvFinQA 让模型学会“怎么在对话中持续做金融推理”。**

| 阶段 | 数据 | 目标 | 说明 |
| --- | --- | --- | --- |
| SFT-1 | FinQA | 学会表文混合数值推理 | 第一阶段 |
| SFT-2 | ConvFinQA | 学会多轮 follow-up 推理 | 第二阶段 |
| DPO（可选） | 从 SFT 样本自动构造 | 优化表达质量、结构、少废话 | 小规模即可 |
| GRPO（推荐） | 基于可验证 reward | 优化答案正确性和推理格式 | 更贴合 reason model |

## SFT 参考数据集

https://finqasite.github.io/ https://github.com/czyssrs/FinQA:  
FinQA 是一个针对金融报告进行 复杂数值推理 的大规模数据集。它由金融专家基于 S&P 500 公司的收益报告编写，包含 8,281 个问答对，旨在测试模型在处理结构化表格和非结构化文本时的计算能力。
- **核心特点**：
  - **专家标注**：11名美国金融专家标注，时薪$20-50
  - **结构化推理**：每个问题附带推理程序（operation步骤）
  - **多模态输入**：62.43%仅需表格，23.42%仅需文本，14.15%需两者结合
- **数据规模**：8,281样本（训练6,251/验证883/测试1,147）
- **推理复杂度**：59.1%单步推理，32.71%两步，8.19%三步+
- **最佳用途**：评估模型**数值推理能力**、**表格理解能力**

https://huggingface.co/datasets/AdaptLLM/ConvFinQA/:  
- **核心特点**：多轮对话 + 金融数值推理
- **最佳用途**：评估/训练模型在连续追问中的推理一致性与上下文保持

---


# SFT 

## 数据处理（已补充）

当前数据处理已统一为“**单入口路由 + family 独立处理器**”：

- 统一入口：`domain/financial/financial_data_router.py`
- 路由参数：
  - `--task {sft,dpo}`
  - `--dataset_family {convfinqa_turn,finqa,fineval,fiqa_qa,auto}`
  - `--source_file / --output_file`
- family 处理器目录：`domain/financial/processors/families/`

### 处理规则

1. `finqa`
- 转为**结构化 SFT 样本：`问题分析 / 关键证据 / 推理程序 / 最终答案`**
- `关键证据` **由 `gold_inds` 摘要生成，不再直接输出原始 JSON 片段** （即清除了数据集中的结构化格式转化成人类自然语言）
- DPO 基于同一 gold 样本构造 chosen/rejected

2. `convfinqa_turn`
- 转为**多轮对话数值推理 SFT 样本（历史对话 + 表文上下文 + 当前问题）**
- 历史问题**先压缩**，再写入 prompt，**避免上下文过长** （每条按 `max_context_chars` 截断）
- `关键证据` 由 `gold_ind/gold_inds` 摘要生成，不再直接输出原始 JSON 片段（即清除了数据集中的结构化格式转化成人类自然语言）
- **默认只保留最终轮**（`--convfinqa_keep_final_only=true`）
- 去重主键：`(filename, qa.question, qa.program, qa.answer)`
- 每个 key **仅保留 `turn_ind` 最大样本**；缺失时回退 `id` 后缀 `_N`
- DPO 与 SFT 共享同一去重口径，避免分布不一致

3. `fineval` / `fiqa_qa`
- 各自独立处理器，维持对应模板与输出结构

```
"问题分析：需要从财报文本和表格中定位相关指标，并依据程序完成数值计算。 关键证据： - {"text_1": "if libor changes by 100 basis points , our annual interest expense would change by $ 3.8 million ."} 推理程序：divide(3.8, divide(100, 100)) 最终答案：3.8"
⬇️
"问题分析：先定位题目涉及的财务指标，再根据给定程序完成数值计算。\n关键证据：\n- text_1: if libor changes by 100 basis points , our annual interest expense would change by $ 3.8 million .\n推理程序：divide(3.8, divide(100, 100))\n最终答案：3.8"

"问题分析：这是一个需要结合历史对话与财务材料进行数值推理的问题。 关键证据： - {"table_1": "the net sales of 2002 is $ 5742 ; the net sales of 2001 is $ 5363 ; the net sales of 2000 is $ 7983 ;"} 推理程序：divide(subtract(5363, 7983), 7983) 最终答案：-32%"
⬇️
"问题分析：需要结合当前问题、历史对话和财务材料定位指标后再做数值计算。\n关键证据：\n- table_1: the net sales of 2002 is $ 5742 ; the net sales of 2001 is $ 5363 ; the net sales of 2000 is $ 7983 ;\n推理程序：divide(subtract(5363, 7983), 7983)\n最终答案：-32%"
```

### 兼容说明

- 旧命令 `fin_to_sharegpt.py` / `fin_to_dpo_pairs.py` 仍可用
- 两者已改为兼容壳层，内部转发到统一路由

---

## SFT v2 数据原则

当前不再预设“联合训练优于分阶段训练”。在旧版 benchmark 中，`sft_merged` 低于 `base`，说明**原始 SFT target 设计存在问题**。

SFT v2 的目标是先修正监督信号，再重新比较 `SFT-1 / SFT-2 / Joint SFT`：
- **不把原始 `gold_ind/gold_inds` JSON 直接监督给模型**
- 保留可验证 `推理程序` 和 `最终答案`
- 对 `ConvFinQA` 历史问题**做压缩，减少长上下文噪声**

## 两阶段 SFT 方案

- **SFT-1：FinQA**（先学会表文混合数值推理）
- **SFT-2：ConvFinQA**（再学会多轮对话 follow-up 推理）

> FinQA 更像“推理教材”，因为它的单题结构更清楚、监督更强，适合先把模型拉进“会算、会推”的状态；  
> ConvFinQA 更像“真实场景”，因为它把推理放进了连续对话里，适合第二阶段把模型从“会做题”推进到“会连续追问下做题”。

### Notebook 中的关键对齐

- `SFT1_DATA_SPECS`：仅 `finqa_train`
- `SFT2_DATA_SPECS`：仅 `fingpt_convfinqa_train`
- 清洗链路保持一致：`clean -> audit -> strict`
- 阶段二不再混入 `fineval/fiqa_qa`，也不做 replay

### SFT v2 审计要求

在重新生成 strict 文件后，训练前至少检查：
- `json_like_evidence_ratio`：答案中残留原始 JSON 证据的比例
- `structured_answer_ratio`：是否稳定包含四个结构锚点
- `avg_prompt_chars` / `avg_answer_chars`：长度是否失控
- preview 样本中是否仍出现 `{"text_` / `{"table_` 一类原始证据块

如果这些指标不通过，不建议直接进入 SFT 训练。

---

# DPO chosen/rejected 数据集构建（文档版 v1）

## 1. 目标与非目标

- 目标：DPO 负责偏好对齐，让已具备推理能力的模型更稳、更规范。
- 非目标：DPO 不承担“教会模型做题”的职责，主干能力学习仍由 SFT-1/SFT-2 完成。
- v1 原则：采用“规则扰动优先”，只使用当前仓库已实现的构造逻辑，不引入额外训练或生成步骤。

## 2. 标准样本格式

DPO 样本输出为 JSONL，每行一个 pair，字段固定如下（与 `python -m domain.financial.financial_data_router --task dpo` 对齐）：

```json
{
  "system": "",
  "history": [],
  "question": "prompt text",
  "response_chosen": "preferred answer",
  "response_rejected": "less preferred answer",
  "source_dataset": "FinQA|ConvFinQA|...",
  "record_id": "sample id",
  "metadata": {}
}
```

- `question`：来自 family 模板化后的 user prompt。
- `response_chosen`：来自 SFT gold 路径（同一条样本的高质量回答）。
- `response_rejected`：由 family 规则构造的“可回答、可比较、但质量更差”回答。
- 训练消费侧字段映射：`training/dpo_training.py` 读取 `system/history/question/response_chosen/response_rejected` 并转为 `prompt/chosen/rejected`。

## 3. rejected 生成规则（v1）

按 family 使用可复现规则：

1. `finqa`
- `chosen`：`build_sft_item` 产出的结构化回答。
- `rejected`：先删除“推理程序”段，再做数值扰动；若无程序段则补 `推理程序：未给出。`
- 目标错误类型：数值错误、程序缺失/不一致。

2. `convfinqa_turn`
- 与 `finqa` 同样的 `chosen/rejected` 生成规则（去程序段 + 数值扰动 + 必要补全）。
- 默认启用 `--convfinqa_keep_final_only=true`，先做最终轮去重，再构造 DPO。
- 去重口径与 SFT 一致：`(filename, qa.question, qa.program, qa.answer)`，避免分布偏移。

3. `fineval` / `fiqa_qa`
- `chosen`：SFT gold 响应。
- `rejected`：模板式弱回答（固定低质量但可回答文本）。
- 风险提示：模板负样本可用于兜底，不宜在主训练集中过高占比，避免退化为“区分正常回答 vs 垃圾回答”。

## 4. pair 难度分层与采样配比

v1 采用三档难度标注口径（用于抽样与验收）：

1. easy
- `rejected` 明显错误（数值偏差大、程序缺失明显）。

2. medium
- `rejected` 看似合理但可验证错误（数值接近但程序不一致，或关键证据缺失）。

3. hard
- `rejected` 表达基本完整，但在程序规范性、历史条件继承或证据精确度上弱于 `chosen`。

建议配比（v1 推荐值）：

- easy:medium:hard = `5:3:2`
- FinQA:ConvFinQA = `1:1`（可在对话目标更强时调整为 `4:6`）
- 若包含 `fineval/fiqa_qa`，其 pair 占比建议 `<= 20%`

## 5. 过滤与质检

### 5.1 过滤规则

- 去重：`question + response_chosen + response_rejected` 完全相同的 pair 仅保留一条。
- 长度：过滤空字段与超长截断后不可比较样本。
- 可比较性：`chosen/rejected` 必须针对同一 `question` 且语义相关。
- 标签稳定性：存在多种等价答案且无法稳定判优的样本剔除。

### 5.2 质检清单（执行前必须通过）

1. 字段完整性：抽样检查所有必填字段存在且类型正确。
2. 负样本质量：`rejected` 不得大面积“胡说八道/与题无关”。
3. 可判别性：人工抽检时能稳定判断 `chosen` 优于 `rejected`。
4. family 对齐：`convfinqa_turn` 的 DPO 与 SFT 使用同一最终轮去重口径。

## 6. 产出文件与统计口径

### 6.1 最小可执行流程

```bash
# 1) raw -> dpo jsonl
python -m domain.financial.financial_data_router \
  --task dpo \
  --source_file <raw.json|raw.jsonl> \
  --output_file <dpo_pairs.jsonl> \
  --dataset_family <finqa|convfinqa_turn|fineval|fiqa_qa|auto> \
  --seed 42

# 2) 训练消费（示例）
python -m training.dpo_training --train_file_dir <dpo_dir> ...
```

流程定义：`raw -> python -m domain.financial.financial_data_router --task dpo -> dpo jsonl -> summary 统计报告 -> python -m training.dpo_training`

### 6.2 summary 字段解释

路由输出统计以 `domain/financial/processors/router.py` 为准：

- 全局：`task, output_file, dataset_family, input_rows, saved_rows, skipped_rows, per_family`
- `convfinqa_turn` 额外：`group_count, dedup_dropped_rows, fallback_selected_rows`

### 6.3 最低通过标准（发布门槛）

1. 每个启用 family 的 `saved_rows > 0`。
2. `skipped_rows / input_rows <= 20%`（`convfinqa_turn` 可放宽至 `35%`，因最终轮去重与样本缺失更常见）。
3. 随机抽检 100 条中，至少 90 条可稳定判定 `chosen` 更优。
4. 三档难度均有覆盖，且任一档占比不低于 `10%`。

---

后续 v2（不在本轮范围）：可加入 SFT checkpoint 采样 rejected、执行器打分和更细粒度对话状态扰动。
