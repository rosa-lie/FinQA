# 金融程序监督推理数据主链

> FinQA 提供表文混合、带结构化程序监督的金融数值推理样本，用于训练模型的多步计算与证据整合能力；ConvFinQA 则进一步将金融推理任务扩展到多轮对话场景，用于训练模型在连续交互中维持上下文并完成 follow-up reasoning。
> **FinQA 让模型学会“怎么做金融推理”，ConvFinQA 让模型学会“怎么在对话中持续做金融推理”。**

本文档描述当前主线的数据方案，仅覆盖 `FinQA` 与 `ConvFinQA`。目标不是做通用金融 QA 清洗，而是构建一条可验证的 reasoning supervision 链路，使训练样本中的 `evidence / program / answer` 三者一致且可审计。

`FinQA` 是单轮表文数值推理主干；`ConvFinQA` 是在此基础上的多轮 follow-up reasoning 扩展。`fineval / fiqa_qa` 暂不在当前主链范围。

## 数据集描述

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


## 主链定位

两阶段 SFT

| 阶段 | 数据 | 目标 |
| --- | --- | --- |
| SFT-1 | FinQA | 学会单轮表文混合数值推理 |
| SFT-2 | ConvFinQA | 学会多轮 follow-up 推理 |


SFT是基于这样的数据前提：数据由`Evidence + Program + Answer`组成  
1. Evidance：evidence 是对齐后的原始英文证据片段，而不是 JSON 片段或自由解释
2. Programe：原始数据中 `program_re` 稳定存在，可作为唯一 program 来源  
注：模板统一使用英文，不再引入中文包装和四段式分析模板

## 数据处理链路

当前统一入口仍为 `financial_data_router.py`，但数据概念上统一为四层：
- `raw`
  - 原始标注层，只读
  - 保留数据集中的 `program_re / program / steps / gold_ind(s) / answer / exe_ans`
- `normalized`
  - 在不改 raw 的前提下补齐派生字段
  - 提供 evidence 对齐结果、program 派生视图、answer 归一化结果
- `audit`
  - 存放不满足 strict 条件的样本及原因
  - 供人工审查、规则修正、失败归因使用
- `strict`
  - 最终进入训练的高置信样本
  - 保证 `evidence / program / answer` 可闭合验证

## 规范化中间表示

`normalized` 层至少需要包含以下字段：

- `program_raw`
- `program_canonical`
- `program_executable`
- `answer_raw`
- `answer_exe`
- `answer_norm`
- `aligned_evidence`
- `evidence_match_type`
- `audit_flags`

推荐同时保留：

- `question`
- `history_questions`
- `context.pre_text`
- `context.table`
- `context.post_text`

原则很简单：

- raw 字段只负责保留原始标注
- 训练、校验、审计全部依赖派生字段
- 派生字段可以变化，raw 字段不能变化

## SFT 目标格式

当前 strict 数据中的 assistant target 固定为英文三段式：

```text
Evidence:
- ...
- ...

Program: ...
Answer: ...
```

注：这个模板是当前唯一主方案。不再包含下述前置迭代的数据格式：
- `问题分析`
- `Analysis`
- `<think>`
- 自由解释性 reasoning 文本
- JSON-like evidence 片段，如 `{"text_1": "..."}`

## Evidence 规则

evidence 来源于原始 `gold_ind / gold_inds`，但不能原样监督给模型。当前规则是“精确对齐后渲染”，不是“摘要改写”。

### 对齐规则

- 文本证据：对齐到 `pre_text / post_text` 中的具体句子
- 表格证据：对齐到具体 row / col / cell，再渲染成紧凑英文片段

例如，表格证据应渲染为类似：

```text
gas production (bcf), 2008 = 78.9
```

而不是整表复制，也不是原始 JSON dict。

### strict 规则

- strict 中 evidence 只保留 `1-3` 条
- 只保留真正参与 program 构造和 answer 求解的证据
- 若 evidence 无法 `exact` 对齐，则样本进入 `audit`，不进入 `strict`
- strict 中 `json_like_evidence_ratio` 必须为 `0`

### 字段建议

`aligned_evidence` 中每条 evidence 至少包含：

- `evidence_type`
- `raw_id`
- `source_location`
- `rendered_text`
- `match_type`

## Program 规则

因为数据中都包含了`program_re`，所以直接以`program_re`为准，不做回退、不做重建。

允许的规范化仅限于：
- 空格统一
- 逗号与括号格式统一
- 不改变语义的轻量字符串规范

不允许：
- 改写运算结构
- 代数重写
- 回填缺失操作
- 用 `program` 或 `steps` 修复 `program_re`

### audit 规则

- 若 `program_re` 无法解析，样本进入 `audit`
- 若 `program_re` 可解析但执行不稳定，样本进入 `audit`
- 若 `program_re` 执行结果与主监督答案不一致，样本进入 `audit`

## Answer 规则

Answer 采用“可执行真值优先”。

### 固定优先级

1. 若 `program_re` 执行值与 `exe_ans` 一致，则 `answer_norm` 取该一致值
2. 若 `exe_ans` 缺失但 `program_re` 可执行，则 `answer_norm` 取执行值
3. 若 `raw answer` 与执行值冲突，只保留 raw，不作为 strict 真值
4. 若主监督答案无法稳定确定，则样本进入 `audit`

### 字段职责

- `answer_raw`
  - 原始 answer 标注
- `answer_exe`
  - 原始 `exe_ans`
- `answer_norm`
  - strict 中真正用于训练和验证的答案

## ConvFinQA 规则

`ConvFinQA` 在当前主线中保留以下特殊处理：

- final-turn dedupe 逻辑保留
- 历史问题压缩逻辑保留
- 历史问题只进入 prompt context，不进入 evidence supervision

其 strict 标准与 `FinQA` 一致：

- evidence 必须 `exact` 对齐
- `program_re` 必须可执行
- `answer_norm` 必须可稳定确定

## 验收与发布门槛

strict 数据发布前，至少满足以下门槛：

- `raw_program_unchanged_ratio = 100%`
- `exact_evidence_alignment_ratio >= 95%`
- `program_answer_match_ratio >= 98%`
- `json_like_evidence_ratio = 0` in strict
- 人工抽检 100 条中，至少 90 条满足 `evidence / program / answer` 一致

这些门槛的含义如下：

- `raw_program_unchanged_ratio = 100%`
  - 证明预处理没有污染 raw `program_re`
- `exact_evidence_alignment_ratio >= 95%`
  - 证明 evidence supervision 主要由可回指的原始证据构成
- `program_answer_match_ratio >= 98%`
  - 证明 program 与 answer 没有大面积自相矛盾
- `json_like_evidence_ratio = 0`
  - 证明 strict 已完全去除旧版 JSON-like evidence 监督
- 人工抽检 `>= 90/100`
  - 证明整条监督链在人工视角下也成立

## 当前不在本轮范围

以下内容不作为当前 `fin_datasets.md` 主方案的一部分：

- `fineval / fiqa_qa` 的详细处理策略
- 四段式 `问题分析 / 关键证据 / 推理程序 / 最终答案`
- 中文默认 SFT 模板
- 基于 `program` 回退或 `steps` 重建的 program 方案
- `<think>` / CoT 作为原始 SFT 监督

如需蒸馏、DPO、GRPO 等后续训练策略，应建立在当前 `strict` 数据稳定之后，再单独设计。
