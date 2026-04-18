# ETD：CoT、PoT、EoT 组合蒸馏路线调研

## 1. 背景与定位

本文件聚焦 reasoning distillation 方向，尤其是如何避免只蒸馏 CoT。用户指出的关键点非常重要：Distilling Mathematical Reasoning Capabilities into Small Language Models（2024）提出 Ensemble Thoughts Distillation（ETD），把 CoT、PoT、EoT 组合起来，并明确把 PoT 作为一种可蒸馏的 reasoning representation。

这一路线与 Shridhar 等 2023 的 reasoning distillation 方向一致：小模型不应只学习最终答案，也不应只学习自然语言解释，而应学习更结构化、更可验证、更容易迁移的 reasoning traces。

本项目已有本地论文：

```text
/root/MedicalGPT/ref/Shridhar 等 - 2023 - Distilling reasoning capabilities into smaller language models.pdf
```

ETD 对 MedicalGPT 的核心启发是：

```text
CoT 负责可读推理
PoT 负责可执行计算
EoT 负责执行校验
```

PoT 的价值不是替代 CoT，而是作为低歧义、可执行、可校验的结构化老师轨迹，与文本 CoT 形成互补。

## 2. 核心方法

### CoT / PoT / EoT 对比表

| 类型 | 全称 | 形式 | 优势 | 风险 | MedicalGPT 用法 |
| --- | --- | --- | --- | --- | --- |
| CoT | Chain of Thought | 自然语言推理 | 可读、解释变量选择 | 计算易错、不可执行 | 简短 `Reasoning:` |
| PoT | Program of Thought | DSL/Python 程序 | 可执行、低歧义、可校验 | 可能硬编码答案 | `Program:` + `Python Program:` |
| EoT | Execution of Thought | 执行结果/校验说明 | 连接程序和答案 | 可能模板化 | `Execution Check:` 或 metadata |

ETD 数据不是简单拼接三种文本，而是用不同 representation 互相校验：

1. CoT 是否解释了正确的证据和运算意图。
2. PoT 是否执行得到正确答案。
3. EoT 是否证明执行结果与 gold answer 一致。

### ETD target schema

推荐 MedicalGPT 使用：

```text
Evidence:
- ...

Reasoning:
Use the current and previous values, subtract the previous value from the current value, then divide by the previous value.

Program:
divide(subtract(6823, 6161), 6161)

Python Program:
cash_current = 6823
cash_previous = 6161
ans = (cash_current - cash_previous) / cash_previous

Execution Check:
ans = 0.10745, which matches Normalized Answer.

Answer: 10.745%
Normalized Answer: 0.10745
```

训练时可以保留 `Execution Check:`，推理时可选择不要求输出该段，只在 evaluator 内部执行校验。

## 3. 与当前 MedicalGPT 的关系

当前主链：

```text
Evidence:
- ...

Program: ...
Answer: ...
Normalized Answer: ...
```

它已经是 PoT-friendly 的，但还不是完整 ETD：

- 缺少显式简短 CoT，即 `Reasoning:`。
- 缺少 Python-level PoT，即 `Python Program:`。
- 缺少 EoT，即 execution check。
- GRPO reward 还没有把 answer correctness、program execution、reasoning compactness 组合起来。

因此 ETD 应作为当前 v2 主链的上层扩展，而不是替换现有 strict-A 数据。

推荐新增：

```text
dual_answer_sft: 当前 baseline
program_executor_sft: 只输出 program，由执行器算答案
ensemble_thought_sft: CoT + PoT + EoT + Answer
```

## 4. 可落地改造方案

### 数据构造

对 FinQA/ConvFinQA：

1. 使用现有 `program_canonical` 作为 DSL PoT。
2. 将 DSL 转成 Python Program。
3. 执行 DSL 和 Python Program，校验结果与 `answer_norm` 一致。
4. 生成简短 CoT：
   - 可由规则模板生成，例如根据 operator sequence 写一句操作说明。
   - 后续可由 teacher 生成，但必须用 program/answer 校验。
5. 生成 EoT：
   - `ans = ..., matches Normalized Answer`
   - 该字段可只进入 metadata，也可进入 SFT target。

对 Fino1/FinCoT：

1. 使用现有 reasoning path 作为 CoT。
2. 如果能和本地 FinQA 样本对齐，则补 gold program 和 Python Program。
3. 如果不能对齐，则标记：
   ```text
   Program: N/A
   Python Program: N/A
   ```
4. 这类样本只参与 CoT SFT，不参与 program reward。

### 过滤规则

保留样本必须满足：

- `Normalized Answer` 可解析。
- DSL Program 可执行，若有。
- Python Program 可执行，若有。
- Python `ans` 与 `Normalized Answer` 一致。
- CoT 不超过长度阈值，避免长篇模板。
- CoT 中至少包含关键数字或运算意图。

剔除样本：

- program 不可执行。
- program 可执行但答案错误。
- `ans` 直接硬编码最终答案。
- CoT 与答案冲突。
- 缺失 `Normalized Answer`。
- 输出残留 JSON-like evidence。

### DPO/GRPO rejected 构造

高质量 rejected 不应只是格式更差，而应覆盖具体推理错误：

| rejected 类型 | 构造方式 | 价值 |
| --- | --- | --- |
| wrong answer | 改错 `Normalized Answer` | 训练答案校验 |
| wrong operator | `divide` 改成 `multiply` 或去掉 subtract | 训练 program 结构 |
| non-executable program | 删除变量或制造除零 | 训练可执行性 |
| hard-coded answer | `ans = gold_answer` | 防 reward hacking |
| CoT-only | 删除 Program/Python Program | 训练 PoT 格式偏好 |
| verbose CoT | 加入冗长无关解释 | 抑制模板化 reasoning |

### reward 设计

推荐 GRPO reward：

| reward | 权重 | 说明 |
| --- | ---: | --- |
| answer correctness | 0.55 | `Normalized Answer` 与 gold 一致 |
| Python execution correctness | 0.25 | `Python Program` 可执行且 `ans` 正确 |
| DSL program consistency | 0.10 | operator sequence 与 gold program 接近 |
| schema format | 0.07 | anchors 完整 |
| reasoning compactness | 0.03 | 简洁、含关键数字、不过长 |

对 `Program: N/A` 的外部样本：

- 跳过 Python execution 和 program consistency。
- 将 answer correctness 和 format reward 重新归一化。

## 5. 实验设计

推荐实验：

| 实验 | 数据 | 目的 |
| --- | --- | --- |
| CoT-only distill | Fino1/FinCoT reasoning path | 验证只蒸 CoT 的上限和风险 |
| PoT-only distill | FinQA/ConvFinQA program executor | 验证程序监督是否提升可计算性 |
| CoT+PoT SFT | FinQA/ConvFinQA ETD without EoT | 验证双轨迹互补 |
| Full ETD SFT | CoT+PoT+EoT | 验证执行校验监督 |
| Full ETD + GRPO | ETD 后可验证 RL | 验证 reward 能否进一步提升 |

评测：

- final answer accuracy
- normalized answer parse rate
- DSL program accuracy
- Python program execute rate
- Python program answer accuracy
- execution-answer agreement rate
- self-consistency pass@k
- average output length

预期现象：

- CoT-only 可能提升输出可读性，但不一定提升 program accuracy。
- PoT-only 可能提升可执行性，但解释性较弱。
- ETD 应在 answer accuracy 与 execute rate 上同时优于 CoT-only。
- GRPO 的收益应主要体现在复杂多步题和 history-dependent ConvFinQA 样本。

## 6. 风险与注意事项

- ETD 不是让模型输出越长越好。金融数值推理中，短 CoT + 可执行程序通常优于长 CoT。
- EoT 如果只作为文本模板，容易被模型空泛模仿。最好以 evaluator 执行结果为准。
- 对没有 gold program 的外部数据，不要强行生成 PoT 后当 gold。可以作为 silver PoT，但必须执行过滤。
- DPO rejected 需要有真实差异。若 chosen/rejected 太像，偏好学习信号弱。
- GRPO reward 需要防止模型学会硬编码答案或只输出 `Normalized Answer` 而忽略程序。

## 7. 参考资料

- Shridhar et al. 2023. Distilling Reasoning Capabilities into Smaller Language Models.
- Distilling Mathematical Reasoning Capabilities into Small Language Models, 2024.
- Chen et al. 2023. Program of Thoughts Prompting.
- Program-of-Thoughts repo: https://github.com/TIGER-AI-Lab/Program-of-Thoughts
- Fino1 repo: https://github.com/The-FinAI/Fino1
- FinCoT: https://huggingface.co/datasets/TheFinAI/FinCoT
- Fino1 Reasoning Path FinQA: https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA

下一步建议：

1. 新增 `ensemble_thought_sft`。
2. 为 FinQA/ConvFinQA 生成 ETD target。
3. 将 Fino1/FinCoT 作为 CoT supplement 接入。
4. 构造 execution-aware DPO rejected。
5. 将 GRPO reward 改成 answer + execution + program + format 的组合。
