# Chen et al. 2023 Program of Thoughts 论文与仓库复用调研

## 1. 背景与定位

Chen et al. 2023 的 Program of Thoughts Prompting 提出将 reasoning 与 computation 解耦：大语言模型负责把问题表达为 Python 程序，外部解释器负责执行计算。该方法特别适合数学和金融数值推理，因为这些任务的失败点往往不是“理解题意”，而是自然语言 CoT 中的算术错误、单位错误和中间变量混乱。

对应官方仓库是 `TIGER-AI-Lab/Program-of-Thoughts`：

```text
https://github.com/TIGER-AI-Lab/Program-of-Thoughts
```

本项目已有本地论文 PDF：

```text
/root/MedicalGPT/ref/Chen 等 - 2023 - Program of thoughts prompting disentangling computation from reasoning for numerical reasoning task.pdf
```

对 MedicalGPT 来说，这篇论文和仓库的价值不是简单复刻 few-shot PoT prompt，而是提供了三个可迁移思想：

- 用程序表达可计算 reasoning。
- 执行程序获得答案。
- 多候选程序执行后做 self-consistency 投票。

## 2. 核心方法

PoT 官方流程：

```text
Question + text/table context
-> LLM generates Python program
-> safe_execute(program)
-> get ans
-> compare ans with gold answer
-> with self-consistency: sample n programs, execute all, majority vote over ans
```

官方 FinQA prompt 形式大致是：

```text
Read the following text and table, and then write code to answer a question:
...
Question: ...
#Python
...
ans = ...
```

官方仓库中，FinQA few-shot prompt 包含四个手写示例，模型输出 Python 代码，最终变量为 `ans`。self-consistency 模式下，仓库会以 temperature 采样多个程序，执行后用 `Counter` 对可执行答案投票。

### 官方 PoT 仓库组件映射表

| 官方组件 | 功能 | MedicalGPT 可复用方式 |
| --- | --- | --- |
| `run_finqa.py` | FinQA PoT prompt、生成、执行、计分 | 借鉴 prompt 和 self-consistency，不直接复制 OpenAI completion 逻辑 |
| `run_convfinqa.py` | ConvFinQA PoT 评测 | 借鉴多轮金融 QA 的 prompt 和执行投票 |
| `tool.py` | `safe_execute`、`floatify_ans`、`finqa_equal` | 参考执行和答案比较思想，但实现更安全的 AST 执行器 |
| `create_finqa_eval.py` | FinQA 数据格式转换 | 对照当前 processor，不作为主入口 |
| `compute_score*.py` | 输出文件评分 | 当前项目已有 benchmark，可扩展 execution 指标 |

官方 README 报告的金融数据结果包括：

| 数据集 | Greedy EM | Self-Consistency EM |
| --- | ---: | ---: |
| FinQA | 0.647 | 0.682 |
| ConvFinQA | 0.665 | 0.714 |
| TAT-QA | 0.689 | 0.702 |

这些结果说明，PoT + self-consistency 在金融数值推理上有效。

## 3. 与当前 MedicalGPT 的关系

当前 MedicalGPT 已具备比官方 PoT baseline 更强的数据条件：

- FinQA 原始数据中有 gold `program` / `program_re` / `exe_ans`。
- ConvFinQA turn-level 数据中有 `cur_program` / `exe_ans`。
- 当前 processor 已 canonicalize program，并执行 DSL 校验。
- 评测已统计 `program_accuracy`，并优先读取 `Normalized Answer`。

因此 MedicalGPT 不应只做 prompt-time PoT，而应做：

```text
Gold DSL Program
-> DSL execution verification
-> DSL-to-Python PoT conversion
-> Python execution verification
-> SFT / DPO / GRPO / benchmark
```

这比官方 PoT 更适合 distillation，因为每条训练样本可以在进入 SFT 前先被执行校验。

## 4. 可落地改造方案

### MedicalGPT 改造清单

| 改造项 | 说明 |
| --- | --- |
| 新增 `ensemble_thought_sft` | target 同时包含 CoT、DSL Program、Python Program、Answer |
| 新增 `program_re_to_python()` | 将 FinQA/ConvFinQA DSL 转 Python |
| 新增 `safe_execute_python_pot()` | AST 白名单执行 Python Program |
| 扩展 benchmark | 执行模型输出的 Python Program，统计 execute rate 和 answer accuracy |
| 增加 PoT self-consistency | 对 pass@k 采样输出执行投票 |
| 改造 GRPO reward | 用 answer correctness + program execution correctness 作为主 reward |

推荐 target：

```text
Evidence:
- ...

Reasoning:
The question asks for percentage change, so use the new value, old value, and divide the difference by the old value.

Program:
divide(subtract(6823, 6161), 6161)

Python Program:
cash_2013 = 6823
cash_2012 = 6161
ans = (cash_2013 - cash_2012) / cash_2012

Answer: 10.745%
Normalized Answer: 0.10745
```

### 安全执行器设计

不建议直接复制官方 `tool.py` 中的通用 `exec`。MedicalGPT 应实现更严格执行器：

- 允许：
  - 数字常量
  - 变量赋值
  - 四则运算
  - 一元负号
  - `min/max/sum/abs/round`
  - 最终读取 `ans`
- 禁止：
  - `import`
  - 函数定义
  - 类定义
  - 属性访问
  - 文件、网络、系统调用
  - `eval/exec/open`
  - 任意下标和复杂对象

执行结果与 `Normalized Answer` 做 tolerance comparison。

### Self-consistency 设计

复用当前 benchmark 中已有 pass@k 采样能力：

1. 对每个样本生成 `k` 个候选。
2. 从候选中解析 `Python Program:`。
3. 执行所有可执行程序。
4. 对执行答案做数值聚类或 rounding 后投票。
5. 使用投票答案计算 `pot_self_consistency_accuracy`。

## 5. 实验设计

推荐实验顺序：

1. `dual_answer_sft` baseline。
2. `program_executor_sft`：只让模型输出 Program，由外部执行器算答案。
3. `ensemble_thought_sft`：CoT + DSL Program + Python Program + Answer。
4. `ensemble_thought_sft + self-consistency`。
5. `ensemble_thought_sft + GRPO`。

推荐指标：

- `answer_accuracy`
- `program_accuracy`
- `normalized_answer_coverage`
- `python_program_parse_rate`
- `python_program_execute_rate`
- `python_program_answer_accuracy`
- `execution_answer_agreement_rate`
- `pot_self_consistency_accuracy`

最低验收：

- Python Program execute rate 高于 0.90。
- ETD/PoT 不降低 `numeric_parse_rate`。
- FinQA answer accuracy 高于当前 dual baseline。
- ConvFinQA 不能出现明显遗忘。

## 6. 风险与注意事项

- 官方 PoT 的 prompt examples 是 few-shot inference 设计，不是直接面向 SFT 的数据格式。
- 官方 safe execution 比较宽松，训练评测中必须加强安全限制。
- PoT self-consistency 可能提升答案准确率，但也可能掩盖模型单次输出格式不稳定，因此要同时报告 parse/execute rate。
- Python Program 可能硬编码最终答案，需要过滤 `ans = 0.10745` 这类没有中间变量的样本。
- 对 ConvFinQA，多轮 history 中不能泄漏当前答案；PoT history 最好只保留前序 question/program/answer。

## 7. 参考资料

- Chen et al. 2023. Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks.
- TIGER-AI-Lab/Program-of-Thoughts: https://github.com/TIGER-AI-Lab/Program-of-Thoughts
- `run_finqa.py`: FinQA PoT prompt、生成、执行、self-consistency。
- `run_convfinqa.py`: ConvFinQA PoT prompt 与评测。
- `tool.py`: safe execution、answer floatify、FinQA answer equality。
- FinQA: https://github.com/czyssrs/FinQA
- ConvFinQA: https://github.com/czyssrs/ConvFinQA

下一步建议：

1. 新增 DSL-to-Python PoT 转换函数。
2. 新增 AST 白名单执行器。
3. 新增 `ensemble_thought_sft` target。
4. 扩展 benchmark，先支持执行模型输出中的 `Python Program:`。
