
以finqa为例
[finqa](/root/autodl-tmp/data/financial_reasoning/raw/finqa/train.json)
[example_finqa](/root/MedicalGPT/docs/example_finqa.json)

样本本质是三层信息：
- 原始材料（text + table）（`supporting fact：text_1`）
- 可验证推理（program + steps）（`gold program：divide(3.8, divide(100,100))`）
- 最终答案（answer）（`gold answer：3.8`）

## 数据处理迭代过程
### 1. **第一版失败（sft逊于base）**的原因
案例（/root/autodl-tmp/data/financial_reasoning/clean/train_sft1_clean_strict.jsonl）
```json
{
    "source_dataset": "FinQA",
    "task_type": "financial_table_text_reasoning",
    "record_id": "ADI/2009/page_49.pdf-1",
    "metadata": {
        "program": "divide(3.8, divide(100, 100))",
        "gold_inds": {
            "text_1": "if libor changes by 100 basis points , our annual interest expense would change by $ 3.8 million ."
        }
    },
    "conversations": [
        {
            "from": "human",
            "value": "你是一名金融表文混合推理助手。请结合文本、表格和问题，给出可执行的推理程序与最终答案。\n\n材料（表格前文本）：\n- interest rate to a variable interest rate based on the three-month libor plus 2.05% ( 2.05 % ) ( 2.34% ( 2.34 % ) as of october 31 , 2009 ) .\n- if libor changes by 100 basis points , our annual interest expense would change by $ 3.8 million .\n- foreign currency exposure as more fully described in note 2i .\n- in the notes to consolidated financial statements contained in item 8 of this annual report on form 10-k , we regularly hedge our non-u.s .\n- dollar-based exposures by entering into forward foreign currency exchange contracts .\n- the terms of these contracts are for periods matching the duration of the underlying exposure and generally range from one month to twelve months .\n\n表格：\n | october 31 2009 | november 1 2008\nfair value of forward exchange contracts asset ( liability ) | $ 6427 | $ -23158 ( 23158 )\nfair value of forward exchange contracts after a 10% ( 10 % ) unfavorable mov... | $ 20132 | $ -9457 ( 9457 )\nfair value of forward exchange contracts after a 10% ( 10 % ) favorable movem... | $ -6781 ( 6781 ) | $ -38294 ( 38294 )\n\n材料（表格后文本）：\n- fair value of forward exchange contracts after a 10% ( 10 % ) unfavorable movement in foreign currency exchange rates asset ( liability ) .\n- .\n- .\n- .\n- .\n- .\n\n问题：what is the the interest expense in 2009?\n\n请按以下结构作答：\n问题分析：...\n关键证据：\n- ...\n推理程序：...\n最终答案：..."
        },
        {
            "from": "gpt",
            "value": "问题分析：先定位题目涉及的财务指标，再根据给定程序完成数值计算。\n关键证据：\n- text_1: if libor changes by 100 basis points , our annual interest expense would change by $ 3.8 million .\n推理程序：divide(3.8, divide(100, 100))\n最终答案：3.8"
        }
    ]
}
```
其中数据处理为：
```
"问题分析：先定位题目涉及的财务指标，再根据给定程序完成数值计算。\n关键证据：\n- text_1: if libor changes by 100 basis points , our annual interest expense would change by $ 3.8 million .\n推理程序：divide(3.8, divide(100, 100))\n最终答案：3.8"
```


1. 问题分析 + 关键证据：不可验证 + 无信息增益 👉 模型学到：写模板句子；复制 evidence；拼接 program token（而不是找对数值+建立运算关系）
2. 推理程序：

### 2. distill数据存在的问题

> 当前的任务场景不是要distill long cot，不是通用的reasoning模型，而是domain-specific + numeric reasoning的金融数值推理模型，要 distill“结构化推理能力”，而不是 distill“自然语言思维链”。

**当前存在的问题：**

`<think>自然语言推理</think>`

```
"<think>\n问题：计算2009年的利息费用。\n关键信息：材料中指出，“如果LIBOR变化100个基点，我们的年利息费用将变化380万美元。”\n推理：该陈述直接给出了利息费用的变化量。问题询问利息费用，因此我们引用这个变化量作为答案，即380万美元。\n</think>\n<answer>\n3.8"
```
teacher 的 CoT 是不可控噪声，甚至产生了幻觉。

用“不可验证的自然语言推理”去训练一个“需要可执行数值推理”的模型

**distill 的作用**

teacher 可以补充：
- 哪些 text 是关键证据
- 哪些数值参与计算
- 如何构造 program

问题在于当前 distill 的是“表达”，不是“推理”，应该distill teacher 的“结构化决策”

**改进的方向：** input → teacher生成 structured program


### 3. 改进的方向

FinQA 的核心 supervision 是 program，不是 CoT，不是 explanation  
这个任务本质是：从表文 → 生成可执行程序 → 执行得到答案  
FinQA 应该当成“Program Learning + Answer Prediction”任务，而不是 CoT / QA 任务  

直接回归到最初的 program + answer 的状态

---

TODO：

**对raw进行的预处理**

1. 规范化中间层：数值归一化（单位；scale）
2. Evidance对齐（精准找到evidance）：把 gold_ind / gold_inds 从“摘要输入”改成“精确对齐后渲染”。
    - 文本证据：对齐到原始 pre_text/post_text 句子
    - 表格证据：对齐到具体 row / col / cell，再渲染成紧凑文本片段
    - strict 中 evidence 只保留 1-3 条
    - evidence 若无法 exact 对齐，样本进 audit，不进 strict
    - strict 中禁止出现原始 JSON-like evidence
3. program：使用`program_re`
    - 如果源数据不存在`program_re`直接舍弃

**得到 program-supervised reasoning dataset**

```
Evidence:
- ...
- ...
Program: {program_canonical}
Answer: {answer_norm}
```

**筛选原则：**




### 展望：CoT的用途

在 GRPO（或 PPO 类方法）里，你可以真正利用 CoT，但方式是：模型内部生成 CoT（类似 reasoning tokens），但：

**✅ 方式1：隐式 CoT（推荐）：** 不输出；不作为监督；只影响 reward

**✅ 方式2：CoT consistency reward：** 生成多条推理路径（不同 CoT）；检查是否得到一致答案
- 一致 → reward高；
- 这是“self-consistency”的 RL 版本

**⚠️ 不推荐：显式 CoT 输出**： 强制：`<think>...</think>`。问题仍然存在：不可验证；容易 reward hacking（写漂亮但错）