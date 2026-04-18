# Fino1、FinCoT 与 Fino1 Reasoning Path 数据接入调研

## 1. 背景与定位

Fino1 是金融 reasoning-enhanced LLM 方向的近作，重点研究通用 reasoning 能力和 reinforcement learning 能否迁移到金融领域。它的路线可以概括为：构造高质量金融 reasoning path 数据，先做 SFT，再使用 GRPO 做强化学习优化。

本项目不应直接用 Fino1 替换当前 MedicalGPT 主链。当前主链的优势是 FinQA/ConvFinQA 有 gold program、exe answer 和 `Normalized Answer`，更适合可验证数值推理。因此推荐定位为：

- MedicalGPT 当前主链：program-supervised financial numerical reasoning。
- Fino1/FinCoT：外部金融 CoT reasoning path 和 RL 数据补充。
- 结合路线：`Program SFT -> Program + Fino1/FinCoT SFT -> Program + Fino1/FinCoT SFT + GRPO`。

这能同时保留当前项目的可验证 PoT 优势，并吸收 Fino1 的金融 reasoning path 和 GRPO 训练经验。

## 2. 核心方法

Fino1 的核心贡献包括：

- 构造 FinCoT 金融 reasoning corpus。
- 使用 SFT 注入金融推理轨迹。
- 使用 GRPO 进行金融推理强化学习。
- 评估金融 reasoning benchmark，覆盖表格、文本、公式和长上下文推理。

相关资源：

| 资源 | 链接 | 用途 |
| --- | --- | --- |
| Fino1 repo | https://github.com/The-FinAI/Fino1 | 训练路线、SFT/RL 组织方式参考 |
| FinCoT | https://huggingface.co/datasets/TheFinAI/FinCoT | 金融 CoT SFT/RL 数据 |
| Fino1 Reasoning Path FinQA | https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA | GPT-4o 生成的 FinQA reasoning path |
| Fino1 paper | `/root/MedicalGPT/ref/Qian 等 - 2025 - Fino1 on the transferability of reasoning-enhanced LLMs and reinforcement learning to finance.pdf` | 理论和实验依据 |

### 数据字段映射表

| 数据源 | 原字段 | MedicalGPT 字段 | 说明 |
| --- | --- | --- | --- |
| Fino1_Reasoning_Path_FinQA | `Open-ended Verifiable Question` | prompt/question | FinQA 风格可验证问题 |
| Fino1_Reasoning_Path_FinQA | `Complex_CoT` | `Reasoning:` | 作为 CoT 辅助监督 |
| Fino1_Reasoning_Path_FinQA | `Ground-True Answer` | `Normalized Answer:` | 标准答案，需数值规范化 |
| Fino1_Reasoning_Path_FinQA | `Response` | `Answer:` | 可读答案 |
| FinCoT | `Question` | prompt/question | 金融 reasoning prompt |
| FinCoT | `Reasoning_process` | `Reasoning:` | CoT supervision |
| FinCoT | `Final_response` | `Answer:` | 最终回答 |
| FinCoT | `Negative_reasoning_process` | rejected reasoning | DPO/GRPO 负样本来源 |
| FinCoT | `Negative_response` | rejected answer | DPO/GRPO 负样本来源 |

### 训练阶段映射表

| Fino1 阶段 | MedicalGPT 对应阶段 | 推荐处理 |
| --- | --- | --- |
| Reasoning SFT | SFT1/SFT2 后的 external reasoning SFT | 以低比例混入，避免覆盖 program 主链 |
| GRPO RL | `training/financial_grpo_training.py` | 用可验证 answer/program reward 改造 |
| FinCoT RL split | verifiable RL dataset | 作为 GRPO 补充，不替代 FinQA/ConvFinQA |
| Negative samples | DPO rejected / RL low-reward candidates | 结合 execution failure 构造更强负样本 |

## 3. 与当前 MedicalGPT 的关系

当前 MedicalGPT 的 v2 主链是：

```text
FinQA strict-A -> SFT1
ConvFinQA turn-level strict-A + FinQA replay -> SFT2
Target: Evidence / Program / Answer / Normalized Answer
Eval: Normalized Answer first, program_accuracy second
```

Fino1/FinCoT 应作为增量数据层：

```text
core_program_sft:
  FinQA + ConvFinQA gold program supervision

external_reasoning_sft:
  Fino1_Reasoning_Path_FinQA + FinCoT SFT split

verifiable_rl:
  FinQA + ConvFinQA + FinCoT RL split
```

核心原则：

- 当前 `Program + Answer + Normalized Answer` 主链不替换。
- Fino1 数据优先提供 CoT reasoning path。
- 如果外部样本能和本地 FinQA id 对齐，再补齐本地 gold `Program` 和 `Python Program`。
- 无法对齐的外部样本不强行伪造 PoT，写 `Program: N/A` 或作为 CoT-only 样本。

## 4. 可落地改造方案

### 新增数据族

新增 processor：

- `financial_data_processors/families/fino1_finqa_path.py`
- `financial_data_processors/families/fincot.py`

并在 family registry 和 router 中注册。

外部样本 target 推荐：

```text
Evidence:
- Not provided in external reasoning path.

Reasoning:
...

Program: N/A

Python Program: N/A

Answer: ...
Normalized Answer: ...
```

若能和本地 FinQA gold program 对齐，则改为：

```text
Evidence:
- ...

Reasoning:
...

Program:
divide(...)

Python Program:
...
ans = ...

Answer: ...
Normalized Answer: ...
```

### Notebook 配置

在 `run_fingpt_v2.ipynb` 中新增：

```python
USE_FINO1_EXTERNAL_DATA = True
FINO1_FINQA_PATH_DATASET = "TheFinAI/Fino1_Reasoning_Path_FinQA"
FINCOT_DATASET = "TheFinAI/FinCoT"
EXTERNAL_REASONING_RATIO = 0.25
FINCOT_SFT_MAX_ROWS = 3000
FINCOT_RL_MAX_ROWS = 1500
```

缓存路径：

```text
/root/autodl-tmp/data/financial_reasoning_v2/raw/fino1_finqa_path/train.jsonl
/root/autodl-tmp/data/financial_reasoning_v2/raw/fincot/sft.jsonl
/root/autodl-tmp/data/financial_reasoning_v2/raw/fincot/rl.jsonl
```

混合产物：

```text
/root/autodl-tmp/data/financial_reasoning_v2/clean/train_external_reasoning_sft.jsonl
/root/autodl-tmp/data/financial_reasoning_v2/clean/train_sft1_dual_plus_fino1.jsonl
/root/autodl-tmp/data/financial_reasoning_v2/clean/train_sft2_dual_plus_fino1.jsonl
```

## 5. 实验设计

推荐实验组：

| 实验 | 数据 | 目的 |
| --- | --- | --- |
| `base` | Qwen2.5-7B-Instruct | 原始基座 |
| `program_sft` | 当前 FinQA/ConvFinQA strict-A | 当前主链 baseline |
| `program_plus_fino1_sft` | 当前主链 + Fino1 FinQA path | 验证 FinQA reasoning path 是否增益 |
| `program_plus_fincot_sft` | 当前主链 + FinCoT SFT | 验证更广金融 CoT 是否增益 |
| `program_plus_fino1_sft_grpo` | 上述 SFT + verifiable GRPO | 验证 Fino1-style RL |

默认混合比例：

```text
core MedicalGPT program data: 75%
external Fino1/FinCoT CoT data: 25%
```

如果 ConvFinQA 明显遗忘，则降到：

```text
core: 90%
external: 10%
```

评测：

- FinQA answer accuracy
- ConvFinQA answer accuracy
- program accuracy
- normalized answer parse rate
- Python Program execute rate，如果启用 ETD
- CFLUE 只作为中文金融泛化补充，不作为主指标

## 6. 风险与注意事项

- Fino1/FinCoT 的 CoT 不等于可验证 program supervision。不能因为 reasoning 看起来合理就直接覆盖 gold program 主链。
- 外部数据可能和 FinQA test/dev 有重叠，需要做 decontamination。
- 外部 reasoning path 较长，可能增加 SFT loss 平滑但降低 generation benchmark，因此要限制长度和比例。
- FinCoT 覆盖任务更广，可能引入非数值题。进入 GRPO 的样本必须能客观判分。
- Fino1-style GRPO 的 reward 不应只奖励格式，应以 `Normalized Answer` correctness 和 program execution correctness 为核心。

## 7. 参考资料

- The-FinAI/Fino1: https://github.com/The-FinAI/Fino1
- TheFinAI/FinCoT: https://huggingface.co/datasets/TheFinAI/FinCoT
- TheFinAI/Fino1_Reasoning_Path_FinQA: https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA
- Qian et al. 2025. Fino1: On the Transferability of Reasoning-Enhanced LLMs and Reinforcement Learning to Finance.
- MedicalGPT current docs: `/root/MedicalGPT/README.md`, `/root/MedicalGPT/docs/fin_datasets.md`, `/root/MedicalGPT/run_fingpt_v2.ipynb`

下一步建议：

1. 新增两个 processor：`fino1_finqa_path` 和 `fincot`。
2. 在 notebook 中增加外部数据下载、缓存、审计和混合单元。
3. 先跑 `program_sft` vs `program_plus_fino1_sft` quick benchmark，再决定是否混入 FinCoT。
4. 将 FinCoT RL split 接入 GRPO 前，先完成 reward schema 改造。
