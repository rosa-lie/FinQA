# Fin-R1 调研：金融 reasoning SFT + RL 的领域迁移路线

## 1. 文献摘要

论文：`Liu 等 - 2026 - Fin-R1 a large language model for financial reasoning through reinforcement learning.pdf`

Fin-R1 是面向金融复杂推理的 7B 级模型，公开信息显示其基于 Qwen2.5-7B-Instruct，使用 DeepSeek-R1 蒸馏出的金融 reasoning 数据进行 SFT，再通过 GRPO/RL 优化。官方仓库为 https://github.com/SUFE-AIFLM-Lab/Fin-R1，论文页为 https://huggingface.co/papers/2503.16252。

Fin-R1 的核心目标是把 DeepSeek-R1 的 reasoning 能力迁移到金融场景，尤其是 FinQA、ConvFinQA、金融知识、市场洞察、金融代码和金融合规等任务。公开 README 中描述 Fin-R1-Data 约 60k 条，覆盖 ConvFinQA、FinQA、Finance-Instruct-500K、FinCUGE、TFNS、FinanceIQ、FinanceQT、Ant_Finance、FinCorpus、FinPEE 等来源。

## 2. 代码框架

Fin-R1 流程可抽象为：

```text
raw financial datasets
-> DeepSeek-R1 distillation
-> answer scoring
-> reasoning scoring
-> good samples for SFT
-> bad/hard samples for RL
-> Qwen2.5-7B-Instruct SFT
-> GRPO with format reward + accuracy reward
-> financial benchmark evaluation
```

| 组件 | 作用 | MedicalGPT 对应 |
| --- | --- | --- |
| DeepSeek-R1 distillation | 生成金融 CoT | `distill/distill_with_teacher.py` |
| answer scoring | 筛掉答案错样本 | `score_distill_candidates.py` 的 answer check |
| reasoning scoring | 检查推理一致性 | 可扩展 program/execution check |
| SFT | 注入金融 reasoning | `training/supervised_finetuning.py` |
| GRPO | 可验证强化学习 | `training/financial_grpo_training.py` |

## 3. 实验方法

Fin-R1 的实验方法包括：

1. 数据蒸馏：使用 DeepSeek-R1 生成金融 CoT reasoning。
2. 两轮筛选：
   - 答案打分：规则匹配或 Qwen2.5-72B-Instruct 判定答案正确性。
   - 推理过程打分：检查内部一致性、术语重叠、步骤数量、逻辑一致性、内容多样性、任务相关性和指令一致性。
3. SFT：使用筛选后的高质量 CoT 数据训练 Qwen2.5-7B-Instruct。
4. RL：在 SFT 模型基础上使用 GRPO，结合格式奖励和准确度奖励。
5. 评测：FinQA、ConvFinQA、Ant_Finance、TFNS、Finance-Instruct-500K 等。

公开结果中，Fin-R1 在 FinQA 和 ConvFinQA 上表现突出，说明“金融领域 CoT 蒸馏 + 可验证 RL”路线有效。

## 4. 如何参考到 MedicalGPT

MedicalGPT 已有 FinQA/ConvFinQA strict-A 数据，不应直接替换为 Fin-R1-Data 风格的长 CoT。推荐借鉴 Fin-R1 的双轮筛选：

```text
teacher candidate
-> answer check
-> reasoning/program check
-> chosen for SFT
-> failed but informative samples for DPO/GRPO
```

当前 `distill/score_distill_candidates.py` 可以扩展：

- `answer_correct`
- `program_consistent`
- `python_program_executable`
- `reasoning_compact`
- `evidence_grounded`

训练路线建议：

| 阶段 | 数据 | 目标 |
| --- | --- | --- |
| SFT1 | FinQA program-supervised strict-A | 学会单轮表文数值推理 |
| SFT2 | ConvFinQA turn-level + FinQA replay | 学会多轮 follow-up reasoning |
| SFT3 optional | Fin-R1-style distilled CoT/ETD | 增强 reasoning 表达 |
| GRPO | hard-but-verifiable FinQA/ConvFinQA | 优化答案正确性和 program execution |

## 5. 对当前项目的具体改造建议

1. 在 distill scoring 中加入“答案+推理/程序”双轮筛选。
2. 不把 rejected 只做成格式差样本，而是构造可学习的错误类型：答案错、程序错、不可执行、硬编码答案。
3. 在 GRPO 数据中优先选择 SFT 模型 pass rate 较低但可判分的样本。
4. 将 Fin-R1 的“bad 数据进入 RL”改造成 MedicalGPT 的 `hard_verifiable_grpo.jsonl`。
5. 与 Fino1/FinCoT 数据结合时，保留 MedicalGPT program 主链，不让外部长 CoT 覆盖 gold program。

## 6. 风险与注意事项

- Fin-R1 主要是 CoT 蒸馏路线，MedicalGPT 的优势是 gold program 和 execution verification，不能退化成只学长 CoT。
- 两轮 judge 若过度依赖 LLM，成本高且可能偏向话术质量。FinQA/ConvFinQA 应优先用 program/exe answer rule-based verifier。
- RL 数据不能太简单；应选择 hard-but-verifiable，而不是普通 SFT 样本重复训练。

## 7. 参考资料

- Fin-R1 GitHub: https://github.com/SUFE-AIFLM-Lab/Fin-R1
- Fin-R1 paper: `Liu 等 - 2026 - Fin-R1 a large language model for financial reasoning through reinforcement learning.pdf`
- Hugging Face paper page: https://huggingface.co/papers/2503.16252
- DeepSeek-R1
- MedicalGPT current pipeline: `run_fingpt_v2.ipynb`
