# DianJin-R1 调研：结构化 CoT SFT 与双重 GRPO 奖励

## 1. 文献摘要

论文：`Zhu 等 - 2025 - DianJin-R1 evaluating and enhancing financial reasoning in large language models.pdf`

DianJin-R1 是通义点金团队提出的金融 reasoning-enhanced 框架，面向金融领域中的专业知识、数值计算和合规推理。公开模型包括 DianJin-R1-7B 和 DianJin-R1-32B，基座分别为 Qwen2.5-7B-Instruct 和 Qwen2.5-32B-Instruct。数据集为 DianJin-R1-Data，构建自 CFLUE、FinQA 和专有 Chinese Compliance Check（CCC）语料；其中公开部分包括增强版 CFLUE 和 Fin-QA。

核心路线：

```text
reasoning-augmented supervision
-> SFT with <think>...</think><answer>...</answer>
-> GRPO
-> format reward + accuracy reward
```

## 2. 代码框架

公开资源：

| 资源 | 链接 | 用途 |
| --- | --- | --- |
| qwen-dianjin GitHub | https://github.com/aliyun/qwen-dianjin | 项目集合与模型/数据入口 |
| DianJin-R1-Data | https://huggingface.co/datasets/DianJin/DianJin-R1-Data | 公开增强 CFLUE 与 Fin-QA 数据 |
| DianJin-R1-7B | https://huggingface.co/DianJin/DianJin-R1-7B | 7B 模型 |
| DianJin-R1-32B | https://huggingface.co/DianJin/DianJin-R1-32B | 32B 模型 |

DianJin-R1 的框架可抽象为：

```text
CFLUE MCQ / CFLUE OE / FinQA / CCC
-> DeepSeek-R1 generate CoT
-> GPT-4o or rule verifier
-> SFT data in <think>/<answer> format
-> Qwen2.5 SFT
-> GRPO with dual reward
-> financial + general reasoning benchmark
```

## 3. 实验方法

- CFLUE MCQ：用 DeepSeek-R1 生成 CoT 和答案，通过 gold answer 校验。
- CFLUE OE：先用 GPT-4o 将选择题转开放题，再用 DeepSeek-R1 生成 CoT，GPT-4o 校验答案和 reasoning 是否一致。
- FinQA：开放式金融数值推理题，用 DeepSeek-R1 生成 CoT 和答案，再用 GPT-4o 筛选正确样本。
- CCC：金融合规场景，因敏感原因不公开。
- SFT target 使用 `<think>...</think>` 和 `<answer>...</answer>`。
- RL 使用 GRPO，reward 包括 format reward 和 accuracy reward。

评测覆盖 CFLUE、FinQA、CCC、MATH-500、GPQA-Diamond，说明 DianJin-R1 同时关注金融任务和通用 reasoning 保持。

## 4. 如何参考到 MedicalGPT

DianJin-R1 使用 `<think>/<answer>`，但 MedicalGPT 当前已统一为：

```text
Evidence:
Program:
Answer:
Normalized Answer:
```

因此不建议直接改成 `<think>/<answer>`。更好的方式是吸收“结构化 reasoning + final answer”思想，扩展为：

```text
Evidence:
Reasoning:
Program:
Python Program:
Answer:
Normalized Answer:
```

DianJin-R1 用 LLM verifier 校验 CoT。MedicalGPT 可以更强：

- FinQA/ConvFinQA 有 gold program，优先 program execution 校验。
- 外部 CoT 数据如 FinCoT/Fino1，可参考 DianJin-R1 的 verifier 思路。
- 中文金融泛化数据如 CFLUE，可采用 DianJin-R1 的 open-ended conversion 和 answer verifier。

MedicalGPT 可复用“format reward + accuracy reward”，但应升级为：

| reward | 来源 |
| --- | --- |
| format reward | 是否包含 v2 anchors |
| answer reward | `Normalized Answer` 是否正确 |
| program reward | DSL/Python Program 是否可执行 |
| evidence reward | 关键数字是否出现在 evidence/context |

## 5. 对当前项目的实验建议

1. 用 DianJin-R1-Data 的公开 Fin-QA 部分作为外部 CoT 补充数据。
2. 对 CFLUE 做中文金融泛化 benchmark，并少量作为 open-ended CoT SFT 补充。
3. 在 GRPO 中复现双重 reward baseline：only format + answer、answer + program execution、answer + program execution + evidence grounding。
4. 比较 MedicalGPT 的 program-supervised schema 与 `<think>/<answer>` schema。

## 6. 风险与注意事项

- `<think>` 长 CoT 可能污染当前短结构化 target，建议作为实验分支而非主链。
- DianJin-R1 的 CCC 不公开，不能复现完整数据分布。
- GPT-4o verifier 成本高，FinQA/ConvFinQA 应尽量用 rule/program verifier。
- 中文 CFLUE 和英文 FinQA 任务差异大，混合训练需要控制比例。

## 7. 参考资料

- DianJin-R1 paper: `Zhu 等 - 2025 - DianJin-R1 evaluating and enhancing financial reasoning in large language models.pdf`
- Hugging Face paper page: https://huggingface.co/papers/2504.15716
- Qwen DianJin GitHub: https://github.com/aliyun/qwen-dianjin
- DianJin-R1-Data: https://huggingface.co/datasets/DianJin/DianJin-R1-Data
- DianJin-R1-7B: https://huggingface.co/DianJin/DianJin-R1-7B
- DianJin-R1-32B: https://huggingface.co/DianJin/DianJin-R1-32B
